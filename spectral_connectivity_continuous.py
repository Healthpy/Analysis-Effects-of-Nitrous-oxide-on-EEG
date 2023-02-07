
from mne_connectivity.viz import plot_sensors_connectivity, plot_connectivity_circle
import mne
from mne_connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import re

def get_sec(time_stp, time_stt, gas_on, gas_off):
    
    h, m, s = time_stp.split(':')
    h1, m1, s1 =  time_stt.split(':')
    h2, m2, s2 =  gas_on.split(':')
    h3, m3, s3 =  gas_off.split(':')
    #total_time = (int(h)-int(h1)) * 3600 + (int(m)-int(m1)) * 60 + int(s)-int(s1)
    gas_time_st = (int(h2)-int(h1)) * 3600 + (int(m2)-int(m1)) * 60 + int(s1)-int(s1) + 1
    gas_time_on = (int(h3)-int(h2)) * 3600 + (int(m3)-int(m2)) * 60 + int(s3)-int(s2) + 1
    gas_time_off = gas_time_st + gas_time_on
    return gas_time_st, gas_time_off



# time_path = r'C:\Users\sqemch\OneDrive - TUNI.fi\Thesis\Notes and Slides\time_stamps.xlsx'
time_path = '/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Notes and Slides/time_stamps.xlsx'
dataframe = pd.read_excel(time_path)
gas_starts = []
gas_stops = []

mainpath = "/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Cleandata/Continuous"

regex = re.compile(r'\d+')

for file in glob(mainpath + '/' + '*.fif'):
    num = [int(x) for x in regex.findall(file)]
    num = num[0]
    i = num - 1
# for i in range(len(dataframe)):
    time_end = str(dataframe.iloc[i,7])
    time_stt = str(dataframe.iloc[i,2])
    gas_on = str(dataframe.iloc[i,3])
    gas_off = str(dataframe.iloc[i,4])
    gas_start, gas_stop = get_sec(time_end, time_stt, gas_on, gas_off)
    gas_starts = round(gas_start/30 + 0.5)
    gas_stops = round(gas_stop/30 + 0.5)
        
# path = r'C:\Users\sqemch\Documents\res\30fulldec\clean_19_epo.fif'
    # path = f"/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Cleandata/Continuous/clean_{x}_epo.fif"


# [1, 4, 7, 11, 19]
# method=['coh', 'plv', 'ciplv', 'ppc', 'pli', 'pli2_unbiased', 'wpli', 'wpli2_debiased']

    real_epochs = mne.read_epochs(file, preload=True)

    indx = []
        
    for a, b in enumerate(real_epochs.selection):
        if b >= gas_starts and b <= gas_stops:
            indx.append(a)
    sfreq = real_epochs.info['sfreq']

    epochs = mne.EpochsArray(real_epochs, real_epochs.info).pick_types(eeg=True)


    method=['wpli']
    #delta , theta, alpha, beta bands
    fmin = (0.5, 4.0, 8.0, 12.0)
    fmax = (4.0, 8.0, 12.0, 40.0)
    
    con_bf = []
    con_dr = []
    con_af = []
    
    vmin = []
    vmax = []
    
    for i in range(len(fmin)):
        con_before = spectral_connectivity_epochs(epochs[0:indx[0]], method=method, 
                                                  sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                                mt_adaptive=False, n_jobs=-1).get_data(
                                                output='dense')[:, :, 0]
        con_during = spectral_connectivity_epochs(epochs[indx[0]:indx[-1]+1], method=method, 
                                                  sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                                mt_adaptive=False, n_jobs=-1).get_data(
                                                    output='dense')[:, :, 0]
        con_after = spectral_connectivity_epochs(epochs[indx[-1]+1:-1], method=method, 
                                                      sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                                    mt_adaptive=False, n_jobs=-1).get_data(
                                                        output='dense')[:, :, 0]
        con_bf.append(con_before)
        con_dr.append(con_during)
        con_af.append(con_after)
        vmin_ = min(conn_con.min() for conn_con in [con_bf[i], con_dr[i], con_af[i]])
        vmax_ = max(conn_con.max() for conn_con in [con_bf[i], con_dr[i], con_af[i]])
        vmin.append(vmin_)
        vmax.append(vmax_)
    


    filenumstr = str(num)
    fig1 = plt.figure(num=None, figsize=(24, 8), facecolor='black')
    plot_connectivity_circle(con_bf[0], n_lines=12, node_names = epochs.ch_names,
                              title='Delta '
                              ' Before $N_{2}O$ (wPLI)', colorbar_pos=(-0.4, 0.0), fontsize_names=10, 
                              fig=fig1, vmin=vmin[0], vmax=vmax[0], colorbar=False, subplot=131)
    
    plot_connectivity_circle(con_dr[0], n_lines=12, node_names = epochs.ch_names,
                              title='Delta '
                              ' During $N_{2}O$ (wPLI)', colorbar_pos=(-0.2, -0.2), fontsize_names=10, 
                              fig=fig1, vmin=vmin[0], vmax=vmax[0], colorbar_size=0.4, fontsize_colorbar=10, subplot=132)
    plot_connectivity_circle(con_af[0], n_lines=12, node_names = epochs.ch_names,
                              title='Delta '
                              ' After $N_{2}O$ (wPLI)', colorbar_pos=(-0.0, -0.2), fontsize_names=10, 
                              fig=fig1, vmin=vmin[0], vmax=vmax[0], colorbar_size=0.4, fontsize_colorbar=10, colorbar=False, subplot=133)
    
    fig1.tight_layout()
    fig1.savefig(filenumstr + '_' + "delta_cont.png")
    
    fig2 = plt.figure(num=None, figsize=(24, 8), facecolor='black')
    plot_connectivity_circle(con_bf[1], n_lines=12, node_names = epochs.ch_names,
                              title='Theta '
                              ' Before $N_{2}O$ (wPLI)', colorbar_pos=(-0.4, 0.0), fontsize_names=10,
                              vmin=vmin[1], vmax=vmax[1], colorbar=False, fig=fig2, subplot=131)
    plot_connectivity_circle(con_dr[1], n_lines=12, node_names = epochs.ch_names,
                              title='Theta '
                              ' During $N_{2}O$ (wPLI)', colorbar_pos=(-0.2, -0.2), colorbar_size=0.4, fontsize_names=10, 
                              vmin=vmin[1], vmax=vmax[1], fig=fig2, subplot=132)
    plot_connectivity_circle(con_af[1], n_lines=12, node_names = epochs.ch_names,
                              title='Theta '
                              ' After $N_{2}O$ (wPLI)', colorbar_pos=(-0.4, 0.0), fontsize_names=10, 
                              vmin=vmin[1], vmax=vmax[1],  colorbar=False, fig=fig2, subplot=133)
    
    fig2.tight_layout()
    fig2.savefig(filenumstr + '_' +"theta_cont.png")
    
    fig3 = plt.figure(num=None, figsize=(24, 8), facecolor='black')
    plot_connectivity_circle(con_bf[2], n_lines=12, node_names = epochs.ch_names,
                              title='Alpha '
                              ' Before $N_{2}O$ (wPLI)', colorbar_pos=(-0.4, 0.0), fontsize_names=10, 
                              vmin=vmin[2], vmax=vmax[2], fig=fig3, colorbar=False, subplot=131)
    plot_connectivity_circle(con_dr[2], n_lines=12, node_names = epochs.ch_names,
                              title='Alpha '
                              ' During $N_{2}O$ (wPLI)', colorbar_pos=(-0.2, -0.2), colorbar_size=0.4, fontsize_names=10, 
                              vmin=vmin[2], vmax=vmax[2], fig=fig3,  subplot=132)
    plot_connectivity_circle(con_af[2], n_lines=12, node_names = epochs.ch_names,
                              title='Alpha '
                              ' After $N_{2}O$ (wPLI)', colorbar_pos=(-0.4, 0.0),  colorbar=False, fontsize_names=10, 
                              vmin=vmin[2], vmax=vmax[2], fig=fig3, subplot=133)
    
    fig3.tight_layout()
    fig3.savefig(filenumstr + '_' +"alpha_cont.png")
    
    fig4 = plt.figure(num=None, figsize=(24, 8), facecolor='black')
    plot_connectivity_circle(con_bf[3], n_lines=12, node_names = epochs.ch_names,
                              title='Beta '
                              ' Before $N_{2}O$ (wPLI)', colorbar_pos=(-0.4, 0.0), fontsize_names=10, 
                              vmin=vmin[3], vmax=vmax[3], fig=fig4, colorbar=False, subplot=131)
    plot_connectivity_circle(con_dr[3], n_lines=12, node_names = epochs.ch_names,
                              title='Beta '
                              ' During $N_{2}O$ (wPLI)', colorbar_pos=(-0.2, -0.2), colorbar_size=0.4, fontsize_names=10,
                              vmin=vmin[3], vmax=vmax[3], fig=fig4, subplot=132)
    plot_connectivity_circle(con_af[3], n_lines=12, node_names = epochs.ch_names,
                              title='Beta '
                              ' After $N_{2}O$ (wPLI)', colorbar_pos=(-0.4, 0.0), fontsize_names=10, 
                              vmin=vmin[3], vmax=vmax[3], fig=fig4, colorbar=False, subplot=133)
    
    fig4.tight_layout()
    fig4.savefig(filenumstr + '_' + "beta_cont.png")