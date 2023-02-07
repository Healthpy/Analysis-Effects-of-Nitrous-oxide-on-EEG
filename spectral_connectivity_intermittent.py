
from mne_connectivity.viz import plot_sensors_connectivity, plot_connectivity_circle
import mne
from mne_connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import re

def get_sec(time_stp, time_stt, gas_on1, gas_off1, gas_on2, gas_off2):

    h, m, s = time_stp.split(':')
    h1, m1, s1 = time_stt.split(':')
    h2, m2, s2 = gas_on1.split(':')
    h3, m3, s3 = gas_off1.split(':')
    h4, m4, s4 = gas_on2.split(':')
    h5, m5, s5 = gas_off2.split(':')

    gas_time_st = (int(h2)-int(h1)) * 3600 + \
        (int(m2)-int(m1)) * 60 + int(s1)-int(s1) + 1
    gas_time_on1 = (int(h3)-int(h2)) * 3600 + \
        (int(m3)-int(m2)) * 60 + int(s3)-int(s2) + 1
    gas_time_off1 = gas_time_st + gas_time_on1
    gas_off1_duration = (int(h4)-int(h3)) * 3600 + \
        (int(m4)-int(m3)) * 60 + int(s4)-int(s3) + 1
    gas_on2_duration = (int(h5)-int(h4)) * 3600 + \
        (int(m5)-int(m4)) * 60 + int(s5)-int(s4) + 1
    gas_time_st2 = gas_time_off1 + gas_off1_duration
    gas_time_off2 = gas_time_st2 + gas_on2_duration

    return gas_time_st, gas_time_off1, gas_time_st2, gas_time_off2

time_path = 'time_stamps.xlsx'
dataframe = pd.read_excel(time_path)
gas_starts = []
gas_stops = []
gas_starts2 = []
gas_stops2 = []

mainpath = "/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Cleandata/Intermittent"

regex = re.compile(r'\d+')

for file in glob(mainpath + '/' + '*.fif'):
    num = [int(x) for x in regex.findall(file)]
    num = num[0]
    i = num - 1

    time_end = str(dataframe.iloc[i, 7])
    time_stt = str(dataframe.iloc[i, 2])
    gas_on1 = str(dataframe.iloc[i, 3])
    gas_off1 = str(dataframe.iloc[i, 4])
    gas_on2 = str(dataframe.iloc[i, 5])
    gas_off2 = str(dataframe.iloc[i, 6])
    gas_start1, gas_stop1, gas_start2, gas_stop2 = get_sec(time_end, time_stt,
                                                           gas_on1, gas_off1,
                                                           gas_on2, gas_off2)
    gas_starts = round(gas_start1/30 + 0.5)
    gas_stops = round(gas_stop1/30 + 0.5)
    gas_starts2 = round(gas_start2/30 + 0.5)
    gas_stops2 = round(gas_stop2/30 + 0.5)

    
    real_epochs = mne.read_epochs(file, preload=True)
    
    indx = []
    indx2 = []
    
    for a, b in enumerate(real_epochs.selection):
        if b >= gas_starts and b <= gas_stops:
            indx.append(a)
        elif b >= gas_starts2 and b <= gas_stops2:
            indx2.append(a)
    
    
    sfreq = real_epochs.info['sfreq']
    
    epochs = mne.EpochsArray(real_epochs, real_epochs.info).pick_types(eeg=True)

    method=['wpli']
    # delta , theta, alpha, beta bands
    fmin = (0.5, 4.0, 8.0, 12.0)
    fmax = (4.0, 8.0, 12.0, 40.0)
    
    con_bf = []
    con_dr1 = []
    con_af1 = []
    con_dr2 = []
    con_af2 = []
    vmin = []
    vmax = []

    for i in range(len(fmin)):
        con_before = spectral_connectivity_epochs(epochs[0:indx[0]+1], method=method,
                                                  sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                                  mt_adaptive=False, n_jobs=-1).get_data(
            output='dense')[:, :, 0]
        con_during1 = spectral_connectivity_epochs(epochs[indx[0]:indx[-1]+1], method=method, sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                                   mt_adaptive=False, n_jobs=-1).get_data(
            output='dense')[:, :, 0]
        con_after1 = spectral_connectivity_epochs(epochs[indx[-1]:indx2[0]+1],
                                                  method=method, sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                                  mt_adaptive=False, n_jobs=-1).get_data(
            output='dense')[:, :, 0]
        con_during2 = spectral_connectivity_epochs(epochs[indx2[0]:indx2[-1]+1], method=method, sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                                   mt_adaptive=False, n_jobs=-1).get_data(
            output='dense')[:, :, 0]
        con_after2 = spectral_connectivity_epochs(epochs[indx2[-1]:-1],
                                                  method=method, sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                                  mt_adaptive=False, n_jobs=-1).get_data(
            output='dense')[:, :, 0]
        con_bf.append(con_before)
        con_dr1.append(con_during1)
        con_af1.append(con_after1)
        con_dr2.append(con_during2)
        con_af2.append(con_after2)
        vmin_ = min(conn_con.min() for conn_con in [con_bf[i], con_dr1[i], con_af1[i], con_dr2[i], con_af2[i]])
        vmax_ = max(conn_con.max() for conn_con in [con_bf[i], con_dr1[i], con_af1[i], con_dr2[i], con_af2[i]])
        vmin.append(vmin_)
        vmax.append(vmax_)
    
    
    
    # plot_sensors_connectivity Circular plots
    filenumstr = str(num)
    
    fig1 = plt.figure(num=None, figsize=(40, 8), facecolor='black')
    plot_connectivity_circle(con_bf[0], n_lines=12, node_names=epochs.ch_names,
                             title='Delta'
                             ' Before $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig1, 
                             vmin=vmin[0], vmax=vmax[0], colorbar=False, subplot=151)
    plot_connectivity_circle(con_dr1[0], n_lines=12, node_names=epochs.ch_names,
                             title='Delta I'
                             ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig1, 
                             vmin=vmin[0], vmax=vmax[0], colorbar=False, subplot=152)
    plot_connectivity_circle(con_af1[0], n_lines=12, node_names=epochs.ch_names,
                             title='Delta I'
                             ' After $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig1,
                             vmin=vmin[0], vmax=vmax[0], colorbar=False, subplot=153)
    plot_connectivity_circle(con_dr2[0], n_lines=12, node_names=epochs.ch_names,
                             title='Delta II'
                             ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig1, 
                             vmin=vmin[0], vmax=vmax[0], colorbar=False, subplot=154)
    plot_connectivity_circle(con_af2[0], n_lines=12, node_names=epochs.ch_names,
                             title='Delta II'
                             ' After $N_{2}O$ (wPLI)', fontsize_names=10, fontsize_colorbar=10,fig=fig1, 
                             vmin=vmin[0], vmax=vmax[0], subplot=155)
    
    fig1.tight_layout()
    fig1.savefig("images\filenumstr" + "_" + "delta_int.png")
    
    fig2 = plt.figure(num=None, figsize=(40, 8), facecolor='black')
    plot_connectivity_circle(con_bf[1], n_lines=12, node_names=epochs.ch_names,
                              title='Theta'
                              ' Before $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig2, 
                              vmin=vmin[1], vmax=vmax[1], colorbar=False, subplot=151)
    plot_connectivity_circle(con_dr1[1], n_lines=12, node_names=epochs.ch_names,
                              title='Theta I'
                              ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig2, 
                              vmin=vmin[1], vmax=vmax[1], colorbar=False, subplot=152)
    plot_connectivity_circle(con_af1[1], n_lines=12, node_names=epochs.ch_names,
                              title='Theta I'
                              ' After $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig2, 
                              vmin=vmin[1], vmax=vmax[1], colorbar=False, subplot=153)
    plot_connectivity_circle(con_dr2[1], n_lines=12, node_names=epochs.ch_names,
                              title='Theta II'
                              ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig2,
                              vmin=vmin[1], vmax=vmax[1], colorbar=False, subplot=154)
    plot_connectivity_circle(con_af2[1], n_lines=12, node_names=epochs.ch_names,
                              title='Theta II'
                              ' After $N_{2}O$ (wPLI)', fontsize_names=10, fontsize_colorbar=10, fig=fig2, 
                              vmin=vmin[1], vmax=vmax[1], subplot=155)
    fig2.tight_layout()
    fig2.savefig("images\filenumstr" + "_" + "theta_int.png")
    
    fig3 = plt.figure(num=None, figsize=(40, 8), facecolor='black')
    plot_connectivity_circle(con_bf[2], n_lines=12, node_names=epochs.ch_names,
                              title='Alpha'
                              ' Before $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig3, 
                              vmin=vmin[2], vmax=vmax[2], colorbar=False, subplot=151)
    plot_connectivity_circle(con_dr1[2], n_lines=12, node_names=epochs.ch_names,
                              title='Alpha I'
                              ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig3, 
                              vmin=vmin[2], vmax=vmax[2], colorbar=False, subplot=152)
    plot_connectivity_circle(con_af1[2], n_lines=12, node_names=epochs.ch_names,
                              title='Alpha I'
                              ' After $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig3, 
                              vmin=vmin[2], vmax=vmax[2], colorbar=False, subplot=153)
    plot_connectivity_circle(con_dr2[2], n_lines=12, node_names=epochs.ch_names,
                              title='Alpha II'
                              ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig3, 
                              vmin=vmin[2], vmax=vmax[2], colorbar=False, subplot=154)
    plot_connectivity_circle(con_af2[2], n_lines=12, node_names=epochs.ch_names,
                              title='Alpha II'
                              ' After $N_{2}O$ (wPLI)', fontsize_names=10, fontsize_colorbar=10, fig=fig3, 
                              vmin=vmin[2], vmax=vmax[2], subplot=155)
    fig3.tight_layout()
    fig3.savefig("images\filenumstr" + "_" + "alpha_int.png")
    
    
    fig4 = plt.figure(num=None, figsize=(40, 8), facecolor='black')
    plot_connectivity_circle(con_bf[3], n_lines=12, node_names=epochs.ch_names,
                              title='Beta'
                              ' Before $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig4, 
                              vmin=vmin[3], vmax=vmax[3], colorbar=False, subplot=151)
    plot_connectivity_circle(con_dr1[3], n_lines=12, node_names=epochs.ch_names,
                              title='Beta I'
                              ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig4, 
                              vmin=vmin[3], vmax=vmax[3], colorbar=False, subplot=152)
    plot_connectivity_circle(con_af1[3], n_lines=12, node_names=epochs.ch_names,
                              title='Beta I'
                              ' After $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig4, 
                              vmin=vmin[3], vmax=vmax[3], colorbar=False, subplot=153)
    plot_connectivity_circle(con_dr2[3], n_lines=12, node_names=epochs.ch_names,
                              title='Beta II'
                              ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig4, 
                              vmin=vmin[3], vmax=vmax[3], colorbar=False, subplot=154)
    plot_connectivity_circle(con_af2[3], n_lines=12, node_names=epochs.ch_names,
                              title='Beta Band II'
                              ' After $N_{2}O$ (wPLI)', fontsize_names=10, fontsize_colorbar=10, fig=fig4,
                              vmin=vmin[3], vmax=vmax[3], subplot=155)
    fig4.tight_layout()
    fig4.savefig("images\filenumstr" + "_" + "beta_int.png")