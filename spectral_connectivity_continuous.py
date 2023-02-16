from mne_connectivity.viz import plot_connectivity_circle
import mne
from mne_connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import re


def get_sec(time_stp, time_stt, gas_on, gas_off):
    """
    The function returns the start and stop times of gas sampling as a list of integers.
    """
    # Split each parameter by ':' to get hour, minute and second
    h, m, s = time_stp.split(':')
    h1, m1, s1 =  time_stt.split(':')
    h2, m2, s2 =  gas_on.split(':')
    h3, m3, s3 =  gas_off.split(':')

    # Calculates the start and stop times of gas sampling 
    gas_time_st = (int(h2)-int(h1)) * 3600 + (int(m2)-int(m1)) * 60 + int(s1)-int(s1) + 1
    gas_time_on = (int(h3)-int(h2)) * 3600 + (int(m3)-int(m2)) * 60 + int(s3)-int(s2) + 1
    gas_time_off = gas_time_st + gas_time_on

    return gas_time_st, gas_time_off


time_path = '/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Notes and Slides/time_stamps.xlsx'
mainpath = "/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Cleandata/Continuous"
dataframe = pd.read_excel(time_path)
regex = re.compile(r'\d+')

for file in glob(mainpath + '/' + '*.fif'):

    gas_starts = []
    gas_stops = []
    # Extracts file number
    num = [int(x) for x in regex.findall(file)][0]
    i = num - 1

    # Extracts time stamps from excel
    time_end = str(dataframe.iloc[i,7])
    time_stt = str(dataframe.iloc[i,2])
    gas_on = str(dataframe.iloc[i,3])
    gas_off = str(dataframe.iloc[i,4])
    
    # Get the start and stop times of gas sampling 
    gas_start, gas_stop = get_sec(time_end, time_stt, gas_on, gas_off)
    gas_starts = round(gas_start/30 + 0.5)
    gas_stops = round(gas_stop/30 + 0.5)
     
    # Creates an epoch for all information
    real_epochs = mne.read_epochs(file, preload=True)
    
    # Finding index between gas start and gas stop times
    indx = [a for a, b in enumerate(real_epochs.selection) if b >= gas_starts and b <= gas_stops]
    
    # Creates an epoch array
    sfreq = real_epochs.info['sfreq']
    
    epochs = mne.EpochsArray(real_epochs, real_epochs.info).pick_types(eeg=True)
    
    # Defines parameters for 
    method=['wpli']
    fmin = (0.5, 4.0, 8.0, 12.0)
    fmax = (4.0, 8.0, 12.0, 40.0)
    conn_cons = []
    vmin = []
    vmax = []

    for i in range(len(fmin)):
        con_bf = spectral_connectivity_epochs(epochs[0:indx[0]], method=method, 
                                            sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                            mt_adaptive=False, n_jobs=-1).get_data(
                                            output='dense')[:, :, 0]
        con_dr = spectral_connectivity_epochs(epochs[indx[0]:indx[-1]+1], method=method, 
                                            sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                            mt_adaptive=False, n_jobs=-1).get_data(
                                                output='dense')[:, :, 0]
        con_af = spectral_connectivity_epochs(epochs[indx[-1]+1:-1], method=method, 
                                            sfreq=sfreq, fmin=fmin[i], fmax=fmax[i],
                                            mt_adaptive=False, n_jobs=-1).get_data(
                                                output='dense')[:, :, 0]
        conn_cons.append([con_bf, con_dr, con_af])
    
        vmin.append(min(conn_cons[i][0].min(), conn_cons[i][1].min(), conn_cons[i][2].min()))
        vmax.append(max(conn_cons[i][0].max(), conn_cons[i][1].max(), conn_cons[i][2].max()))
    
    GRAPH_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta']
    for i, graph_name in enumerate(GRAPH_NAMES):
        filenumstr = str(num)
        fig = plt.figure(num=None, figsize=(24, 8), facecolor='black')
        plot_connectivity_circle(conn_cons[i][0], n_lines=12, node_names = epochs.ch_names,
                                title= graph_name + ' '
                                ' Before $N_{2}O$ (wPLI)', colorbar_pos=(-0.4, 0.0), fontsize_names=10, 
                                fig=fig, vmin=vmin[i], vmax=vmax[i], colorbar=False, subplot=131)
        plot_connectivity_circle(conn_cons[i][1], n_lines=12, node_names = epochs.ch_names,
                                title= graph_name + ' '
                                ' During $N_{2}O$ (wPLI)', colorbar_pos=(-0.2, -0.2), fontsize_names=10, 
                                fig=fig, vmin=vmin[i], vmax=vmax[i], colorbar_size=0.4, fontsize_colorbar=10, subplot=132)
        plot_connectivity_circle(conn_cons[i][2], n_lines=12, node_names = epochs.ch_names,
                                title= graph_name + ' '
                                ' After $N_{2}O$ (wPLI)', colorbar_pos=(-0.0, -0.2), fontsize_names=10, 
                                fig=fig, vmin=vmin[i], vmax=vmax[i], colorbar_size=0.4, fontsize_colorbar=10, colorbar=False, subplot=133)
        
        fig.tight_layout()
        fig.savefig(f"images/{filenumstr}_{graph_name}_cont.png")
        plt.close(fig)