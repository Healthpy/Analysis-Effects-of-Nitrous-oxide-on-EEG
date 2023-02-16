
from mne_connectivity.viz import plot_connectivity_circle
import mne
from mne_connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import re

def get_sec(time_stp, time_stt, gas_on1, gas_off1, gas_on2, gas_off2):
    """
    The function returns the start and stop times of gas sampling as a list of integers.
    """

    # Split each parameter by ':' to get hour, minute and second
    # h, m, s = time_stp.split(':')
    h1, m1, s1 = time_stt.split(':')
    h2, m2, s2 = gas_on1.split(':')
    h3, m3, s3 = gas_off1.split(':')
    h4, m4, s4 = gas_on2.split(':')
    h5, m5, s5 = gas_off2.split(':')

    # Calculates the start and stop times of gas sampling 
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
mainpath = "/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Cleandata/Intermittent"
dataframe = pd.read_excel(time_path)
regex = re.compile(r'\d+')


for file in glob(mainpath + '/' + '*.fif'):

    gas_starts = []
    gas_stops = []
    gas_starts2 = []
    gas_stops2 = []

    # Extracts file number
    num = [int(x) for x in regex.findall(file)][0]
    i = num - 1

    # Extracts time stamps from excel
    time_end = str(dataframe.iloc[i, 7])
    time_stt = str(dataframe.iloc[i, 2])
    gas_on1 = str(dataframe.iloc[i, 3])
    gas_off1 = str(dataframe.iloc[i, 4])
    gas_on2 = str(dataframe.iloc[i, 5])
    gas_off2 = str(dataframe.iloc[i, 6])

    # Get the start and stop times of gas sampling 
    gas_start1, gas_stop1, gas_start2, gas_stop2 = get_sec(time_end, time_stt,
                                                           gas_on1, gas_off1,
                                                           gas_on2, gas_off2)
    gas_starts = round(gas_start1/30 + 0.5)
    gas_stops = round(gas_stop1/30 + 0.5)
    gas_starts2 = round(gas_start2/30 + 0.5)
    gas_stops2 = round(gas_stop2/30 + 0.5)

    # Creates an epoch for all information
    real_epochs = mne.read_epochs(file, preload=True)
    
    # Finding index between gas start and gas stop times
    indx = [a for a, b in enumerate(real_epochs.selection) if b >= gas_starts and b <= gas_stops]
    indx2 = [a for a, b in enumerate(real_epochs.selection) if b >= gas_starts2 and b <= gas_stops2]

    sfreq = real_epochs.info['sfreq']
    
    epochs = mne.EpochsArray(real_epochs, real_epochs.info).pick_types(eeg=True)

    method=['wpli']
    # delta , theta, alpha, beta bands
    fmin = (0.5, 4.0, 8.0, 12.0)
    fmax = (4.0, 8.0, 12.0, 40.0)

    conn_cons = []
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

        conn_cons.append([con_before, con_during1, con_after1, con_during2, con_after2])
        vmin.append(min(conn_cons[i][0].min(), conn_cons[i][1].min(), conn_cons[i][2].min(),
                        conn_cons[i][3].min(), conn_cons[i][4].min()))
        vmax.append(max(conn_cons[i][0].max(), conn_cons[i][1].max(), conn_cons[i][2].max(),
                        conn_cons[i][3].max(), conn_cons[i][4].max()))

    GRAPH_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta']
    
    for i, graph_name in enumerate(GRAPH_NAMES):
    # plot_sensors_connectivity Circular plots
        filenumstr = str(num)
        fig = plt.figure(num=None, figsize=(40, 8), facecolor='black')
        plot_connectivity_circle(conn_cons[i][0], n_lines=12, node_names=epochs.ch_names,
                                title= graph_name +
                                ' Before $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig, 
                                vmin=vmin[i], vmax=vmax[i], colorbar=False, subplot=151)
        plot_connectivity_circle(conn_cons[i][1], n_lines=12, node_names=epochs.ch_names,
                                title=graph_name +
                                ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig, 
                                vmin=vmin[0], vmax=vmax[i], colorbar=False, subplot=152)
        plot_connectivity_circle(conn_cons[i][2], n_lines=12, node_names=epochs.ch_names,
                                title=graph_name +
                                ' After $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig,
                                vmin=vmin[0], vmax=vmax[i], colorbar=False, subplot=153)
        plot_connectivity_circle(conn_cons[i][3], n_lines=12, node_names=epochs.ch_names,
                                title=graph_name +
                                ' During $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig, 
                                vmin=vmin[0], vmax=vmax[0], colorbar=False, subplot=154)
        plot_connectivity_circle(conn_cons[i][4], n_lines=12, node_names=epochs.ch_names,
                                title=graph_name +
                                ' After $N_{2}O$ (wPLI)', fontsize_names=10, fig=fig, 
                                fontsize_colorbar=10, vmin=vmin[i], vmax=vmax[i], subplot=155)
        
        fig.tight_layout()
        fig.savefig(f"images/{filenumstr}_{graph_name}_int.png")
        plt.close(fig)