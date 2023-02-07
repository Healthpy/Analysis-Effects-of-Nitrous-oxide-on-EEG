# -*- coding: utf-8 -*
"""
Created on Tue Jul 19 15:27:22 2022

@author: sqemch
"""

import re
from mne_connectivity import spectral_connectivity_epochs

import numpy as np
import pandas as pd
import mne
from glob import glob
from numpy.random import permutation
from scipy import stats


regex = re.compile(r'\d+')
#path = r'C:\Users\sqemch\Documents\res\Cleandata\intermitent'
# path = r'C:\Users\sqemch\OneDrive - TUNI.fi\Thesis\Cleandata\Intermitent'
path = "/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Cleandata/Intermitent"


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


# time_path = r'C:\Users\sqemch\OneDrive - TUNI.fi\Thesis\Notes and Slides\time_stamps.xlsx'
time_path = "/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Notes and Slides/time_stamps.xlsx"

dataframe = pd.read_excel(time_path)


def read_data_to_dict(path):
    clean_all_dic = {}
    #ch_dic = {}
    index1 = {}
    index2 = {}
    gas_starts = []
    gas_stops = []
    gas_starts2 = []
    gas_stops2 = []
    for f, file in enumerate(glob(path + '/' + '*.fif')):

        # if os.path.basename(file) != filename:
        num = [int(x) for x in regex.findall(file)]
        for i in range(len(dataframe)):
            if dataframe.iloc[i, 1] == num[0]:

                time_end = str(dataframe.iloc[i, 7])
                time_stt = str(dataframe.iloc[i, 2])
                gas_on1 = str(dataframe.iloc[i, 3])
                gas_off1 = str(dataframe.iloc[i, 4])
                gas_on2 = str(dataframe.iloc[i, 5])
                gas_off2 = str(dataframe.iloc[i, 6])
                gas_start1, gas_stop1, gas_start2, gas_stop2 = get_sec(time_end, time_stt,
                                                                       gas_on1, gas_off1,
                                                                       gas_on2, gas_off2)
                gas_starts.append(round(gas_start1/30 + 0.5))
                gas_stops.append(round(gas_stop1/30 + 0.5))
                gas_starts2.append(round(gas_start2/30 + 0.5))
                gas_stops2.append(round(gas_stop2/30 + 0.5))

        cleani = mne.read_epochs(file).pick_types(eeg=True)
        sfreq = cleani.info['sfreq']

        indx = []
        indx2 = []

        for a, b in enumerate(cleani.selection):
            if b >= gas_starts[f] and b <= gas_stops[f]:
                indx.append(a)
            elif b >= gas_starts2[f] and b <= gas_stops2[f]:
                indx2.append(a)

   ########################## continue here #########################

        index1[num[0]] = indx
        index2[num[0]] = indx2

        # cleani = mne.EpochsArray(cleani, cleani.info).pick_types(eeg=True)
        # ch_dic[(num[0])] = [c for c in cleani.info["ch_names"]]
        clean_all_dic[(num[0])] = cleani.get_data().astype(np.float32)

    return clean_all_dic, index1, index2, sfreq


data_all,  index1, index2, sfreq = read_data_to_dict(path)


def compute_sep(data_all, index1, index2):
    data_before = [0, 0, 0, 0]
    data_during1 = [0, 0, 0, 0]
    data_after1 = [0, 0, 0, 0]
    data_during2 = [0, 0, 0, 0]
    data_after2 = [0, 0, 0, 0]

    for i, j in enumerate(index1.keys()):
        if j == list(data_all.keys())[i]:
            indxx1 = index1.get(j)
            indxx2 = index2.get(j)
            data_al = np.array(data_all.get(j)).astype('float64')
            data_before[i] = data_al[0:indxx1[0]]
            data_during1[i] = data_al[indxx1[0]:indxx1[-1]+1]
            data_after1[i] = data_al[indxx1[-1]+1:indxx2[0]+1]
            data_during2[i] = data_al[indxx2[0]+1:indxx2[-1]+1]
            data_after2[i] = data_al[indxx2[-1]+1:-1]

    return data_before, data_during1, data_after1, data_during2, data_after2


def compute_con(data_before_i, data_during_i1, data_after_i1, data_during_i2, data_after_i2, band):

    con_before = spectral_connectivity_epochs(data_before_i, method=method,
                                              sfreq=sfreq, fmin=band[0], fmax=band[1],
                                              mt_adaptive=False, n_jobs=-1).get_data(
        output='dense')[:, :, 0]
    con_during1 = spectral_connectivity_epochs(data_during_i1, method=method,
                                               sfreq=sfreq, fmin=band[0], fmax=band[1],
                                               mt_adaptive=False, n_jobs=-1).get_data(
        output='dense')[:, :, 0]
    con_after1 = spectral_connectivity_epochs(data_after_i1, method=method,
                                              sfreq=sfreq, fmin=band[0], fmax=band[1],
                                              mt_adaptive=False, n_jobs=-1).get_data(
        output='dense')[:, :, 0]
    con_during2 = spectral_connectivity_epochs(data_during_i2, method=method,
                                               sfreq=sfreq, fmin=band[0], fmax=band[1],
                                               mt_adaptive=False, n_jobs=-1).get_data(
        output='dense')[:, :, 0]
    con_after2 = spectral_connectivity_epochs(data_after_i2, method=method,
                                              sfreq=sfreq, fmin=band[0], fmax=band[1],
                                              mt_adaptive=False, n_jobs=-1).get_data(
        output='dense')[:, :, 0]

    return con_before, con_during1, con_after1, con_during2, con_after2


def compute_surrogate_con(condition, band):

    con_delta = []
    con_theta = []
    con_alpha = []
    con_beta = []
    # fmin = (0.5, 4.0, 8.0, 13.0)
    # fmax = (4.0, 8.0, 13.0, 30.0)
    for k in permutation(condition):

        choice_data = k
        n_random_picks = 1
        # for i in range(5)]
        idn = np.random.choice(choice_data.shape[1], 50, replace=True)
        idn = idn.reshape(-1, 1)

        for l in idn:

            for i, j in enumerate(condition):
                if np.array(j != choice_data).all():
                    data = condition[i]
                    # np.random.seed(l)
                    idx = np.random.choice(
                        data.shape[1], n_random_picks, replace=False)
                    if idx != l:
                        if len(choice_data) <= len(data):
                            picked = np.hstack(
                                (data[:len(choice_data), idx, :], choice_data[:, l, :]))
                            con_delt = spectral_connectivity_epochs(picked, method=method,
                                                                    sfreq=sfreq, fmin=band[0][0],
                                                                    fmax=band[0][1],
                                                                    mt_adaptive=False, n_jobs=-1).get_data(
                                output='dense')[1, 0, 0]
                            con_delta.append(con_delt)

                            con_thet = spectral_connectivity_epochs(picked, method=method,
                                                                    sfreq=sfreq, fmin=band[1][0],
                                                                    fmax=band[1][1],
                                                                    mt_adaptive=False, n_jobs=-1).get_data(
                                output='dense')[1, 0, 0]
                            con_theta.append(con_thet)

                            con_alph = spectral_connectivity_epochs(picked, method=method,
                                                                    sfreq=sfreq, fmin=band[2][0],
                                                                    fmax=band[2][1],
                                                                    mt_adaptive=False, n_jobs=-1).get_data(
                                output='dense')[1, 0, 0]
                            con_alpha.append(con_alph)

                            con_bet = spectral_connectivity_epochs(picked, method=method,
                                                                   sfreq=sfreq, fmin=band[3][0],
                                                                   fmax=band[3][1],
                                                                   mt_adaptive=False, n_jobs=-1).get_data(
                                                                       output='dense')[1, 0, 0]
                            con_beta.append(con_bet)

                        else:
                            picked = np.hstack(
                                (data[:, idx, :], choice_data[:len(data), l, :]))

                            con_delt = spectral_connectivity_epochs(picked, method=method,
                                                                    sfreq=sfreq, fmin=band[0][0],
                                                                    fmax=band[0][1],
                                                                    mt_adaptive=False, n_jobs=-1).get_data(
                                output='dense')[1, 0, 0]
                            con_delta.append(con_delt)

                            con_thet = spectral_connectivity_epochs(picked, method=method,
                                                                    sfreq=sfreq, fmin=band[1][0],
                                                                    fmax=band[1][1],
                                                                    mt_adaptive=False, n_jobs=-1).get_data(
                                output='dense')[1, 0, 0]
                            con_theta.append(con_thet)

                            con_alph = spectral_connectivity_epochs(picked, method=method,
                                                                    sfreq=sfreq, fmin=band[2][0],
                                                                    fmax=band[2][1],
                                                                    mt_adaptive=False, n_jobs=-1).get_data(
                                output='dense')[1, 0, 0]
                            con_alpha.append(con_alph)

                            con_bet = spectral_connectivity_epochs(picked, method=method,
                                                                   sfreq=sfreq, fmin=band[3][0],
                                                                   fmax=band[3][1],
                                                                   mt_adaptive=False, n_jobs=-1).get_data(
                                                                       output='dense')[1, 0, 0]
                            con_beta.append(con_bet)

    # sur_conn_delta = np.unique(con_delta)
    # sur_conn_theta = np.unique(con_theta)
    # sur_conn_alpha = np.unique(con_alpha)
    # sur_conn_beta = np.unique(con_beta)

    sur_conn_delta = con_delta
    sur_conn_theta = con_theta
    sur_conn_alpha = con_alpha
    sur_conn_beta = con_beta

    return sur_conn_delta, sur_conn_theta, sur_conn_alpha, sur_conn_beta


delta = [0.5, 4.0]
theta = [4, 8.0]
alpha = [8.0, 12.0]
beta = [12.0, 40.0]

band = [delta, theta, alpha, beta]

method = 'wpli'

data_before, data_during1, data_after1, data_during2, data_after2 = compute_sep(
    data_all, index1, index2)

# Null distribution of surrogate connectivity before anesthesia
null_delta_bf, null_theta_bf, null_alpha_bf, null_beta_bf = compute_surrogate_con(
    data_before, band)
data_bf = {'null_delta_bf': null_delta_bf, 'null_theta_bf': null_theta_bf,
           'null_alpha_bf': null_alpha_bf, 'null_beta_bf': null_beta_bf}

dataframe_bf = pd.DataFrame(data_bf)
dataframe_bf.to_excel('dataframe_bf_intermittent.xlsx')

# Null distribution of surrogate connectivity during anesthesia
null_delta_dr1, null_theta_dr1, null_alpha_dr1, null_beta_dr1 = compute_surrogate_con(
    data_during1, band)
data_dr1 = {'null_delta_dr1': null_delta_dr1, 'null_theta_dr1': null_theta_dr1,
            'null_alpha_dr1': null_alpha_dr1, 'null_beta_dr1': null_beta_dr1}

dataframe_dr1 = pd.DataFrame(data_dr1)
dataframe_dr1.to_excel('dataframe_dr1_intermittent.xlsx')
# Null distribution of surrogate connectivity after anesthesia
null_delta_af1, null_theta_af1, null_alpha_af1, null_beta_af1 = compute_surrogate_con(
    data_after1, band)
data_af1 = {'null_delta_af1': null_delta_af1, 'null_theta_af1': null_theta_af1,
            'null_alpha_af1': null_alpha_af1, 'null_beta_af1': null_beta_af1}

dataframe_af1 = pd.DataFrame(data_af1)
dataframe_af1.to_excel('dataframe_af1_intermittent.xlsx')

# Null distribution of surrogate connectivity during anesthesia
null_delta_dr2, null_theta_dr2, null_alpha_dr2, null_beta_dr2 = compute_surrogate_con(
    data_during2, band)
data_dr2 = {'null_delta_dr2': null_delta_dr2, 'null_theta_dr2': null_theta_dr2,
            'null_alpha_dr2': null_alpha_dr2, 'null_beta_dr2': null_beta_dr2}

dataframe_dr2 = pd.DataFrame(data_dr2)
dataframe_dr2.to_excel('dataframe_dr2_intermittent.xlsx')

# Null distribution of surrogate connectivity after anesthesia
null_delta_af2, null_theta_af2, null_alpha_af2, null_beta_af2 = compute_surrogate_con(
    data_after2, band)
data_af2 = {'null_delta_af2': null_delta_af2, 'null_theta_af2': null_theta_af2,
            'null_alpha_af2': null_alpha_af2, 'null_beta_af2': null_beta_af2}

dataframe_af2 = pd.DataFrame(data_af2)
dataframe_af2.to_excel('dataframe_af2_intermittent.xlsx')
# Compute the eeg connectivity of a subject according to a band
