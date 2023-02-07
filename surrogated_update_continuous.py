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
import pandas as pd

regex = re.compile(r'\d+')

path = "/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Cleandata/Continous"

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
time_path = 'time_stamps.xlsx'

dataframe = pd.read_excel(time_path)



def read_data_to_dict(path):
    clean_all_dic = {}
    ch_dic = {}
    index = {}
    gas_starts = []
    gas_stops = []
    for f, file in enumerate(glob(path + '/' + '*.fif')):
        # if os.path.basename(file) != filename:
        num = [int(x) for x in regex.findall(file)]
        for i in range(len(dataframe)):
            if dataframe.iloc[i,1] == num[0]:
                time_end = str(dataframe.iloc[i,7])
                time_stt = str(dataframe.iloc[i,2])
                gas_on = str(dataframe.iloc[i,3])
                gas_off = str(dataframe.iloc[i,4])
                gas_start, gas_stop = get_sec(time_end, time_stt, gas_on, gas_off)
                gas_starts.append(round(gas_start/30 + 0.5))
                gas_stops.append(round(gas_stop/30 + 0.5))
                
        cleani = mne.read_epochs(file).pick_types(eeg=True)
        sfreq = cleani.info['sfreq']
        indx = []
        
        for a, b in enumerate(cleani.selection):
            if b >= gas_starts[f] and b <= gas_stops[f]:
                indx.append(a)
        
        index[num[0]] = indx
                

        # cleani = mne.EpochsArray(cleani, cleani.info).pick_types(eeg=True)
        ch_dic[(num[0])] = [c for c in cleani.info["ch_names"]]
        clean_all_dic[(num[0])] = cleani.get_data().astype(np.float32)
        
    return clean_all_dic, index, sfreq

data_all,  index, sfreq = read_data_to_dict(path)



def compute_sep(data_all, index):
    data_before = [0] * len(data_all)
    data_during = [0] * len(data_all)
    data_after = [0] * len(data_all)
    
    
    for i, j in enumerate(index.keys()):
        if j == list(data_all.keys())[i]:
            indxx = index.get(j)
            data_al = np.array(data_all.get(j)).astype('float64')
            data_before[i] = data_al[0:indxx[0]]
            data_during[i] = data_al[indxx[0]:indxx[-1]+1]
            data_after[i] = data_al[indxx[-1]+1:-1]
            
    return data_before, data_during, data_after
        
        
            
def compute_con(data_before_i, data_during_i, data_after_i, band):
    
        con_before = spectral_connectivity_epochs(data_before_i, method=method, 
                                                  sfreq=sfreq, fmin=band[0], fmax=band[1],
                                                mt_adaptive=False, n_jobs=-1).get_data(
                                                output='dense')[:, :, 0]
        con_during = spectral_connectivity_epochs(data_during_i, method=method, 
                                                  sfreq=sfreq, fmin=band[0], fmax=band[1],
                                                mt_adaptive=False, n_jobs=-1).get_data(
                                                    output='dense')[:, :, 0]
        con_after = spectral_connectivity_epochs(data_after_i, method=method, 
                                                      sfreq=sfreq, fmin=band[0], fmax=band[1],
                                                    mt_adaptive=False, n_jobs=-1).get_data(
                                                        output='dense')[:, :, 0]
                                                        
                                             
        return con_before, con_during, con_after



def compute_surrogate_con(condition, band):

    con_delta = []
    con_theta = []
    con_alpha = []
    con_beta = []
    # fmin = (0.1, 4.0, 8.0, 13.0)
    # fmax = (4.0, 8.0, 13.0, 30.0)
    for k in permutation(condition):
        
       choice_data = k
       n_random_picks = 1
       idn = np.random.choice(choice_data.shape[1], 50, replace=True) #for i in range(5)]
       idn = idn.reshape(-1,1)
      

       for l in idn:
           
           for i, j in enumerate(condition):
               if np.array(j != choice_data).all():
                   data = condition[i]
                   # np.random.seed(l)
                   idx = np.random.choice(data.shape[1], n_random_picks, replace=False)
                   if idx != l:
                       if len(choice_data) <= len(data):
                           picked = np.hstack((data[:len(choice_data),idx,:], choice_data[:,l,:]))
                           con_delt = spectral_connectivity_epochs(picked, method='wpli', 
                                                                  sfreq=sfreq, fmin=band[0][0], 
                                                                  fmax=band[0][1], 
                                                                  mt_adaptive=False, n_jobs=-1).get_data(
                                                                      output='dense')[1, 0, 0]
                           con_delta.append(con_delt)
                           
                           con_thet = spectral_connectivity_epochs(picked, method='wpli', 
                                                                  sfreq=sfreq, fmin=band[1][0], 
                                                                  fmax=band[1][1], 
                                                                  mt_adaptive=False, n_jobs=-1).get_data(
                                                                      output='dense')[1, 0, 0]
                           con_theta.append(con_thet)
                           
                           con_alph = spectral_connectivity_epochs(picked, method='wpli', 
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
                           picked = np.hstack((data[:,idx,:], choice_data[:len(data), l,:]))
                           
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
                           
                           
    sur_conn_delta = con_delta #np.unique(con_delta)
    sur_conn_theta = con_theta #np.unique(con_theta)
    sur_conn_alpha = con_alpha #np.unique(con_alpha)
    sur_conn_beta = con_beta #np.unique(con_beta)
    
    return sur_conn_delta, sur_conn_theta, sur_conn_alpha, sur_conn_beta


delta = [0.5, 4.0]
theta = [4.0, 8.0]
alpha = [8.0, 12.0]
beta = [12.0, 40.0]

band = [delta, theta, alpha, beta]

method = 'wpli' #or 'wpli'

data_before, data_during, data_after = compute_sep(data_all, index)

# Null distribution of surrogate connectivity before anesthesia
null_delta_bf, null_theta_bf, null_alpha_bf, null_beta_bf = compute_surrogate_con(data_before, band)
data_bf = {'null_delta_bf':null_delta_bf, 'null_theta_bf':null_theta_bf, 
           'null_alpha_bf':null_alpha_bf, 'null_beta_bf':null_beta_bf}

dataframe_bf = pd.DataFrame(data_bf)
dataframe_bf.to_excel('dataframe_bf_continuous.xlsx')

# Null distribution of surrogate connectivity during anesthesia
null_delta_dr, null_theta_dr, null_alpha_dr, null_beta_dr = compute_surrogate_con(data_during, band)
data_dr = {'null_delta_dr':null_delta_dr, 'null_theta_dr':null_theta_dr, 
            'null_alpha_dr':null_alpha_dr, 'null_beta_dr':null_beta_dr}

dataframe_dr = pd.DataFrame(data_dr)
dataframe_dr.to_excel('dataframe_dr_continuous.xlsx')
# Null distribution of surrogate connectivity after anesthesia
null_delta_af, null_theta_af, null_alpha_af, null_beta_af = compute_surrogate_con(data_after, band)
data_af = {'null_delta_af':null_delta_af, 'null_theta_af':null_theta_af, 
            'null_alpha_af':null_alpha_af, 'null_beta_af':null_beta_af}

dataframe_af = pd.DataFrame(data_af)
dataframe_af.to_excel('dataframe_af_continuous.xlsx')

# Compute the eeg connectivity of a subject according to a band
conn_before, conn_during, conn_after = compute_con(data_before[2], data_during[2], data_after[2], band[0])

newcon = conn_before[np.where(conn_during!=0)]

p_value = stats.ttest_ind(newcon, null_delta_bf).pvalue