import re
from mne_connectivity import spectral_connectivity_epochs

import numpy as np
import pandas as pd
import mne
from glob import glob
from numpy.random import permutation
from scipy import stats
import matplotlib.pyplot as plt

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

time_path = '/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Notes and Slides/time_stamps.xlsx'

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
        clean_all_dic[(num[0])] = cleani.get_data().astype(np.float64)
        
    return clean_all_dic, index, sfreq

data_all,  index, sfreq = read_data_to_dict(path)


def compute_sep(data_all, index):
    data_before = [0, 0, 0, 0, 0]
    data_during = [0, 0, 0, 0, 0]
    data_after = [0, 0, 0, 0, 0]
    
    
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

delta = [0.5, 4.0]
theta = [4.0, 8.0]
alpha = [8.0, 12.0]
beta = [12.0, 40.0]

band = [delta, theta, alpha, beta]

method = 'wpli'

data_before, data_during, data_after = compute_sep(data_all, index)

# Column 0 - delta, 1-theta, 2-alpha, 3-beta

dataframe_bf = pd.read_excel('dataframe_bf_continuous.xlsx').drop('Unnamed: 0', axis=1).values 
dataframe_dr = pd.read_excel('dataframe_dr_continuous.xlsx').drop('Unnamed: 0', axis=1).values 
dataframe_af = pd.read_excel('dataframe_af_continuous.xlsx').drop('Unnamed: 0', axis=1).values 

plt.subplot(3, 1, 1)
plt.hist(dataframe_bf)
plt.subplot(3, 1, 2)
plt.hist(dataframe_dr)
plt.subplot(3, 1, 3)
plt.hist(dataframe_af)

# delta_null_natural_log_bf = np.log(null_delta_bf)
# plt.hist(delta_null_natural_log_bf)


# Compute the eeg connectivity of a subject according to a band

# p_values  =  {}

# for k, m in enumerate(data_all):
#     p_value_bf = [0] * len(band)
#     p_value_dr = [0] * len(band)
#     p_value_af = [0] * len(band)
#     for l in range(len(band)):
#         conn_before, conn_during, conn_after = compute_con(data_before[k], data_during[k], data_after[k], band[l])
#         # newcon_b = conn_before[np.where(conn_during!=0)]
#         p_value_bf[l] = stats.ttest_ind(conn_before[np.where(conn_before!=0)], dataframe_bf[:,l]).pvalue
#         p_value_dr[l] = stats.ttest_ind(conn_during[np.where(conn_during!=0)], dataframe_dr[:,l]).pvalue
#         p_value_af[l] = stats.ttest_ind(conn_after[np.where(conn_after!=0)], dataframe_af[:,l]).pvalue
#     columns = ['delta', 'theta', 'alpha', 'beta']
#     rows = ['before', 'during', 'after']
#     p_values[m] = pd.DataFrame([p_value_bf, p_value_dr, p_value_af], index=rows, columns=columns)

# print(p_values)

