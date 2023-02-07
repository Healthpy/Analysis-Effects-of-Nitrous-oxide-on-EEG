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

time_path = r'C:\Users\sqemch\OneDrive - TUNI.fi\Thesis\Notes and Slides\time_stamps.xlsx'
# time_path = '/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUNI.fi/Thesis/Notes and Slides/time_stamps.xlsx'

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
            if dataframe.iloc[i,1] == num[0]:

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
                   
        
        index1[num[0]] = indx
        index2[num[0]] = indx2

        # cleani = mne.EpochsArray(cleani, cleani.info).pick_types(eeg=True)
        # ch_dic[(num[0])] = [c for c in cleani.info["ch_names"]]
        clean_all_dic[(num[0])] = cleani.get_data().astype(np.float32)
        
    return clean_all_dic, index1, index2, sfreq

data_all,  index1, index2, sfreq = read_data_to_dict(path)



def compute_sep(data_all, index1, index2):
    data_before = [0] * len(data_all)
    data_during1 = [0] * len(data_all)
    data_after1 = [0] * len(data_all)
    data_during2 = [0] * len(data_all)
    data_after2 = [0] * len(data_all)
    
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

delta = [0.5, 4.0]
theta = [4, 8.0]
alpha = [8.0, 12.0]
beta = [12.0, 40.0]

band = [delta, theta, alpha, beta]

method = 'wpli'

data_before, data_during1, data_after1, data_during2, data_after2 = compute_sep(data_all, index1, index2)

# Column 0 - delta, 1-theta, 2-alpha, 3-beta

dataframe_bf = pd.read_excel('dataframe_bf_intermittent.xlsx').drop('Unnamed: 0', axis=1).values 
dataframe_dr1 = pd.read_excel('dataframe_dr1_intermittent.xlsx').drop('Unnamed: 0', axis=1).values 
dataframe_af1 = pd.read_excel('dataframe_af1_intermittent.xlsx').drop('Unnamed: 0', axis=1).values
dataframe_dr2 = pd.read_excel('dataframe_dr2_intermittent.xlsx').drop('Unnamed: 0', axis=1).values 
dataframe_af2 = pd.read_excel('dataframe_af2_intermittent.xlsx').drop('Unnamed: 0', axis=1).values 


# Compute the eeg connectivity of a subject according to a band

p_values_interm  =  {}

for k, m in enumerate(data_all):
    p_value_bf = [0] * len(band)
    p_value_dr1 = [0] * len(band)
    p_value_af1 = [0] * len(band)
    p_value_dr2 = [0] * len(band)
    p_value_af2 = [0] * len(band)
    for l in range(len(band)):
        conn_before, conn_during1, conn_after1, conn_during2, conn_after2 = compute_con(data_before[k], 
                                                                                        data_during1[k], data_after1[k], data_during2[k], data_after2[k], band[l])

        # newcon_b = conn_before[np.where(conn_during!=0)]
        p_value_bf[l] = stats.ttest_ind(conn_before[np.where(conn_before!=0)], np.unique(dataframe_bf[:,l])).pvalue
        p_value_dr1[l] = stats.ttest_ind(conn_during1[np.where(conn_during1!=0)], np.unique(dataframe_dr1[:,l])).pvalue
        p_value_af1[l] = stats.ttest_ind(conn_after1[np.where(conn_after1!=0)], np.unique(dataframe_af1[:,l])).pvalue
        p_value_dr2[l] = stats.ttest_ind(conn_during2[np.where(conn_during2!=0)], np.unique(dataframe_dr2[:,l])).pvalue
        p_value_af2[l] = stats.ttest_ind(conn_after2[np.where(conn_after2!=0)], np.unique(dataframe_af2[:,l])).pvalue
    columns = ['delta', 'theta', 'alpha', 'beta']
    rows = ['before', 'during1', 'after1', 'during2', 'after2']
    p_values_interm[m] = pd.DataFrame([p_value_bf, p_value_dr1, p_value_af1, p_value_dr2, p_value_af2], 
                                      index=rows, columns=columns)

print(p_values_interm)

#p_value_tables = pd.DataFrame.from_dict(p_values)
#print(p_value_tables)
#p_value_tables.to_excel('p_values_tables_continuous.xlsx')
# creating Panel
# panel = pd.Pan #Panel.from_dict(p_values, orient ='minor')
# print("panel['1'] is - \n\n", panel[1])
 
# print("\nShape of panel['b'] is - ",  panel[1].shape)

# panel.to_excel('p_values_tables_continuous.xlsx')