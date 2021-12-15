# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:04:54 2021

@author: ThanhVi
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import math
import os
from automatic_signal_F0 import read_result, normalize, mean, stand, ACF

def statistic(X,Fs,W, v, uv, bounds):
    samples = len(X)
    frames = math.floor(samples/W)
    # Compute ACF_max all frames
    ACF_maxs = []
    for i in range(frames):
        ACF_vals = normalize(np.array([ACF(X, W, i*W, l) for l in range(*bounds)]))
        val_max = ACF_vals[int(Fs/450) + np.argmax(ACF_vals[int(Fs/450): int(Fs/70)])]
        index = W*i
        ACF_maxs.append([index, val_max])
    ACF_maxs = np.array(ACF_maxs)
    # ACF_maxs[:,1] = normalize(ACF_maxs[:,1])
    
    #statistic voice 
    acf_v_max = [] # variable contains magnitude acf_max of v
    for i in range(len(v)):
        rs = np.where((ACF_maxs[:,0] >= v[i][0]) & (ACF_maxs[:,0] < v[i][1]))
        acf_v = ACF_maxs[rs,1]
        acf_v = np.array([acf_v]).reshape(-1,)
        acf_v_max = np.concatenate((acf_v_max, acf_v), axis=0)

    acf_v_mean = mean(acf_v_max)
    acf_v_std = stand(acf_v_max)
    print('ACF_voice_mean: ' + str(acf_v_mean))
    print('ACF_voice_std: ' + str(acf_v_std) + '\n')
    
    #statistic voice 
    acf_uv_max = [] # variable contains magnitude acf_max of v
    for i in range(len(uv)):
        rs = np.where((ACF_maxs[:,0]> uv[i][0]) & (ACF_maxs[:,0] < uv[i][1])) 
        acf_uv = ACF_maxs[rs,1]
        acf_uv = np.array([acf_uv]).reshape(-1,)
        acf_uv_max = np.concatenate((acf_uv_max, acf_uv), axis=0)
        
    acf_uv_mean = mean(acf_uv_max)
    acf_uv_std = stand(acf_uv_max)
    print('ACF_uvoice_mean: ' + str(acf_uv_mean))
    print('ACF_uvoice_std: ' + str(acf_uv_std) + '\n')
    
    # plot all local max of voice region
    plt.figure(figsize=(40, 10))
    plt.subplot(2,1,1)
    plt.stem(np.arange(acf_v_max.shape[0]), acf_v_max)
    plt.title("ACF_voice")
    plt.xlabel("Sample")
    plt.ylabel("ACF")
    
    # plot all local max of unvoice region
    plt.subplot(2,1,2)
    plt.stem(np.arange(acf_uv_max.shape[0]), acf_uv_max)
    plt.title("ACF_uvoice")
    plt.xlabel("Sample")
    plt.ylabel("ACF")
    plt.show()
    return acf_v_max, acf_uv_max

# statistic voice and unvoice
PATH_X = 'train_signal/X'
PATH_Y = 'train_signal/Y'
FILE_X = os.listdir(PATH_X)
FILE_Y = os.listdir(PATH_Y)

acf_v_maxs = []
acf_uv_maxs = []
for i in range(len(FILE_X)):
    frame = 25/1000
    X, Fs = sf.read(PATH_X + '/' + FILE_X[i])
    v, uv, sil, Fc = read_result(PATH_Y + '/' + FILE_Y[i])
    v = (v.astype(float)*Fs).astype(int)
    uv = (uv.astype(float)*Fs).astype(int)
    X = normalize(X)
    window_size = (int)(frame*Fs)
    bounds = [0, window_size - 1] # F0 is from 70Hz to 400 -> range of lag (floor(Fs/400), window_size)
    print(FILE_X[i])
    acf_v_max, acf_uv_max = statistic(X,Fs,window_size, v, uv, bounds)
    acf_v_maxs = np.concatenate((acf_v_maxs, acf_v_max), axis = 0)
    acf_uv_maxs = np.concatenate((acf_uv_maxs, acf_uv_max), axis = 0)
    
acf_v_mean_nor = mean(acf_v_maxs)
acf_uv_mean_nor = mean(acf_uv_maxs)
acf_v_mean_std = stand(acf_v_maxs)
acf_uv_mean_std = stand(acf_uv_maxs)
print('mean voice normalize: ' + str(acf_v_mean_nor))
print('standard deviation: ' + str(acf_v_mean_std))
print('mean uvoice normalize: ' + str(acf_uv_mean_nor))
print('standard deviation: ' + str(acf_uv_mean_std))

# mean voice normalize: 0.6110656614197735
# standard deviation: 0.16393152966411356
# mean uvoice normalize: 0.31269835424435327
# standard deviation: 0.15931473565907728