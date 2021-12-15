# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:57:36 2021

@author: ThanhVi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from automatic_processing_signal import read_result, normalize, mean, detect_pitch_ACF, detect_threshold_mean_STE, ShortTermEnergy, SegmentSpeech, intermediate_frame, mean, stand

if __name__ == '__main__':
    PATH_X = 'test_signal/X'
    PATH_Y = 'test_signal/Y'
    FILE_X = os.listdir(PATH_X)
    FILE_Y = os.listdir(PATH_Y)
    frame_shift = 10/1000
    frame_size = 20/1000
    threshold = 0.0023955459484534683
    N_FFT = 1024 * 8
    Fmin = 70
    Fmax = 400
    detect_threshold_mean_STE(frame_size, frame_shift)

    for i in range(len(FILE_X)):
        Fs, X = wavfile.read(PATH_X + '/' + FILE_X[i])
        X = normalize(X)
        STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
        STE = normalize(STE)
        v, sil, f = read_result(PATH_Y + '/' + FILE_Y[i])
        v = v.astype(float) * Fs
        v = (v/(frame_shift*Fs)).astype(int).reshape(-1)
        speech, speech_segment = SegmentSpeech(X, Fs, frame_size, frame_shift, threshold)
        freq, dftx, dfft, env, ACF_vals, peaks_ACF = intermediate_frame(X, Fs, N_FFT, frame_size, frame_shift, speech, Fmin, Fmax)
        
        _ = np.zeros((len(X)))
        F00 = detect_pitch_ACF(X, Fs, Fmin, Fmax, frame_size, frame_shift, speech, N_FFT)
        F00[:,0] = F00[:,0]*frame_shift
        F = np.mean(F00)
        print('Thay cho: ' + str(f))
        print('Ket qua: ' + str(F))
        
        t = np.linspace(0, len(X)/Fs, num = len(X))
        tt = np.linspace(0, len(X)/Fs, num = len(STE))
        ttt = speech*(frame_shift*Fs)/Fs
        plt.figure(figsize=(40, 40))
        
        # plt.subplot(2, 2, 1)
        # plt.title("Energy-based Speech/Silence discrimination")
        # plt.plot(t, X)
        # plt.plot(tt, STE, '-')
        # # plt.plot(ttt, F00/100, '.')
        # for i in range(tt[speech_segment].shape[0]):
        #     plt.axvline(tt[speech_segment][i], color='r', linestyle=':', linewidth=2)
        # for i in range(tt[v].shape[0]):
        #     plt.axvline(tt[v][i], color='b', linestyle=':', linewidth=2)
        
        # ax1 = plt.subplot(2, 2, 3)
        # ax1.plot(t, _)
        # ax1.set_ylim([0,400])
        # ax1.plot(F00[:,0], F00[:,1], 'r.')
        
        # plt.title("frequency")
        # plt.xlabel("Time")
        # plt.ylabel("Hz")
        
        # plt.subplot(2, 2, 2)
        # l1, = plt.plot(freq[:1000], dftx[:1000], label= 'sdfsd')
        # l2, = plt.plot(freq[:1000], dfft[:1000], '-')
        # l3, = plt.plot(freq[:1000], env[:1000])
        
        # plt.legend((l1,l2,l3), ["zero-crossing spectrum", "spectrum", "envelope spectrum"])
        # plt.title("Analytics spectrum")
        # plt.xlabel("Hz")
        # plt.ylabel("Magnitude")
        # plt.subplot(2, 2, 4)
        # plt.plot(np.arange(ACF_vals.size), ACF_vals)
        # plt.plot(np.arange(ACF_vals.size)[peaks_ACF[0]], ACF_vals[peaks_ACF[0]], '.')
        # plt.title("Spectral Autocorrelation")
        # plt.xlabel("Lag")
        # plt.ylabel("Magnitude")
        # plt.show()
        
        plt.subplot(2, 1, 1)
        plt.title("Energy-based Speech/Silence discrimination")
        plt.plot(t, X)
        plt.plot(tt, STE, '-')
        # plt.plot(ttt, F00/100, '.')
        for i in range(tt[speech_segment].shape[0]):
            if i != 0:
                plt.axvline(tt[speech_segment][i], color='r', linestyle='-', linewidth=1)
            else: 
                plt.axvline(tt[speech_segment][i], color='r', linestyle='-', linewidth=1, label='Biên chuẩn')
        for i in range(tt[v].shape[0]):
            if i != 0:
                plt.axvline(tt[v][i], color='b', linestyle='-', linewidth=1)
            else:
                plt.axvline(tt[v][i], color='b', linestyle='-', linewidth=1, label='Biên tự động')
        plt.legend(loc='upper right')
        
        ax1 = plt.subplot(2, 1, 2)
        ax1.plot(t, _)
        ax1.set_ylim([0,400])
        ax1.plot(F00[:,0], F00[:,1], 'r.')
        
        plt.title("frequency")
        plt.xlabel("Time")
        plt.ylabel("Hz")
        
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.show()