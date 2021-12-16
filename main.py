# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:57:36 2021

@author: ThanhVi
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import wavfile
import os
from automatic_processing_signal import read_result, normalize, detect_pitch_ACF, detect_threshold_mean_STE, ShortTermEnergy, SegmentSpeech, intermediate_frame, mean, stand

if __name__ == '__main__':
    PATH_X = 'test_signal/X'
    PATH_Y = 'test_signal/Y'
    FILE_X = os.listdir(PATH_X)
    FILE_Y = os.listdir(PATH_Y)
    frame_shift = 10/1000
    frame_size = 20/1000
    N_FFT = 1024 * 8
    Fmin = 70
    Fmax = 400
    # threshold = 0.0024
    threshold = detect_threshold_mean_STE(frame_size, frame_shift)

    for i in range(len(FILE_X)):
        Fs, X = wavfile.read(PATH_X + '/' + FILE_X[i])
        X = normalize(X)
        STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
        STE = normalize(STE)
        
        v, s, f = read_result(PATH_Y + '/' + FILE_Y[i])
        v = v * Fs
        v_segment = (v/(frame_shift*Fs)).astype(int)
        v = v_segment.reshape(-1)
        speech, speech_segment = SegmentSpeech(X, Fs, frame_size, frame_shift, threshold)
        freq, dftx, dfft, env, ACF_vals, peaks_ACF = intermediate_frame(X, Fs, N_FFT, frame_size, frame_shift, speech, Fmin, Fmax)
        
        _ = np.zeros((len(X)))
        F00 = detect_pitch_ACF(X, Fs, Fmin, Fmax, frame_size, frame_shift, speech, N_FFT)
        F00[:,0] = F00[:,0]*frame_shift
        F = mean(F00[:,1])
        Fstd = stand(F00[:,1])
        
        print(FILE_X[i])
        print('Câu 1: Tìm biên nguyên âm')
        if v_segment.size == speech_segment.size:
            ae = 0
            for i in range(v_segment.size):
                ae += np.sum(np.abs(v_segment - speech_segment))
                ae *= frame_shift
            print('Sai số tuyệt đối giữa các biên: ' + str(ae) + ' ms')
        
        F_delta = math.fabs(float(f[1][1])-F)
        Fstd_delta = math.fabs(float(f[0][1]) - Fstd)
        _F = (F_delta/F)*100
        _Fstd = (Fstd_delta/Fstd)*100
        
        print('Câu 2: Tìm tần số')
        print('F lab: ' + str(f[1][1]))
        print('Fstd lab: '+ str(f[0][1]))
        print('F tìm được: ' + str(F))
        print('Fstd: ' + str(Fstd))
        print('delta: ' + str(F_delta))
        print('Sai số tương đối của F: ' + str(_F) + '%')
        print('Sai số tương đối của Fstd: ' + str(_Fstd) + '%')
        print('\n')
        
        t = np.linspace(0, len(X)/Fs, num = len(X))
        tt = np.linspace(0, len(X)/Fs, num = len(STE))
        ttt = speech*(frame_shift*Fs)/Fs
        speech_segment = speech_segment.reshape(-1)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(2, 2, 1)
        plt.title("Energy-based Speech/Silence discrimination")
        plt.plot(t, X)
        plt.plot(tt, STE, '-')
        
        for i in range(tt[speech_segment].shape[0]):
            if i != 0:
                plt.axvline(tt[speech_segment][i], color='r', linestyle='-', linewidth=1)
            else: 
                plt.axvline(tt[speech_segment][i], color='r', linestyle='-', linewidth=1, label='Biên tự động')
        for i in range(tt[v].shape[0]):
            if i != 0:
                plt.axvline(tt[v][i], color='b', linestyle='-', linewidth=1)
            else:
                plt.axvline(tt[v][i], color='b', linestyle='-', linewidth=1, label='Biên chuẩn')
        plt.legend(loc='upper right')
        
        ax1 = plt.subplot(2, 2, 3)
        ax1.plot(t, _)
        ax1.set_ylim([0,400])
        ax1.plot(F00[:,0], F00[:,1], 'r.')
        
        plt.title("frequency")
        plt.xlabel("Time")
        plt.ylabel("Hz")
        
        plt.subplot(2, 2, 2)
        l1, = plt.plot(freq[:500], dftx[:500])
        l2, = plt.plot(freq[:500], dfft[:500])
        l3, = plt.plot(freq[:500], env[:500])
        
        plt.legend((l1,l2,l3), ["zero-crossing spectrum", "spectrum", "envelope spectrum"], loc='upper right')
        plt.title("Analytics spectrum")
        plt.xlabel("Hz")
        plt.ylabel("Magnitude")
        plt.subplot(2, 2, 4)
        plt.plot(np.arange(ACF_vals.size), ACF_vals)
        plt.plot(np.arange(ACF_vals.size)[peaks_ACF[0]], ACF_vals[peaks_ACF[0]], '.')
        plt.title("Spectral Autocorrelation")
        plt.xlabel("Lag")
        plt.ylabel("Magnitude")
        
        # plt.subplot(2, 1, 1)
        # plt.title("Energy-based Speech/Silence discrimination")
        # plt.plot(t, X)
        # plt.plot(tt, STE, '-')
        # # plt.plot(ttt, F00/100, '.')
        # for i in range(tt[speech_segment].shape[0]):
        #     if i != 0:
        #         plt.axvline(tt[speech_segment][i], color='r', linestyle='-', linewidth=1)
        #     else: 
        #         plt.axvline(tt[speech_segment][i], color='r', linestyle='-', linewidth=1, label='Biên tự động')
        # for i in range(tt[v].shape[0]):
        #     if i != 0:
        #         plt.axvline(tt[v][i], color='b', linestyle='-', linewidth=1)
        #     else:
        #         plt.axvline(tt[v][i], color='b', linestyle='-', linewidth=1, label='Biên chuẩn')
        # plt.legend(loc='upper right')
        
        # ax1 = plt.subplot(2, 1, 2)
        # ax1.plot(t, _)
        # ax1.set_ylim([0,400])
        # ax1.plot(F00[:,0], F00[:,1], 'r.')
        
        # plt.title("frequency")
        # plt.xlabel("Time")
        # plt.ylabel("Hz")
        
        # plt.subplots_adjust(left=0.1,
        #             bottom=0.1, 
        #             right=0.9, 
        #             top=0.9, 
        #             wspace=0.4, 
        #             hspace=0.4)
        plt.tight_layout(h_pad=0.544, w_pad=0.221)
        plt.show()