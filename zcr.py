# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:56:09 2021

@author: ThanhVi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import math

def normalize(X):
    X = X.reshape(-1,)
    # return X/32676
    return X/X[np.argmax(np.fabs(X))]

def sgn(X):
    X = np.where(X >= 0, 1, -1)
    return X

def ZeroCrossingRate(X, Fs, frame_size, frame_shift):
    frame_size = frame_size/1000
    frame_shift = frame_shift/1000
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    ZCR = []
    summ = 0
    X = sgn(X)
    X_shift = np.append(np.ones(1), X[:len(X) - 1])
    for i in range(math.floor(len(X)/sample_shift) - math.ceil(window_length/sample_shift)):
        summ = np.sum(np.fabs(X[(i*sample_shift):(i*sample_shift) + window_length] - X_shift[(i*sample_shift):(i*sample_shift) + window_length]))
        ZCR.append(summ)
    return np.array(ZCR)

def ShortTermEnergy(X, Fs, frame_size, frame_shift):
    frame_size = frame_size/1000
    frame_shift = frame_shift/1000
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    len_X = len(X)
    frame_sample = (int)(frame_size*Fs)
    STE = []
    summ = 0
    for i in range(math.floor(len(X)/sample_shift) - math.ceil(window_length/sample_shift)):
        summ = np.sum(X[(i*sample_shift):(i*sample_shift) + window_length]**2)
        STE.append(summ)
    return np.array(STE)
    # t = np.linspace(frame_size, math.floor(len_X/frame_sample)*frame_size, num = len(energy))

def SegmentSpeech(X, Fs, frame_size, frame_shift, threshold):
    STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
    STE = normalize(STE)
    SEG = np.where(STE > threshold, 1, 0)
    # 0 0 0 > 300 ms silence else speech => 300/20 = 15 frame
    # k = 300/frame_size
    for i in range(6, len(STE)-6):
        if np.any(SEG[i-6:i] == 1) and np.any(SEG[i:i+7] == 1):
            SEG[i] = 1
    SP = []
    for i in range(len(SEG) - 1):
        if (SEG[i] == 0 and SEG[i + 1] == 1) or (SEG[i] == 1 and SEG[i+1] == 0):
            SP.append(i)
    return np.array(SP)

    

Fs, X = wavfile.read('TinHieuHuanLuyen/01MDA.wav')
# Fs, X = wavfile.read('TinHieuHuanLuyen/02FVA.wav')
# Fs, X = wavfile.read('TinHieuHuanLuyen/03MAB.wav')
# Fs, X = wavfile.read('TinHieuHuanLuyen/06FTB.wav')
# Fs, X = wavfile.read('TinHieuHuanLuyen/phone_male.wav')
# Fs, X = wavfile.read('TinHieuHuanLuyen/studio_male.wav')
X = normalize(X)
long = len(X)/Fs
t = np.linspace(0, long, num = len(X))

frame_shift = 10
frame_size = 20
threshold = 0.0030
ZCR = 0
ZCR = ZeroCrossingRate(X, Fs, 20, 10)
STE = ShortTermEnergy(X, Fs, 20, 10)
ZCR = normalize(ZCR)
STE = normalize(STE)
SP = SegmentSpeech(X, Fs, 20, 10, threshold)

tt = np.linspace(0, long, num = len(ZCR))
ttt = np.linspace(0, long, num = len(STE))
# print(len(ZCR))
    

sd.play(X, Fs)

plt.figure(figsize=(50, 20))
plt.subplot(2, 1, 1)
plt.plot(t, X)
plt.plot(tt, ZCR, '-')
plt.plot(tt, STE, '-')
# plt.plot(tt, SEG, '.')
print(SP)
for i in range(tt[SP].shape[0]):
    plt.axvline(tt[SP][i], color='b', linestyle=':', linewidth=2)
plt.show()

# print(ZCR)





