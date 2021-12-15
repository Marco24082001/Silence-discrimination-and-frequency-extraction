# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:18:59 2021

@author: ThanhVi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft, rfft
from scipy.signal import find_peaks
import sounddevice as sd
import math
import os
import soundfile as sf
def normalize(X):
    X = X.reshape(-1,)
    # return X/32676
    return X/X[np.argmax(np.fabs(X))]

def downsample(X, factor):
    return X[::factor]
# get median
def median(x):
    midx = int(x.shape[0]/2)    # middle point
    x= np.sort(x)   # sort ascending
    return x[midx]  # return middle value

def median_filter(x):
    n = len(x)  # length of x
    for i in range(0, 2):
        x[i] = median(x[i:i+5])
    for i in range(2, n-2): 
        x[i] = median(x[i-2:i+3])   # select slide window = 5
    return x

def mean(x):
    return np.sum(x)/x.shape[0]

def stand(x):
    y = mean(x)
    return np.sqrt(np.sum((x-y)**2)/(x.shape[0]))

def ShortTermEnergy(X, Fs, frame_size, frame_shift):
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    STE = []
    summ = 0
    for i in range(math.floor(len(X)/sample_shift) - math.ceil(window_length/sample_shift)):
        summ = np.sum(X[(i*sample_shift):(i*sample_shift) + window_length]**2)
        STE.append(summ)
    return np.array(STE)

def SegmentSpeech(X, Fs, frame_size, frame_shift, threshold):
    STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
    STE = normalize(STE)
    SEG = np.where(STE > threshold, 1, 0)
    # 0 0 0 > 300 ms silence else speech => 300/20 = 15 frame
    # k = 300/frame_size

    for i in range(6, len(STE)-6):
        if np.any(SEG[i-6:i] == 1) and np.any(SEG[i:i+7] == 1):
            SEG[i] = 1
    speech_segment = []
    speech = []
    for j in range(len(SEG) - 1):
        if (SEG[j] == 0 and SEG[j + 1] == 1) or (SEG[j] == 1 and SEG[j+1] == 0):
            speech_segment.append(j+1)
        if (SEG[j] == 1):
            speech.append(j)
    return np.array(speech),np.array(speech_segment)

def read_result(x):
    f = open(x, 'r', encoding = 'utf-8')
    r = f.readlines()
    r = [s.replace('\n', ' ') for s in r]
    r = [s.replace('\t', ' ') for s in r]
    r = [s.split(' ') for s in r]
    [s.remove('') for s in r if len(s)!=2]
    
    F = np.array(r[:-3:-1]) # frequency standard and mean
    V = np.array(r[:-2])
    v = np.delete(V, np.where(V[:,2]=='sil'), axis=0)[:,:-1]  # voice region
    sil = np.delete(V, np.where(V[:,2]!='sil'), axis=0)[:,:-1]  # silence region 
    return v, sil, F

def findThreadSTE(X, Fs, frame_size, frame_shift):
    
    pass

def findPeak(dfft, idx):
    n = dfft.size
    for i in range(idx, n-4):
        if np.argmax(dfft[i:i+5]) == 2:
            if dfft[i + 1] > dfft[i] and dfft[i + 4] < dfft[i + 3]:
                return i + 3
    
   
def ACF(X, W, t, lag):
    # m=0 -> N-1-n
    return np.sum(
            X[t : t + W - 1 - lag] *
            X[lag + t : lag + t + W - 1 - lag]
            )

def detect_pitch_ACF(X, Fs, Fmin, Fmax, frame_size, frame_shift, speech_segment, N_FFT):
    N_FFT = 1024*8
    resolution = Fs/N_FFT
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    w = np.hamming(window_length)   # window hamming
    freq = np.fft.rfftfreq(N_FFT, d=1./Fs)
    F00 = []
    F0 = 0
    
    # # x = X[(speech_segment[134]*sample_shift):(speech_segment[134]*sample_shift) + window_length] * w
    # dftx = np.abs(rfft(x, N_FFT))
    # dftx = normalize(dftx)
    # bounds = [int(Fmin/resolution) + 5 , int(Fmax/resolution) ]
    # ACF_vals = np.array([ACF(dftx, dftx.size, 0, l) for l in range(*bounds)])
    # print(bounds[0] + np.argmax(ACF_vals[0:int(Fmax/resolution)]))
    # F0 = freq[bounds[0] + np.argmax(ACF_vals[0:int(Fmax/resolution)])]
    # print(F0)
    # print(resolution)
    for i in range(len(speech_segment)):
        # print(int(Fmin/resolution))
        x = X[(speech_segment[i]*sample_shift):(speech_segment[i]*sample_shift) + window_length] * w
        bounds = [int(Fmin/resolution), int(Fmax/resolution)]
        # print(bounds)
        dftx = np.abs(rfft(x, N_FFT))
        dftx = normalize(dftx)
        ACF_vals = normalize(np.array([ACF(dftx, dftx.size, 0, l) for l in range(*bounds)]))
        # F0 = freq[bounds[0] + np.argmax(ACF_vals[0:int(Fmax/resolution)])]
        # print(F0)
        
        peaks = find_peaks(ACF_vals, height= 0.4)
        if len(peaks[0]) != 0:
            F0 = freq[bounds[0] + peaks[0][0]]
        else:
            F0 = freq[0]
        # idx = np.argmax(ACF_vals[20: int(Fmax/resolution)])
        # next_peak = findPeak(dftx, int(Fmin/resolution) + 1)
        # F0 = freq[20 + idx]
        F00.append(F0)
        plt.figure(figsize=(50, 20))
        plt.subplot(2, 1, 1)
        plt.plot(freq, dftx)
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(ACF_vals.size), ACF_vals)
        plt.plot(np.arange(ACF_vals.size)[peaks[0]], ACF_vals[peaks[0]], 'ro')
        # plt.plot(np.arange(ACF_vals.size)[20 + idx], ACF_vals[20 + idx], 'ro')
        # break
    F00 = median_filter(np.array(F00))
        
    return np.array(F00)

    

def detect_pitch_HPS(X, Fs, Fmin, Fmax, frame_size, frame_shift, speech_segment, N_FFT): # Fmin = 70 Fmax = 450
    noise_value = 0.05
    N_FFT = int(X.size/2)
    resolution = Fs/N_FFT
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    w = np.hamming(window_length)   # window hamming
    iOrder = 9
    freq = np.fft.rfftfreq(N_FFT, d=1./Fs)
    F00 = []
    F0 = 0
    loop = 2
    for i in range(len(speech_segment)):
        x = X[(speech_segment[i]*sample_shift):(speech_segment[i]*sample_shift) + window_length] * w
        
        dftx = np.abs(fft(x, N_FFT))
        # dftx = np.log(dftx)
        dftx = normalize(dftx)
        # dftx[0] = 0
        # dftx[1] = 0
        # dftx[2] = 0
        iLen = int((dftx.shape[0]) / (iOrder-1))
        afHps = dftx[np.arange(0, iLen)]
        t = np.linspace(0,Fs, num= N_FFT)
        # t = t
        
        bound_left = int(Fmin/resolution)
        afHps[:bound_left] = 0
        # bound_right = int(Fmax/resolution) + 1
        
        for j in range(2, iOrder):
            _dftx = downsample(dftx, j)
            afHps *= _dftx[np.arange(0, iLen)]
        
        # low_value_indexs = dftx < noise_value
        # dftx[low_value_indexs] = 0
        idx = np.argmax(dftx)
        next_peak = findPeak(dftx, idx)
        F0 = freq[next_peak] - freq[idx]
        # print(F0)
        
        plt.figure(figsize=(50, 20))
        plt.subplot(3, 1, 1)
        plt.plot(freq[:int(iLen)], dftx[: int(iLen)])
        plt.plot(freq[idx], dftx[idx], 'ro')
        plt.plot(freq[next_peak], dftx[next_peak], 'ro')
        plt.subplot(3, 1, 2)
        plt.plot(freq[:int(iLen)], afHps)
        # plt.subplot(3, 1, 3)
        # plt.plot(t[:int(iLen)], afHps)
        # plt.title(str(i))
        
        
        
        # print(F0)
        
        F00.append(F0)
        # loop = loop - 1
        # if loop == 0:
        #     break

    return np.array(F00)



def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def PlotfftFrame(X, Fs, frame_size, frame_shift, N_FFT):
    N_FFT = X.size
    resolution = Fs/N_FFT
    g = int(70/resolution)
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    x_frame = X[59*sample_shift:(59*sample_shift) + window_length]
    w = np.hamming(len(x_frame))
    x_frame = x_frame*w
    tt = np.linspace(0,Fs, num= N_FFT)
    t = np.linspace(0, 2, num = 2*Fs)
    freq = np.fft.rfftfreq(N_FFT, d=1./Fs)
    # dfty = np.abs(fft(X*np.hamming(len(X)), N_FFT))
    dfty = np.abs(rfft(x_frame, N_FFT))
    signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    n = signal.size
    sample_rate = 100
    print(freq)

    x_frame2 = downsample(dfty, 2)
    x_frame3 = downsample(dfty, 3)
    x_frame4 = downsample(dfty, 4)
    x_frame2 = np.append(x_frame2, np.zeros(len(dfty) - len(x_frame2)))
    x_frame3 = np.append(x_frame3, np.zeros(len(dfty) - len(x_frame3)))
    x_frame4 = np.append(x_frame4, np.zeros(len(dfty) - len(x_frame4)))
    
    hps = dfty * x_frame2 * x_frame3 * x_frame4
        
    plt.figure(figsize=(200, 20))
    plt.subplot(3, 1, 1)
    plt.plot(freq[:int(N_FFT/2 + 1)], dfty[:int(N_FFT/2 + 1)])
    plt.subplot(3, 1, 2)
    plt.plot(freq[:int(N_FFT/2 + 1)], x_frame2[:int(N_FFT/2 + 1)])
    # plt.subplot(3, 1, 3)
    # plt.plot(t[:int(N_FFT)], X[:int(N_FFT)])
    plt.show()
    

    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(dfty))  # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(dfty)), i)[0]
    print(true_i)
    print(Fs * true_i / len(x_frame))
    # Convert to equivalent frequency
    return Fs * true_i / len(x_frame)
    

frame_shift = 10/1000
frame_size = 20/1000
threshold = 0.013
N_FFT = 1024 * 4
Fmin = 70
Fmax = 450
    

Fs = 10000
F1 = 200
F2 = 100
duration = 5
nSamples = 100

t = np.linspace(0, 2, num = 2*Fs)
# Fs, X = wavfile.read('TinHieuHuanLuyen/01MDA.wav') # 135.5
# Fs, X = wavfile.read('TinHieuHuanLuyen/02FVA.wav') # 239.7
# Fs, X = wavfile.read('TinHieuHuanLuyen/03MAB.wav') # 115.0
Fs, X = wavfile.read('TinHieuHuanLuyen/06FTB.wav') # 202.9
# Fs, X = wavfile.read('TinHieuKiemThu/30FTN.wav') # 233.2
# Fs, X = wavfile.read('TinHieuKiemThu/42FQT.wav') # 242.7
# Fs, X = wavfile.read('TinHieuKiemThu/44MTT.wav') # 125.7
# Fs, X = wavfile.read('TinHieuKiemThu/45MDV.wav') # 177.8


X = normalize(X)
speech, speech_segment = SegmentSpeech(X, Fs, frame_size, frame_shift, threshold)
STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
STE = normalize(STE)
tt = np.linspace(0, len(X)/Fs, num = len(STE))

# X = 2*np.sin(2*np.pi*F1*t) + 5*np.sin(5*np.pi*F1*t)+ 5*np.sin(6*np.pi*F1*t) + 5*np.sin(7*np.pi*F1*t)+ 5*np.sin(10*np.pi*F1*t)

v, sil, F = read_result("TinHieuHuanLuyen/06FTB.lab")
v = v.astype(float) * Fs
v = (v/(frame_shift*Fs)).astype(int).reshape(-1)
# PlotfftFrame(X, Fs, frame_size, frame_shift, N_FFT)
F00 = detect_pitch_ACF(X, Fs, Fmin, Fmax, frame_size, frame_shift, speech, N_FFT)
# F00 = detect_pitch_HPS(X, Fs, Fmin, Fmax, frame_size, frame_shift, speech, N_FFT)

F = np.sum(F00)/ len(F00)

print('result: ')
print(F)
# print(F00/200)

t = np.linspace(0, len(X)/Fs, num = len(X))
ttt = speech*(frame_shift*Fs)/Fs
plt.figure(figsize=(100, 20))
plt.subplot(2, 1, 1)
plt.plot(t, X)
plt.plot(tt, STE, '-')
plt.plot(ttt, F00/100, '.')
for i in range(tt[speech_segment].shape[0]):
    plt.axvline(tt[speech_segment][i], color='r', linestyle=':', linewidth=2)
for i in range(tt[v].shape[0]):
    plt.axvline(tt[v][i], color='b', linestyle=':', linewidth=2)
plt.show()

# Y = X
# X = normalize(X)


# STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
# STE = normalize(STE)

# speech, speech_segment = SegmentSpeech(X, Fs, frame_size, frame_shift, threshold)



# F00 = detect_pitch_HPS(Y, Fs, Fmin, Fmax, frame_size, frame_shift, speech, N_FFT)
# F00 = median_filter(F00)
# F = np.sum(F00)/ len(F00)

# print('result')

# print(F)

# tt = np.linspace(0, len(X)/Fs, num = len(STE))
# ttt = speech*(frame_shift/1000*Fs)/Fs

# plt.figure(figsize=(50, 20))
# plt.subplot(2, 1, 1)
# plt.plot(t, X)
# plt.plot(tt, STE, '-')
# plt.plot(ttt, F00/200, '.')

# for i in range(tt[speech_segment].shape[0]):
#     plt.axvline(tt[speech_segment][i], color='b', linestyle=':', linewidth=2)
# plt.show()
# PlotfftFram(X, Fs, frame_size, frame_shift)











# # sd.play(y, Fs)

# y = normalize(y)

# plt.figure(figsize=(40, 20))
# plt.subplot(2, 1, 1)
# plt.plot(t[:nSamples], y[:nSamples])
# N_FFT = 1024
# dfty = np.abs(fft(y, N_FFT))
# tt = np.linspace(0,Fs, num= N_FFT)
# plt.subplot(2,1,2)
# plt.plot(tt[:int(N_FFT/2 + 1)], dfty[:int(N_FFT/2 + 1)])




