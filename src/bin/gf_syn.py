#!/usr/bin/env python

import numpy as np
import librosa
import soundfile as sf

f = open('melsp.txt', 'r')

lines = f.readlines()

melmagsp = None
for line in lines:
    vals = np.expand_dims(np.array(line.strip().split(' ')).astype(np.float), axis=0)
    if melmagsp is not None:
        melmagsp = np.append(melmagsp, vals, axis=0)
    else:
        melmagsp = vals
#    print(vals)
    print(melmagsp.shape)

#fs = 22050
fs = 24000
#fftl = 1024
fftl = 2048
mel_dim = 80
shiftms = 5
#shiftms = 4.9886621315192743764172335600907
#shiftms = 10
#shiftms = 9.9773242630385487528344671201814
winms = 27.5
hop_length = int((fs/1000)*shiftms)
win_length = int((fs/1000)*winms)

melfb_t = np.linalg.pinv(librosa.filters.mel(fs, fftl, n_mels=mel_dim))
print(melfb_t.shape)
recmagsp = np.matmul(melfb_t, melmagsp.T)
print(recmagsp.shape)
wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
            win_length=win_length, window='hann'), -1, 0.999969482421875)
print(wav.shape)
sf.write('melsp_syn.wav', wav, fs, 'PCM_16')
