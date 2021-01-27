#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
from distutils.util import strtobool
import multiprocessing as mp
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

import torch

from pqmf import PQMF

from utils import find_files
from utils import read_txt

##FS = 16000
#FS = 22050
FS = 24000
##FS = 44100
##FS = 48000
ALPHA = 0.85
N_BANDS = 5


def deemphasis(x, alpha=ALPHA):
    b = np.array([1.], x.dtype)
    a = np.array([1., -alpha], x.dtype)
    return lfilter(b, a, x)


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument(
        "--waveforms", default=None,
        help="directory or list of filename of input wavfile")
    parser.add_argument(
        "--writedir", default=None,
        help="directory to save preprocessed wav file")
    parser.add_argument(
        "--writesyndir", default=None,
        help="directory to save preprocessed wav file")
    parser.add_argument(
        "--fs", default=FS,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--n_bands", default=N_BANDS,
        type=int, help="number of bands for multiband analysis")
    parser.add_argument(
        "--alpha", default=ALPHA,
        type=float, help="coefficient of pre-emphasis")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")
    parser.add_argument(
        '--n_jobs', default=1,
        type=int, help="number of parallel jobs")
    args = parser.parse_args()

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)

    # check directory existence
    if not os.path.exists(args.writedir):
        os.makedirs(args.writedir)
    if not os.path.exists(args.writesyndir):
        os.makedirs(args.writesyndir)

    def noise_shaping(wav_list):
        pqmf = PQMF(args.n_bands)
        print(f'{pqmf.subbands} {pqmf.A} {pqmf.taps} {pqmf.cutoff_ratio} {pqmf.beta}')
        for wav_name in wav_list:
            x, fs = sf.read(wav_name)

            ## check sampling frequency
            if not fs == args.fs:
                print("ERROR: sampling frequency is not matched.")
                sys.exit(1)

            x_bands_ana = pqmf.analysis(torch.FloatTensor(x).unsqueeze(0).unsqueeze(0))
            print(x_bands_ana.shape)
            x_bands_syn = pqmf.synthesis(x_bands_ana)
            print(x_bands_syn.shape)
            for i in range(args.n_bands):
                wav = np.clip(x_bands_ana[0,i].data.numpy(), -1, 0.999969482421875)
                if args.n_bands < 10:
                    wavpath = os.path.join(args.writedir, os.path.basename(wav_name).split(".")[0]+"_B-"+str(i+1)+".wav")
                else:
                    if i < args.n_bands - 1:
                        wavpath = os.path.join(args.writedir, os.path.basename(wav_name).split(".")[0]+"_B-0"+str(i+1)+".wav")
                    else:
                        wavpath = os.path.join(args.writedir, os.path.basename(wav_name).split(".")[0]+"_B-"+str(i+1)+".wav")
                print(wavpath)
                sf.write(wavpath, wav, fs, 'PCM_16')
            wav = np.clip(x_bands_syn[0,0].data.numpy(), -1, 0.999969482421875)
            wav = deemphasis(wav, alpha=args.alpha)
            wavpath = os.path.join(args.writesyndir, os.path.basename(wav_name))
            print(wavpath)
            sf.write(wavpath, wav, fs, 'PCM_16')


    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    for f in file_lists:
        p = mp.Process(target=noise_shaping, args=(f,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
