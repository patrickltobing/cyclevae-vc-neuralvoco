#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
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

from utils import find_files
from utils import read_txt

##FS = 16000
#FS = 22050
FS = 24000
##FS = 44100
##FS = 48000
ALPHA = 0.85

def preemphasis(x, alpha=ALPHA):
    b = np.array([1., -alpha], x.dtype)
    a = np.array([1.], x.dtype)
    return lfilter(b, a, x)


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
        "--fs", default=FS,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--alpha", default=ALPHA,
        type=float, help="coefficient of pre-emphasis")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")
    parser.add_argument(
        '--n_jobs', default=1,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        '--inv', default=False, type=strtobool,
        help="if True, inverse filtering will be performed")
    args = parser.parse_args()

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)

    # check directory existence
    if not os.path.exists(args.writedir):
        os.makedirs(args.writedir)

    def noise_shaping(wav_list):
        for wav_name in wav_list:
            # load wavfile and apply low cut filter
            x, fs = sf.read(wav_name)

            ## check sampling frequency
            if not fs == args.fs:
                print("ERROR: sampling frequency is not matched.")
                sys.exit(1)

            ## synthesis and write
            if not args.inv:
                x_ns = preemphasis(x, alpha=args.alpha)
            else:
                x_ns = deemphasis(x, alpha=args.alpha)
            write_name = args.writedir + "/" + os.path.basename(wav_name)
            sf.write(write_name, np.clip(x_ns, -1, 0.999969482421875), args.fs, 'PCM_16')

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
