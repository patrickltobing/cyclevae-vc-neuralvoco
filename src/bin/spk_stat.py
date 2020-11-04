#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on a VC implementation by Kazuhiro Kobayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
from pathlib import Path
import logging

import matplotlib
import numpy as np
from utils import check_hdf5
from utils import read_hdf5
from utils import read_txt
from utils import write_hdf5

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument("--expdir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/spk_stat.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/spk_stat.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/spk_stat.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    filenames = read_txt(args.feats)
    logging.info("number of training utterances = %d" % len(filenames))

    npows = np.empty((0))
    f0s = np.empty((0))
    # process over all of data
    for filename in filenames:
        logging.info(filename)
        f0 = read_hdf5(filename, "/f0")
        npow = read_hdf5(filename, "/npow")
        nonzero_indices = np.nonzero(f0)
        logging.info(f0[nonzero_indices].shape)
        logging.info(f0s.shape)
        f0s = np.concatenate([f0s,f0[nonzero_indices]])
        logging.info(f0s.shape)
        logging.info(npows.shape)
        npows = np.concatenate([npows,npow])
        logging.info(npows.shape)

    spkr = os.path.basename(args.feats).split('.')[0].split('-')[-1]

    plt.rcParams["figure.figsize"] = (20,11.25) #1920x1080

    # create a histogram to visualize F0 range of the speaker
    f0histogrampath = os.path.join(args.expdir, spkr + '_f0histogram.png')
    f0hist, f0bins, _ = plt.hist(f0s, bins=500, range=(50, 550),
        density=True, histtype="stepfilled")
    # plot with matplotlib
    plt.xlabel('Fundamental frequency [Hz]')
    plt.ylabel("Probability")
    plt.xticks(np.arange(50, 551, 10), rotation=45)
    figure_dir = os.path.dirname(f0histogrampath)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(f0histogrampath)
    plt.close()
    # save values to txt
    f0histogrampath = os.path.join(args.expdir, spkr + '_f0histogram.txt')
    f = open(f0histogrampath, 'w')
    for i in range(f0hist.shape[0]):
        f.write('%d %.9f\n' % (f0bins[i], f0hist[i]))
    f.close()

    # create a histogram to visualize npow range of the speaker
    npowhistogrampath = os.path.join(args.expdir, spkr + '_npowhistogram.png')
    npowhist, npowbins, _ = plt.hist(npows, bins=120, range=(-50, 10),
        density=True, histtype="stepfilled")
    # plot with matplotlib
    plt.xlabel('Frame power [dB]')
    plt.ylabel("Probability")
    plt.xticks(np.arange(-50, 11, 1), rotation=45)
    figure_dir = os.path.dirname(npowhistogrampath)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(npowhistogrampath)
    plt.close()
    # save values to txt
    npowhistogrampath = os.path.join(args.expdir, spkr + '_npowhistogram.txt')
    f = open(npowhistogrampath, 'w')
    for i in range(npowhist.shape[0]):
        f.write('%.1f %.9f\n' % (npowbins[i], npowhist[i]))
    f.close()


if __name__ == '__main__':
    main()
