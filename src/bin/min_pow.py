#!/usr/bin/env python

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import logging

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save log")
    parser.add_argument("--featdir", required=True,
                        type=str, help="directory of feature extraction log")
    parser.add_argument("--confdir", required=True,
                        type=str, help="directory of speaker config.")
    parser.add_argument("--spk_list", required=True,
                        type=str, help="speaker list")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/min_pow.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/min_pow.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/min_pow.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    spks = args.spk_list.split('@')
    folder = args.featdir
    conf = args.confdir
    logging.info(spks)
    logging.info(folder)
    logging.info(conf)
    for spk in spks:
        logging.info(spk)
        in_file = os.path.join(folder,spk+'_npowhistogram.txt')
        logging.info(in_file)
        arr_data = np.loadtxt(in_file)
    
        length = arr_data.shape[0]
        peak_1 = -999999999
        peak_1_idx = 0
        global_min = 999999999
        global_min_idx = length // 2 - 1
        peak_2 = -999999999
        peak_2_idx = length-1
        list_min_global_idx = []
    
        for i in range(length // 2 - 2):
            if arr_data[i][1] > peak_1:
                peak_1_idx = i
                peak_1 = arr_data[i][1]
        for i in range(length-1,(length - length // 3),-1):
            if arr_data[i][1] > peak_2:
                peak_2_idx = i
                peak_2 = arr_data[i][1]
        for i in range(length):
            if arr_data[i][1] <= global_min and i > peak_1_idx and i < peak_2_idx:
                global_min_idx = i
                if arr_data[i][1] == global_min:
                    list_min_global_idx.append(arr_data[i][0])
                else:
                    list_min_global_idx = []
                    list_min_global_idx.append(arr_data[i][0])
                global_min = arr_data[i][1]
        min_pow = np.mean(list_min_global_idx)
    
        logging.info('%d %d %lf' % (peak_1_idx, arr_data[peak_1_idx][0], peak_1))
        logging.info('%d %d %lf' % (global_min_idx, arr_data[global_min_idx][0], global_min))
        logging.info('%d %d %lf' % (peak_2_idx, arr_data[peak_2_idx][0], peak_2))
        logging.info(list_min_global_idx)
        logging.info(min_pow)
        out_file = os.path.join(conf,spk+'.pow')
        logging.info(out_file)
        f = open(out_file, 'w')
        f.write('%.1f\n' % (min_pow))
        f.close()


if __name__ == "__main__":
    main()
