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
                            filename=args.expdir + "/f0_range.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/f0_range.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/f0_range.log")
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
        in_file = os.path.join(folder,spk+'_f0histogram.txt')
        logging.info(in_file)
        arr_data = np.loadtxt(in_file)
    
        length = arr_data.shape[0]
        left_min = 999999999
        left_min_idx = -1
        right_min = 999999999
        right_min_idx = -1

        freqs = arr_data[:,0]
        idx_80 = np.where(freqs > 80)
        #logging.info(arr_data[idx_80,1])
        peak = np.max(arr_data[idx_80,1])
        peak_idx_80 = np.argmax(arr_data[idx_80,1])
        f0_peak_idx = arr_data[idx_80][peak_idx_80,0]
        peak_idx = np.where(arr_data[:,0] == f0_peak_idx)[0][0]
        logging.info(peak_idx_80)
        logging.info(f0_peak_idx)
        logging.info(peak_idx)

        #peak = np.max(arr_data[:,1])
        #peak_idx = np.argmax(arr_data[:,1])
    
        # left min
        if arr_data[peak_idx,0] > 90:
            left_left_f0 = arr_data[peak_idx,0]//2+40-15+1
            logging.info(left_left_f0)
            if left_left_f0 > 150:
                left_left_f0 = 130
                left_left_f0_idx = int(left_left_f0)-40+1
                left_left_max = np.max(arr_data[:left_left_f0_idx,1])
                left_left_max_idx = np.argmax(arr_data[:left_left_f0_idx,1])
            else:
                left_left_f0_idx = int(left_left_f0)-40+1
                left_left_max = np.max(arr_data[:left_left_f0_idx,1])
                left_left_max_idx = np.argmax(arr_data[:left_left_f0_idx,1])
                if left_left_max >= 0.0045:
                    left_left_f0 -= 20
                    left_left_f0_idx = int(left_left_f0)-40+1
                    left_left_max = np.max(arr_data[:left_left_f0_idx,1])
                    left_left_max_idx = np.argmax(arr_data[:left_left_f0_idx,1])
                    while left_left_max < 0.000045:
                        left_left_max_idx += 1
                        left_left_max = arr_data[left_left_max_idx,1]
            logging.info('%lf %d %d' % (left_left_max, left_left_max_idx, arr_data[left_left_max_idx,0]))
            logging.info('%lf %d %d' % (peak, peak_idx, arr_data[peak_idx,0]))
            left_right_min = np.min(arr_data[left_left_max_idx+1:peak_idx,1])
            left_right_min_idx = np.argmin(arr_data[left_left_max_idx+1:peak_idx,1])+left_left_max_idx
            if left_left_max - left_right_min >= 0.001: #saddle min
                left_min = left_right_min
                left_min_idx = left_right_min_idx
            else:
                for i in range(left_left_max_idx-1,-1,-1):
                    if left_min_idx == -1 and arr_data[i,1] < 0.0006:
                        left_min = arr_data[i+1,1]
                        left_min_idx = i+1
                    elif left_min_idx != -1 and arr_data[i,1] >= 0.001:
                        left_min = 999999999
                        left_min_idx = -1
            logging.info('%lf %d %d' % (left_right_min, left_right_min_idx, arr_data[left_right_min_idx,0]))
        else:
            for i in range(peak_idx-1,-1,-1):
                if left_min_idx == -1 and arr_data[i,1] < 0.0006:
                    left_min = arr_data[i+1,1]
                    left_min_idx = i+1
                elif left_min_idx != -1 and arr_data[i,1] >= 0.001:
                    left_min = 999999999
                    left_min_idx = -1
    
        # right min
        flag_count = 0
        tmp_right_min_idx = -1
        tmp_right_min = 99999999
        for i in range(peak_idx+1,length):
            #if right_min_idx == -1 and arr_data[i,1] < 0.00009:
            if flag_count < 4 and right_min_idx == -1 and arr_data[i,1] <= 0.00013:
                flag_count += 1
            elif flag_count >= 4 and arr_data[i,1] <= 0.00013:
                right_min = arr_data[i-1,1]
                right_min_idx = i-1
                break
            #elif right_min_idx != -1 and arr_data[i,1] >= 0.00013:
            #    right_min = 999999999
            #    right_min_idx = -1
    
        logging.info('%d %d %lf' % (left_min_idx, arr_data[left_min_idx][0], left_min))
        logging.info('%d %d %lf' % (peak_idx, arr_data[peak_idx][0], peak))
        logging.info('%d %d %lf' % (right_min_idx, arr_data[right_min_idx][0], right_min))
        out_file = os.path.join(conf,spk+'.f0')
        logging.info(out_file)
        f = open(out_file, 'w')
        f.write('%d %d\n' % (arr_data[left_min_idx,0], arr_data[right_min_idx,0]))
        f.close()


if __name__ == "__main__":
    main()
