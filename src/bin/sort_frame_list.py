#!/usr/bin/env python

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import multiprocessing as mp

import argparse
import os
import sys

import logging
import time

#from collections import ChainMap
from functools import reduce

from utils import find_files
from utils import shape_hdf5
from utils import read_txt

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save log")
    parser.add_argument("--feats", required=True,
                        type=str, help="feat list")
    parser.add_argument("--waveforms", required=True,
                        type=str, help="wav list")
    parser.add_argument("--spk_list", required=True,
                        type=str, help="wav list")
    parser.add_argument(
        "--n_jobs", default=10,
        type=int, help="number of parallel jobs")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/sort_frame_list.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/sort_frame_list.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/f0_range.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    if os.path.isdir(args.waveforms):
        filenames = sorted(find_files(args.waveforms, "*.wav", use_dir_name=False))
        wav_list = [args.waveforms + "/" + filename for filename in filenames]
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)
    if os.path.isdir(args.feats):
        feat_list = [args.feats + "/" + filename for filename in filenames]
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)
    assert len(wav_list) == len(feat_list)

    def get_max_frame(l, feat_list, wav_list, cpu, spk_list, spk_dict):
        tmp_spk_dict = {}
        for spk in spk_list:
            tmp_spk_dict[spk] = {}
        #count = 0
        for feat, wav in  zip(feat_list, wav_list):
            n_frame = shape_hdf5(feat, '/f0_range')[0]
            spk = os.path.basename(os.path.dirname(feat))
            tmp_spk_dict[spk][feat+"@"+wav] = n_frame
            logging.info(f'{cpu} {spk} {feat} {wav} {n_frame}')
        #    logging.info(tmp_spk_dict)
            #break
        #    count += 1
        #    if count > 2:
        #    if count > 5:
        #        break
        #time.sleep(10)
        l.acquire()
        try:
            for spk in spk_list:
                if bool(tmp_spk_dict[spk]):
                    spk_dict[spk].append(tmp_spk_dict[spk])
        finally:
            l.release()
    #    logging.info(spk_dict)

    # divide list
    wav_lists = np.array_split(wav_list, args.n_jobs)
    wav_lists = [f_list.tolist() for f_list in wav_lists]
    feat_lists = np.array_split(feat_list, args.n_jobs)
    feat_lists = [f_list.tolist() for f_list in feat_lists]

    for i in range(len(feat_lists)):
        logging.info("%d %d" % (i+1, len(feat_lists[i])))

    spk_list = args.spk_list.split('@')
    n_spk = len(spk_list)
    logging.info(spk_list)

    # multi processing
    with mp.Manager() as manager:
        processes = []
        spk_dict = manager.dict()
        for spk in spk_list:
            spk_dict[spk] = manager.list()
        lock = mp.Lock()
        for i, (feat_list, wav_list) in enumerate(zip(feat_lists, wav_lists)):
            p = mp.Process(target=get_max_frame, args=(lock, feat_list, wav_list, i+1, spk_list, spk_dict,))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        count = 0
        spk_dict_res = {}
        flatten = lambda x:[i for row in x for i in row]
        data_dir = os.path.dirname(args.feats)
        sort_feat = os.path.join(data_dir, os.path.basename(args.feats).split(".")[0]+"_sort.scp")
        sort_wav = os.path.join(data_dir, os.path.basename(args.waveforms).split(".")[0]+"_sort.scp")
        logging.info(sort_feat)
        logging.info(sort_wav)
        file_sort_feat = open(sort_feat, "w")
        file_sort_wav = open(sort_wav, "w")
        for key in spk_dict:
            #if bool(spk_dict[key]):
            if len(spk_dict[key]) > 0:
                count += 1
                #for i in range(len(spk_dict[key])):
        #        logging.info(spk_dict[key])
                #logging.info(flatten(spk_dict[key]))
                #final_map = ChainMap(*spk_dict[key])
                final_dict = reduce(lambda d, src: d.update(src) or d, spk_dict[key], {})
                sort_dict = (dict(reversed(sorted(final_dict.items(), key=lambda item: item[1]))))
                #final_dict = dict(d.items()[0] for d in spk_dict[key])
                #logging.info(final_dict)
        #        logging.info(sort_dict)
                for paths, frame in sort_dict.items():
                    paths_split = paths.split("@")
                    feat_key = paths_split[0]
                    wav_key = paths_split[1]
                    logging.info(f'{feat_key} {wav_key} {frame}')
                    file_sort_feat.write(feat_key+"\n")
                    file_sort_wav.write(wav_key+"\n")
                logging.info(f'{count} {key} {len(spk_dict[key])}')
        file_sort_feat.close()
        file_sort_wav.close()
        #exit()

    #    logging.info(spk_dict)
        #data_dir = os.path.dirname(args.feats)
        #sort_feat = os.path.join(data_dir, os.path.basename(args.feats).split(".")[0]+"_sort.h5")
        #sort_wav = os.path.join(data_dir, os.path.basename(args.waveforms).split(".")[0]+"_sort.h5")
        #logging.info(sort_feat)
        #logging.info(sort_wav)
        #file_sort_feat = open(sort_feat, "w")
        #file_sort_wav = open(sort_wav, "w")
        #count = 0
        #for spk in spk_list:
        #    if bool(spk_dict[spk]):
        #        count += 1
        #        sort_dict = (dict(reversed(sorted(spk_dict[spk].items(), key=lambda item: item[1]))))
        #        for key, value in sort_dict.items():
        #            key_split = key.split("@")
        #            feat_key = key_split[0]
        #            wav_key = key_split[1]
        #        #    logging.info(f'{feat_key} {wav_key} {value}')
        #            file_sort_feat.write(feat_key+"\n")
        #            file_sort_wav.write(wav_key+"\n")
        #logging.info(f'{count}')
        #file_sort_feat.close()
        #file_sort_wav.close()


if __name__ == "__main__":
    main()
