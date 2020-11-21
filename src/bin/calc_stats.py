#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import argparse
import multiprocessing as mp
import logging
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import check_hdf5
from utils import read_hdf5
from utils import read_txt
from utils import write_hdf5

from multiprocessing import Array


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--stats", default=None, required=True,
        help="filename of stats for hdf5 format")
    parser.add_argument("--expdir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument("--mcep_dim", default=50,
        type=int, help="dimension of mel-cepstrum")
    parser.add_argument(
        "--n_jobs", default=10,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # read list and define scaler
    filenames = read_txt(args.feats)
    logging.info("number of training utterances = "+str(len(filenames)))

    def calc_stats(filenames, cpu, feat_mceplf0cap_list, feat_orglf0_list, varmcep_list, f0_list,
            melsp_list, varmelsp_list, melworldsp_list, varmelworldsp_list):
        feat_mceplf0cap_arr = None
        feat_orglf0_arr = None
        varmcep_arr = None
        f0_arr = None
        melsp_arr = None
        varmelsp_arr = None
        melworldsp_arr = None
        varmelworldsp_arr = None
        count = 0
        # process over all of data
        for filename in filenames:
            logging.info(filename)
            feat_mceplf0cap = read_hdf5(filename, "/feat_mceplf0cap")
            logging.info(feat_mceplf0cap.shape)
            feat_orglf0 = read_hdf5(filename, "/feat_org_lf0")
            logging.info(feat_orglf0.shape)
            melsp = read_hdf5(filename, "/log_1pmelmagsp")
            logging.info(melsp.shape)
            melworldsp = read_hdf5(filename, "/log_1pmelworldsp")
            logging.info(melworldsp.shape)
            if feat_mceplf0cap_arr is not None:
                feat_mceplf0cap_arr = np.r_[feat_mceplf0cap_arr, feat_mceplf0cap]
            else:
                feat_mceplf0cap_arr = feat_mceplf0cap
            if feat_orglf0_arr is not None:
                feat_orglf0_arr = np.r_[feat_orglf0_arr, feat_orglf0]
            else:
                feat_orglf0_arr = feat_orglf0
            logging.info('feat')
            logging.info(feat_mceplf0cap_arr.shape)
            logging.info(feat_orglf0_arr.shape)
            if varmcep_arr is not None:
                varmcep_arr = np.r_[varmcep_arr, np.var(feat_mceplf0cap[:,-args.mcep_dim:], \
                                        axis=0, keepdims=True)]
            else:
                varmcep_arr = np.var(feat_mceplf0cap[:,-args.mcep_dim:], axis=0, keepdims=True)
            logging.info('var')
            logging.info(varmcep_arr.shape)
            logging.info('f0')
            f0 = read_hdf5(filename, "/f0_range")
            logging.info(f0.shape)
            logging.info('f0 > 0')
            f0 = f0[np.nonzero(f0)]
            logging.info(f0.shape)
            if f0_arr is not None:
                f0_arr = np.r_[f0_arr, f0]
            else:
                f0_arr = f0
            logging.info(f0_arr.shape)
            if melsp_arr is not None:
                melsp_arr = np.r_[melsp_arr, melsp]
            else:
                melsp_arr = melsp
            logging.info(melsp_arr.shape)
            if varmelsp_arr is not None:
                varmelsp_arr = np.r_[varmelsp_arr, np.var((np.exp(melsp)-1)/10000, axis=0, \
                                        keepdims=True)]
            else:
                varmelsp_arr = np.var((np.exp(melsp)-1)/10000, axis=0, keepdims=True)
            logging.info('var melsp')
            logging.info(varmelsp_arr.shape)
            if melworldsp_arr is not None:
                melworldsp_arr = np.r_[melworldsp_arr, melworldsp]
            else:
                melworldsp_arr = melworldsp
            logging.info(melworldsp_arr.shape)
            if varmelworldsp_arr is not None:
                varmelworldsp_arr = np.r_[varmelworldsp_arr, np.var((np.exp(melworldsp)-1)/10000, axis=0, \
                                        keepdims=True)]
            else:
                varmelworldsp_arr = np.var((np.exp(melworldsp)-1)/10000, axis=0, keepdims=True)
            logging.info('var melworldsp')
            logging.info(varmelworldsp_arr.shape)
            count += 1
            logging.info("cpu %d %d %d %d %d %d %d %d %d %d" % (cpu, count, len(feat_mceplf0cap_arr),
                    len(feat_orglf0_arr), len(varmcep_arr), len(f0_arr), len(melsp_arr),
                        len(varmelsp_arr), len(melworldsp_arr), len(varmelworldsp_arr)))
            #if count >= 5:
            #    break

        feat_mceplf0cap_list.append(feat_mceplf0cap_arr)
        feat_orglf0_list.append(feat_orglf0_arr)
        varmcep_list.append(varmcep_arr)
        f0_list.append(f0_arr)
        melsp_list.append(melsp_arr)
        varmelsp_list.append(varmelsp_arr)
        melworldsp_list.append(melworldsp_arr)
        varmelworldsp_list.append(varmelworldsp_arr)

    # divie list
    feat_lists = np.array_split(filenames, args.n_jobs)
    feat_lists = [f_list.tolist() for f_list in feat_lists]

    for i in range(len(feat_lists)):
        logging.info("%d %d" % (i+1, len(feat_lists[i])))

    # multi processing
    with mp.Manager() as manager:
        processes = []
        feat_mceplf0cap_list = manager.list()
        feat_orglf0_list = manager.list()
        varmcep_list = manager.list()
        f0_list = manager.list()
        melsp_list = manager.list()
        varmelsp_list = manager.list()
        melworldsp_list = manager.list()
        varmelworldsp_list = manager.list()
        for i, feat_list in enumerate(feat_lists):
            p = mp.Process(target=calc_stats, args=(feat_list, i+1, feat_mceplf0cap_list,
                        feat_orglf0_list, varmcep_list, f0_list, melsp_list, varmelsp_list,
                        melworldsp_list, varmelworldsp_list,))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        feat_mceplf0cap = None
        for i in range(len(feat_mceplf0cap_list)):
            if feat_mceplf0cap_list[i] is not None:
                logging.info(i)
                logging.info(feat_mceplf0cap_list[i].shape)
                if feat_mceplf0cap is not None:
                    feat_mceplf0cap = np.r_[feat_mceplf0cap, feat_mceplf0cap_list[i]]
                else:
                    feat_mceplf0cap = feat_mceplf0cap_list[i]
        logging.info('feat mceplf0cap: %d' % (len(feat_mceplf0cap)))
        logging.info(feat_mceplf0cap.shape)

        feat_orglf0 = None
        for i in range(len(feat_orglf0_list)):
            if feat_orglf0_list[i] is not None:
                logging.info(i)
                logging.info(feat_orglf0_list[i].shape)
                if feat_orglf0 is not None:
                    feat_orglf0 = np.r_[feat_orglf0, feat_orglf0_list[i]]
                else:
                    feat_orglf0 = feat_orglf0_list[i]
        logging.info('feat orglf0: %d' % (len(feat_orglf0)))
        logging.info(feat_orglf0.shape)

        var_range = None
        for i in range(len(varmcep_list)):
            if varmcep_list[i] is not None:
                logging.info(i)
                logging.info(varmcep_list[i].shape)
                if var_range is not None:
                    var_range = np.r_[var_range, varmcep_list[i]]
                else:
                    var_range = varmcep_list[i]
        logging.info('var mcep: %d' % (len(var_range)))
        logging.info(var_range.shape)

        f0s_range = None
        for i in range(len(f0_list)):
            if f0_list[i] is not None:
                logging.info(i)
                logging.info(f0_list[i].shape)
                if f0s_range is not None:
                    f0s_range = np.r_[f0s_range, f0_list[i]]
                else:
                    f0s_range = f0_list[i]
        logging.info('f0: %d' % (len(f0s_range)))
        logging.info(f0s_range.shape)

        melsp = None
        for i in range(len(melsp_list)):
            if melsp_list[i] is not None:
                logging.info(i)
                logging.info(melsp_list[i].shape)
                if melsp is not None:
                    melsp = np.r_[melsp, melsp_list[i]]
                else:
                    melsp = melsp_list[i]
        logging.info('melsp: %d' % (len(melsp)))
        logging.info(melsp.shape)

        var_melsp = None
        for i in range(len(varmelsp_list)):
            if varmelsp_list[i] is not None:
                logging.info(i)
                logging.info(varmelsp_list[i].shape)
                if var_melsp is not None:
                    var_melsp = np.r_[var_melsp, varmelsp_list[i]]
                else:
                    var_melsp = varmelsp_list[i]
        logging.info('var melsp: %d' % (len(var_melsp)))
        logging.info(var_melsp.shape)

        melworldsp = None
        for i in range(len(melworldsp_list)):
            if melworldsp_list[i] is not None:
                logging.info(i)
                logging.info(melworldsp_list[i].shape)
                if melworldsp is not None:
                    melworldsp = np.r_[melworldsp, melworldsp_list[i]]
                else:
                    melworldsp = melworldsp_list[i]
        logging.info('melworldsp: %d' % (len(melworldsp)))
        logging.info(melworldsp.shape)

        var_melworldsp = None
        for i in range(len(varmelworldsp_list)):
            if varmelworldsp_list[i] is not None:
                logging.info(i)
                logging.info(varmelworldsp_list[i].shape)
                if var_melworldsp is not None:
                    var_melworldsp = np.r_[var_melworldsp, varmelworldsp_list[i]]
                else:
                    var_melworldsp = varmelworldsp_list[i]
        logging.info('var melworldsp: %d' % (len(var_melworldsp)))
        logging.info(var_melworldsp.shape)

        scaler_feat_mceplf0cap = StandardScaler()
        scaler_feat_orglf0 = StandardScaler()

        logging.info(feat_mceplf0cap.shape)
        #min_mcep = np.min(feat_mceplf0cap[:,-args.mcep_dim:], axis=0)
        #max_mcep = np.max(feat_mceplf0cap[:,-args.mcep_dim:], axis=0)
        #logging.info(min_mcep)
        #logging.info(max_mcep)
        #write_hdf5(args.stats, "/min_mcep", min_mcep)
        #write_hdf5(args.stats, "/max_mcep", max_mcep)

        scaler_feat_mceplf0cap.partial_fit(feat_mceplf0cap)
        scaler_feat_orglf0.partial_fit(feat_orglf0)

        logging.info(melsp.shape)
        #min_melsp = np.min(melsp, axis=0)
        #max_melsp = np.max(melsp, axis=0)
        #logging.info(min_melsp)
        #logging.info(max_melsp)
        #write_hdf5(args.stats, "/min_melsp", min_melsp)
        #write_hdf5(args.stats, "/max_melsp", max_melsp)

        scaler_melsp = StandardScaler()
        scaler_melsp.partial_fit(melsp)

        mean_feat_mceplf0cap = scaler_feat_mceplf0cap.mean_
        scale_feat_mceplf0cap = scaler_feat_mceplf0cap.scale_

        #logging.info("mcep_bound")
        #min_mcep_bound = min_mcep-scale_feat_mceplf0cap[-args.mcep_dim:]
        #max_mcep_bound = max_mcep+scale_feat_mceplf0cap[-args.mcep_dim:]
        #logging.info(min_mcep_bound)
        #logging.info(max_mcep_bound)
        #write_hdf5(args.stats, "/min_mcep_bound", min_mcep_bound)
        #write_hdf5(args.stats, "/max_mcep_bound", max_mcep_bound)

        mean_feat_orglf0 = scaler_feat_orglf0.mean_
        scale_feat_orglf0 = scaler_feat_orglf0.scale_
        gv_range_mean = np.mean(np.array(var_range), axis=0)
        gv_range_var = np.var(np.array(var_range), axis=0)
        logging.info(gv_range_mean)
        logging.info(gv_range_var)
        f0_range_mean = np.mean(f0s_range)
        f0_range_std = np.std(f0s_range)
        logging.info(f0_range_mean)
        logging.info(f0_range_std)
        lf0_range_mean = np.mean(np.log(f0s_range))
        lf0_range_std = np.std(np.log(f0s_range))
        logging.info(lf0_range_mean)
        logging.info(lf0_range_std)

        logging.info(mean_feat_mceplf0cap)
        logging.info(scale_feat_mceplf0cap)
        write_hdf5(args.stats, "/mean_feat_mceplf0cap", mean_feat_mceplf0cap)
        write_hdf5(args.stats, "/scale_feat_mceplf0cap", scale_feat_mceplf0cap)
        logging.info(mean_feat_orglf0)
        logging.info(scale_feat_orglf0)
        write_hdf5(args.stats, "/mean_feat_org_lf0", mean_feat_orglf0)
        write_hdf5(args.stats, "/scale_feat_org_lf0", scale_feat_orglf0)
        write_hdf5(args.stats, "/gv_range_mean", gv_range_mean)
        write_hdf5(args.stats, "/gv_range_var", gv_range_var)
        write_hdf5(args.stats, "/f0_range_mean", f0_range_mean)
        write_hdf5(args.stats, "/f0_range_std", f0_range_std)
        write_hdf5(args.stats, "/lf0_range_mean", lf0_range_mean)
        write_hdf5(args.stats, "/lf0_range_std", lf0_range_std)

        mean_melsp = scaler_melsp.mean_
        scale_melsp = scaler_melsp.scale_

        #logging.info("melsp_bound")
        #min_melsp_bound = min_melsp-scale_melsp
        #max_melsp_bound = max_melsp+scale_melsp
        #logging.info(min_melsp_bound)
        #logging.info(max_melsp_bound)
        #write_hdf5(args.stats, "/min_melsp_bound", min_melsp_bound)
        #write_hdf5(args.stats, "/max_melsp_bound", max_melsp_bound)

        gv_melsp_mean = np.mean(np.array(var_melsp), axis=0)
        gv_melsp_var = np.var(np.array(var_melsp), axis=0)
        logging.info(gv_melsp_mean)
        logging.info(gv_melsp_var)
        logging.info(mean_melsp)
        logging.info(scale_melsp)
        write_hdf5(args.stats, "/mean_melsp", mean_melsp)
        write_hdf5(args.stats, "/scale_melsp", scale_melsp)
        write_hdf5(args.stats, "/gv_melsp_mean", gv_melsp_mean)
        write_hdf5(args.stats, "/gv_melsp_var", gv_melsp_var)

        scaler_melworldsp = StandardScaler()
        scaler_melworldsp.partial_fit(melworldsp)

        mean_melworldsp = scaler_melworldsp.mean_
        scale_melworldsp = scaler_melworldsp.scale_

        #logging.info("melworldsp_bound")
        #min_melworldsp_bound = min_melworldsp-scale_melworldsp
        #max_melworldsp_bound = max_melworldsp+scale_melworldsp
        #logging.info(min_melworldsp_bound)
        #logging.info(max_melworldsp_bound)
        #write_hdf5(args.stats, "/min_melworldsp_bound", min_melworldsp_bound)
        #write_hdf5(args.stats, "/max_melworldsp_bound", max_melworldsp_bound)

        gv_melworldsp_mean = np.mean(np.array(var_melworldsp), axis=0)
        gv_melworldsp_var = np.var(np.array(var_melworldsp), axis=0)
        logging.info(gv_melworldsp_mean)
        logging.info(gv_melworldsp_var)
        logging.info(mean_melworldsp)
        logging.info(scale_melworldsp)
        write_hdf5(args.stats, "/mean_melworldsp", mean_melworldsp)
        write_hdf5(args.stats, "/scale_melworldsp", scale_melworldsp)
        write_hdf5(args.stats, "/gv_melworldsp_mean", gv_melworldsp_mean)
        write_hdf5(args.stats, "/gv_melworldsp_var", gv_melworldsp_var)


if __name__ == "__main__":
    main()
