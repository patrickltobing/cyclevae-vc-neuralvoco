#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER
from utils import find_files, read_hdf5, read_txt, write_hdf5, check_hdf5

from dtw_c import dtw_c as dtw

import torch.nn.functional as F
import h5py

#np.set_printoptions(threshold=np.inf)


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--spk", required=True,
                        type=str, help="speaker name to be reconstructed")
    parser.add_argument("--model", required=True,
                        type=str, help="model file")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save log")
    parser.add_argument("--string_path", required=True,
                        type=str, help="path of h5 generated feature")
    # other setting
    parser.add_argument("--GPU_device", default=None,
                        type=int, help="selection of GPU device")
    parser.add_argument("--GPU_device_str", default=None,
                        type=str, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")

    args = parser.parse_args()

    if args.GPU_device is not None or args.GPU_device_str is not None:
        os.environ["CUDA_DEVICE_ORDER"]		= "PCI_BUS_ID"
        if args.GPU_device_str is None:
            os.environ["CUDA_VISIBLE_DEVICES"]	= str(args.GPU_device)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]	= args.GPU_device_str

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S', filemode='w',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S', filemode='w',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S', filemode='w',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # load config
    config = torch.load(args.config)

    # get source feat list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]
    for i in range(args.n_gpus):
        logging.info('%d: %d' % (i+1, len(feat_lists[i])))

    spk_list = config.spk_list.split('@')
    n_spk = len(spk_list)
    spk_idx = spk_list.index(args.spk)

    stats_list = config.stats_list.split('@')
    assert(n_spk == len(stats_list))

    spk_stat = stats_list[spk_idx]
    gv_mean = read_hdf5(spk_stat, "/gv_range_mean")[1:]

    model_epoch = os.path.basename(args.model).split('.')[0].split('-')[1]
    logging.info('epoch: '+model_epoch)

    model_name = os.path.basename(os.path.dirname(args.model)).split('_')[1]
    logging.info('mdl_name: '+model_name)

    logging.info(config)
    # define gpu decode function
    def gpu_decode(feat_list, gpu, cvlist=None, mcdpow_cvlist=None, mcdpowstd_cvlist=None, mcd_cvlist=None, \
                    mcdstd_cvlist=None, cvlist_dv=None, mcdpow_cvlist_dv=None, mcdpowstd_cvlist_dv=None, \
                    mcd_cvlist_dv=None, mcdstd_cvlist_dv=None, \
                    cvlist_cyc=None, mcdpow_cvlist_cyc=None, mcdpowstd_cvlist_cyc=None, mcd_cvlist_cyc=None, \
                    mcdstd_cvlist_cyc=None, cvlist_cyc_dv=None, mcdpow_cvlist_cyc_dv=None, mcdpowstd_cvlist_cyc_dv=None, \
                    mcd_cvlist_cyc_dv=None, mcdstd_cvlist_cyc_dv=None):
        with torch.cuda.device(gpu):
            # define model and load parameters
            with torch.no_grad():
                model_encoder = GRU_VAE_ENCODER(
                    in_dim=config.mcep_dim+config.excit_dim,
                    n_spk=n_spk,
                    lat_dim=config.lat_dim,
                    hidden_layers=config.hidden_layers_enc,
                    hidden_units=config.hidden_units_enc,
                    kernel_size=config.kernel_size_enc,
                    dilation_size=config.dilation_size_enc,
                    causal_conv=config.causal_conv_enc,
                    bi=False,
                    ar=False,
                    pad_first=True,
                    right_size=config.right_size_enc)
                logging.info(model_encoder)
                model_decoder = GRU_SPEC_DECODER(
                    feat_dim=config.lat_dim,
                    out_dim=config.mcep_dim,
                    n_spk=n_spk,
                    hidden_layers=config.hidden_layers_dec,
                    hidden_units=config.hidden_units_dec,
                    kernel_size=config.kernel_size_dec,
                    dilation_size=config.dilation_size_dec,
                    causal_conv=config.causal_conv_dec,
                    bi=False,
                    ar=False,
                    pad_first=True,
                    right_size=config.right_size_dec)
                logging.info(model_decoder)
                model_encoder.load_state_dict(torch.load(args.model)["model_encoder"])
                model_decoder.load_state_dict(torch.load(args.model)["model_decoder"])
                model_encoder.remove_weight_norm()
                model_decoder.remove_weight_norm()
                model_encoder.cuda()
                model_decoder.cuda()
                model_encoder.eval()
                model_decoder.eval()
                for param in model_encoder.parameters():
                    param.requires_grad = False
                for param in model_decoder.parameters():
                    param.requires_grad = False
            count = 0
            pad_left = (model_encoder.pad_left + model_decoder.pad_left)*2
            pad_right = (model_encoder.pad_right + model_decoder.pad_right)*2
            outpad_lefts = [None]*3
            outpad_rights = [None]*3
            outpad_lefts[0] = pad_left-model_encoder.pad_left
            outpad_rights[0] = pad_right-model_encoder.pad_right
            outpad_lefts[1] = outpad_lefts[0]-model_decoder.pad_left
            outpad_rights[1] = outpad_rights[0]-model_decoder.pad_right
            outpad_lefts[2] = outpad_lefts[1]-model_encoder.pad_left
            outpad_rights[2] = outpad_rights[1]-model_encoder.pad_right
            for feat_file in feat_list:
                # convert mcep
                logging.info("recmcep " + feat_file)

                feat_org = read_hdf5(feat_file, "/feat_mceplf0cap")
                logging.info(feat_org.shape)
                mcep = np.array(feat_org[:,-config.mcep_dim:])

                with torch.no_grad():
                    feat = F.pad(torch.FloatTensor(feat_org).cuda().unsqueeze(0).transpose(1,2), (pad_left,pad_right), "replicate").transpose(1,2)
                    feat_excit = torch.FloatTensor(feat_org[:,:config.excit_dim]).cuda().unsqueeze(0)

                    spk_logits, _, lat_src, _ = model_encoder(feat, sampling=False)
                    logging.info('input spkpost')
                    if outpad_rights[0] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[0]:-outpad_rights[0]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[0]:], dim=-1), 1))

                    cvmcep_src, _ = model_decoder((torch.ones((1, lat_src.shape[1]))*spk_idx).cuda().long(), lat_src)
                    spk_logits, _, lat_rec, _ = model_encoder(torch.cat((F.pad(feat_excit.transpose(1,2), \
                                        (outpad_lefts[1],outpad_rights[1]), "replicate").transpose(1,2), cvmcep_src), 2), 
                                                        sampling=False)
                    logging.info('rec spkpost')
                    if outpad_rights[2] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:-outpad_rights[2]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:], dim=-1), 1))

                    cvmcep_cyc, _ = model_decoder((torch.ones((1, lat_rec.shape[1]))*spk_idx).cuda().long(), lat_rec)

                    if outpad_rights[1] > 0:
                        feat_rec = torch.cat((feat_excit, cvmcep_src[:,outpad_lefts[1]:-outpad_rights[1]]), 2)[0].cpu().data.numpy()
                    else:
                        feat_rec = torch.cat((feat_excit, cvmcep_src[:,outpad_lefts[1]:]), 2)[0].cpu().data.numpy()
                    feat_cyc = torch.cat((feat_excit, cvmcep_cyc), 2)[0].cpu().data.numpy()

                    cvmcep_src = np.array(cvmcep_src[0].cpu().data.numpy(), dtype=np.float64)
                    cvmcep_cyc = np.array(cvmcep_cyc[0].cpu().data.numpy(), dtype=np.float64)

                logging.info(cvmcep_src.shape)
                logging.info(cvmcep_cyc.shape)
 
                spcidx = read_hdf5(feat_file, "/spcidx_range")[0]

                _, _, _, mcdpow_arr = dtw.dtw_org_to_trg(np.array(cvmcep_src[np.array(spcidx),:], \
                                            dtype=np.float64), np.array(mcep[np.array(spcidx),:], dtype=np.float64))
                _, _, _, mcd_arr = dtw.dtw_org_to_trg(np.array(cvmcep_src[np.array(spcidx),1:], \
                                            dtype=np.float64), np.array(mcep[np.array(spcidx),1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_rec: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_rec: %.6f dB +- %.6f" % (mcd_mean, mcd_std))

                _, _, _, mcdpow_arr = dtw.dtw_org_to_trg(np.array(cvmcep_cyc[np.array(spcidx),:], \
                                            dtype=np.float64), np.array(mcep[np.array(spcidx),:], dtype=np.float64))
                _, _, _, mcd_arr = dtw.dtw_org_to_trg(np.array(cvmcep_cyc[np.array(spcidx),1:], \
                                            dtype=np.float64), np.array(mcep[np.array(spcidx),1:], dtype=np.float64))
                mcdpow_mean_cyc = np.mean(mcdpow_arr)
                mcdpow_std_cyc = np.std(mcdpow_arr)
                mcd_mean_cyc = np.mean(mcd_arr)
                mcd_std_cyc = np.std(mcd_arr)
                logging.info("mcdpow_cyc: %.6f dB +- %.6f" % (mcdpow_mean_cyc, mcdpow_std_cyc))
                logging.info("mcd_cyc: %.6f dB +- %.6f" % (mcd_mean_cyc, mcd_std_cyc))
            
                dataset = feat_file.split('/')[1].split('_')[0]
                if 'tr' in dataset:
                    logging.info('trn')
                    mcdpow_cvlist.append(mcdpow_mean)
                    mcdpowstd_cvlist.append(mcdpow_std)
                    mcd_cvlist.append(mcd_mean)
                    mcdstd_cvlist.append(mcd_std)
                    cvlist.append(np.var(cvmcep_src[:,1:], axis=0))
                    logging.info(len(cvlist))
                    mcdpow_cvlist_cyc.append(mcdpow_mean_cyc)
                    mcdpowstd_cvlist_cyc.append(mcdpow_std_cyc)
                    mcd_cvlist_cyc.append(mcd_mean_cyc)
                    mcdstd_cvlist_cyc.append(mcd_std_cyc)
                    cvlist_cyc.append(np.var(cvmcep_cyc[:,1:], axis=0))
                elif 'dv' in dataset:
                    logging.info('dev')
                    mcdpow_cvlist_dv.append(mcdpow_mean)
                    mcdpowstd_cvlist_dv.append(mcdpow_std)
                    mcd_cvlist_dv.append(mcd_mean)
                    mcdstd_cvlist_dv.append(mcd_std)
                    cvlist_dv.append(np.var(cvmcep_src[:,1:], axis=0))
                    logging.info(len(cvlist_dv))
                    mcdpow_cvlist_cyc_dv.append(mcdpow_mean_cyc)
                    mcdpowstd_cvlist_cyc_dv.append(mcdpow_std_cyc)
                    mcd_cvlist_cyc_dv.append(mcd_mean_cyc)
                    mcdstd_cvlist_cyc_dv.append(mcd_std_cyc)
                    cvlist_cyc_dv.append(np.var(cvmcep_cyc[:,1:], axis=0))

                logging.info('write rec to h5')
                outh5dir = os.path.join(os.path.dirname(os.path.dirname(feat_file)), args.spk+"-"+args.spk)
                if not os.path.exists(outh5dir):
                    os.makedirs(outh5dir)
                feat_file = os.path.join(outh5dir, os.path.basename(feat_file))
                logging.info(feat_file + ' ' + args.string_path)
                logging.info(feat_rec.shape)
                write_hdf5(feat_file, args.string_path, feat_rec)

                logging.info('write cyc to h5')
                outh5dir = os.path.join(os.path.dirname(os.path.dirname(feat_file)), args.spk+"-"+args.spk+"-"+args.spk)
                if not os.path.exists(outh5dir):
                    os.makedirs(outh5dir)
                feat_file = os.path.join(outh5dir, os.path.basename(feat_file))
                logging.info(feat_file + ' ' + args.string_path)
                logging.info(feat_cyc.shape)
                write_hdf5(feat_file, args.string_path, feat_cyc)

                count += 1
                #if count >= 5:
                #if count >= 3:
                #    break


    # parallel decode training
    with mp.Manager() as manager:
        gpu = 0
        processes = []
        cvlist = manager.list()
        mcd_cvlist = manager.list()
        mcdstd_cvlist = manager.list()
        mcdpow_cvlist = manager.list()
        mcdpowstd_cvlist = manager.list()
        cvlist_dv = manager.list()
        mcd_cvlist_dv = manager.list()
        mcdstd_cvlist_dv = manager.list()
        mcdpow_cvlist_dv = manager.list()
        mcdpowstd_cvlist_dv = manager.list()
        cvlist_cyc = manager.list()
        mcd_cvlist_cyc = manager.list()
        mcdstd_cvlist_cyc = manager.list()
        mcdpow_cvlist_cyc = manager.list()
        mcdpowstd_cvlist_cyc = manager.list()
        cvlist_cyc_dv = manager.list()
        mcd_cvlist_cyc_dv = manager.list()
        mcdstd_cvlist_cyc_dv = manager.list()
        mcdpow_cvlist_cyc_dv = manager.list()
        mcdpowstd_cvlist_cyc_dv = manager.list()
        for i, feat_list in enumerate(feat_lists):
            logging.info(i)
            p = mp.Process(target=gpu_decode, args=(feat_list, gpu, cvlist, mcdpow_cvlist, mcdpowstd_cvlist, \
                                                    mcd_cvlist, mcdstd_cvlist, cvlist_dv, mcdpow_cvlist_dv, \
                                                    mcdpowstd_cvlist_dv, mcd_cvlist_dv, mcdstd_cvlist_dv,\
                                                    cvlist_cyc, mcdpow_cvlist_cyc, mcdpowstd_cvlist_cyc, \
                                                    mcd_cvlist_cyc, mcdstd_cvlist_cyc, cvlist_cyc_dv, mcdpow_cvlist_cyc_dv, \
                                                    mcdpowstd_cvlist_cyc_dv, mcd_cvlist_cyc_dv, mcdstd_cvlist_cyc_dv,))
            p.start()
            processes.append(p)
            gpu += 1
            if (i + 1) % args.n_gpus == 0:
                gpu = 0
        # wait for all process
        for p in processes:
            p.join()

        # calculate cv_gv statistics
        if len(mcdpow_cvlist) > 0:
            logging.info("mcdpow_rec: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist)), \
                        np.std(np.array(mcdpow_cvlist)),np.mean(np.array(mcdpowstd_cvlist)),\
                        np.std(np.array(mcdpowstd_cvlist))))
            logging.info("mcd_rec: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist)), \
                        np.std(np.array(mcd_cvlist)),np.mean(np.array(mcdstd_cvlist)),\
                        np.std(np.array(mcdstd_cvlist))))
            cvgv_mean = np.mean(np.array(cvlist), axis=0)
            cvgv_var = np.var(np.array(cvlist), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("mcdpow_cyc: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist_cyc)), \
                        np.std(np.array(mcdpow_cvlist_cyc)),np.mean(np.array(mcdpowstd_cvlist_cyc)),\
                        np.std(np.array(mcdpowstd_cvlist_cyc))))
            logging.info("mcd_cyc: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_cyc)), \
                        np.std(np.array(mcd_cvlist_cyc)),np.mean(np.array(mcdstd_cvlist_cyc)),\
                        np.std(np.array(mcdstd_cvlist_cyc))))
            cvgv_mean = np.mean(np.array(cvlist_cyc), axis=0)
            cvgv_var = np.var(np.array(cvlist_cyc), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))

            cvgv_mean = np.mean(np.array(np.r_[cvlist,cvlist_cyc]), axis=0)
            cvgv_var = np.var(np.array(np.r_[cvlist,cvlist_cyc]), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))

            string_path = model_name+"-"+str(config.n_half_cyc)+"-"+str(config.lat_dim)+"-"+model_epoch
            logging.info(string_path)

            string_mean = "/recgv_mean_"+string_path
            string_var = "/recgv_var_"+string_path
            write_hdf5(spk_stat, string_mean, cvgv_mean)
            write_hdf5(spk_stat, string_var, cvgv_var)

        if len(mcdpow_cvlist_dv) > 0:
            logging.info("mcdpow_rec_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist_dv)), \
                        np.std(np.array(mcdpow_cvlist_dv)),np.mean(np.array(mcdpowstd_cvlist_dv)),\
                        np.std(np.array(mcdpowstd_cvlist_dv))))
            logging.info("mcd_rec_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_dv)), \
                        np.std(np.array(mcd_cvlist_dv)),np.mean(np.array(mcdstd_cvlist_dv)),\
                        np.std(np.array(mcdstd_cvlist_dv))))
            cvgv_mean = np.mean(np.array(cvlist_dv), axis=0)
            cvgv_var = np.var(np.array(cvlist_dv), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("mcdpow_cyc_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist_cyc_dv)), \
                        np.std(np.array(mcdpow_cvlist_cyc_dv)),np.mean(np.array(mcdpowstd_cvlist_cyc_dv)),\
                        np.std(np.array(mcdpowstd_cvlist_cyc_dv))))
            logging.info("mcd_cyc_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_cyc_dv)), \
                        np.std(np.array(mcd_cvlist_cyc_dv)),np.mean(np.array(mcdstd_cvlist_cyc_dv)),\
                        np.std(np.array(mcdstd_cvlist_cyc_dv))))
            cvgv_mean = np.mean(np.array(cvlist_cyc_dv), axis=0)
            cvgv_var = np.var(np.array(cvlist_cyc_dv), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))


if __name__ == "__main__":
    main()
