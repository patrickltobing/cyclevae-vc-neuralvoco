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
from distutils.util import strtobool

import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.nn.functional as F

from utils import find_files
from utils import read_hdf5
from utils import read_txt
from utils import check_hdf5
from utils import write_hdf5

#import matplotlib.pyplot as plt

import soundfile as sf

from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER, GRU_SPK, GRU_POST_NET
from vcneuvoco import GRU_EXCIT_DECODER, SPKID_TRANSFORM_LAYER
from feature_extract import convert_f0
from dtw_c import dtw_c as dtw

#import pysptk as ps
#import pyworld as pw
import librosa
#from pysptk.synthesis import MLSADF

#np.set_printoptions(threshold=np.inf)

#FS = 16000
#FS = 22050
FS = 24000
N_GPUS = 1
SHIFT_MS = 5.0
#SHIFT_MS = 10.0
WIN_MS = 27.5
#FFTL = 1024
FFTL = 2048
IRLEN = 1024
VERBOSE = 1
GV_COEFF = 0.9


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--model", required=True,
                        type=str, help="model file")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--fs", default=FS,
                        type=int, help="sampling rate")
    parser.add_argument("--spk_trg", required=True,
                        type=str, help="speaker target")
    parser.add_argument("--n_gpus", default=N_GPUS,
                        type=int, help="number of gpus")
    parser.add_argument("--string_path", required=True,
                        type=str, help="directory to save generated samples")
    # other setting
    parser.add_argument("--n_interp", default=0,
                        type=int, help="number of interpolation points if using cont. spk-code (if 0, just rec. and cv.)")
    parser.add_argument("--shiftms", default=SHIFT_MS,
                        type=float, help="frame shift")
    parser.add_argument("--winms", default=WIN_MS,
                        type=float, help="frame shift")
    parser.add_argument("--fftl", default=FFTL,
                        type=int, help="FFT length")
    parser.add_argument("--GPU_device", default=None,
                        type=int, help="selection of GPU device")
    parser.add_argument("--GPU_device_str", default=None,
                        type=str, help="selection of GPU device")
    parser.add_argument("--verbose", default=VERBOSE,
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
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # load config
    config = torch.load(args.config)

    # get file list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    spk_list = config.spk_list.split('@')
    n_spk = len(spk_list)
    trg_idx = spk_list.index(args.spk_trg)

    stats_list = config.stats_list.split('@')
    assert(n_spk == len(stats_list))

    spk_src = os.path.basename(os.path.dirname(feat_list[0]))
    src_idx = spk_list.index(spk_src)

    src_f0_mean = read_hdf5(stats_list[src_idx], "/lf0_range_mean")
    src_f0_std = read_hdf5(stats_list[src_idx], "/lf0_range_std")
    logging.info(src_f0_mean)
    logging.info(src_f0_std)
    trg_f0_mean = read_hdf5(stats_list[trg_idx], "/lf0_range_mean")
    trg_f0_std = read_hdf5(stats_list[trg_idx], "/lf0_range_std")
    logging.info(trg_f0_mean)
    logging.info(trg_f0_std)

    model_epoch = os.path.basename(args.model).split('.')[0].split('-')[1]
    logging.info('epoch: '+model_epoch)

    model_name = os.path.basename(os.path.dirname(args.model)).split('_')[1]
    logging.info('mdl_name: '+model_name)

    gv_mean_trg = read_hdf5(stats_list[trg_idx], "/gv_melsp_mean")
    if args.n_interp > 0:
        gv_mean_trgs = []
        cvgv_means = []
        for i in range(n_spk):
            gv_mean_trgs.append(read_hdf5(stats_list[i], "/gv_melsp_mean"))
            cvgv_means.append(read_hdf5(stats_list[i], "/gv_melsp_mean"))

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]
    for i in range(args.n_gpus):
        logging.info('gpu: %d'+str(i+1)+' : '+str(len(feat_lists[i])))

    ### GRU-RNN decoding ###
    logging.info(config)
    def decode_RNN(feat_list, gpu, cvlist=None,
            lsd_cvlist_src=None, lsdstd_cvlist_src=None,
            f0rmse_cvlist_src=None, f0corr_cvlist_src=None, caprmse_cvlist_src=None,
            lsd_cvlist_cyc=None, lsdstd_cvlist_cyc=None,
            f0rmse_cvlist_cyc=None, f0corr_cvlist_cyc=None, caprmse_cvlist_cyc=None,
            f0rmse_cvlist_cv=None, f0corr_cvlist_cv=None,
            lsd_cvlist=None, lsdstd_cvlist=None,
            lat_dist_rmse_list=None, lat_dist_cosim_list=None):
        with torch.cuda.device(gpu):
            # define model and load parameters
            with torch.no_grad():
                model_encoder_melsp = GRU_VAE_ENCODER(
                    in_dim=config.mel_dim,
                    n_spk=n_spk,
                    lat_dim=config.lat_dim,
                    hidden_layers=config.hidden_layers_enc,
                    hidden_units=config.hidden_units_enc,
                    kernel_size=config.kernel_size_enc,
                    dilation_size=config.dilation_size_enc,
                    causal_conv=config.causal_conv_enc,
                    pad_first=True,
                    right_size=config.right_size_enc)
                logging.info(model_encoder_melsp)
                model_decoder_melsp_fix = GRU_SPEC_DECODER(
                    feat_dim=config.lat_dim+config.lat_dim_e,
                    excit_dim=config.excit_dim,
                    out_dim=config.mel_dim,
                    n_spk=n_spk,
                    aux_dim=n_spk,
                    hidden_layers=config.hidden_layers_dec,
                    hidden_units=config.hidden_units_dec,
                    kernel_size=config.kernel_size_dec,
                    dilation_size=config.dilation_size_dec,
                    causal_conv=config.causal_conv_dec,
                    pad_first=True,
                    right_size=config.right_size_dec)
                logging.info(model_decoder_melsp_fix)
                model_decoder_melsp = GRU_SPEC_DECODER(
                    feat_dim=config.lat_dim+config.lat_dim_e,
                    excit_dim=config.excit_dim,
                    out_dim=config.mel_dim,
                    n_spk=n_spk,
                    aux_dim=n_spk,
                    hidden_layers=config.hidden_layers_dec,
                    hidden_units=config.hidden_units_dec,
                    kernel_size=config.kernel_size_dec,
                    dilation_size=config.dilation_size_dec,
                    causal_conv=config.causal_conv_dec,
                    pad_first=True,
                    right_size=config.right_size_dec)
                logging.info(model_decoder_melsp)
                model_encoder_excit = GRU_VAE_ENCODER(
                    in_dim=config.mel_dim,
                    n_spk=n_spk,
                    lat_dim=config.lat_dim_e,
                    hidden_layers=config.hidden_layers_enc,
                    hidden_units=config.hidden_units_enc,
                    kernel_size=config.kernel_size_enc,
                    dilation_size=config.dilation_size_enc,
                    causal_conv=config.causal_conv_enc,
                    pad_first=True,
                    right_size=config.right_size_enc)
                logging.info(model_encoder_excit)
                model_decoder_excit = GRU_EXCIT_DECODER(
                    feat_dim=config.lat_dim_e,
                    cap_dim=config.cap_dim,
                    n_spk=n_spk,
                    aux_dim=n_spk,
                    hidden_layers=config.hidden_layers_lf0,
                    hidden_units=config.hidden_units_lf0,
                    kernel_size=config.kernel_size_lf0,
                    dilation_size=config.dilation_size_lf0,
                    causal_conv=config.causal_conv_lf0,
                    pad_first=True,
                    right_size=config.right_size_lf0)
                logging.info(model_decoder_excit)
                if (config.spkidtr_dim > 0):
                    model_spkidtr = SPKID_TRANSFORM_LAYER(
                        n_spk=n_spk,
                        spkidtr_dim=config.spkidtr_dim)
                    logging.info(model_spkidtr)
                model_spk = GRU_SPK(
                    n_spk=n_spk,
                    feat_dim=config.lat_dim+config.lat_dim_e,
                    hidden_units=32)
                logging.info(model_spk)
                model_post = GRU_POST_NET(
                    spec_dim=config.mel_dim,
                    excit_dim=config.excit_dim+config.cap_dim+1,
                    n_spk=n_spk,
                    aux_dim=n_spk,
                    hidden_layers=config.hidden_layers_post,
                    hidden_units=config.hidden_units_post,
                    kernel_size=config.kernel_size_post,
                    dilation_size=config.dilation_size_post,
                    causal_conv=config.causal_conv_post,
                    pad_first=True,
                    right_size=config.right_size_post,
                    res=True,
                    laplace=True)
                logging.info(model_post)
                model_encoder_melsp.load_state_dict(torch.load(args.model)["model_encoder_melsp"])
                model_decoder_melsp_fix.load_state_dict(torch.load(args.model)["model_decoder_melsp_fix"])
                model_decoder_melsp.load_state_dict(torch.load(args.model)["model_decoder_melsp"])
                model_encoder_excit.load_state_dict(torch.load(args.model)["model_encoder_excit"])
                model_decoder_excit.load_state_dict(torch.load(args.model)["model_decoder_excit"])
                if (config.spkidtr_dim > 0):
                    model_spkidtr.load_state_dict(torch.load(args.model)["model_spkidtr"])
                model_spk.load_state_dict(torch.load(args.model)["model_spk"])
                model_post.load_state_dict(torch.load(args.model)["model_post"])
                model_encoder_melsp.cuda()
                model_decoder_melsp_fix.cuda()
                model_decoder_melsp.cuda()
                model_encoder_excit.cuda()
                model_decoder_excit.cuda()
                if (config.spkidtr_dim > 0):
                    model_spkidtr.cuda()
                model_spk.cuda()
                model_post.cuda()
                model_encoder_melsp.eval()
                model_decoder_melsp_fix.eval()
                model_decoder_melsp.eval()
                model_encoder_excit.eval()
                model_decoder_excit.eval()
                if (config.spkidtr_dim > 0):
                    model_spkidtr.eval()
                model_spk.eval()
                model_post.eval()
                for param in model_encoder_melsp.parameters():
                    param.requires_grad = False
                for param in model_decoder_melsp_fix.parameters():
                    param.requires_grad = False
                for param in model_decoder_melsp.parameters():
                    param.requires_grad = False
                for param in model_encoder_excit.parameters():
                    param.requires_grad = False
                for param in model_decoder_excit.parameters():
                    param.requires_grad = False
                if (config.spkidtr_dim > 0):
                    for param in model_spkidtr.parameters():
                        param.requires_grad = False
                for param in model_spk.parameters():
                    param.requires_grad = False
                for param in model_post.parameters():
                    param.requires_grad = False
            # interpolated spk-code
            #if args.n_interp > 0:
            #    feat = torch.LongTensor(np.arange(n_spk)).cuda().unsqueeze(0)
            #    logging.info(feat)
            #    z = model_decoder_mcep.spkidtr_conv(F.one_hot(feat, num_classes=n_spk).float().transpose(1,2)).transpose(1,2)
            #    z_e = model_decoder_excit.spkidtr_conv(F.one_hot(feat, num_classes=n_spk).float().transpose(1,2)).transpose(1,2)

            #    # 1 x 1 x 2
            #    z_src = z[:,src_idx:src_idx+1,:]
            #    z_e_src = z_e[:,src_idx:src_idx+1,:]
            #    logging.info("source %d %s" % (src_idx+1, spk_list[src_idx]))
            #    logging.info(z_src)
            #    logging.info(z_e_src)

            #    z_trg = z[:,trg_idx:trg_idx+1,:]
            #    z_e_trg = z_e[:,trg_idx:trg_idx+1,:]
            #    logging.info("target %d %s" % (trg_idx+1, spk_list[trg_idx]))
            #    logging.info(z_trg)
            #    logging.info(z_e_trg)

            #    n_delta = args.n_interp

            #    delta_z = (z_trg - z_src)/n_delta
            #    delta_z_e = (z_e_trg - z_e_src)/n_delta
            #    logging.info("delta")
            #    logging.info(delta_z)
            #    logging.info(delta_z_e)
            count = 0
            pad_left = (model_encoder_melsp.pad_left + model_decoder_excit.pad_left + model_decoder_melsp.pad_left + model_post.pad_left)*2
            pad_right = (model_encoder_melsp.pad_right + model_decoder_excit.pad_right + model_decoder_melsp.pad_right + model_post.pad_right)*2
            outpad_lefts = [None]*7
            outpad_rights = [None]*7
            outpad_lefts[0] = pad_left-model_encoder_melsp.pad_left
            outpad_rights[0] = pad_right-model_encoder_melsp.pad_right
            outpad_lefts[1] = outpad_lefts[0]-model_decoder_excit.pad_left
            outpad_rights[1] = outpad_rights[0]-model_decoder_excit.pad_right
            outpad_lefts[2] = outpad_lefts[1]-model_decoder_melsp.pad_left
            outpad_rights[2] = outpad_rights[1]-model_decoder_melsp.pad_right
            outpad_lefts[3] = outpad_lefts[2]-model_post.pad_left
            outpad_rights[3] = outpad_rights[2]-model_post.pad_right
            outpad_lefts[4] = outpad_lefts[3]-model_encoder_excit.pad_left
            outpad_rights[4] = outpad_rights[3]-model_encoder_excit.pad_right
            outpad_lefts[5] = outpad_lefts[4]-model_decoder_excit.pad_left
            outpad_rights[5] = outpad_rights[4]-model_decoder_excit.pad_right
            outpad_lefts[6] = outpad_lefts[5]-model_decoder_melsp.pad_left
            outpad_rights[6] = outpad_rights[5]-model_decoder_melsp.pad_right
            melfb_t = np.linalg.pinv(librosa.filters.mel(args.fs, args.fftl, n_mels=config.mel_dim))
            for feat_file in feat_list:
                # convert melsp
                spk_src = os.path.basename(os.path.dirname(feat_file))
                logging.info('%s --> %s' % (spk_src, args.spk_trg))

                file_trg = os.path.join(os.path.dirname(os.path.dirname(feat_file)), args.spk_trg, os.path.basename(feat_file))
                trg_exist = False
                if os.path.exists(file_trg):
                    logging.info('exist: %s' % (file_trg))
                    feat_trg = read_hdf5(file_trg, "/log_1pmelmagsp")
                    logging.info(feat_trg.shape)
                    trg_exist = True

                feat_org = read_hdf5(feat_file, "/log_1pmelmagsp")
                logging.info(feat_org.shape)

                logging.info("generate")
                with torch.no_grad():
                    feat = F.pad(torch.FloatTensor(feat_org).cuda().unsqueeze(0).transpose(1,2), (pad_left,pad_right), "replicate").transpose(1,2)

                    spk_logits, _, lat_src, _ = model_encoder_melsp(feat, sampling=False)
                    spk_logits_e, _, lat_src_e, _ = model_encoder_excit(feat, sampling=False)
                    logging.info('input spkpost')
                    if outpad_rights[0] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[0]:-outpad_rights[0]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[0]:], dim=-1), 1))
                    logging.info('input spkpost_e')
                    if outpad_rights[0] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[0]:-outpad_rights[0]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[0]:], dim=-1), 1))

                    if trg_exist:
                        spk_trg_logits, _, lat_trg, _ = model_encoder_melsp(F.pad(torch.FloatTensor(feat_trg).cuda().unsqueeze(0).transpose(1,2),
                                                                        (model_encoder_melsp.pad_left,model_encoder_melsp.pad_right), "replicate").transpose(1,2), sampling=False)
                        spk_trg_logits_e, _, lat_trg_e, _ = model_encoder_excit(F.pad(torch.FloatTensor(feat_trg).cuda().unsqueeze(0).transpose(1,2),
                                                                        (model_encoder_excit.pad_left,model_encoder_excit.pad_right), "replicate").transpose(1,2), sampling=False)
                        logging.info('target spkpost')
                        logging.info(torch.mean(F.softmax(spk_trg_logits, dim=-1), 1))
                        logging.info('target spkpost_e')
                        logging.info(torch.mean(F.softmax(spk_trg_logits_e, dim=-1), 1))

                    if args.n_interp == 0: # if just reconstructed and conversion
                        if config.spkidtr_dim > 0:
                            src_code = model_spkidtr((torch.ones((1, lat_src_e.shape[1]))*src_idx).cuda().long())
                        else:
                            src_code = (torch.ones((1, lat_src_e.shape[1]))*src_idx).cuda().long()
                        if config.spkidtr_dim > 0:
                            trg_code = model_spkidtr((torch.ones((1, lat_src_e.shape[1]))*trg_idx).cuda().long())
                        else:
                            trg_code = (torch.ones((1, lat_src_e.shape[1]))*trg_idx).cuda().long()
                        lat_cat = torch.cat((lat_src_e, lat_src), 2)
                        trj_src_code, _ = model_spk(src_code, z=lat_cat)
                        trj_trg_code, _ = model_spk(trg_code, z=lat_cat)
                        cvlf0_src, _ = model_decoder_excit(lat_src_e, y=src_code, aux=trj_src_code)
                        cvlf0, _ = model_decoder_excit(lat_src_e, y=trg_code, aux=trj_trg_code)

                        # reconstruction and conversion using updated decoder-melsp and postnet
                        if model_decoder_excit.pad_right > 0:
                            lat_cat = lat_cat[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                            src_code = src_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                            trg_code = trg_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                            trj_src_code = trj_src_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                            trj_trg_code = trj_trg_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                        else:
                            lat_cat = lat_cat[:,model_decoder_excit.pad_left:]
                            src_code = src_code[:,model_decoder_excit.pad_left:]
                            trg_code = trg_code[:,model_decoder_excit.pad_left:]
                            trj_src_code = trj_src_code[:,model_decoder_excit.pad_left:]
                            trj_trg_code = trj_trg_code[:,model_decoder_excit.pad_left:]
                        cvmelsp_src, _ = model_decoder_melsp(lat_cat, y=src_code, aux=trj_src_code, e=cvlf0_src[:,:,:config.excit_dim])
                        cvmelsp, _ = model_decoder_melsp(lat_cat, y=trg_code, aux=trj_trg_code, e=cvlf0[:,:,:config.excit_dim])

                        if model_decoder_melsp.pad_right > 0:
                            e_post = cvlf0_src[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                            e_cv_post = cvlf0[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                            src_code_ = src_code[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                            trg_code_ = trg_code[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                            trj_src_code_ = trj_src_code[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                            trj_trg_code_ = trj_trg_code[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                        else:
                            e_post = cvlf0_src[:,model_decoder_melsp.pad_left:]
                            e_cv_post = cvlf0[:,model_decoder_melsp.pad_left:]
                            src_code_ = src_code[:,model_decoder_melsp.pad_left:]
                            trg_code_ = trg_code[:,model_decoder_melsp.pad_left:]
                            trj_src_code_ = trj_src_code[:,model_decoder_melsp.pad_left:]
                            trj_trg_code_ = trj_trg_code[:,model_decoder_melsp.pad_left:]
                        pdf, cvmelsp_src_post, _ = model_post(cvmelsp_src, y=src_code_, aux=trj_src_code_, e=e_post)
                        pdf_cv, cvmelsp_post, _ = model_post(cvmelsp, y=trg_code_, aux=trj_trg_code_, e=e_cv_post)
                        #cvmelsp_src_post = pdf[:,:,:config.mel_dim]
                        #cvmelsp_post = pdf_cv[:,:,:config.mel_dim]

                        # cyclic flow using fixed decoder for 1st rec./conv.
                        cvmelsp_src, _ = model_decoder_melsp_fix(lat_cat, y=src_code, aux=trj_src_code, e=cvlf0_src[:,:,:config.excit_dim])
                        if model_post.pad_right > 0:
                            cvmelsp_src = cvmelsp_src[:,model_post.pad_left:-model_post.pad_right]
                        else:
                            cvmelsp_src = cvmelsp_src[:,model_post.pad_left:]
                        spk_logits, _, lat_rec, _ = model_encoder_melsp(cvmelsp_src, sampling=False)
                        spk_logits_e, _, lat_rec_e, _ = model_encoder_excit(cvmelsp_src, sampling=False)
                        logging.info('rec spkpost')
                        if outpad_rights[4] > 0:
                            logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[4]:-outpad_rights[4]], dim=-1), 1))
                        else:
                            logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[4]:], dim=-1), 1))
                        logging.info('rec spkpost_e')
                        if outpad_rights[4] > 0:
                            logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[4]:-outpad_rights[4]], dim=-1), 1))
                        else:
                            logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[4]:], dim=-1), 1))

                        cvmelsp, _ = model_decoder_melsp_fix(lat_cat, y=trg_code, aux=trj_trg_code, e=cvlf0[:,:,:config.excit_dim])
                        if model_post.pad_right > 0:
                            cvmelsp = cvmelsp[:,model_post.pad_left:-model_post.pad_right]
                        else:
                            cvmelsp = cvmelsp[:,model_post.pad_left:]
                        spk_logits, _, lat_cv, _ = model_encoder_melsp(cvmelsp, sampling=False)
                        spk_logits_e, _, lat_cv_e, _ = model_encoder_excit(cvmelsp, sampling=False)
                        logging.info('cv spkpost')
                        if outpad_rights[4] > 0:
                            logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[4]:-outpad_rights[4]], dim=-1), 1))
                        else:
                            logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[4]:], dim=-1), 1))
                        logging.info('cv spkpost_e')
                        if outpad_rights[4] > 0:
                            logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[4]:-outpad_rights[4]], dim=-1), 1))
                        else:
                            logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[4]:], dim=-1), 1))

                        if config.spkidtr_dim > 0:
                            src_code = model_spkidtr((torch.ones((1, lat_cv_e.shape[1]))*src_idx).cuda().long())
                        else:
                            src_code = (torch.ones((1, lat_cv_e.shape[1]))*src_idx).cuda().long()
                        lat_cat = torch.cat((lat_cv_e, lat_cv), 2)
                        trj_src_code, _ = model_spk(src_code, z=lat_cat)
                        cvlf0_cyc, _ = model_decoder_excit(lat_cv_e, y=src_code, aux=trj_src_code)

                        if model_decoder_excit.pad_right > 0:
                            lat_cat = lat_cat[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                            src_code = src_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                            trj_src_code = trj_src_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                        else:
                            lat_cat = lat_cat[:,model_decoder_excit.pad_left:]
                            src_code = src_code[:,model_decoder_excit.pad_left:]
                            trj_src_code = trj_src_code[:,model_decoder_excit.pad_left:]
                        cvmelsp_cyc, _ = model_decoder_melsp(lat_cat, y=src_code, aux=trj_src_code, e=cvlf0_cyc[:,:,:config.excit_dim])
                        
                        if model_decoder_melsp.pad_right > 0:
                            e_post = cvlf0_cyc[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                            src_code = src_code[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                            trj_src_code = trj_src_code[:,model_decoder_melsp.pad_left:-model_decoder_melsp.pad_right]
                        else:
                            e_post = cvlf0_cyc[:,model_decoder_melsp.pad_left:]
                            src_code = src_code[:,model_decoder_melsp.pad_left:]
                            trj_src_code = trj_src_code[:,model_decoder_melsp.pad_left:]
                        pdf, cvmelsp_cyc_post, _ = model_post(cvmelsp_cyc, y=src_code, aux=trj_src_code, e=e_post)
                        #cvmelsp_cyc_post = pdf[:,:,:config.mel_dim]
                    #else: # if using interpolated spk-code
                    #    z_interpolate = []
                    #    z_e_interpolate = []

                    #    z_src = z[:,src_idx:src_idx+1,:]
                    #    logging.info(z_src)
                    #    z_interpolate.append(z[0,src_idx,:].cpu().data.numpy())
                    #    src_code = torch.repeat_interleave(z_src, lat_src.shape[1], dim=1)
                    #    cvmcep_src, _ = model_decoder_mcep(src_code, lat_src)
                    #    z_e_src = z_e[:,src_idx:src_idx+1,:]
                    #    logging.info(z_e_src)
                    #    z_e_interpolate.append(z_e[0,src_idx,:].cpu().data.numpy())
                    #    src_code = torch.repeat_interleave(z_e_src, lat_src_e.shape[1], dim=1)
                    #    cvlf0_src, _ = model_decoder_excit(src_code, lat_src_e)

                    #    spk_prob_interpolate = []
                    #    spk_interpolate = []
                    #    spk_idx_interpolate = []
                    #    spk_prob_e_interpolate = []
                    #    spk_e_interpolate = []

                    #    if config.ar_enc:
                    #        spk_logits, _, lat_cv, _, _ = model_encoder_mcep(torch.cat((cvlf0_src, cvmcep_src), 2), 
                    #                                            yz_in=yz_in, sampling=False)
                    #    else:
                    #        spk_logits, _, lat_cv, _ = model_encoder_mcep(torch.cat((cvlf0_src, cvmcep_src), 2), 
                    #                                            sampling=False)
                    #    logging.info('cv-0 spkpost')
                    #    if outpad_rights[2] > 0:
                    #        spk_prob = torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:-outpad_rights[2]], dim=-1), 1)
                    #    else:
                    #        spk_prob = torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:], dim=-1), 1)
                    #    logging.info(spk_prob)
                    #    max_prob = torch.max(spk_prob).cpu().data.item()
                    #    max_prob_idx = torch.argmax(spk_prob).cpu().data.item()
                    #    spk_prob_interpolate.append(max_prob*100)
                    #    spk_interpolate.append(spk_list[max_prob_idx])
                    #    spk_idx_interpolate.append(max_prob_idx)
                    #    logging.info(spk_prob_interpolate[0])
                    #    logging.info(spk_interpolate[0])
                    #    logging.info(spk_idx_interpolate[0])

                    #    if config.ar_enc:
                    #        spk_logits_e, _, lat_cv_e, _, _ = model_encoder_excit(torch.cat((cvlf0_src, cvmcep_src), 2), 
                    #                                            yz_in=yz_in, sampling=False)
                    #    else:
                    #        spk_logits_e, _, lat_cv_e, _ = model_encoder_excit(torch.cat((cvlf0_src, cvmcep_src), 2), 
                    #                                            sampling=False)
                    #    logging.info('cv-0 spkpost_e')
                    #    if outpad_rights[2] > 0:
                    #        spk_prob = torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[2]:-outpad_rights[2]], dim=-1), 1)
                    #    else:
                    #        spk_prob = torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[2]:], dim=-1), 1)
                    #    logging.info(spk_prob)
                    #    max_prob = torch.max(spk_prob).cpu().data.item()
                    #    max_prob_idx = torch.argmax(spk_prob).cpu().data.item()
                    #    spk_prob_e_interpolate.append(max_prob*100)
                    #    spk_e_interpolate.append(spk_list[max_prob_idx])
                    #    logging.info(spk_prob_e_interpolate[0])
                    #    logging.info(spk_e_interpolate[0])

                    #    cvmcep_interpolate = []
                    #    cvlf0_interpolate = []
                    #    for i in range(n_delta):
                    #        logging.info("delta %d" % (i+1))

                    #        cv_code = torch.repeat_interleave(((i+1)*delta_z)+z_src, lat_src.shape[1], dim=1)
                    #        logging.info(cv_code[0,0])
                    #        z_interpolate.append(cv_code[0,0,:].cpu().data.numpy())
                    #        cvmcep, _ = model_decoder_mcep(cv_code, lat_src)
                    #        cv_e_code = torch.repeat_interleave(((i+1)*delta_z_e)+z_e_src, lat_src_e.shape[1], dim=1)
                    #        logging.info(cv_e_code[0,0])
                    #        z_e_interpolate.append(cv_e_code[0,0,:].cpu().data.numpy())
                    #        cvlf0, _ = model_decoder_excit(cv_e_code, lat_src_e)
                    #        if i < n_delta-1:
                    #            if outpad_rights[1] > 0:
                    #                cvmcep_interpolate.append(np.array(cvmcep[0,outpad_lefts[1]:-outpad_rights[1]].cpu().data.numpy(), dtype=np.float64))
                    #                cvlf0_interpolate.append(np.array(cvlf0[0,outpad_lefts[1]:-outpad_rights[1]].cpu().data.numpy(), dtype=np.float64))
                    #            else:
                    #                cvmcep_interpolate.append(np.array(cvmcep[0,outpad_lefts[1]:].cpu().data.numpy(), dtype=np.float64))
                    #                cvlf0_interpolate.append(np.array(cvlf0[0,outpad_lefts[1]:].cpu().data.numpy(), dtype=np.float64))

                    #        if config.ar_enc:
                    #            spk_logits, _, lat_cv, _, _ = model_encoder_mcep(torch.cat((cvlf0, cvmcep), 2), 
                    #                                                yz_in=yz_in, sampling=False)
                    #        else:
                    #            spk_logits, _, lat_cv, _ = model_encoder_mcep(torch.cat((cvlf0, cvmcep), 2), 
                    #                                                sampling=False)
                    #        logging.info('cv-%d spkpost' % (i+1))
                    #        if outpad_rights[2] > 0:
                    #            spk_prob = torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:-outpad_rights[2]], dim=-1), 1)
                    #        else:
                    #            spk_prob = torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:], dim=-1), 1)
                    #        logging.info(spk_prob)
                    #        max_prob = torch.max(spk_prob).cpu().data.item()
                    #        max_prob_idx = torch.argmax(spk_prob).cpu().data.item()
                    #        spk_prob_interpolate.append(max_prob*100)
                    #        spk_interpolate.append(spk_list[max_prob_idx])
                    #        spk_idx_interpolate.append(max_prob_idx)
                    #        logging.info(spk_prob_interpolate[i+1])
                    #        logging.info(spk_interpolate[i+1])
                    #        logging.info(spk_idx_interpolate[i+1])

                    #        if config.ar_enc:
                    #            spk_logits_e, _, lat_cv_e, _, _ = model_encoder_excit(torch.cat((cvlf0, cvmcep), 2), 
                    #                                                yz_in=yz_in, sampling=False)
                    #        else:
                    #            spk_logits_e, _, lat_cv_e, _ = model_encoder_excit(torch.cat((cvlf0, cvmcep), 2), 
                    #                                                sampling=False)
                    #        logging.info('cv-%d spkpost_e' % (i+1))
                    #        if outpad_rights[2] > 0:
                    #            spk_prob = torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[2]:-outpad_rights[2]], dim=-1), 1)
                    #        else:
                    #            spk_prob = torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[2]:], dim=-1), 1)
                    #        logging.info(spk_prob)
                    #        max_prob = torch.max(spk_prob).cpu().data.item()
                    #        max_prob_idx = torch.argmax(spk_prob).cpu().data.item()
                    #        spk_prob_e_interpolate.append(max_prob*100)
                    #        spk_e_interpolate.append(spk_list[max_prob_idx])
                    #        logging.info(spk_prob_e_interpolate[i+1])
                    #        logging.info(spk_e_interpolate[i+1])

                    #    src_code = torch.repeat_interleave(z_src, lat_cv.shape[1], dim=1)
                    #    cvmcep_cyc, _ = model_decoder_mcep(src_code, lat_cv)
                    #    src_code = torch.repeat_interleave(z_e_src, lat_cv_e.shape[1], dim=1)
                    #    cvlf0_cyc, _ = model_decoder_excit(src_code, lat_cv_e)

                    if outpad_rights[1] > 0:
                        cvlf0_src = cvlf0_src[:,outpad_lefts[1]:-outpad_rights[1]]
                        cvlf0 = cvlf0[:,outpad_lefts[1]:-outpad_rights[1]]
                    else:
                        cvlf0_src = cvlf0_src[:,outpad_lefts[1]:]
                        cvlf0 = cvlf0[:,outpad_lefts[1]:]
                    if outpad_rights[3] > 0:
                        cvmelsp_src_post = cvmelsp_src_post[:,outpad_lefts[3]:-outpad_rights[3]]
                        cvmelsp_post = cvmelsp_post[:,outpad_lefts[3]:-outpad_rights[3]]
                    else:
                        cvmelsp_src_post = cvmelsp_src_post[:,outpad_lefts[3]:]
                        cvmelsp_post = cvmelsp_post[:,outpad_lefts[3]:]
                    if outpad_rights[5] > 0:
                        cvlf0_cyc = cvlf0_cyc[:,outpad_lefts[5]:-outpad_rights[5]]
                    else:
                        cvlf0_cyc = cvlf0_cyc[:,outpad_lefts[5]:]

                    feat_cv = cvmelsp[0].cpu().data.numpy()

                    cvmelsp_src = np.array(cvmelsp_src_post[0].cpu().data.numpy(), dtype=np.float64)
                    cvlf0_src = np.array(cvlf0_src[0].cpu().data.numpy(), dtype=np.float64)

                    cvmelsp = np.array(cvmelsp_post[0].cpu().data.numpy(), dtype=np.float64)
                    cvlf0 = np.array(cvlf0[0].cpu().data.numpy(), dtype=np.float64)

                    cvmelsp_cyc = np.array(cvmelsp_cyc_post[0].cpu().data.numpy(), dtype=np.float64)
                    cvlf0_cyc = np.array(cvlf0_cyc[0].cpu().data.numpy(), dtype=np.float64)

                    if trg_exist:
                        if outpad_rights[0] > 0:
                            lat_src = torch.cat((lat_src_e, lat_src), 2)[:,outpad_lefts[0]:-outpad_rights[0]]
                        else:
                            lat_src = torch.cat((lat_src_e, lat_src), 2)[:,outpad_lefts[0]:]
                        lat_trg = torch.cat((lat_trg_e, lat_trg), 2)

                logging.info(cvlf0_src.shape)
                logging.info(cvmelsp_src.shape)

                logging.info(cvlf0.shape)
                logging.info(cvmelsp.shape)

                logging.info(cvlf0_cyc.shape)
                logging.info(cvmelsp_cyc.shape)

                melsp = np.array(feat_org)

                feat_world = read_hdf5(feat_file, "/feat_mceplf0cap")
                f0 = np.array(np.rint(feat_world[:,0])*np.exp(feat_world[:,1]))
                codeap = np.array(np.rint(feat_world[:,2:3])*(-np.exp(feat_world[:,3:config.full_excit_dim])))
                if trg_exist:
                    logging.info(lat_src.shape)
                    logging.info(lat_trg.shape)
                    melsp_trg = np.array(feat_trg)
                    feat_world_trg = read_hdf5(file_trg, "/feat_mceplf0cap")
                    f0_trg = np.array(np.rint(feat_world_trg[:,0])*np.exp(feat_world_trg[:,1]))
                    codeap_trg = np.array(np.rint(feat_world_trg[:,2:3])*(-np.exp(feat_world_trg[:,3:config.full_excit_dim])))
 
                cvf0_src = np.array(np.rint(cvlf0_src[:,0])*np.exp(cvlf0_src[:,1]))
                cvcodeap_src = np.array(np.rint(cvlf0_src[:,2:3])*(-np.exp(cvlf0_src[:,3:])))
                f0_rmse = np.sqrt(np.mean((cvf0_src-f0)**2))
                logging.info('F0_rmse: %lf Hz' % (f0_rmse))
                f0rmse_cvlist_src.append(f0_rmse)
                cvf0_src_mean = np.mean(cvf0_src)
                f0_mean = np.mean(f0)
                f0_corr = np.sum((cvf0_src-cvf0_src_mean)*(f0-f0_mean))/(np.sqrt(np.sum((cvf0_src-cvf0_src_mean)**2))*np.sqrt(np.sum((f0-f0_mean)**2)))
                logging.info('F0_corr: %lf' % (f0_corr))
                f0corr_cvlist_src.append(f0_corr)
                codeap_rmse = np.sqrt(np.mean((cvcodeap_src-codeap)**2, axis=0))
                for i in range(codeap_rmse.shape[-1]):
                    logging.info('codeap-%d_rmse: %lf dB' % (i+1, codeap_rmse[i]))
                caprmse_cvlist_src.append(codeap_rmse)

                cvf0_cyc = np.array(np.rint(cvlf0_cyc[:,0])*np.exp(cvlf0_cyc[:,1]))
                cvcodeap_cyc = np.array(np.rint(cvlf0_cyc[:,2:3])*(-np.exp(cvlf0_cyc[:,3:])))
                f0_rmse = np.sqrt(np.mean((cvf0_cyc-f0)**2))
                logging.info('F0_rmse_cyc: %lf Hz' % (f0_rmse))
                f0rmse_cvlist_cyc.append(f0_rmse)
                cvf0_cyc_mean = np.mean(cvf0_cyc)
                f0_mean = np.mean(f0)
                f0_corr = np.sum((cvf0_cyc-cvf0_cyc_mean)*(f0-f0_mean))/(np.sqrt(np.sum((cvf0_cyc-cvf0_cyc_mean)**2))*np.sqrt(np.sum((f0-f0_mean)**2)))
                logging.info('F0_corr_cyc: %lf' % (f0_corr))
                f0corr_cvlist_cyc.append(f0_corr)
                codeap_rmse = np.sqrt(np.mean((cvcodeap_cyc-codeap)**2, axis=0))
                for i in range(codeap_rmse.shape[-1]):
                    logging.info('codeap-%d_rmse_cyc: %lf dB' % (i+1, codeap_rmse[i]))
                caprmse_cvlist_cyc.append(codeap_rmse)

                cvf0 = np.array(np.rint(cvlf0[:,0])*np.exp(cvlf0[:,1]))
                cvcodeap = np.array(np.rint(cvlf0[:,2:3])*(-np.exp(cvlf0[:,3:])))

                #if trg_exist:
                #    for i in range(codeap_rmse.shape[-1]):
                #        figname = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5","_cap-"+str(i+1)+".png"))
                #        plt.subplot(3, 1, 1)
                #        plt.plot(codeap[:,i])
                #        plt.title("source codeap-"+str(i+1))
                #        plt.subplot(3, 1, 2)
                #        plt.plot(codeap_trg[:,i])
                #        plt.title("target codeap-"+str(i+1))
                #        plt.subplot(3, 1, 3)
                #        plt.plot(cvcodeap[:,i])
                #        plt.title("converted codeap-"+str(i+1))
                #        plt.tight_layout()
                #        plt.savefig(figname)
                #        plt.close()
                #else:
                #    for i in range(codeap_rmse.shape[-1]):
                #        figname = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5","_cap-"+str(i+1)+".png"))
                #        plt.subplot(3, 1, 1)
                #        plt.plot(codeap[:,i])
                #        plt.title("source codeap-"+str(i+1))
                #        plt.subplot(3, 1, 2)
                #        plt.plot(cvcodeap_src[:,i])
                #        plt.title("reconstructed codeap-"+str(i+1))
                #        plt.subplot(3, 1, 3)
                #        plt.plot(cvcodeap[:,i])
                #        plt.title("converted codeap-"+str(i+1))
                #        plt.tight_layout()
                #        plt.savefig(figname)
                #        plt.close()

                logging.info("cvf0lin")
                cvf0_lin = convert_f0(f0, src_f0_mean, src_f0_std, trg_f0_mean, trg_f0_std)
                #uv_range_lin, cont_f0_range_lin = convert_continuos_f0(np.array(cvf0_lin))
                #unique, counts = np.unique(uv_range_lin, return_counts=True)
                #logging.info(dict(zip(unique, counts)))
                #cont_f0_lpf_range_lin = \
                #    low_pass_filter(cont_f0_range_lin, int(1.0 / (args.shiftms * 0.001)), cutoff=20)
                #uv_range_lin = np.expand_dims(uv_range_lin, axis=-1)
                #cont_f0_lpf_range_lin = np.expand_dims(cont_f0_lpf_range_lin, axis=-1)

                f0_rmse = np.sqrt(np.mean((cvf0-cvf0_lin)**2))
                logging.info('F0_rmse_cv: %lf Hz' % (f0_rmse))
                f0rmse_cvlist_cv.append(f0_rmse)
                cvf0_mean = np.mean(cvf0)
                f0_mean = np.mean(cvf0_lin)
                f0_corr = np.sum((cvf0-cvf0_mean)*(cvf0_lin-f0_mean))/(np.sqrt(np.sum((cvf0-cvf0_mean)**2))*np.sqrt(np.sum((cvf0_lin-f0_mean)**2)))
                logging.info('F0_corr_cv: %lf' % (f0_corr))
                f0corr_cvlist_cv.append(f0_corr)

                #if trg_exist:
                #    figname = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5","_f0.png"))
                #    plt.subplot(3, 1, 1)
                #    plt.plot(f0)
                #    plt.title("source f0")
                #    plt.subplot(3, 1, 2)
                #    plt.plot(f0_trg)
                #    plt.title("target f0")
                #    plt.subplot(3, 1, 3)
                #    plt.plot(cvf0)
                #    plt.title("converted f0")
                #    plt.tight_layout()
                #    plt.savefig(figname)
                #    plt.close()
                #else:
                #    figname = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5","_f0.png"))
                #    plt.subplot(3, 1, 1)
                #    plt.plot(f0)
                #    plt.title("source f0")
                #    plt.subplot(3, 1, 2)
                #    plt.plot(cvf0_lin)
                #    plt.title("linear converted f0")
                #    plt.subplot(3, 1, 3)
                #    plt.plot(cvf0)
                #    plt.title("converted f0")
                #    plt.tight_layout()
                #    plt.savefig(figname)
                #    plt.close()

                spcidx = np.array(read_hdf5(feat_file, "/spcidx_range")[0])

                melsp_rest = (np.exp(melsp)-1)/10000
                melsp_cv_rest = (np.exp(cvmelsp)-1)/10000
                melsp_src_rest = (np.exp(cvmelsp_src)-1)/10000
                melsp_cyc_rest = (np.exp(cvmelsp_cyc)-1)/10000

                cvlist.append(np.var(melsp_cv_rest, axis=0))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_src_rest[spcidx], a_min=1e-16, a_max=None))\
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_mean = np.mean(lsd_arr)
                lsd_std = np.std(lsd_arr)
                logging.info("lsd_src_cv: %.6f dB +- %.6f" % (lsd_mean, lsd_std))
                lsd_cvlist_src.append(lsd_mean)
                lsdstd_cvlist_src.append(lsd_std)

                if trg_exist:
                    melsp_trg_rest = (np.exp(melsp_trg)-1)/10000

                    spcidx_trg = np.array(read_hdf5(file_trg, "/spcidx_range")[0])

                    _, twf_melsp, _, _ = dtw.dtw_org_to_trg(np.array(melsp_cv_rest[spcidx], \
                                                dtype=np.float64), np.array(melsp_trg_rest[spcidx_trg], dtype=np.float64), mcd=-1)
                    twf_melsp = np.array(twf_melsp[:,0])
                    lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_cv_rest[twf_melsp], a_min=1e-16, a_max=None))\
                                                             -np.log10(np.clip(melsp_rest[twf_melsp], a_min=1e-16, a_max=None))))**2, axis=-1))
                    lsd_mean = np.mean(lsd_arr)
                    lsd_std = np.std(lsd_arr)
                    logging.info("lsd_trg: %.6f dB +- %.6f" % (lsd_mean, lsd_std))
                    lsd_cvlist.append(lsd_mean)
                    lsdstd_cvlist.append(lsd_std)

                    spcidx_src = torch.LongTensor(spcidx).cuda()
                    spcidx_trg = torch.LongTensor(spcidx_trg).cuda()

                    trj_lat_src = np.array(torch.index_select(lat_src[0],0,spcidx_src).cpu().data.numpy(), dtype=np.float64)
                    trj_lat_trg = np.array(torch.index_select(lat_trg[0],0,spcidx_trg).cpu().data.numpy(), dtype=np.float64)
                    aligned_lat_srctrg, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src, trj_lat_trg)
                    lat_dist_srctrg = np.mean(np.sqrt(np.mean((aligned_lat_srctrg-trj_lat_trg)**2, axis=0)))
                    _, _, lat_cdist_srctrg, _ = dtw.dtw_org_to_trg(trj_lat_trg, trj_lat_src, mcd=0)
                    aligned_lat_trgsrc, _, _, _ = dtw.dtw_org_to_trg(trj_lat_trg, trj_lat_src)
                    lat_dist_trgsrc = np.mean(np.sqrt(np.mean((aligned_lat_trgsrc-trj_lat_src)**2, axis=0)))
                    _, _, lat_cdist_trgsrc, _ = dtw.dtw_org_to_trg(trj_lat_src, trj_lat_trg, mcd=0)
                    logging.info("%lf %lf %lf %lf" % (lat_dist_srctrg, lat_cdist_srctrg, lat_dist_trgsrc, lat_cdist_trgsrc))
                    lat_dist_rmse = (lat_dist_srctrg+lat_dist_trgsrc)/2
                    lat_dist_cosim = (lat_cdist_srctrg+lat_cdist_trgsrc)/2
                    lat_dist_rmse_list.append(lat_dist_rmse)
                    lat_dist_cosim_list.append(lat_dist_cosim)
                    logging.info("lat_dist: %.6f %.6f" % (lat_dist_rmse, lat_dist_cosim))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_cyc_rest[spcidx], a_min=1e-16, a_max=None))\
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_mean_cyc = np.mean(lsd_arr)
                lsd_std_cyc = np.std(lsd_arr)
                logging.info("lsd_cyc: %.6f dB +- %.6f" % (lsd_mean_cyc, lsd_std_cyc))
                lsd_cvlist_cyc.append(lsd_mean_cyc)
                lsdstd_cvlist_cyc.append(lsd_std_cyc)
            
                logging.info('org f0')
                logging.info(f0[10:15])
                logging.info('rec f0')
                logging.info(cvf0_src[10:15])
                logging.info('cyc f0')
                logging.info(cvf0_cyc[10:15])
                logging.info('cv f0')
                logging.info(cvf0[10:15])
                logging.info('lin f0')
                logging.info(cvf0_lin[10:15])
                logging.info('org cap')
                logging.info(codeap[10:15])
                logging.info('rec cap')
                logging.info(cvcodeap_src[10:15])
                logging.info('cyc cap')
                logging.info(cvcodeap_cyc[10:15])
                logging.info('cv cap')
                logging.info(cvcodeap[10:15])

                logging.info("synth anasyn")
                magsp = np.matmul(melfb_t, melsp_rest.T)
                logging.info(magsp.shape)
                hop_length = int((args.fs/1000)*args.shiftms)
                win_length = int((args.fs/1000)*args.winms)
                wav = np.clip(librosa.core.griffinlim(magsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir,os.path.basename(feat_file).replace(".h5","_anasyn.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                #if trg_exist:
                #    logging.info("synth anasyn_trg")
                #    wav = np.clip(pw.synthesize(f0_trg, sp_trg, ap_trg, fs, frame_period=args.shiftms), -1, 1)
                #    wavpath = os.path.join(args.outdir,os.path.basename(feat_file).replace(".h5","_anasyn_trg.wav"))
                #    sf.write(wavpath, wav, fs, 'PCM_16')
                #    logging.info(wavpath)

                if args.n_interp == 0:
                    logging.info("synth gf rec")
                    recmagsp = np.matmul(melfb_t, melsp_src_rest.T)
                    logging.info(recmagsp.shape)
                    wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                                win_length=win_length, window='hann'), -1, 0.999969482421875)
                    wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_rec.wav"))
                    logging.info(wavpath)
                    sf.write(wavpath, wav, args.fs, 'PCM_16')

                    logging.info("synth gf cv")
                    cvmagsp = np.matmul(melfb_t, melsp_cv_rest.T)
                    logging.info(cvmagsp.shape)
                    wav = np.clip(librosa.core.griffinlim(cvmagsp, hop_length=hop_length,
                                win_length=win_length, window='hann'), -1, 0.999969482421875)
                    wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_cv.wav"))
                    logging.info(wavpath)
                    sf.write(wavpath, wav, args.fs, 'PCM_16')

                    #logging.info("synth gf cv GV")
                    #datamean = np.mean(melsp_cv_rest, axis=0)
                    #cvmelsp_gv =  args.gv_coeff*(np.sqrt(gv_mean_trg/cvgv_mean) * \
                    #                    (melsp_cv_rest-datamean) + datamean) + (1-args.gv_coeff)*melsp_cv_rest
                    #logging.info(cvmelsp_gv.shape)
                    #cvmagsp_gv = np.matmul(melfb_t, cvmelsp_gv.T)
                    #logging.info(cvmagsp_gv.shape)
                    #wav = np.clip(librosa.core.griffinlim(cvmagsp, hop_length=hop_length,
                    #            win_length=win_length, window='hann'), -1, 0.999969482421875)
                    #wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_cvGV.wav"))
                    #logging.info(wavpath)
                    #sf.write(wavpath, wav, args.fs, 'PCM_16')
                #else:
                #    logging.info("synth voco rec")
                #    cvsp_src = ps.mc2sp(cvmcep_src, args.mcep_alpha, fft_size)
                #    cvap_src = pw.decode_aperiodicity(cvcodeap_src, args.fs, args.fftl)
                #    logging.info(cvsp_src.shape)
                #    logging.info(cvap_src.shape)
                #    wav = np.clip(pw.synthesize(cvf0_src, cvsp_src, cvap_src, fs, frame_period=args.shiftms), -1, 1)
                #    if args.n_interp < 10:
                #        cvstr = "cv0"
                #    elif args.n_interp < 100:
                #        cvstr = "cv00"
                #    elif args.n_interp < 1000:
                #        cvstr = "cv000"
                #    elif args.n_interp < 10000:
                #        cvstr = "cv0000"
                #    wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_"+cvstr+"_"+str(round(z_interpolate[0][0], 3))+"_"+str(round(z_interpolate[0][1], 3)) \
                #                +"_"+str(round(z_e_interpolate[0][0], 3))+"_"+str(round(z_e_interpolate[0][1], 3)) \
                #                +"_spec-"+str(spk_interpolate[0])+"-"+str(round(spk_prob_interpolate[0], 2))+"_exct-"+str(spk_e_interpolate[0])+"-"+str(round(spk_prob_e_interpolate[0], 2))+".wav"))
                #    sf.write(wavpath, wav, fs, 'PCM_16')
                #    logging.info(wavpath)

                #    for i in range(n_delta-1):
                #        if n_delta < 10:
                #            cvstr = "cv"
                #        elif n_delta < 100:
                #            if i+1 < 10:
                #                cvstr = "cv0"
                #            else:
                #                cvstr = "cv"
                #        elif n_delta < 1000:
                #            if i+1 < 10:
                #                cvstr = "cv00"
                #            elif i+1 < 100:
                #                cvstr = "cv0"
                #            else:
                #                cvstr = "cv"
                #        elif n_delta < 10000:
                #            if i+1 < 10:
                #                cvstr = "cv000"
                #            elif i+1 < 100:
                #                cvstr = "cv00"
                #            elif i+1 < 1000:
                #                cvstr = "cv0"
                #            else:
                #                cvstr = "cv"

                #        logging.info("synth voco interpolate-%d" % (i+1))
                #        cvmcep_ = cvmcep_interpolate[i]
                #        cvsp = ps.mc2sp(cvmcep_, args.mcep_alpha, fft_size)
                #        cvlf0_ = cvlf0_interpolate[i]
                #        cvf0_ = np.array(np.rint(cvlf0_[:,0])*np.exp(cvlf0_[:,1]))
                #        cvcodeap_ = np.array(np.rint(cvlf0_[:,2:3])*(-np.exp(cvlf0_[:,3:])))
                #        cvap = pw.decode_aperiodicity(cvcodeap_, args.fs, args.fftl)
                #        logging.info(cvsp.shape)
                #        logging.info(cvap.shape)
                #        wav = np.clip(pw.synthesize(cvf0_, cvsp, cvap, fs, frame_period=args.shiftms), -1, 1)
                #        wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_"+cvstr+str(i+1)+"_"+str(round(z_interpolate[i+1][0], 3))+"_"+str(round(z_interpolate[i+1][1], 3)) \
                #                        +"_"+str(round(z_e_interpolate[i+1][0], 3))+"_"+str(round(z_e_interpolate[i+1][1], 3)) \
                #                        +"_spec-"+str(spk_interpolate[i+1])+"-"+str(round(spk_prob_interpolate[i+1], 2))+"_exct-"+str(spk_e_interpolate[i+1])+"-"+str(round(spk_prob_e_interpolate[i+1], 2))+".wav"))
                #        sf.write(wavpath, wav, fs, 'PCM_16')
                #        logging.info(wavpath)

                #        logging.info("synth voco cv GV interpolate-%d" % (i+1))
                #        datamean = np.mean(cvmcep_[:,1:], axis=0)
                #        cvmcep_gv =  np.c_[cvmcep_[:,0], args.gv_coeff*(np.sqrt(gv_mean_trgs[spk_idx_interpolate[i]]/cvgv_means[spk_idx_interpolate[i]]) * \
                #                            (cvmcep_[:,1:]-datamean) + datamean) + (1-args.gv_coeff)*cvmcep_[:,1:]]
                #        cvmcep_gv = mod_pow(cvmcep_gv, cvmcep_, alpha=args.mcep_alpha, irlen=IRLEN)
                #        cvsp_gv = ps.mc2sp(cvmcep_gv, args.mcep_alpha, fft_size)
                #        logging.info(cvsp_gv.shape)
                #        wav = np.clip(pw.synthesize(cvf0_, cvsp_gv, cvap, fs, frame_period=args.shiftms), -1, 1)
                #        wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_"+cvstr+str(i+1)+"_"+str(round(z_interpolate[i+1][0], 3))+"_"+str(round(z_interpolate[i+1][1], 3)) \
                #                        +"_"+str(round(z_e_interpolate[i+1][0], 3))+"_"+str(round(z_e_interpolate[i+1][1], 3)) \
                #                        +"_spec-"+str(spk_interpolate[i+1])+"-"+str(round(spk_prob_interpolate[i+1], 2))+"_exct-"+str(spk_e_interpolate[i+1])+"-"+str(round(spk_prob_e_interpolate[i+1], 2))+"_GV.wav"))
                #        sf.write(wavpath, wav, fs, 'PCM_16')
                #        logging.info(wavpath)

                #    logging.info("synth voco cv")
                #    cvsp = ps.mc2sp(cvmcep, args.mcep_alpha, fft_size)
                #    cvap = pw.decode_aperiodicity(cvcodeap, args.fs, args.fftl)
                #    logging.info(cvsp.shape)
                #    logging.info(cvap.shape)
                #    wav = np.clip(pw.synthesize(cvf0, cvsp, cvap, fs, frame_period=args.shiftms), -1, 1)
                #    wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_cv"+str(n_delta)+"_"+str(round(z_interpolate[n_delta][0], 3))+"_"+str(round(z_interpolate[n_delta][1], 3))\
                #                +"_"+str(round(z_e_interpolate[n_delta][0], 3))+"_"+str(round(z_e_interpolate[n_delta][1], 3))\
                #                +"_spec-"+str(spk_interpolate[n_delta])+"-"+str(round(spk_prob_interpolate[n_delta], 2))+"_exct-"+str(spk_e_interpolate[n_delta])+"-"+str(round(spk_prob_e_interpolate[n_delta], 2))+".wav"))
                #    sf.write(wavpath, wav, fs, 'PCM_16')
                #    logging.info(wavpath)

                #    logging.info("synth voco cv GV")
                #    datamean = np.mean(cvmcep[:,1:], axis=0)
                #    cvmcep_gv =  np.c_[cvmcep[:,0], args.gv_coeff*(np.sqrt(gv_mean_trg/cvgv_mean) * \
                #                        (cvmcep[:,1:]-datamean) + datamean) + (1-args.gv_coeff)*cvmcep[:,1:]]
                #    cvmcep_gv = mod_pow(cvmcep_gv, cvmcep, alpha=args.mcep_alpha, irlen=IRLEN)
                #    cvsp_gv = ps.mc2sp(cvmcep_gv, args.mcep_alpha, fft_size)
                #    logging.info(cvsp_gv.shape)
                #    wav = np.clip(pw.synthesize(cvf0, cvsp_gv, cvap, fs, frame_period=args.shiftms), -1, 1)
                #    wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_cv"+str(n_delta)+"_"+str(round(z_interpolate[n_delta][0], 3))+"_"+str(round(z_interpolate[n_delta][1], 3))\
                #                +"_"+str(round(z_e_interpolate[n_delta][0], 3))+"_"+str(round(z_e_interpolate[n_delta][1], 3))\
                #                +"_spec-"+str(spk_interpolate[n_delta])+"-"+str(round(spk_prob_interpolate[n_delta], 2))+"_exct-"+str(spk_e_interpolate[n_delta])+"-"+str(round(spk_prob_e_interpolate[n_delta], 2))+"_GV.wav"))
                #    sf.write(wavpath, wav, fs, 'PCM_16')
                #    logging.info(wavpath)

                #logging.info("write lat")
                #outTxtDir = os.path.join(args.outdir, os.path.basename(os.path.dirname(feat_file)))
                #if not os.path.exists(outTxtDir):
                #    os.mkdir(outTxtDir)
                #outTxt = os.path.join(outTxtDir, os.path.basename(feat_file).replace(".wav", ".txt"))
                #logging.info(outTxt)
                #g = open(outTxt, "wt")
                #idx_frm = 0 
                #nfrm = trj_lat_src.shape[0]
                #dim = trj_lat_src.shape[1]
                #if not args.time_flag:
                ##if True:
                #    while idx_frm < nfrm:
                #        idx_elmt = 1 
                #        for elmt in trj_lat_src[idx_frm]:
                #            if idx_elmt < dim:
                #                g.write("%lf " % (elmt))
                #            else:
                #                g.write("%lf\n" % (elmt))
                #            idx_elmt += 1
                #        idx_frm += 1
                #else:
                #    while idx_frm < nfrm:
                #        idx_elmt = 1 
                #        for elmt in trj_lat_src[idx_frm]:
                #            if idx_elmt < dim:
                #                if idx_elmt > 1:
                #                    g.write("%lf " % (elmt))
                #                else:
                #                    g.write("%lf %lf " % (time_axis[idx_frm], elmt))
                #            else:
                #                g.write("%lf\n" % (elmt))
                #            idx_elmt += 1
                #        idx_frm += 1
                #g.close()

                logging.info('write to h5')
                outh5dir = os.path.join(os.path.dirname(os.path.dirname(feat_file)), spk_src+"-"+args.spk_trg)
                if not os.path.exists(outh5dir):
                    os.makedirs(outh5dir)
                feat_file = os.path.join(outh5dir, os.path.basename(feat_file))
                # cv
                write_path = args.string_path
                logging.info(feat_file + ' ' + write_path)
                logging.info(feat_cv.shape)
                write_hdf5(feat_file, write_path, feat_cv)

                count += 1
                #if count >= 3:
                #    break


    with mp.Manager() as manager:
        logging.info("GRU-RNN decoding")
        processes = []
        cvlist = manager.list()
        lsd_cvlist_src = manager.list()
        lsdstd_cvlist_src = manager.list()
        f0rmse_cvlist_src = manager.list()
        f0corr_cvlist_src = manager.list()
        caprmse_cvlist_src = manager.list()
        lsd_cvlist_cyc = manager.list()
        lsdstd_cvlist_cyc = manager.list()
        f0rmse_cvlist_cyc = manager.list()
        f0corr_cvlist_cyc = manager.list()
        caprmse_cvlist_cyc = manager.list()
        f0rmse_cvlist_cv = manager.list()
        f0corr_cvlist_cv = manager.list()
        lsd_cvlist = manager.list()
        lsdstd_cvlist = manager.list()
        lat_dist_rmse_list = manager.list()
        lat_dist_cosim_list = manager.list()
        gpu = 0
        for i, feat_list in enumerate(feat_lists):
            logging.info(i)
            p = mp.Process(target=decode_RNN, args=(feat_list, gpu, cvlist,
                lsd_cvlist_src, lsdstd_cvlist_src,
                    f0rmse_cvlist_src, f0corr_cvlist_src, caprmse_cvlist_src,
                lsd_cvlist_cyc, lsdstd_cvlist_cyc,
                    f0rmse_cvlist_cyc, f0corr_cvlist_cyc, caprmse_cvlist_cyc,
                f0rmse_cvlist_cv, f0corr_cvlist_cv,
                    lsd_cvlist, lsdstd_cvlist,
                lat_dist_rmse_list, lat_dist_cosim_list,))
            p.start()
            processes.append(p)
            gpu += 1
            if (i + 1) % args.n_gpus == 0:
                gpu = 0

        # wait for all process
        for p in processes:
            p.join()

        # calculate statistics
        logging.info("== summary rec. acc. ==")
        logging.info("lsd_src_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist_src)),\
        np.std(np.array(lsd_cvlist_src)),np.mean(np.array(lsdstd_cvlist_src)),np.std(np.array(lsdstd_cvlist_src))))
        logging.info("f0rmse_src_cv: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_src)),np.std(np.array(f0rmse_cvlist_src))))
        logging.info("f0corr_src_cv: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_src)),np.std(np.array(f0corr_cvlist_src))))
        caprmse_cvlist_src = np.array(caprmse_cvlist_src)
        for i in range(caprmse_cvlist_src.shape[-1]):
            logging.info("caprmse-%d_src_cv: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist_src[:,i]),np.std(caprmse_cvlist_src[:,i])))
        logging.info("=== summary cyc. acc. ===")
        logging.info("lsd_cyc_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist_cyc)),\
        np.std(np.array(lsd_cvlist_cyc)),np.mean(np.array(lsdstd_cvlist_cyc)),np.std(np.array(lsdstd_cvlist_cyc))))
        logging.info("f0rmse_cyc_cv: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_cyc)),np.std(np.array(f0rmse_cvlist_cyc))))
        logging.info("f0corr_cyc_cv: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_cyc)),np.std(np.array(f0corr_cvlist_cyc))))
        caprmse_cvlist_cyc = np.array(caprmse_cvlist_cyc)
        for i in range(caprmse_cvlist_cyc.shape[-1]):
            logging.info("caprmse-%d_cyc_cv: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist_cyc[:,i]),np.std(caprmse_cvlist_cyc[:,i])))
        logging.info("=== summary cv. acc. ===")
        logging.info("f0rmse_cv: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_cv)),np.std(np.array(f0rmse_cvlist_cv))))
        logging.info("f0corr_cv: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_cv)),np.std(np.array(f0corr_cvlist_cv))))
        cvgv_mean = np.mean(np.array(cvlist), axis=0)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg))))))
        if len(lsd_cvlist) > 0:
            logging.info("lsd_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist)),\
            np.std(np.array(lsd_cvlist)),np.mean(np.array(lsdstd_cvlist)),np.std(np.array(lsdstd_cvlist))))
            logging.info("lat_dist_rmse: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_rmse_list)),np.std(np.array(lat_dist_rmse_list))))
            logging.info("lat_dist_cosim: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_cosim_list)),np.std(np.array(lat_dist_cosim_list))))
  
 
if __name__ == "__main__":
    main()
