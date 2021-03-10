#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
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

from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER, GRU_SPK
from vcneuvoco import SPKID_TRANSFORM_LAYER
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
            lsd_cvlist_cyc=None, lsdstd_cvlist_cyc=None,
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
                model_decoder_melsp = GRU_SPEC_DECODER(
                    feat_dim=config.lat_dim+config.lat_dim_e,
                    out_dim=config.mel_dim,
                    n_spk=n_spk,
                    aux_dim=n_spk,
                    hidden_layers=config.hidden_layers_dec,
                    hidden_units=config.hidden_units_dec,
                    kernel_size=config.kernel_size_dec,
                    dilation_size=config.dilation_size_dec,
                    causal_conv=config.causal_conv_dec,
                    pad_first=True,
                    right_size=config.right_size_dec,
                    red_dim_upd=config.mel_dim,
                    pdf=True)
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
                if (config.spkidtr_dim > 0):
                    model_spkidtr = SPKID_TRANSFORM_LAYER(
                        n_spk=n_spk,
                        spkidtr_dim=config.spkidtr_dim)
                    logging.info(model_spkidtr)
                else:
                    model_spkidtr = None
                model_spk = GRU_SPK(
                    n_spk=n_spk,
                    feat_dim=config.lat_dim+config.lat_dim_e,
                    hidden_units=32,
                    kernel_size=config.kernel_size_spk,
                    dilation_size=config.dilation_size_spk,
                    causal_conv=config.causal_conv_spk,
                    pad_first=True,
                    right_size=config.right_size_spk,
                    red_dim=config.mel_dim)
                logging.info(model_spk)
                model_encoder_melsp.load_state_dict(torch.load(args.model)["model_encoder_melsp"])
                model_decoder_melsp.load_state_dict(torch.load(args.model)["model_decoder_melsp"])
                model_encoder_excit.load_state_dict(torch.load(args.model)["model_encoder_excit"])
                if (config.spkidtr_dim > 0):
                    model_spkidtr.load_state_dict(torch.load(args.model)["model_spkidtr"])
                model_spk.load_state_dict(torch.load(args.model)["model_spk"])
                model_encoder_melsp.cuda()
                model_decoder_melsp.cuda()
                model_encoder_excit.cuda()
                if (config.spkidtr_dim > 0):
                    model_spkidtr.cuda()
                model_spk.cuda()
                model_encoder_melsp.eval()
                model_decoder_melsp.eval()
                model_encoder_excit.eval()
                if (config.spkidtr_dim > 0):
                    model_spkidtr.eval()
                model_spk.eval()
                model_encoder_melsp.remove_weight_norm()
                model_decoder_melsp.remove_weight_norm()
                model_encoder_excit.remove_weight_norm()
                if (config.spkidtr_dim > 0):
                    model_spkidtr.remove_weight_norm()
                model_spk.remove_weight_norm()
                for param in model_encoder_melsp.parameters():
                    param.requires_grad = False
                for param in model_decoder_melsp.parameters():
                    param.requires_grad = False
                for param in model_encoder_excit.parameters():
                    param.requires_grad = False
                if (config.spkidtr_dim > 0):
                    for param in model_spkidtr.parameters():
                        param.requires_grad = False
                for param in model_spk.parameters():
                    param.requires_grad = False
            count = 0
            pad_left = (model_encoder_melsp.pad_left + model_spk.pad_left + model_decoder_melsp.pad_left)*2
            pad_right = (model_encoder_melsp.pad_right + model_spk.pad_right + model_decoder_melsp.pad_right)*2
            outpad_lefts = [None]*5
            outpad_rights = [None]*5
            outpad_lefts[0] = pad_left-model_encoder_melsp.pad_left
            outpad_rights[0] = pad_right-model_encoder_melsp.pad_right
            outpad_lefts[1] = outpad_lefts[0]-model_spk.pad_left
            outpad_rights[1] = outpad_rights[0]-model_spk.pad_right
            outpad_lefts[2] = outpad_lefts[1]-model_decoder_melsp.pad_left
            outpad_rights[2] = outpad_rights[1]-model_decoder_melsp.pad_right
            outpad_lefts[3] = outpad_lefts[2]-model_encoder_melsp.pad_left
            outpad_rights[3] = outpad_rights[2]-model_encoder_melsp.pad_right
            outpad_lefts[4] = outpad_lefts[3]-model_spk.pad_left
            outpad_rights[4] = outpad_rights[3]-model_spk.pad_right
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
                    
                    if model_spk.pad_right > 0:
                        lat_cat = lat_cat[:,model_spk.pad_left:-model_spk.pad_right]
                        src_code = src_code[:,model_spk.pad_left:-model_spk.pad_right]
                        trg_code = trg_code[:,model_spk.pad_left:-model_spk.pad_right]
                    else:
                        lat_cat = lat_cat[:,model_spk.pad_left:]
                        src_code = src_code[:,model_spk.pad_left:]
                        trg_code = trg_code[:,model_spk.pad_left:]

                    _, cvmelsp_src, _ = model_decoder_melsp(lat_cat, y=src_code, aux=trj_src_code)
                    _, cvmelsp, _ = model_decoder_melsp(lat_cat, y=trg_code, aux=trj_trg_code)
                    trj_lat_cat = lat_cat_src = lat_cat

                    spk_logits, _, lat_cv, _ = model_encoder_melsp(cvmelsp, sampling=False)
                    spk_logits_e, _, lat_cv_e, _ = model_encoder_excit(cvmelsp, sampling=False)
                    logging.info('cv spkpost')
                    if outpad_rights[3] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[3]:-outpad_rights[3]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[3]:], dim=-1), 1))
                    logging.info('cv spkpost_e')
                    if outpad_rights[3] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[3]:-outpad_rights[3]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[3]:], dim=-1), 1))

                    if config.spkidtr_dim > 0:
                        src_code = model_spkidtr((torch.ones((1, lat_cv_e.shape[1]))*src_idx).cuda().long())
                    else:
                        src_code = (torch.ones((1, lat_cv_e.shape[1]))*src_idx).cuda().long()
                    lat_cat = torch.cat((lat_cv_e, lat_cv), 2)
                    trj_src_code, _ = model_spk(src_code, z=lat_cat)
    
                    if model_spk.pad_right > 0:
                        lat_cat = lat_cat[:,model_spk.pad_left:-model_spk.pad_right]
                        src_code = src_code[:,model_spk.pad_left:-model_spk.pad_right]
                        trg_code = trg_code[:,model_spk.pad_left:-model_spk.pad_right]
                    else:
                        lat_cat = lat_cat[:,model_spk.pad_left:]
                        src_code = src_code[:,model_spk.pad_left:]
                        trg_code = trg_code[:,model_spk.pad_left:]

                    _, cvmelsp_cyc, _ = model_decoder_melsp(lat_cat, y=src_code, aux=trj_src_code)

                    #if outpad_rights[1] > 0:
                    #    trj_lat_cat = trj_lat_cat[:,outpad_lefts[1]:-outpad_rights[1]]
                    #else:
                    #    trj_lat_cat = trj_lat_cat[:,outpad_lefts[1]:-outpad_rights[1]]
                    if outpad_rights[2] > 0:
                        cvmelsp_src = cvmelsp_src[:,outpad_lefts[2]:-outpad_rights[2]]
                        cvmelsp = cvmelsp[:,outpad_lefts[2]:-outpad_rights[2]]
                    else:
                        cvmelsp_src = cvmelsp_src[:,outpad_lefts[2]:]
                        cvmelsp = cvmelsp[:,outpad_lefts[2]:]

                    feat_cv = cvmelsp[0].cpu().data.numpy()
                    #feat_lat = trj_lat_cat[0].cpu().data.numpy()

                    cvmelsp_src = np.array(cvmelsp_src[0].cpu().data.numpy(), dtype=np.float64)
                    cvmelsp = np.array(cvmelsp[0].cpu().data.numpy(), dtype=np.float64)
                    cvmelsp_cyc = np.array(cvmelsp_cyc[0].cpu().data.numpy(), dtype=np.float64)

                    if trg_exist:
                        if outpad_rights[1] > 0:
                            lat_src = lat_cat_src[:,outpad_lefts[1]:-outpad_rights[1]]
                        else:
                            lat_src = lat_cat_src[:,outpad_lefts[1]:]
                        lat_trg = torch.cat((lat_trg_e, lat_trg), 2)

                logging.info(cvmelsp_src.shape)
                logging.info(cvmelsp.shape)
                logging.info(cvmelsp_cyc.shape)

                melsp = np.array(feat_org)

                if trg_exist:
                    logging.info(lat_src.shape)
                    logging.info(lat_trg.shape)
                    melsp_trg = np.array(feat_trg)
 
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

                #logging.info('write lat to h5')
                #logging.info(feat_file + ' ' + args.string_path+'_lat')
                #logging.info(feat_lat.shape)
                #write_hdf5(feat_file, args.string_path+'_lat', feat_lat)

                count += 1
                #if count >= 3:
                #    break


    with mp.Manager() as manager:
        logging.info("GRU-RNN decoding")
        processes = []
        cvlist = manager.list()
        lsd_cvlist_src = manager.list()
        lsdstd_cvlist_src = manager.list()
        lsd_cvlist_cyc = manager.list()
        lsdstd_cvlist_cyc = manager.list()
        lsd_cvlist = manager.list()
        lsdstd_cvlist = manager.list()
        lat_dist_rmse_list = manager.list()
        lat_dist_cosim_list = manager.list()
        gpu = 0
        for i, feat_list in enumerate(feat_lists):
            logging.info(i)
            p = mp.Process(target=decode_RNN, args=(feat_list, gpu, cvlist,
                lsd_cvlist_src, lsdstd_cvlist_src,
                lsd_cvlist_cyc, lsdstd_cvlist_cyc,
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
        logging.info("=== summary cyc. acc. ===")
        logging.info("lsd_cyc_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist_cyc)),\
        np.std(np.array(lsd_cvlist_cyc)),np.mean(np.array(lsdstd_cvlist_cyc)),np.std(np.array(lsdstd_cvlist_cyc))))
        logging.info("=== summary cv. acc. ===")
        cvgv_mean = np.mean(np.array(cvlist), axis=0)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg))))))
        if len(lsd_cvlist) > 0:
            logging.info("lsd_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist)),\
            np.std(np.array(lsd_cvlist)),np.mean(np.array(lsdstd_cvlist)),np.std(np.array(lsdstd_cvlist))))
            logging.info("lat_dist_rmse: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_rmse_list)),np.std(np.array(lat_dist_rmse_list))))
            logging.info("lat_dist_cosim: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_cosim_list)),np.std(np.array(lat_dist_cosim_list))))
  
 
if __name__ == "__main__":
    main()
