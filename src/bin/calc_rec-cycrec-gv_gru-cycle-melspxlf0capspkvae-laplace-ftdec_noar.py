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

import numpy as np
import torch
import torch.multiprocessing as mp

from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER, GRU_SPK
from vcneuvoco import GRU_EXCIT_DECODER, SPKID_TRANSFORM_LAYER
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
    gv_mean = read_hdf5(spk_stat, "/gv_melsp_mean")

    model_epoch = os.path.basename(args.model).split('.')[0].split('-')[1]
    logging.info('epoch: '+model_epoch)

    model_name = os.path.basename(os.path.dirname(args.model)).split('_')[1]
    logging.info('mdl_name: '+model_name)

    logging.info(config)
    # define gpu decode function
    def gpu_decode(feat_list, gpu, cvlist=None, lsd_cvlist=None,
                    lsdstd_cvlist=None, cvlist_dv=None,
                    lsd_cvlist_dv=None, lsdstd_cvlist_dv=None,
                    f0rmse_cvlist=None, f0corr_cvlist=None, caprmse_cvlist=None,
                    f0rmse_cvlist_dv=None, f0corr_cvlist_dv=None, caprmse_cvlist_dv=None,
                    cvlist_cyc=None, lsd_cvlist_cyc=None,
                    lsdstd_cvlist_cyc=None, cvlist_cyc_dv=None,
                    lsd_cvlist_cyc_dv=None, lsdstd_cvlist_cyc_dv=None,
                    f0rmse_cvlist_cyc=None, f0corr_cvlist_cyc=None, caprmse_cvlist_cyc=None,
                    f0rmse_cvlist_cyc_dv=None, f0corr_cvlist_cyc_dv=None, caprmse_cvlist_cyc_dv=None):
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
                    right_size=config.right_size_dec,
                    red_dim=config.mel_dim)
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
                    right_size=config.right_size_dec,
                    red_dim=config.mel_dim)
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
                    right_size=config.right_size_lf0,
                    red_dim=config.mel_dim)
                logging.info(model_decoder_excit)
                if (config.spkidtr_dim > 0):
                    model_spkidtr = SPKID_TRANSFORM_LAYER(
                        n_spk=n_spk,
                        spkidtr_dim=config.spkidtr_dim)
                    logging.info(model_spkidtr)
                model_spk = GRU_SPK(
                    n_spk=n_spk,
                    feat_dim=config.lat_dim+config.lat_dim_e,
                    kernel_size=config.kernel_size_spk,
                    dilation_size=config.dilation_size_spk,
                    causal_conv=config.causal_conv_spk,
                    pad_first=True,
                    right_size=config.right_size_spk,
                    hidden_units=32,
                    red_dim=config.mel_dim)
                logging.info(model_spk)
                model_encoder_melsp.load_state_dict(torch.load(args.model)["model_encoder_melsp"])
                model_decoder_melsp.load_state_dict(torch.load(args.model)["model_decoder_melsp"])
                model_decoder_melsp_fix.load_state_dict(torch.load(args.model)["model_decoder_melsp_fix"])
                model_encoder_excit.load_state_dict(torch.load(args.model)["model_encoder_excit"])
                model_decoder_excit.load_state_dict(torch.load(args.model)["model_decoder_excit"])
                if (config.spkidtr_dim > 0):
                    model_spkidtr.load_state_dict(torch.load(args.model)["model_spkidtr"])
                model_spk.load_state_dict(torch.load(args.model)["model_spk"])
                model_encoder_melsp.cuda()
                model_decoder_melsp.cuda()
                model_decoder_melsp_fix.cuda()
                model_encoder_excit.cuda()
                model_decoder_excit.cuda()
                if (config.spkidtr_dim > 0):
                    model_spkidtr.cuda()
                model_spk.cuda()
                model_encoder_melsp.eval()
                model_decoder_melsp.eval()
                model_decoder_melsp_fix.eval()
                model_encoder_excit.eval()
                model_decoder_excit.eval()
                if (config.spkidtr_dim > 0):
                    model_spkidtr.eval()
                model_spk.eval()
                for param in model_encoder_melsp.parameters():
                    param.requires_grad = False
                for param in model_decoder_melsp.parameters():
                    param.requires_grad = False
                for param in model_decoder_melsp_fix.parameters():
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
            count = 0
            pad_left = (model_encoder_melsp.pad_left + model_spk.pad_left + model_decoder_excit.pad_left + model_decoder_melsp.pad_left)*2
            pad_right = (model_encoder_melsp.pad_right + model_spk.pad_right + model_decoder_excit.pad_right + model_decoder_melsp.pad_right)*2
            outpad_lefts = [None]*7
            outpad_rights = [None]*7
            outpad_lefts[0] = pad_left-model_encoder_melsp.pad_left
            outpad_rights[0] = pad_right-model_encoder_melsp.pad_right
            outpad_lefts[1] = outpad_lefts[0]-model_spk.pad_left
            outpad_rights[1] = outpad_rights[0]-model_spk.pad_right
            outpad_lefts[2] = outpad_lefts[1]-model_decoder_excit.pad_left
            outpad_rights[2] = outpad_rights[1]-model_decoder_excit.pad_right
            outpad_lefts[3] = outpad_lefts[2]-model_decoder_melsp.pad_left
            outpad_rights[3] = outpad_rights[2]-model_decoder_melsp.pad_right
            outpad_lefts[4] = outpad_lefts[3]-model_encoder_melsp.pad_left
            outpad_rights[4] = outpad_rights[3]-model_encoder_melsp.pad_right
            outpad_lefts[5] = outpad_lefts[4]-model_spk.pad_left
            outpad_rights[5] = outpad_rights[4]-model_spk.pad_right
            outpad_lefts[6] = outpad_lefts[5]-model_decoder_excit.pad_left
            outpad_rights[6] = outpad_rights[5]-model_decoder_excit.pad_right
            for feat_file in feat_list:
                # reconst. melsp
                logging.info("recmelsp " + feat_file)

                feat_org = read_hdf5(feat_file, "/log_1pmelmagsp")
                logging.info(feat_org.shape)

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

                    if config.spkidtr_dim > 0:
                        src_code = model_spkidtr((torch.ones((1, lat_src_e.shape[1]))*spk_idx).cuda().long())
                    else:
                        src_code = (torch.ones((1, lat_src_e.shape[1]))*spk_idx).cuda().long()
                    lat_cat = torch.cat((lat_src_e, lat_src), 2)
                    trj_src_code, _ = model_spk(src_code, z=lat_cat)

                    if model_spk.pad_right > 0:
                        lat_cat = lat_cat[:,model_spk.pad_left:-model_spk.pad_right]
                        lat_src_e = lat_src_e[:,model_spk.pad_left:-model_spk.pad_right]
                        src_code = src_code[:,model_spk.pad_left:-model_spk.pad_right]
                    else:
                        lat_cat = lat_cat[:,model_spk.pad_left:]
                        lat_src_e = lat_src_e[:,model_spk.pad_left:]
                        src_code = src_code[:,model_spk.pad_left:]
                    cvlf0_src, _ = model_decoder_excit(lat_src_e, y=src_code, aux=trj_src_code)

                    if model_decoder_excit.pad_right > 0:
                        lat_cat = lat_cat[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                        src_code = src_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                        trj_src_code = trj_src_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                    else:
                        lat_cat = lat_cat[:,model_decoder_excit.pad_left:]
                        src_code = src_code[:,model_decoder_excit.pad_left:]
                        trj_src_code = trj_src_code[:,model_decoder_excit.pad_left:]
                    cvmelsp_src, _ = model_decoder_melsp(lat_cat, y=src_code, aux=trj_src_code, e=cvlf0_src[:,:,:config.excit_dim])
                    cvmelsp_src_fix, _ = model_decoder_melsp_fix(lat_cat, y=src_code, aux=trj_src_code, e=cvlf0_src[:,:,:config.excit_dim])
                    trj_lat_cat = lat_cat

                    spk_logits, _, lat_rec, _ = model_encoder_melsp(cvmelsp_src_fix, sampling=False)
                    spk_logits_e, _, lat_rec_e, _ = model_encoder_excit(cvmelsp_src_fix, sampling=False)
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

                    if config.spkidtr_dim > 0:
                        src_code = model_spkidtr((torch.ones((1, lat_rec_e.shape[1]))*spk_idx).cuda().long())
                    else:
                        src_code = (torch.ones((1, lat_rec_e.shape[1]))*spk_idx).cuda().long()
                    lat_cat = torch.cat((lat_rec_e, lat_rec), 2)
                    trj_src_code, _ = model_spk(src_code, z=lat_cat)

                    if model_spk.pad_right > 0:
                        lat_cat = lat_cat[:,model_spk.pad_left:-model_spk.pad_right]
                        lat_rec_e = lat_rec_e[:,model_spk.pad_left:-model_spk.pad_right]
                        src_code = src_code[:,model_spk.pad_left:-model_spk.pad_right]
                    else:
                        lat_cat = lat_cat[:,model_spk.pad_left:]
                        lat_rec_e = lat_rec_e[:,model_spk.pad_left:]
                        src_code = src_code[:,model_spk.pad_left:]
                    cvlf0_cyc, _ = model_decoder_excit(lat_rec_e, y=src_code, aux=trj_src_code)

                    if model_decoder_excit.pad_right > 0:
                        lat_cat = lat_cat[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                        src_code = src_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                        trj_src_code = trj_src_code[:,model_decoder_excit.pad_left:-model_decoder_excit.pad_right]
                    else:
                        lat_cat = lat_cat[:,model_decoder_excit.pad_left:]
                        src_code = src_code[:,model_decoder_excit.pad_left:]
                        trj_src_code = trj_src_code[:,model_decoder_excit.pad_left:]
                    cvmelsp_cyc, _ = model_decoder_melsp(lat_cat, y=src_code, aux=trj_src_code, e=cvlf0_cyc[:,:,:config.excit_dim])
                    #trj_lat_cat_cyc = lat_cat

                    #if outpad_rights[0] > 0:
                    #    lat_src = lat_src[:,outpad_lefts[0]:-outpad_rights[0]]
                    #else:
                    #    lat_src = lat_src[:,outpad_lefts[0]:]
                    #if outpad_rights[1] > 0:
                    #    lat_src_e = lat_src_e[:,outpad_lefts[1]:-outpad_rights[1]]
                    #else:
                    #    lat_src_e = lat_src_e[:,outpad_lefts[1]:]
                    if outpad_rights[2] > 0:
                        cvlf0_src = cvlf0_src[:,outpad_lefts[2]:-outpad_rights[2]]
                        trj_lat_cat = trj_lat_cat[:,outpad_lefts[2]:-outpad_rights[2]]
                    else:
                        cvlf0_src = cvlf0_src[:,outpad_lefts[2]:]
                        trj_lat_cat = trj_lat_cat[:,outpad_lefts[2]:-outpad_rights[2]]
                    if outpad_rights[3] > 0:
                        cvmelsp_src = cvmelsp_src[:,outpad_lefts[3]:-outpad_rights[3]]
                    else:
                        cvmelsp_src = cvmelsp_src[:,outpad_lefts[3]:]
                    #if outpad_rights[4] > 0:
                    #    lat_rec = lat_rec[:,outpad_lefts[4]:-outpad_rights[4]]
                    #else:
                    #    lat_rec = lat_rec[:,outpad_lefts[4]:]
                    #if outpad_rights[5] > 0:
                    #    lat_rec_e = lat_rec_e[:,outpad_lefts[5]:-outpad_rights[5]]
                    #else:
                    #    lat_rec_e = lat_rec_e[:,outpad_lefts[5]:]
                    if outpad_rights[6] > 0:
                        cvlf0_cyc = cvlf0_cyc[:,outpad_lefts[6]:-outpad_rights[6]]
                    #    trj_lat_cat_cyc = trj_lat_cat_cyc[:,outpad_lefts[6]:-outpad_rights[6]]
                    else:
                        cvlf0_cyc = cvlf0_cyc[:,outpad_lefts[6]:]
                    #    trj_lat_cat_cyc = trj_lat_cat_cyc[:,outpad_lefts[6]:]

                    feat_rec = cvmelsp_src[0].cpu().data.numpy()
                    feat_cyc = cvmelsp_cyc[0].cpu().data.numpy()
                    feat_lat = trj_lat_cat[0].cpu().data.numpy()
                    #feat_lat_cyc = trj_lat_cat_cyc[0].cpu().data.numpy()

                    #lat_src = lat_src[0].cpu().data.numpy()
                    #lat_src_e = lat_src_e[0].cpu().data.numpy()

                    cvmelsp_src = np.array(cvmelsp_src[0].cpu().data.numpy(), dtype=np.float64)
                    cvlf0_src = np.array(cvlf0_src[0].cpu().data.numpy(), dtype=np.float64)

                    #lat_rec = lat_rec[0].cpu().data.numpy()
                    #lat_rec_e = lat_rec_e[0].cpu().data.numpy()

                    cvmelsp_cyc = np.array(cvmelsp_cyc[0].cpu().data.numpy(), dtype=np.float64)
                    cvlf0_cyc = np.array(cvlf0_cyc[0].cpu().data.numpy(), dtype=np.float64)

                logging.info(cvlf0_src.shape)
                logging.info(cvmelsp_src.shape)

                logging.info(cvlf0_cyc.shape)
                logging.info(cvmelsp_cyc.shape)

                melsp = np.array(feat_org)

                feat_world = read_hdf5(feat_file, "/feat_mceplf0cap")
                f0 = np.array(np.rint(feat_world[:,0])*np.exp(feat_world[:,1]))
                codeap = np.array(np.rint(feat_world[:,2:3])*(-np.exp(feat_world[:,3:config.full_excit_dim])))
 
                cvf0_src = np.array(np.rint(cvlf0_src[:,0])*np.exp(cvlf0_src[:,1]))
                cvcodeap_src = np.array(np.rint(cvlf0_src[:,2:3])*(-np.exp(cvlf0_src[:,3:])))
                f0_rmse = np.sqrt(np.mean((cvf0_src-f0)**2))
                logging.info('F0_rmse_rec: %lf Hz' % (f0_rmse))
                cvf0_src_mean = np.mean(cvf0_src)
                f0_mean = np.mean(f0)
                f0_corr = np.sum((cvf0_src-cvf0_src_mean)*(f0-f0_mean))/\
                            (np.sqrt(np.sum((cvf0_src-cvf0_src_mean)**2))*np.sqrt(np.sum((f0-f0_mean)**2)))
                logging.info('F0_corr_rec: %lf' % (f0_corr))

                codeap_rmse = np.sqrt(np.mean((cvcodeap_src-codeap)**2, axis=0))
                for i in range(codeap_rmse.shape[-1]):
                    logging.info('codeap-%d_rmse_rec: %lf dB' % (i+1, codeap_rmse[i]))

                cvf0_cyc = np.array(np.rint(cvlf0_cyc[:,0])*np.exp(cvlf0_cyc[:,1]))
                cvcodeap_cyc = np.array(np.rint(cvlf0_cyc[:,2:3])*(-np.exp(cvlf0_cyc[:,3:])))
                f0_rmse_cyc = np.sqrt(np.mean((cvf0_cyc-f0)**2))
                logging.info('F0_rmse_cyc: %lf Hz' % (f0_rmse_cyc))
                cvf0_cyc_mean = np.mean(cvf0_cyc)
                f0_mean = np.mean(f0)
                f0_corr_cyc = np.sum((cvf0_cyc-cvf0_cyc_mean)*(f0-f0_mean))/\
                            (np.sqrt(np.sum((cvf0_cyc-cvf0_cyc_mean)**2))*np.sqrt(np.sum((f0-f0_mean)**2)))
                logging.info('F0_corr_cyc: %lf' % (f0_corr_cyc))

                codeap_rmse_cyc = np.sqrt(np.mean((cvcodeap_cyc-codeap)**2, axis=0))
                for i in range(codeap_rmse_cyc.shape[-1]):
                    logging.info('codeap-%d_rmse_cyc: %lf dB' % (i+1, codeap_rmse_cyc[i]))

                spcidx = np.array(read_hdf5(feat_file, "/spcidx_range")[0])

                melsp_rest = (np.exp(melsp)-1)/10000
                melsp_src_rest = (np.exp(cvmelsp_src)-1)/10000
                melsp_cyc_rest = (np.exp(cvmelsp_cyc)-1)/10000

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_src_rest[spcidx], a_min=1e-16, a_max=None))\
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_mean = np.mean(lsd_arr)
                lsd_std = np.std(lsd_arr)
                logging.info("lsd_rec: %.6f dB +- %.6f" % (lsd_mean, lsd_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_cyc_rest[spcidx], a_min=1e-16, a_max=None))\
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_mean_cyc = np.mean(lsd_arr)
                lsd_std_cyc = np.std(lsd_arr)
                logging.info("lsd_cyc: %.6f dB +- %.6f" % (lsd_mean_cyc, lsd_std_cyc))
            
                logging.info('org f0')
                logging.info(f0[10:15])
                logging.info('rec f0')
                logging.info(cvf0_src[10:15])
                logging.info('cyc f0')
                logging.info(cvf0_cyc[10:15])
                logging.info('org cap')
                logging.info(codeap[10:15])
                logging.info('rec cap')
                logging.info(cvcodeap_src[10:15])
                logging.info('cyc cap')
                logging.info(cvcodeap_cyc[10:15])

                dataset = feat_file.split('/')[1].split('_')[0]
                if 'tr' in dataset:
                    logging.info('trn')
                    f0rmse_cvlist.append(f0_rmse)
                    f0corr_cvlist.append(f0_corr)
                    caprmse_cvlist.append(codeap_rmse)
                    lsd_cvlist.append(lsd_mean)
                    lsdstd_cvlist.append(lsd_std)
                    cvlist.append(np.var(melsp_src_rest, axis=0))
                    logging.info(len(cvlist))
                    f0rmse_cvlist_cyc.append(f0_rmse_cyc)
                    f0corr_cvlist_cyc.append(f0_corr_cyc)
                    caprmse_cvlist_cyc.append(codeap_rmse_cyc)
                    lsd_cvlist_cyc.append(lsd_mean_cyc)
                    lsdstd_cvlist_cyc.append(lsd_std_cyc)
                    cvlist_cyc.append(np.var(melsp_cyc_rest, axis=0))
                elif 'dv' in dataset:
                    logging.info('dev')
                    f0rmse_cvlist_dv.append(f0_rmse)
                    f0corr_cvlist_dv.append(f0_corr)
                    caprmse_cvlist_dv.append(codeap_rmse)
                    lsd_cvlist_dv.append(lsd_mean)
                    lsdstd_cvlist_dv.append(lsd_std)
                    cvlist_dv.append(np.var(melsp_src_rest, axis=0))
                    logging.info(len(cvlist_dv))
                    f0rmse_cvlist_cyc_dv.append(f0_rmse_cyc)
                    f0corr_cvlist_cyc_dv.append(f0_corr_cyc)
                    caprmse_cvlist_cyc_dv.append(codeap_rmse_cyc)
                    lsd_cvlist_cyc_dv.append(lsd_mean_cyc)
                    lsdstd_cvlist_cyc_dv.append(lsd_std_cyc)
                    cvlist_cyc_dv.append(np.var(melsp_cyc_rest, axis=0))

                logging.info('write lat to h5')
                outh5dir = os.path.join(os.path.dirname(os.path.dirname(feat_file)), args.spk)
                feat_file = os.path.join(outh5dir, os.path.basename(feat_file))
                logging.info(feat_file + ' ' + args.string_path+'_lat')
                logging.info(feat_lat.shape)
                write_hdf5(feat_file, args.string_path+'_lat', feat_lat)

                logging.info('write rec to h5')
                outh5dir = os.path.join(os.path.dirname(os.path.dirname(feat_file)), args.spk+"-"+args.spk)
                if not os.path.exists(outh5dir):
                    os.makedirs(outh5dir)
                feat_file = os.path.join(outh5dir, os.path.basename(feat_file))
                logging.info(feat_file + ' ' + args.string_path)
                logging.info(feat_rec.shape)
                write_hdf5(feat_file, args.string_path, feat_rec)

                logging.info('write lat to h5 rec')
                logging.info(feat_file + ' ' + args.string_path+'_lat')
                logging.info(feat_lat.shape)
                write_hdf5(feat_file, args.string_path+'_lat', feat_lat)

                logging.info('write cyc to h5')
                outh5dir = os.path.join(os.path.dirname(os.path.dirname(feat_file)), args.spk+"-"+args.spk+"-"+args.spk)
                if not os.path.exists(outh5dir):
                    os.makedirs(outh5dir)
                feat_file = os.path.join(outh5dir, os.path.basename(feat_file))
                logging.info(feat_file + ' ' + args.string_path)
                logging.info(feat_cyc.shape)
                write_hdf5(feat_file, args.string_path, feat_cyc)

                logging.info('write lat to h5 cyc')
                logging.info(feat_file + ' ' + args.string_path+'_lat')
                logging.info(feat_lat.shape)
                write_hdf5(feat_file, args.string_path+'_lat', feat_lat)

                count += 1
                #if count >= 5:
                #    break


    # parallel decode training
    with mp.Manager() as manager:
        gpu = 0
        processes = []
        cvlist = manager.list()
        lsd_cvlist = manager.list()
        lsdstd_cvlist = manager.list()
        f0rmse_cvlist = manager.list()
        f0corr_cvlist = manager.list()
        caprmse_cvlist = manager.list()
        cvlist_dv = manager.list()
        lsd_cvlist_dv = manager.list()
        lsdstd_cvlist_dv = manager.list()
        f0rmse_cvlist_dv = manager.list()
        f0corr_cvlist_dv = manager.list()
        caprmse_cvlist_dv = manager.list()
        cvlist_cyc = manager.list()
        lsd_cvlist_cyc = manager.list()
        lsdstd_cvlist_cyc = manager.list()
        f0rmse_cvlist_cyc = manager.list()
        f0corr_cvlist_cyc = manager.list()
        caprmse_cvlist_cyc = manager.list()
        cvlist_cyc_dv = manager.list()
        lsd_cvlist_cyc_dv = manager.list()
        lsdstd_cvlist_cyc_dv = manager.list()
        f0rmse_cvlist_cyc_dv = manager.list()
        f0corr_cvlist_cyc_dv = manager.list()
        caprmse_cvlist_cyc_dv = manager.list()
        for i, feat_list in enumerate(feat_lists):
            logging.info(i)
            p = mp.Process(target=gpu_decode, args=(feat_list, gpu, cvlist, 
                                                    lsd_cvlist, lsdstd_cvlist, cvlist_dv, 
                                                    lsd_cvlist_dv, lsdstd_cvlist_dv,
                                                    f0rmse_cvlist, f0corr_cvlist, caprmse_cvlist,
                                                    f0rmse_cvlist_dv, f0corr_cvlist_dv, caprmse_cvlist_dv,
                                                    cvlist_cyc, lsd_cvlist_cyc, lsdstd_cvlist_cyc, cvlist_cyc_dv,
                                                    lsd_cvlist_cyc_dv, lsdstd_cvlist_cyc_dv,
                                                    f0rmse_cvlist_cyc, f0corr_cvlist_cyc, caprmse_cvlist_cyc,
                                                    f0rmse_cvlist_cyc_dv, f0corr_cvlist_cyc_dv, caprmse_cvlist_cyc_dv,))
            p.start()
            processes.append(p)
            gpu += 1
            if (i + 1) % args.n_gpus == 0:
                gpu = 0
        # wait for all process
        for p in processes:
            p.join()

        # calculate cv_gv statistics
        if len(lsd_cvlist) > 0:
            logging.info("lsd_rec: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist)), \
                        np.std(np.array(lsd_cvlist)),np.mean(np.array(lsdstd_cvlist)),\
                        np.std(np.array(lsdstd_cvlist))))
            cvgv_mean = np.mean(np.array(cvlist), axis=0)
            cvgv_var = np.var(np.array(cvlist), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("f0rmse_rec: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist)),np.std(np.array(f0rmse_cvlist))))
            logging.info("f0corr_rec: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist)),np.std(np.array(f0corr_cvlist))))
            caprmse_cvlist = np.array(caprmse_cvlist)
            for i in range(caprmse_cvlist.shape[-1]):
                logging.info("caprmse-%d_rec: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist[:,i]),np.std(caprmse_cvlist[:,i])))
            logging.info("lsd_cyc: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist_cyc)), \
                        np.std(np.array(lsd_cvlist_cyc)),np.mean(np.array(lsdstd_cvlist_cyc)),\
                        np.std(np.array(lsdstd_cvlist_cyc))))
            cvgv_mean = np.mean(np.array(cvlist_cyc), axis=0)
            cvgv_var = np.var(np.array(cvlist_cyc), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("f0rmse_cyc: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_cyc)),np.std(np.array(f0rmse_cvlist_cyc))))
            logging.info("f0corr_cyc: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_cyc)),np.std(np.array(f0corr_cvlist_cyc))))
            caprmse_cvlist_cyc = np.array(caprmse_cvlist_cyc)
            for i in range(caprmse_cvlist_cyc.shape[-1]):
                logging.info("caprmse-%d_cyc: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist_cyc[:,i]),np.std(caprmse_cvlist_cyc[:,i])))

            cvgv_mean = np.mean(np.array(np.r_[cvlist,cvlist_cyc]), axis=0)
            cvgv_var = np.var(np.array(np.r_[cvlist,cvlist_cyc]), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))

            string_path = model_name+"-"+str(config.n_half_cyc)+"-"+str(config.lat_dim)+"-"+str(config.lat_dim_e)\
                            +"-"+str(config.spkidtr_dim)+"-"+model_epoch
            logging.info(string_path)

            string_mean = "/recgv_mean_"+string_path
            string_var = "/recgv_var_"+string_path
            write_hdf5(spk_stat, string_mean, cvgv_mean)
            write_hdf5(spk_stat, string_var, cvgv_var)

        if len(lsd_cvlist_dv) > 0:
            logging.info("lsd_rec_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist_dv)), \
                        np.std(np.array(lsd_cvlist_dv)),np.mean(np.array(lsdstd_cvlist_dv)),\
                        np.std(np.array(lsdstd_cvlist_dv))))
            cvgv_mean = np.mean(np.array(cvlist_dv), axis=0)
            cvgv_var = np.var(np.array(cvlist_dv), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("f0rmse_rec_dv: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_dv)),np.std(np.array(f0rmse_cvlist_dv))))
            logging.info("f0corr_rec_dv: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_dv)),np.std(np.array(f0corr_cvlist_dv))))
            caprmse_cvlist_dv = np.array(caprmse_cvlist_dv)
            for i in range(caprmse_cvlist.shape[-1]):
                logging.info("caprmse-%d_rec_dv: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist_dv[:,i]),np.std(caprmse_cvlist_dv[:,i])))
            logging.info("lsd_cyc_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist_cyc_dv)), \
                        np.std(np.array(lsd_cvlist_cyc_dv)),np.mean(np.array(lsdstd_cvlist_cyc_dv)),\
                        np.std(np.array(lsdstd_cvlist_cyc_dv))))
            cvgv_mean = np.mean(np.array(cvlist_cyc_dv), axis=0)
            cvgv_var = np.var(np.array(cvlist_cyc_dv), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("f0rmse_cyc_dv: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_cyc_dv)),np.std(np.array(f0rmse_cvlist_cyc_dv))))
            logging.info("f0corr_cyc_dv: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_cyc_dv)),np.std(np.array(f0corr_cvlist_cyc_dv))))
            caprmse_cvlist_cyc_dv = np.array(caprmse_cvlist_cyc_dv)
            for i in range(caprmse_cvlist_cyc_dv.shape[-1]):
                logging.info("caprmse-%d_cyc_dv: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist_cyc_dv[:,i]),np.std(caprmse_cvlist_cyc_dv[:,i])))


if __name__ == "__main__":
    main()
