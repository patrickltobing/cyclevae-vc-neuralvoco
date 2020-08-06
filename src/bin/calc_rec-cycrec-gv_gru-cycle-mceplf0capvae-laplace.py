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

from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER, GRU_EXCIT_DECODER
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

    spk_f0_mean = read_hdf5(spk_stat, "/lf0_range_mean")
    spk_f0_std = read_hdf5(spk_stat, "/lf0_range_std")
    logging.info(spk_f0_mean)
    logging.info(spk_f0_std)

    model_epoch = os.path.basename(args.model).split('.')[0].split('-')[1]
    logging.info('epoch: '+model_epoch)

    model_name = os.path.basename(os.path.dirname(args.model)).split('_')[1]
    logging.info('mdl_name: '+model_name)

    logging.info(config)
    # define gpu decode function
    def gpu_decode(feat_list, gpu, cvlist=None, mcdpow_cvlist=None, mcdpowstd_cvlist=None, mcd_cvlist=None, \
                    mcdstd_cvlist=None, cvlist_dv=None, mcdpow_cvlist_dv=None, mcdpowstd_cvlist_dv=None, \
                    mcd_cvlist_dv=None, mcdstd_cvlist_dv=None, \
                    f0rmse_cvlist=None, f0corr_cvlist=None, caprmse_cvlist=None, \
                    f0rmse_cvlist_dv=None, f0corr_cvlist_dv=None, caprmse_cvlist_dv=None, \
                    cvlist_cyc=None, mcdpow_cvlist_cyc=None, mcdpowstd_cvlist_cyc=None, mcd_cvlist_cyc=None, \
                    mcdstd_cvlist_cyc=None, cvlist_cyc_dv=None, mcdpow_cvlist_cyc_dv=None, mcdpowstd_cvlist_cyc_dv=None, \
                    mcd_cvlist_cyc_dv=None, mcdstd_cvlist_cyc_dv=None, \
                    f0rmse_cvlist_cyc=None, f0corr_cvlist_cyc=None, caprmse_cvlist_cyc=None, \
                    f0rmse_cvlist_cyc_dv=None, f0corr_cvlist_cyc_dv=None, caprmse_cvlist_cyc_dv=None):
        with torch.cuda.device(gpu):
            # define model and load parameters
            with torch.no_grad():
                model_encoder_mcep = GRU_VAE_ENCODER(
                    in_dim=config.mcep_dim+config.excit_dim,
                    n_spk=n_spk,
                    lat_dim=config.lat_dim,
                    hidden_layers=config.hidden_layers_enc,
                    hidden_units=config.hidden_units_enc,
                    kernel_size=config.kernel_size_enc,
                    dilation_size=config.dilation_size_enc,
                    causal_conv=config.causal_conv_enc,
                    bi=config.bi_enc,
                    pad_first=True,
                    right_size=config.right_size,
                    ar=config.ar_enc)
                logging.info(model_encoder_mcep)
                model_decoder_mcep = GRU_SPEC_DECODER(
                    feat_dim=config.lat_dim,
                    out_dim=config.mcep_dim,
                    n_spk=n_spk,
                    hidden_layers=config.hidden_layers_dec,
                    hidden_units=config.hidden_units_dec,
                    kernel_size=config.kernel_size_dec,
                    dilation_size=config.dilation_size_dec,
                    causal_conv=config.causal_conv_dec,
                    bi=config.bi_dec,
                    spkidtr_dim=config.spkidtr_dim,
                    pad_first=True,
                    diff=config.diff,
                    ar=config.ar_dec)
                logging.info(model_decoder_mcep)
                model_encoder_excit = GRU_VAE_ENCODER(
                    in_dim=config.mcep_dim+config.excit_dim,
                    n_spk=n_spk,
                    lat_dim=config.lat_dim_e,
                    hidden_layers=config.hidden_layers_enc,
                    hidden_units=config.hidden_units_enc,
                    kernel_size=config.kernel_size_enc,
                    dilation_size=config.dilation_size_enc,
                    causal_conv=config.causal_conv_enc,
                    bi=config.bi_enc,
                    pad_first=True,
                    right_size=config.right_size,
                    ar=config.ar_enc)
                logging.info(model_encoder_excit)
                model_decoder_excit = GRU_EXCIT_DECODER(
                    feat_dim=config.lat_dim_e,
                    cap_dim=config.cap_dim,
                    n_spk=n_spk,
                    hidden_layers=config.hidden_layers_lf0,
                    hidden_units=config.hidden_units_lf0,
                    kernel_size=config.kernel_size_lf0,
                    dilation_size=config.dilation_size_lf0,
                    causal_conv=config.causal_conv_lf0,
                    bi=config.bi_lf0,
                    spkidtr_dim=config.spkidtr_dim,
                    pad_first=True,
                    ar=config.ar_f0)
                logging.info(model_decoder_excit)
                model_encoder_mcep.load_state_dict(torch.load(args.model)["model_encoder_mcep"])
                model_decoder_mcep.load_state_dict(torch.load(args.model)["model_decoder_mcep"])
                model_encoder_excit.load_state_dict(torch.load(args.model)["model_encoder_excit"])
                model_decoder_excit.load_state_dict(torch.load(args.model)["model_decoder_excit"])
                model_encoder_mcep.cuda()
                model_decoder_mcep.cuda()
                model_encoder_excit.cuda()
                model_decoder_excit.cuda()
                model_encoder_mcep.eval()
                model_decoder_mcep.eval()
                model_encoder_excit.eval()
                model_decoder_excit.eval()
                for param in model_encoder_mcep.parameters():
                    param.requires_grad = False
                for param in model_decoder_mcep.parameters():
                    param.requires_grad = False
                for param in model_encoder_excit.parameters():
                    param.requires_grad = False
                for param in model_decoder_excit.parameters():
                    param.requires_grad = False
                if config.ar_enc:
                    yz_in = torch.zeros((1, 1, n_spk+config.lat_dim*2)).cuda()
                    yz_in_e = torch.zeros((1, 1, n_spk+config.lat_dim_e*2)).cuda()
                if config.ar_dec or config.ar_f0:
                    mean_stats = torch.FloatTensor(read_hdf5(config.stats, "/mean_"+config.string_path.replace("/","")))
                    scale_stats = torch.FloatTensor(read_hdf5(config.stats, "/scale_"+config.string_path.replace("/","")))
                if config.ar_dec:
                    x_in = ((torch.zeros((1, 1, config.mcep_dim))-mean_stats[config.excit_dim:])/scale_stats[config.excit_dim:]).cuda()
                if config.ar_f0:
                    e_in = torch.cat((torch.zeros(1,1,1), (torch.zeros(1,1,1)-mean_stats[1:2])/scale_stats[1:2], \
                                    torch.zeros(1,1,1), (torch.zeros(1,1,config.cap_dim)-mean_stats[3:config.excit_dim])/scale_stats[3:config.excit_dim]), 2).cuda()
            count = 0
            pad_left = (model_encoder_mcep.pad_left + model_decoder_mcep.pad_left)*2
            pad_right = (model_encoder_mcep.pad_right + model_decoder_mcep.pad_right)*2
            outpad_lefts = [None]*3
            outpad_rights = [None]*3
            outpad_lefts[0] = pad_left-model_encoder_mcep.pad_left
            outpad_rights[0] = pad_right-model_encoder_mcep.pad_right
            outpad_lefts[1] = outpad_lefts[0]-model_decoder_mcep.pad_left
            outpad_rights[1] = outpad_rights[0]-model_decoder_mcep.pad_right
            outpad_lefts[2] = outpad_lefts[1]-model_encoder_mcep.pad_left
            outpad_rights[2] = outpad_rights[1]-model_encoder_mcep.pad_right
            for feat_file in feat_list:
                # convert mcep
                logging.info("recmcep " + feat_file)

                feat_org = read_hdf5(feat_file, "/feat_mceplf0cap")
                logging.info(feat_org.shape)

                with torch.no_grad():
                    feat = F.pad(torch.FloatTensor(feat_org).cuda().unsqueeze(0).transpose(1,2), (pad_left,pad_right), "replicate").transpose(1,2)

                    if config.ar_enc:
                        spk_logits, _, lat_src, _, _ = model_encoder_mcep(feat, yz_in=yz_in, sampling=False)
                        spk_logits_e, _, lat_src_e, _, _ = model_encoder_excit(feat, yz_in=yz_in, sampling=False)
                    else:
                        spk_logits, _, lat_src, _ = model_encoder_mcep(feat, sampling=False)
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

                    src_code = (torch.ones((1, lat_src.shape[1]))*spk_idx).cuda().long()
                    if config.ar_dec:
                        cvmcep_src, _, _ = model_decoder_mcep(src_code, lat_src, x_in=x_in)
                    else:
                        cvmcep_src, _ = model_decoder_mcep(src_code, lat_src)
                    src_code = (torch.ones((1, lat_src_e.shape[1]))*spk_idx).cuda().long()
                    if config.ar_f0:
                        cvlf0_src, _, _ = model_decoder_excit(src_code, lat_src_e, e_in=e_in)
                    else:
                        cvlf0_src, _ = model_decoder_excit(src_code, lat_src_e)

                    cv_feat = torch.cat((cvlf0_src, cvmcep_src), 2)
                    if config.ar_enc:
                        spk_logits, _, lat_rec, _, _ = model_encoder_mcep(cv_feat, yz_in=yz_in, sampling=False)
                        spk_logits_e, _, lat_rec_e, _, _ = model_encoder_excit(cv_feat, yz_in=yz_in, sampling=False)
                    else:
                        spk_logits, _, lat_rec, _ = model_encoder_mcep(cv_feat, sampling=False)
                        spk_logits_e, _, lat_rec_e, _ = model_encoder_excit(cv_feat, sampling=False)
                    logging.info('rec spkpost')
                    if outpad_rights[0] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[0]:-outpad_rights[0]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[0]:], dim=-1), 1))
                    logging.info('rec spkpost_e')
                    if outpad_rights[0] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[0]:-outpad_rights[0]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits_e[:,outpad_lefts[0]:], dim=-1), 1))

                    src_code = (torch.ones((1, lat_rec.shape[1]))*spk_idx).cuda().long()
                    if config.ar_dec:
                        cvmcep_cyc, _, _ = model_decoder_mcep(src_code, lat_rec, x_in=x_in)
                    else:
                        cvmcep_cyc, _ = model_decoder_mcep(src_code, lat_rec)
                    src_code = (torch.ones((1, lat_rec_e.shape[1]))*spk_idx).cuda().long()
                    if config.ar_f0:
                        cvlf0_cyc, _, _ = model_decoder_excit(src_code, lat_rec_e, e_in=e_in)
                    else:
                        cvlf0_cyc, _ = model_decoder_excit(src_code, lat_rec_e)

                    if outpad_rights[1] > 0:
                        cvmcep_src = cvmcep_src[:,outpad_lefts[1]:-outpad_rights[1]]
                        cvlf0_src = cvlf0_src[:,outpad_lefts[1]:-outpad_rights[1]]
                    else:
                        cvmcep_src = cvmcep_src[:,outpad_lefts[1]:]
                        cvlf0_src = cvlf0_src[:,outpad_lefts[1]:]

                    feat_rec = torch.cat((torch.round(cvlf0_src[:,:,:1]), cvlf0_src[:,:,1:2], \
                                            torch.round(cvlf0_src[:,:,2:3]), cvlf0_src[:,:,3:], cvmcep_src), \
                                                2)[0].cpu().data.numpy()
                    feat_cyc = torch.cat((torch.round(cvlf0_cyc[:,:,:1]), cvlf0_cyc[:,:,1:2], \
                                            torch.round(cvlf0_cyc[:,:,2:3]), cvlf0_cyc[:,:,3:], cvmcep_cyc), \
                                                2)[0].cpu().data.numpy()

                    cvmcep_src = np.array(cvmcep_src[0].cpu().data.numpy(), dtype=np.float64)
                    cvlf0_src = np.array(cvlf0_src[0].cpu().data.numpy(), dtype=np.float64)

                    cvmcep_cyc = np.array(cvmcep_cyc[0].cpu().data.numpy(), dtype=np.float64)
                    cvlf0_cyc = np.array(cvlf0_cyc[0].cpu().data.numpy(), dtype=np.float64)

                logging.info(cvlf0_src.shape)
                logging.info(cvmcep_src.shape)

                logging.info(cvlf0_cyc.shape)
                logging.info(cvmcep_cyc.shape)

                mcep = np.array(feat_org[:,-model_decoder_mcep.out_dim:])
                f0 = np.array(np.rint(feat_org[:,0])*np.exp(feat_org[:,1]))
                codeap = np.array(np.rint(feat_org[:,2:3])*(-np.exp(feat_org[:,3:feat_org.shape[-1]-model_decoder_mcep.out_dim])))
 
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
                    mcdpow_cvlist.append(mcdpow_mean)
                    mcdpow_cvlist.append(mcdpow_mean)
                    mcdpowstd_cvlist.append(mcdpow_std)
                    mcd_cvlist.append(mcd_mean)
                    mcdstd_cvlist.append(mcd_std)
                    cvlist.append(np.var(cvmcep_src[:,1:], axis=0))
                    logging.info(len(cvlist))
                    f0rmse_cvlist_cyc.append(f0_rmse_cyc)
                    f0corr_cvlist_cyc.append(f0_corr_cyc)
                    caprmse_cvlist_cyc.append(codeap_rmse_cyc)
                    mcdpow_cvlist_cyc.append(mcdpow_mean_cyc)
                    mcdpow_cvlist_cyc.append(mcdpow_mean_cyc)
                    mcdpowstd_cvlist_cyc.append(mcdpow_std_cyc)
                    mcd_cvlist_cyc.append(mcd_mean_cyc)
                    mcdstd_cvlist_cyc.append(mcd_std_cyc)
                    cvlist_cyc.append(np.var(cvmcep_cyc[:,1:], axis=0))
                elif 'dv' in dataset:
                    logging.info('dev')
                    f0rmse_cvlist_dv.append(f0_rmse)
                    f0corr_cvlist_dv.append(f0_corr)
                    caprmse_cvlist_dv.append(codeap_rmse)
                    mcdpow_cvlist_dv.append(mcdpow_mean)
                    mcdpowstd_cvlist_dv.append(mcdpow_std)
                    mcd_cvlist_dv.append(mcd_mean)
                    mcdstd_cvlist_dv.append(mcd_std)
                    cvlist_dv.append(np.var(cvmcep_src[:,1:], axis=0))
                    logging.info(len(cvlist_dv))
                    f0rmse_cvlist_cyc_dv.append(f0_rmse_cyc)
                    f0corr_cvlist_cyc_dv.append(f0_corr_cyc)
                    caprmse_cvlist_cyc_dv.append(codeap_rmse_cyc)
                    mcdpow_cvlist_cyc_dv.append(mcdpow_mean_cyc)
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
        f0rmse_cvlist = manager.list()
        f0corr_cvlist = manager.list()
        caprmse_cvlist = manager.list()
        cvlist_dv = manager.list()
        mcd_cvlist_dv = manager.list()
        mcdstd_cvlist_dv = manager.list()
        mcdpow_cvlist_dv = manager.list()
        mcdpowstd_cvlist_dv = manager.list()
        f0rmse_cvlist_dv = manager.list()
        f0corr_cvlist_dv = manager.list()
        caprmse_cvlist_dv = manager.list()
        cvlist_cyc = manager.list()
        mcd_cvlist_cyc = manager.list()
        mcdstd_cvlist_cyc = manager.list()
        mcdpow_cvlist_cyc = manager.list()
        mcdpowstd_cvlist_cyc = manager.list()
        f0rmse_cvlist_cyc = manager.list()
        f0corr_cvlist_cyc = manager.list()
        caprmse_cvlist_cyc = manager.list()
        cvlist_cyc_dv = manager.list()
        mcd_cvlist_cyc_dv = manager.list()
        mcdstd_cvlist_cyc_dv = manager.list()
        mcdpow_cvlist_cyc_dv = manager.list()
        mcdpowstd_cvlist_cyc_dv = manager.list()
        f0rmse_cvlist_cyc_dv = manager.list()
        f0corr_cvlist_cyc_dv = manager.list()
        caprmse_cvlist_cyc_dv = manager.list()
        for i, feat_list in enumerate(feat_lists):
            logging.info(i)
            p = mp.Process(target=gpu_decode, args=(feat_list, gpu, cvlist, mcdpow_cvlist, mcdpowstd_cvlist, \
                                                    mcd_cvlist, mcdstd_cvlist, cvlist_dv, mcdpow_cvlist_dv, \
                                                    mcdpowstd_cvlist_dv, mcd_cvlist_dv, mcdstd_cvlist_dv,\
                                                    f0rmse_cvlist, f0corr_cvlist, caprmse_cvlist,\
                                                    f0rmse_cvlist_dv, f0corr_cvlist_dv, caprmse_cvlist_dv,\
                                                    cvlist_cyc, mcdpow_cvlist_cyc, mcdpowstd_cvlist_cyc, \
                                                    mcd_cvlist_cyc, mcdstd_cvlist_cyc, cvlist_cyc_dv, mcdpow_cvlist_cyc_dv, \
                                                    mcdpowstd_cvlist_cyc_dv, mcd_cvlist_cyc_dv, mcdstd_cvlist_cyc_dv,\
                                                    f0rmse_cvlist_cyc, f0corr_cvlist_cyc, caprmse_cvlist_cyc,\
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
            logging.info("f0rmse_rec: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist)),np.std(np.array(f0rmse_cvlist))))
            logging.info("f0corr_rec: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist)),np.std(np.array(f0corr_cvlist))))
            caprmse_cvlist = np.array(caprmse_cvlist)
            for i in range(caprmse_cvlist.shape[-1]):
                logging.info("caprmse-%d_rec: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist[:,i]),np.std(caprmse_cvlist[:,i])))
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
            logging.info("f0rmse_cyc: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_cyc)),np.std(np.array(f0rmse_cvlist_cyc))))
            logging.info("f0corr_cyc: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_cyc)),np.std(np.array(f0corr_cvlist_cyc))))
            caprmse_cvlist_cyc = np.array(caprmse_cvlist_cyc)
            for i in range(caprmse_cvlist_cyc.shape[-1]):
                logging.info("caprmse-%d_cyc: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist_cyc[:,i]),np.std(caprmse_cvlist_cyc[:,i])))

            cvgv_mean = np.mean(np.array(np.r_[cvlist,cvlist_cyc]), axis=0)
            cvgv_var = np.var(np.array(np.r_[cvlist,cvlist_cyc]), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))), \
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))

            string_path = model_name+"-"+str(config.detach)+"-"+str(config.n_half_cyc)+"-"+str(config.lat_dim)\
                            +"-"+str(config.spkidtr_dim)+"-"+str(config.ar_enc)+"-"+str(config.ar_dec)+"-"+str(config.ar_f0)+"-"+str(config.diff)+"-"+model_epoch
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
            logging.info("f0rmse_rec_dv: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_dv)),np.std(np.array(f0rmse_cvlist_dv))))
            logging.info("f0corr_rec_dv: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_dv)),np.std(np.array(f0corr_cvlist_dv))))
            caprmse_cvlist_dv = np.array(caprmse_cvlist_dv)
            for i in range(caprmse_cvlist.shape[-1]):
                logging.info("caprmse-%d_rec_dv: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist_dv[:,i]),np.std(caprmse_cvlist_dv[:,i])))
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
            logging.info("f0rmse_cyc_dv: %.6f Hz (+- %.6f)" % (np.mean(np.array(f0rmse_cvlist_cyc_dv)),np.std(np.array(f0rmse_cvlist_cyc_dv))))
            logging.info("f0corr_cyc_dv: %.6f (+- %.6f)" % (np.mean(np.array(f0corr_cvlist_cyc_dv)),np.std(np.array(f0corr_cvlist_cyc_dv))))
            caprmse_cvlist_cyc_dv = np.array(caprmse_cvlist_cyc_dv)
            for i in range(caprmse_cvlist_cyc_dv.shape[-1]):
                logging.info("caprmse-%d_cyc_dv: %.6f dB (+- %.6f)" % (i+1, np.mean(caprmse_cvlist_cyc_dv[:,i]),np.std(caprmse_cvlist_cyc_dv[:,i])))


if __name__ == "__main__":
    main()
