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

from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER
from feature_extract import convert_f0, convert_continuos_f0, low_pass_filter, mod_pow
#from feature_extract import convert_continuos_codeap
from dtw_c import dtw_c as dtw

import pysptk as ps
import pyworld as pw
#from pysptk.synthesis import MLSADF

#np.set_printoptions(threshold=np.inf)

#FS = 16000
#FS = 22050
FS = 24000
N_GPUS = 1
SHIFT_MS = 5.0
#SHIFT_MS = 10.0
#MCEP_ALPHA = 0.41000000000000003
#MCEP_ALPHA = 0.455
MCEP_ALPHA = 0.466
#FFTL = 1024
FFTL = 2048
IRLEN = 1024
VERBOSE = 1
GV_COEFF = 0.9
MAX_CODEAP = -8.6856974912498e-12


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
    parser.add_argument("--gv_coeff", default=GV_COEFF,
                        type=float, help="weighting coefficient for GV postfilter")
    parser.add_argument("--shiftms", default=SHIFT_MS,
                        type=float, help="frame shift")
    parser.add_argument("--mcep_alpha", default=MCEP_ALPHA,
                        type=float, help="mcep alpha coeff.")
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

    string_path = model_name+"-"+str(config.n_half_cyc)+"-"+str(config.lat_dim)+"-"+model_epoch
    cvgv_mean = read_hdf5(stats_list[trg_idx], "/recgv_mean_"+string_path)
    gv_mean_trg = read_hdf5(stats_list[trg_idx], "/gv_range_mean")[1:]

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]
    for i in range(args.n_gpus):
        logging.info('gpu: %d'+str(i+1)+' : '+str(len(feat_lists[i])))

    ### GRU-RNN decoding ###
    logging.info(config)
    def decode_RNN(feat_list, gpu, cvlist=None,
            mcd_cvlist_src=None, mcdstd_cvlist_src=None, mcdpow_cvlist_src=None, mcdpowstd_cvlist_src=None,\
            mcd_cvlist_cyc=None, mcdstd_cvlist_cyc=None, mcdpow_cvlist_cyc=None, mcdpowstd_cvlist_cyc=None,\
            mcd_cvlist=None, mcdstd_cvlist=None, mcdpow_cvlist=None, mcdpowstd_cvlist=None, \
            lat_dist_rmse_list=None, lat_dist_cosim_list=None):
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
                spk_src = os.path.basename(os.path.dirname(feat_file))
                src_idx = spk_list.index(spk_src)
                logging.info('%s --> %s' % (spk_src, args.spk_trg))

                file_trg = os.path.join(os.path.dirname(os.path.dirname(feat_file)), args.spk_trg, os.path.basename(feat_file))
                trg_exist = False
                if os.path.exists(file_trg):
                    logging.info('exist: %s' % (file_trg))
                    feat_trg = read_hdf5(file_trg, config.string_path)
                    mcep_trg = feat_trg[:,-config.mcep_dim:]
                    logging.info(mcep_trg.shape)
                    trg_exist = True

                feat_org = read_hdf5(feat_file, config.string_path)
                mcep = np.array(feat_org[:,-config.mcep_dim:])
                codeap = np.array(np.rint(feat_org[:,2:3])*(-np.exp(feat_org[:,3:config.excit_dim])))
                sp = np.array(ps.mc2sp(mcep, args.mcep_alpha, args.fftl))
                ap = pw.decode_aperiodicity(codeap, args.fs, args.fftl)
                feat_cvf0_lin = np.expand_dims(convert_f0(np.exp(feat_org[:,1]), src_f0_mean, src_f0_std, trg_f0_mean, trg_f0_std), axis=-1)
                feat_cv = np.c_[feat_org[:,:1], np.log(feat_cvf0_lin), feat_org[:,2:config.excit_dim]]

                logging.info("generate")
                with torch.no_grad():
                    feat = F.pad(torch.FloatTensor(feat_org).cuda().unsqueeze(0).transpose(1,2), (pad_left,pad_right), "replicate").transpose(1,2)
                    feat_excit = torch.FloatTensor(feat_org[:,:config.excit_dim]).cuda().unsqueeze(0)
                    feat_excit_cv = torch.FloatTensor(feat_cv).cuda().unsqueeze(0)

                    spk_logits, _, lat_src, _ = model_encoder(feat, sampling=False)
                    logging.info('input spkpost')
                    if outpad_rights[0] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[0]:-outpad_rights[0]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[0]:], dim=-1), 1))

                    if trg_exist:
                        spk_trg_logits, _, lat_trg, _ = model_encoder(F.pad(torch.FloatTensor(feat_trg).cuda().unsqueeze(0).transpose(1,2), \
                                                            (model_encoder.pad_left,model_encoder.pad_right), "replicate").transpose(1,2), sampling=False)
                        logging.info('target spkpost')
                        logging.info(torch.mean(F.softmax(spk_trg_logits, dim=-1), 1))

                    cvmcep_src, _ = model_decoder((torch.ones((1, lat_src.shape[1]))*src_idx).cuda().long(), lat_src)
                    spk_logits, _, lat_rec, _ = model_encoder(torch.cat((F.pad(feat_excit.transpose(1,2), \
                                        (outpad_lefts[1],outpad_rights[1]), "replicate").transpose(1,2), cvmcep_src), 2), 
                                                        sampling=False)
                    logging.info('rec spkpost')
                    if outpad_rights[2] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:-outpad_rights[2]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:], dim=-1), 1))

                    cvmcep, _ = model_decoder((torch.ones((1, lat_src.shape[1]))*trg_idx).cuda().long(), lat_src)
                    spk_logits, _, lat_cv, _ = model_encoder(torch.cat((F.pad(torch.FloatTensor(feat_cv).cuda().unsqueeze(0).transpose(1,2), \
                                        (outpad_lefts[1],outpad_rights[1]), "replicate").transpose(1,2), cvmcep_src), 2), 
                                                        sampling=False)
                    logging.info('cv spkpost')
                    if outpad_rights[2] > 0:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:-outpad_rights[2]], dim=-1), 1))
                    else:
                        logging.info(torch.mean(F.softmax(spk_logits[:,outpad_lefts[2]:], dim=-1), 1))

                    src_code = (torch.ones((1, lat_cv.shape[1]))*src_idx).cuda().long()
                    cvmcep_cyc, _ = model_decoder(src_code, lat_cv)

                    if outpad_rights[1] > 0:
                        cvmcep_src = np.array(cvmcep_src[0,outpad_lefts[1]:-outpad_rights[1]].cpu().data.numpy(), dtype=np.float64)
                        cvmcep = np.array(cvmcep[0,outpad_lefts[1]:-outpad_rights[1]].cpu().data.numpy(), dtype=np.float64)
                    else:
                        cvmcep_src = np.array(cvmcep_src[0,outpad_lefts[1]:].cpu().data.numpy(), dtype=np.float64)
                        cvmcep = np.array(cvmcep[0,outpad_lefts[1]:].cpu().data.numpy(), dtype=np.float64)
                    cvmcep_cyc = np.array(cvmcep_cyc[0].cpu().data.numpy(), dtype=np.float64)

                    if trg_exist:
                        if outpad_rights[0] > 0:
                            lat_src = lat_src[:,outpad_lefts[0]:-outpad_rights[0]]
                        else:
                            lat_src = lat_src[:,outpad_lefts[0]:]

                logging.info(cvmcep_src.shape)
                logging.info(cvmcep.shape)
                logging.info(cvmcep_cyc.shape)

                if trg_exist:
                    logging.info(lat_src.shape)
                    logging.info(lat_trg.shape)
 
                cvlist.append(np.var(cvmcep[:,1:], axis=0))

                logging.info("cvf0lin")
                f0_range = read_hdf5(feat_file, "/f0_range")
                cvf0_range_lin = convert_f0(f0_range, src_f0_mean, src_f0_std, trg_f0_mean, trg_f0_std)
                uv_range_lin, cont_f0_range_lin = convert_continuos_f0(np.array(cvf0_range_lin))
                unique, counts = np.unique(uv_range_lin, return_counts=True)
                logging.info(dict(zip(unique, counts)))
                cont_f0_lpf_range_lin = \
                    low_pass_filter(cont_f0_range_lin, int(1.0 / (args.shiftms * 0.001)), cutoff=20)
                uv_range_lin = np.expand_dims(uv_range_lin, axis=-1)
                cont_f0_lpf_range_lin = np.expand_dims(cont_f0_lpf_range_lin, axis=-1)
                # plain converted feat for neural vocoder
                feat_cv = np.c_[uv_range_lin, np.log(cont_f0_lpf_range_lin), feat_cv[:,2:config.excit_dim], cvmcep]
                logging.info(feat_cv.shape)

                logging.info("mcd acc")
                spcidx = np.array(read_hdf5(feat_file, "/spcidx_range")[0])
                _, mcdpow_arr = dtw.calc_mcd(np.array(mcep[spcidx], dtype=np.float64), np.array(cvmcep_src[spcidx], dtype=np.float64))
                _, mcd_arr = dtw.calc_mcd(np.array(mcep[spcidx,1:], dtype=np.float64), np.array(cvmcep_src[spcidx,1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_src_cv: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_src_cv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpow_cvlist_src.append(mcdpow_mean)
                mcdpowstd_cvlist_src.append(mcdpow_std)
                mcd_cvlist_src.append(mcd_mean)
                mcdstd_cvlist_src.append(mcd_std)
                if trg_exist:
                    spcidx_trg = np.array(read_hdf5(file_trg, "/spcidx_range")[0])
                    _, _, _, mcdpow_arr = dtw.dtw_org_to_trg(np.array(cvmcep[spcidx], \
                                                dtype=np.float64), np.array(mcep_trg[spcidx_trg], dtype=np.float64))
                    _, _, _, mcd_arr = dtw.dtw_org_to_trg(np.array(cvmcep[spcidx,1:], \
                                                dtype=np.float64), np.array(mcep_trg[spcidx_trg,1:], dtype=np.float64))
                    mcdpow_mean = np.mean(mcdpow_arr)
                    mcdpow_std = np.std(mcdpow_arr)
                    mcd_mean = np.mean(mcd_arr)
                    mcd_std = np.std(mcd_arr)
                    logging.info("mcdpow_trg: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                    logging.info("mcd_trg: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                    mcdpow_cvlist.append(mcdpow_mean)
                    mcdpowstd_cvlist.append(mcdpow_std)
                    mcd_cvlist.append(mcd_mean)
                    mcdstd_cvlist.append(mcd_std)
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
                _, mcdpow_arr = dtw.calc_mcd(np.array(mcep[spcidx], dtype=np.float64), np.array(cvmcep_cyc[spcidx], dtype=np.float64))
                _, mcd_arr = dtw.calc_mcd(np.array(mcep[spcidx,1:], dtype=np.float64), np.array(cvmcep_cyc[spcidx,1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_cyc_cv: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_cyc_cv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpow_cvlist_cyc.append(mcdpow_mean)
                mcdpowstd_cvlist_cyc.append(mcdpow_std)
                mcd_cvlist_cyc.append(mcd_mean)
                mcdstd_cvlist_cyc.append(mcd_std)

                logging.info("synth anasyn")
                wav = np.clip(pw.synthesize(f0_range, sp, ap, args.fs, frame_period=args.shiftms), -1, 1)
                wavpath = os.path.join(args.outdir,os.path.basename(feat_file).replace(".h5","_anasyn.wav"))
                sf.write(wavpath, wav, args.fs, 'PCM_16')
                logging.info(wavpath)

                logging.info("synth voco rec")
                cvsp_src = ps.mc2sp(cvmcep_src, args.mcep_alpha, args.fftl)
                logging.info(cvsp_src.shape)
                wav = np.clip(pw.synthesize(f0_range, cvsp_src, ap, args.fs, frame_period=args.shiftms), -1, 1)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_rec.wav"))
                sf.write(wavpath, wav, args.fs, 'PCM_16')
                logging.info(wavpath)

                logging.info("synth voco cv")
                cvsp = ps.mc2sp(cvmcep, args.mcep_alpha, args.fftl)
                logging.info(cvsp.shape)
                wav = np.clip(pw.synthesize(cvf0_range_lin, cvsp, ap, args.fs, frame_period=args.shiftms), -1, 1)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_cv.wav"))
                sf.write(wavpath, wav, args.fs, 'PCM_16')
                logging.info(wavpath)

                logging.info("synth voco cv GV")
                datamean = np.mean(cvmcep[:,1:], axis=0)
                cvmcep_gv =  np.c_[cvmcep[:,0], args.gv_coeff*(np.sqrt(gv_mean_trg/cvgv_mean) * \
                                    (cvmcep[:,1:]-datamean) + datamean) + (1-args.gv_coeff)*cvmcep[:,1:]]
                cvmcep_gv = mod_pow(cvmcep_gv, cvmcep, alpha=args.mcep_alpha, irlen=IRLEN)
                cvsp_gv = ps.mc2sp(cvmcep_gv, args.mcep_alpha, args.fftl)
                logging.info(cvsp_gv.shape)
                wav = np.clip(pw.synthesize(cvf0_range_lin, cvsp_gv, ap, args.fs, frame_period=args.shiftms), -1, 1)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_cvGV.wav"))
                sf.write(wavpath, wav, args.fs, 'PCM_16')
                logging.info(wavpath)

                #logging.info("synth diffGV")
                #shiftl = int(args.fs/1000*args.shiftms)
                #mc_cv_diff = cvmcep_gv-mcep
                #b = np.apply_along_axis(ps.mc2b, 1, mc_cv_diff, args.mcep_alpha)
                #logging.info(b.shape)
                #assert np.isfinite(b).all
                #mlsa_fil = ps.synthesis.Synthesizer(MLSADF(mcep_dim, alpha=args.mcep_alpha), shiftl)
                #x, fs_ = sf.read(os.path.join(os.path.dirname(feat_file).replace("hdf5", "wav_filtered"), os.path.basename(feat_file).replace(".h5", ".wav")))
                #assert(fs_ == args.fs)
                #wav = mlsa_fil.synthesis(x, b)
                #wav = np.clip(wav, -1, 1)
                #wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_DiffGV.wav"))
                #sf.write(wavpath, wav, args.fs, 'PCM_16')
                #logging.info(wavpath)

                #logging.info("synth diffGVF0")
                #time_axis = read_hdf5(feat_file, "/time_axis")
                #sp_diff = pw.cheaptrick(wav, f0_range, time_axis, args.fs, fft_size=args.fftl)
                #logging.info(sp_diff.shape)
                #ap_diff = pw.d4c(wav, f0_range, time_axis, args.fs, fft_size=args.fftl)
                #logging.info(ap_diff.shape)
                #wav = pw.synthesize(cvf0_range_lin, sp_diff, ap_diff, args.fs, frame_period=args.shiftms)
                #wav = np.clip(wav, -1, 1)
                #wavpath = os.path.join(args.outdir,os.path.basename(feat_file).replace(".h5", "_DiffGVF0.wav"))
                #sf.write(wavpath, wav, args.fs, 'PCM_16')
                #logging.info(wavpath)

                #logging.info("analysis diffGVF0")
                #sp_diff_anasyn = pw.cheaptrick(wav, cvf0_range_lin, time_axis, args.fs, fft_size=args.fftl)
                #logging.info(sp_diff_anasyn.shape)
                #mc_cv_diff_anasyn = ps.sp2mc(sp_diff_anasyn, mcep_dim, args.mcep_alpha)
                #ap_diff_anasyn = pw.d4c(wav, cvf0_range_lin, time_axis, args.fs, fft_size=args.fftl)
                #code_ap_diff_anasyn = pw.code_aperiodicity(ap_diff_anasyn, args.fs)
                ## convert to continouos codeap with uv
                #for i in range(code_ap_diff_anasyn.shape[-1]):
                #    logging.info('codeap: %d' % (i+1))
                #    uv_codeap_i, cont_codeap_i = convert_continuos_codeap(np.array(code_ap_diff_anasyn[:,i]))
                #    cont_codeap_i = np.log(-np.clip(cont_codeap_i, a_min=np.amin(cont_codeap_i), a_max=MAX_CODEAP))
                #    if i > 0:
                #        cont_codeap = np.c_[cont_codeap, np.expand_dims(cont_codeap_i, axis=-1)]
                #    else:
                #        uv_codeap = np.expand_dims(uv_codeap_i, axis=-1)
                #        cont_codeap = np.expand_dims(cont_codeap_i, axis=-1)
                #    uv_codeap_i = np.expand_dims(uv_codeap_i, axis=-1)
                #    unique, counts = np.unique(uv_codeap_i, return_counts=True)
                #    logging.info(dict(zip(unique, counts)))
                ## postprocessed converted feat for neural vocoder
                #feat_diffgv_anasyn = np.c_[feat_cv[:,:2], uv_codeap, cont_codeap, mc_cv_diff_anasyn]

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
                ## diffGVF0
                #write_path = args.string_path+"_diffgvf0"
                #logging.info(feat_file + ' ' + write_path)
                #logging.info(feat_diffgv_anasyn.shape)
                #write_hdf5(feat_file, write_path, feat_diffgv_anasyn)

                count += 1
                #if count >= 5:
                #if count >= 3:
                #    break


    with mp.Manager() as manager:
        logging.info("GRU-RNN decoding")
        processes = []
        cvlist = manager.list()
        mcd_cvlist_src = manager.list()
        mcdstd_cvlist_src = manager.list()
        mcdpow_cvlist_src = manager.list()
        mcdpowstd_cvlist_src = manager.list()
        mcd_cvlist_cyc = manager.list()
        mcdstd_cvlist_cyc = manager.list()
        mcdpow_cvlist_cyc = manager.list()
        mcdpowstd_cvlist_cyc = manager.list()
        mcd_cvlist = manager.list()
        mcdstd_cvlist = manager.list()
        mcdpow_cvlist = manager.list()
        mcdpowstd_cvlist = manager.list()
        lat_dist_rmse_list = manager.list()
        lat_dist_cosim_list = manager.list()
        gpu = 0
        for i, feat_list in enumerate(feat_lists):
            logging.info(i)
            p = mp.Process(target=decode_RNN, args=(feat_list, gpu, cvlist, \
                mcd_cvlist_src, mcdstd_cvlist_src, mcdpow_cvlist_src, mcdpowstd_cvlist_src, \
                mcd_cvlist_cyc, mcdstd_cvlist_cyc, mcdpow_cvlist_cyc, mcdpowstd_cvlist_cyc, \
                    mcd_cvlist, mcdstd_cvlist, mcdpow_cvlist, mcdpowstd_cvlist,\
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
        logging.info("mcdpow_rec_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist_src)),\
        np.std(np.array(mcdpow_cvlist_src)),np.mean(np.array(mcdpowstd_cvlist_src)),np.std(np.array(mcdpowstd_cvlist_src))))
        logging.info("mcd_rec_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_src)),\
        np.std(np.array(mcd_cvlist_src)),np.mean(np.array(mcdstd_cvlist_src)),np.std(np.array(mcdstd_cvlist_src))))
        logging.info("=== summary cyc. acc. ===")
        logging.info("mcdpow_cyc_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist_cyc)),\
        np.std(np.array(mcdpow_cvlist_cyc)),np.mean(np.array(mcdpowstd_cvlist_cyc)),np.std(np.array(mcdpowstd_cvlist_cyc))))
        logging.info("mcd_cyc_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_cyc)),\
        np.std(np.array(mcd_cvlist_cyc)),np.mean(np.array(mcdstd_cvlist_cyc)),np.std(np.array(mcdstd_cvlist_cyc))))
        logging.info("=== summary cv. acc. ===")
        cvgv_mean = np.mean(np.array(cvlist), axis=0)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg))))))
        if len(mcdpow_cvlist) > 0:
            logging.info("mcdpow_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist)),\
            np.std(np.array(mcdpow_cvlist)),np.mean(np.array(mcdpowstd_cvlist)),np.std(np.array(mcdpowstd_cvlist))))
            logging.info("mcd_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist)),\
            np.std(np.array(mcd_cvlist)),np.mean(np.array(mcdstd_cvlist)),np.std(np.array(mcdstd_cvlist))))
            logging.info("lat_dist_rmse: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_rmse_list)),np.std(np.array(lat_dist_rmse_list))))
            logging.info("lat_dist_cosim: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_cosim_list)),np.std(np.array(lat_dist_cosim_list))))
  
 
if __name__ == "__main__":
    main()
