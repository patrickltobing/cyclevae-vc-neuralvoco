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

from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER, GRU_POST_NET
from utils import find_files, read_hdf5, read_txt, write_hdf5, check_hdf5

from dtw_c import dtw_c as dtw

import torch.nn.functional as F
import h5py

import librosa
import soundfile as sf

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
    parser.add_argument("--n_gpus", default=N_GPUS,
                        type=int, help="number of gpus")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save log")
    parser.add_argument("--string_path", required=True,
                        type=str, help="path of h5 generated feature")
    parser.add_argument("--fs", default=FS,
                        type=int, help="frame shift")
    parser.add_argument("--shiftms", default=SHIFT_MS,
                        type=float, help="frame shift")
    parser.add_argument("--winms", default=WIN_MS,
                        type=float, help="frame shift")
    parser.add_argument("--fftl", default=FFTL,
                        type=int, help="FFT length")
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

    str_split = os.path.basename(os.path.dirname(args.model)).split('_')
    model_name = str_split[1]+"_"+str_split[2]
    logging.info('mdl_name: '+model_name)

    logging.info(config)
    # define gpu decode function
    def gpu_decode(feat_list, gpu, cvlist=None, lsd_cvlist=None,
                    lsdstd_cvlist=None, cvlist_dv=None,
                    lsd_cvlist_dv=None, lsdstd_cvlist_dv=None,
                    cvlist_res=None, lsd_res_cvlist=None,
                    lsdstd_res_cvlist=None, cvlist_res_dv=None,
                    lsd_res_cvlist_dv=None, lsdstd_res_cvlist_dv=None,
                    cvlist_smpl=None, lsd_smpl_cvlist=None,
                    lsdstd_smpl_cvlist=None, cvlist_smpl_dv=None,
                    lsd_smpl_cvlist_dv=None, lsdstd_smpl_cvlist_dv=None):
        with torch.cuda.device(gpu):
            # define model and load parameters
            with torch.no_grad():
                model_encoder_clean = GRU_VAE_ENCODER(
                    in_dim=config.mel_dim,
                    lat_dim=config.lat_dim,
                    hidden_layers=config.hidden_layers_enc,
                    hidden_units=config.hidden_units_enc,
                    kernel_size=config.kernel_size_enc,
                    dilation_size=config.dilation_size_enc,
                    causal_conv=config.causal_conv_enc,
                    pad_first=True,
                    right_size=config.right_size_enc,
                    n_spk=None)
                logging.info(model_encoder_clean)
                model_encoder_noise = GRU_VAE_ENCODER(
                    in_dim=config.mel_dim,
                    lat_dim=config.lat_dim,
                    hidden_layers=config.hidden_layers_enc,
                    hidden_units=config.hidden_units_enc,
                    kernel_size=config.kernel_size_enc,
                    dilation_size=config.dilation_size_enc,
                    causal_conv=config.causal_conv_enc,
                    pad_first=True,
                    right_size=config.right_size_enc,
                    n_spk=None)
                logging.info(model_encoder_noise)
                model_decoder = GRU_SPEC_DECODER(
                    feat_dim=config.lat_dim,
                    out_dim=config.mel_dim,
                    hidden_layers=config.hidden_layers_dec,
                    hidden_units=config.hidden_units_dec,
                    kernel_size=config.kernel_size_dec,
                    dilation_size=config.dilation_size_dec,
                    causal_conv=config.causal_conv_dec,
                    pad_first=True,
                    right_size=config.right_size_dec,
                    n_spk=None)
                logging.info(model_decoder)
                model_post = GRU_POST_NET(
                    spec_dim=config.mel_dim,
                    excit_dim=None,
                    n_spk=None,
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
                model_encoder_clean.load_state_dict(torch.load(args.model)["model_encoder_noise"])
                model_encoder_noise.load_state_dict(torch.load(args.model)["model_encoder_clean"])
                model_decoder.load_state_dict(torch.load(args.model)["model_decoder"])
                model_post.load_state_dict(torch.load(args.model)["model_post"])
                model_encoder_clean.cuda()
                model_encoder_noise.cuda()
                model_decoder.cuda()
                model_post.cuda()
                model_encoder_clean.eval()
                model_encoder_noise.eval()
                model_decoder.eval()
                model_post.eval()
                for param in model_encoder_clean.parameters():
                    param.requires_grad = False
                for param in model_encoder_noise.parameters():
                    param.requires_grad = False
                for param in model_decoder.parameters():
                    param.requires_grad = False
                for param in model_post.parameters():
                    param.requires_grad = False
            count = 0
            pad_left = model_encoder_clean.pad_left + model_decoder.pad_left + model_post.pad_left
            pad_right = model_encoder_clean.pad_right + model_decoder.pad_right + model_post.pad_right
            melfb_t = np.linalg.pinv(librosa.filters.mel(args.fs, args.fftl, n_mels=config.mel_dim))
            hop_length = int((args.fs/1000)*args.shiftms)
            win_length = int((args.fs/1000)*args.winms)
            for feat_file in feat_list:
                # reconst. melsp
                logging.info("recmelsp " + feat_file)

                feat_org = read_hdf5(feat_file, "/log_1pmelmagsp")
                logging.info(feat_org.shape)

                with torch.no_grad():
                    feat = F.pad(torch.FloatTensor(feat_org).cuda().unsqueeze(0).transpose(1,2), (pad_left,pad_right), "replicate").transpose(1,2)

                    _, lat_rec, _ = model_encoder_clean(feat, sampling=False)
                    _, lat_rec_n, _ = model_encoder_noise(feat, sampling=False)
                    melsp_rec, _ = model_decoder(lat_rec)
                    melsp_rec_n, _ = model_decoder(lat_rec_n)
                    melsp_pdf, melsp_smpl, _ = model_post(melsp_rec)
                    melsp_res = melsp_pdf[:,:,:config.mel_dim]
                    melsp_pdf_n, melsp_smpl_n, _ = model_post(melsp_rec_n)
                    melsp_res_n = melsp_pdf_n[:,:,:config.mel_dim]
                    melsp_rec_sum = torch.log(torch.clamp((((melsp_rec.exp()-1)/10000)+((melsp_rec_n.exp()-1)/10000)), min=1e-13)*10000+1)
                    melsp_pdf_sum, melsp_smpl_sum, _ = model_post(melsp_rec_sum)
                    melsp_res_sum = melsp_pdf_sum[:,:,:config.mel_dim]

                    if model_post.pad_right > 0:
                        melsp_rec = melsp_rec[0,model_post.pad_left:-model_post.pad_right].cpu().data.numpy()
                        melsp_rec_n = melsp_rec_n[0,model_post.pad_left:-model_post.pad_right].cpu().data.numpy()
                        melsp_rec_sum = melsp_rec_sum[0,model_post.pad_left:-model_post.pad_right].cpu().data.numpy()
                    else:
                        melsp_rec = melsp_rec[0,model_post.pad_left:].cpu().data.numpy()
                        melsp_rec_n = melsp_rec_n[0,model_post.pad_left:].cpu().data.numpy()
                        melsp_rec_sum = melsp_rec_sum[0,model_post.pad_left:].cpu().data.numpy()
                    melsp_res = melsp_res[0].cpu().data.numpy()
                    melsp_res_n = melsp_res_n[0].cpu().data.numpy()
                    melsp_res_sum = melsp_res_sum[0].cpu().data.numpy()
                    melsp_smpl = melsp_smpl[0].cpu().data.numpy()
                    melsp_smpl_n = melsp_smpl_n[0].cpu().data.numpy()
                    melsp_smpl_sum = melsp_smpl_sum[0].cpu().data.numpy()

                logging.info(melsp_rec.shape)
                logging.info(melsp_rec_n.shape)
                logging.info(melsp_rec_sum.shape)
                logging.info(melsp_res.shape)
                logging.info(melsp_res_n.shape)
                logging.info(melsp_res_sum.shape)
                logging.info(melsp_smpl.shape)
                logging.info(melsp_smpl_n.shape)
                logging.info(melsp_smpl_sum.shape)

                melsp = np.array(feat_org)

                spcidx = np.array(read_hdf5(feat_file, "/spcidx_range")[0])

                melsp_rest = (np.exp(melsp)-1)/10000
                melsp_rec_rest = (np.exp(melsp_rec)-1)/10000
                melsp_rec_n_rest = (np.exp(melsp_rec_n)-1)/10000
                melsp_rec_sum_rest = (np.exp(melsp_rec_sum)-1)/10000
                melsp_res_rest = (np.exp(melsp_res)-1)/10000
                melsp_res_n_rest = (np.exp(melsp_res_n)-1)/10000
                melsp_res_sum_rest = (np.exp(melsp_res_sum)-1)/10000
                melsp_smpl_rest = (np.exp(melsp_smpl)-1)/10000
                melsp_smpl_n_rest = (np.exp(melsp_smpl_n)-1)/10000
                melsp_smpl_sum_rest = (np.exp(melsp_smpl_sum)-1)/10000

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_rec_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_rec_mean = np.mean(lsd_arr)
                lsd_rec_std = np.std(lsd_arr)
                logging.info("lsd_rec: %.6f dB +- %.6f" % (lsd_rec_mean, lsd_rec_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_rec_n_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_rec_n_mean = np.mean(lsd_arr)
                lsd_rec_n_std = np.std(lsd_arr)
                logging.info("lsd_rec_n: %.6f dB +- %.6f" % (lsd_rec_n_mean, lsd_rec_n_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_rec_sum_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_rec_sum_mean = np.mean(lsd_arr)
                lsd_rec_sum_std = np.std(lsd_arr)
                logging.info("lsd_rec_sum: %.6f dB +- %.6f" % (lsd_rec_sum_mean, lsd_rec_sum_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_res_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_res_mean = np.mean(lsd_arr)
                lsd_res_std = np.std(lsd_arr)
                logging.info("lsd_res: %.6f dB +- %.6f" % (lsd_res_mean, lsd_res_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_res_n_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_res_n_mean = np.mean(lsd_arr)
                lsd_res_n_std = np.std(lsd_arr)
                logging.info("lsd_res_n: %.6f dB +- %.6f" % (lsd_res_n_mean, lsd_res_n_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_res_sum_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_res_sum_mean = np.mean(lsd_arr)
                lsd_res_sum_std = np.std(lsd_arr)
                logging.info("lsd_res_sum: %.6f dB +- %.6f" % (lsd_res_sum_mean, lsd_res_sum_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_smpl_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_smpl_mean = np.mean(lsd_arr)
                lsd_smpl_std = np.std(lsd_arr)
                logging.info("lsd_smpl: %.6f dB +- %.6f" % (lsd_smpl_mean, lsd_smpl_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_smpl_n_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_smpl_n_mean = np.mean(lsd_arr)
                lsd_smpl_n_std = np.std(lsd_arr)
                logging.info("lsd_smpl_n: %.6f dB +- %.6f" % (lsd_smpl_n_mean, lsd_smpl_n_std))

                lsd_arr = np.sqrt(np.mean((20*(np.log10(np.clip(melsp_smpl_sum_rest[spcidx], a_min=1e-16, a_max=None))
                                                         -np.log10(np.clip(melsp_rest[spcidx], a_min=1e-16, a_max=None))))**2, axis=-1))
                lsd_smpl_sum_mean = np.mean(lsd_arr)
                lsd_smpl_sum_std = np.std(lsd_arr)
                logging.info("lsd_smpl_sum: %.6f dB +- %.6f" % (lsd_smpl_sum_mean, lsd_smpl_sum_std))

                dataset = feat_file.split('/')[1].split('_')[0]
                if 'tr' in dataset or 'ts' in dataset:
                    logging.info('trn')
                    lsd_cvlist.append(lsd_rec_mean)
                    lsdstd_cvlist.append(lsd_rec_std)
                    cvlist.append(np.var(melsp_rec_rest, axis=0))
                    logging.info(len(cvlist))
                    lsd_res_cvlist.append(lsd_res_mean)
                    lsdstd_res_cvlist.append(lsd_res_std)
                    cvlist_res.append(np.var(melsp_res_rest, axis=0))
                    logging.info(len(cvlist_res))
                    lsd_smpl_cvlist.append(lsd_smpl_mean)
                    lsdstd_smpl_cvlist.append(lsd_smpl_std)
                    cvlist_smpl.append(np.var(melsp_smpl_rest, axis=0))
                    logging.info(len(cvlist_smpl))
                elif 'dv' in dataset:
                    logging.info('dev')
                    lsd_cvlist_dv.append(lsd_rec_mean)
                    lsdstd_cvlist_dv.append(lsd_rec_std)
                    cvlist_dv.append(np.var(melsp_rec_rest, axis=0))
                    logging.info(len(cvlist_dv))
                    lsd_res_cvlist_dv.append(lsd_res_mean)
                    lsdstd_res_cvlist_dv.append(lsd_res_std)
                    cvlist_res_dv.append(np.var(melsp_res_rest, axis=0))
                    logging.info(len(cvlist_res_dv))
                    lsd_smpl_cvlist_dv.append(lsd_smpl_mean)
                    lsdstd_smpl_cvlist_dv.append(lsd_smpl_std)
                    cvlist_smpl_dv.append(np.var(melsp_smpl_rest, axis=0))
                    logging.info(len(cvlist_smpl_dv))

                logging.info("synth gf anasyn")
                recmagsp = np.matmul(melfb_t, melsp_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_anasyn.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf rec")
                recmagsp = np.matmul(melfb_t, melsp_rec_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_rec.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf rec_n")
                recmagsp = np.matmul(melfb_t, melsp_rec_n_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_rec_n.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf rec_sum")
                recmagsp = np.matmul(melfb_t, melsp_rec_sum_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_rec_sum.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf res")
                recmagsp = np.matmul(melfb_t, melsp_res_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_res.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf res_n")
                recmagsp = np.matmul(melfb_t, melsp_res_n_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_res_n.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf res_sum")
                recmagsp = np.matmul(melfb_t, melsp_res_sum_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_res_sum.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf smpl")
                recmagsp = np.matmul(melfb_t, melsp_smpl_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_smpl.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf smpl_n")
                recmagsp = np.matmul(melfb_t, melsp_smpl_n_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_smpl_n.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info("synth gf smpl_sum")
                recmagsp = np.matmul(melfb_t, melsp_smpl_sum_rest.T)
                logging.info(recmagsp.shape)
                wav = np.clip(librosa.core.griffinlim(recmagsp, hop_length=hop_length,
                            win_length=win_length, window='hann'), -1, 0.999969482421875)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_smpl_sum.wav"))
                logging.info(wavpath)
                sf.write(wavpath, wav, args.fs, 'PCM_16')

                logging.info('write rec smpl to h5')
                outh5dir = os.path.join(os.path.dirname(os.path.dirname(feat_file)), args.spk+"-"+args.spk)
                if not os.path.exists(outh5dir):
                    os.makedirs(outh5dir)
                feat_file = os.path.join(outh5dir, os.path.basename(feat_file))
                logging.info(feat_file + ' ' + args.string_path)
                logging.info(melsp_smpl.shape)
                write_hdf5(feat_file, args.string_path, melsp_smpl)

                logging.info('write rec smpl_n to h5')
                logging.info(feat_file + ' ' + args.string_path+"_n")
                logging.info(melsp_smpl_n.shape)
                write_hdf5(feat_file, args.string_path+"_n", melsp_smpl_n)

                logging.info('write rec smpl_sum to h5')
                logging.info(feat_file + ' ' + args.string_path+"_sum")
                logging.info(melsp_smpl_sum.shape)
                write_hdf5(feat_file, args.string_path+"_sum", melsp_smpl_sum)

                count += 1
                if count >= 3:
                    break


    # parallel decode training
    with mp.Manager() as manager:
        gpu = 0
        processes = []
        cvlist = manager.list()
        lsd_cvlist = manager.list()
        lsdstd_cvlist = manager.list()
        cvlist_res = manager.list()
        lsd_res_cvlist = manager.list()
        lsdstd_res_cvlist = manager.list()
        cvlist_smpl = manager.list()
        lsd_smpl_cvlist = manager.list()
        lsdstd_smpl_cvlist = manager.list()
        cvlist_dv = manager.list()
        lsd_cvlist_dv = manager.list()
        lsdstd_cvlist_dv = manager.list()
        cvlist_res_dv = manager.list()
        lsd_res_cvlist_dv = manager.list()
        lsdstd_res_cvlist_dv = manager.list()
        cvlist_smpl_dv = manager.list()
        lsd_smpl_cvlist_dv = manager.list()
        lsdstd_smpl_cvlist_dv = manager.list()
        for i, feat_list in enumerate(feat_lists):
            logging.info(i)
            p = mp.Process(target=gpu_decode, args=(feat_list, gpu, cvlist, 
                                                    lsd_cvlist, lsdstd_cvlist, cvlist_dv, 
                                                    lsd_cvlist_dv, lsdstd_cvlist_dv,
                                                    cvlist_res, 
                                                    lsd_res_cvlist, lsdstd_res_cvlist, cvlist_res_dv, 
                                                    lsd_res_cvlist_dv, lsdstd_res_cvlist_dv,
                                                    cvlist_smpl, 
                                                    lsd_smpl_cvlist, lsdstd_smpl_cvlist, cvlist_smpl_dv, 
                                                    lsd_smpl_cvlist_dv, lsdstd_smpl_cvlist_dv,))
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
            logging.info("lsd_rec: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist)),
                        np.std(np.array(lsd_cvlist)),np.mean(np.array(lsdstd_cvlist)),
                        np.std(np.array(lsdstd_cvlist))))
            cvgv_mean = np.mean(np.array(cvlist), axis=0)
            cvgv_var = np.var(np.array(cvlist), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))),
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("lsd_rec_res: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_res_cvlist)),
                        np.std(np.array(lsd_res_cvlist)),np.mean(np.array(lsdstd_res_cvlist)),
                        np.std(np.array(lsdstd_res_cvlist))))
            cvgv_mean = np.mean(np.array(cvlist_res), axis=0)
            cvgv_var = np.var(np.array(cvlist_res), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))),
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("lsd_rec_smpl: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_smpl_cvlist)),
                        np.std(np.array(lsd_smpl_cvlist)),np.mean(np.array(lsdstd_smpl_cvlist)),
                        np.std(np.array(lsdstd_smpl_cvlist))))
            cvgv_mean = np.mean(np.array(cvlist_smpl), axis=0)
            cvgv_var = np.var(np.array(cvlist_smpl), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))),
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))

            #string_path = model_name+"-"+str(config.n_half_cyc)+"-"+str(config.lat_dim)+"-"+str(config.lat_dim_e)\
            #                +"-"+str(config.spkidtr_dim)+"-"+model_epoch
            #logging.info(string_path)

            #string_mean = "/recgv_mean_"+string_path
            #string_var = "/recgv_var_"+string_path
            #write_hdf5(spk_stat, string_mean, cvgv_mean)
            #write_hdf5(spk_stat, string_var, cvgv_var)

        if len(lsd_cvlist_dv) > 0:
            logging.info("lsd_rec_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_cvlist_dv)),
                        np.std(np.array(lsd_cvlist_dv)),np.mean(np.array(lsdstd_cvlist_dv)),
                        np.std(np.array(lsdstd_cvlist_dv))))
            cvgv_mean = np.mean(np.array(cvlist_dv), axis=0)
            cvgv_var = np.var(np.array(cvlist_dv), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))),
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("lsd_rec_res_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_res_cvlist_dv)),
                        np.std(np.array(lsd_res_cvlist_dv)),np.mean(np.array(lsdstd_res_cvlist_dv)),
                        np.std(np.array(lsdstd_res_cvlist_dv))))
            cvgv_mean = np.mean(np.array(cvlist_res_dv), axis=0)
            cvgv_var = np.var(np.array(cvlist_res_dv), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))),
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))
            logging.info("lsd_rec_smpl_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(lsd_smpl_cvlist_dv)),
                        np.std(np.array(lsd_smpl_cvlist_dv)),np.mean(np.array(lsdstd_smpl_cvlist_dv)),
                        np.std(np.array(lsdstd_smpl_cvlist_dv))))
            cvgv_mean = np.mean(np.array(cvlist_smpl_dv), axis=0)
            cvgv_var = np.var(np.array(cvlist_smpl_dv), axis=0)
            logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean)))),
                                        np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean))))))


if __name__ == "__main__":
    main()
