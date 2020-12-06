#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
from dateutil.relativedelta import relativedelta
from distutils.util import strtobool
import logging
import os
import sys
import time

from collections import defaultdict
from tensorboardX import SummaryWriter

import numpy as np
import six
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import torch.nn.functional as F

from utils import find_files
from utils import read_hdf5
from utils import read_txt
from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER, GRU_LAT_FEAT_CLASSIFIER
from vcneuvoco import GRU_EXCIT_DECODER, SPKID_TRANSFORM_LAYER, GRU_SPK, GRU_POST_NET
from vcneuvoco import kl_laplace, ModulationSpectrumLoss, kl_categorical_categorical_logits, LaplaceLoss

import torch_optimizer as optim

from dataset import FeatureDatasetCycMceplf0WavVAE, FeatureDatasetEvalCycMceplf0WavVAE, padding

import librosa
from dtw_c import dtw_c as dtw

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)


def train_generator(dataloader, device, batch_size, n_cv, limit_count=None):
    """TRAINING BATCH GENERATOR

    Args:
        wav_list (str): list of wav files
        feat_list (str): list of feat files
        batch_size (int): batch size
        wav_transform (func): preprocessing function for waveform

    Return:
        (object): generator instance
    """
    while True:
        # process over all of files
        c_idx = 0
        count = 0
        for idx, batch in enumerate(dataloader):
            flens = batch['flen'].data.numpy()
            max_flen = np.max(flens) ## get max samples length
            feat = batch['feat'][:,:max_flen].to(device)
            feat_magsp = batch['feat_magsp'][:,:max_flen].to(device)
            sc = batch['src_codes'][:,:max_flen].to(device)
            sc_cv = [None]*n_cv
            feat_cv = [None]*n_cv
            for i in range(n_cv):
                sc_cv[i] = batch['src_trg_codes_list'][i][:,:max_flen].to(device)
                feat_cv[i] = batch['feat_cv_list'][i][:,:max_flen].to(device)
            featfiles = batch['featfile']
            spk_cv = batch['pair_spk_list']
            n_batch_utt = feat.size(0)

            len_frm = max_flen
            f_ss = 0
            f_bs = batch_size
            delta_frm = batch_size
            flens_acc = np.array(flens)
            while True:
                del_index_utt = []
                idx_select = []
                idx_select_full = []
                for i in range(n_batch_utt):
                    if flens_acc[i] <= 0:
                        del_index_utt.append(i)
                if len(del_index_utt) > 0:
                    flens = np.delete(flens, del_index_utt, axis=0)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_magsp = torch.FloatTensor(np.delete(feat_magsp.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc = torch.LongTensor(np.delete(sc.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    for j in range(n_cv):
                        sc_cv[j] = torch.LongTensor(np.delete(sc_cv[j].cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                        feat_cv[j] = torch.FloatTensor(np.delete(feat_cv[j].cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                        spk_cv[j] = np.delete(spk_cv[j], del_index_utt, axis=0)
                    featfiles = np.delete(featfiles, del_index_utt, axis=0)
                    flens_acc = np.delete(flens_acc, del_index_utt, axis=0)
                    n_batch_utt -= len(del_index_utt)
                for i in range(n_batch_utt):
                    if flens_acc[i] < f_bs:
                        idx_select.append(i)
                if len(idx_select) > 0:
                    idx_select_full = torch.LongTensor(np.delete(np.arange(n_batch_utt), idx_select, axis=0)).to(device)
                    idx_select = torch.LongTensor(idx_select).to(device)
                yield feat, feat_magsp, sc, sc_cv, feat_cv, c_idx, idx, featfiles, f_bs, f_ss, flens, \
                    n_batch_utt, del_index_utt, max_flen, spk_cv, idx_select, idx_select_full, flens_acc
                for i in range(n_batch_utt):
                    flens_acc[i] -= delta_frm

                count += 1
                if limit_count is not None and count > limit_count:
                    break
                len_frm -= delta_frm
                if len_frm > 0:
                    f_ss += delta_frm
                else:
                    break

            if limit_count is not None and count > limit_count:
                break
            c_idx += 1
            #if c_idx > 0:
            #if c_idx > 1:
            #if c_idx > 2:
            #    break

        yield [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], []


def eval_generator(dataloader, device, batch_size, limit_count=None, spcidx=True):
    """TRAINING BATCH GENERATOR

    Args:
        wav_list (str): list of wav files
        feat_list (str): list of feat files
        batch_size (int): batch size
        wav_transform (func): preprocessing function for waveform

    Return:
        (object): generator instance
    """
    while True:
        # process over all of files
        c_idx = 0
        count = 0
        for idx, batch in enumerate(dataloader):
            flens = batch['flen_src'].data.numpy()
            flens_trg = batch['flen_src_trg'].data.numpy()
            flens_spc_src = batch['flen_spc_src'].data.numpy()
            flens_spc_src_trg = batch['flen_spc_src_trg'].data.numpy()
            max_flen = np.max(flens) ## get max samples length
            max_flen_trg = np.max(flens_trg) ## get max samples length
            max_flen_spc_src = np.max(flens_spc_src) ## get max samples length
            max_flen_spc_src_trg = np.max(flens_spc_src_trg) ## get max samples length
            feat = batch['h_src'][:,:max_flen].to(device)
            feat_magsp = batch['h_src_magsp'][:,:max_flen].to(device)
            feat_trg = batch['h_src_trg'][:,:max_flen_trg].to(device)
            sc = batch['src_code'][:,:max_flen].to(device)
            sc_cv = batch['src_trg_code'][:,:max_flen].to(device)
            feat_cv = batch['cv_src'][:,:max_flen].to(device)
            spcidx_src = batch['spcidx_src'][:,:max_flen_spc_src].to(device)
            spcidx_src_trg = batch['spcidx_src_trg'][:,:max_flen_spc_src_trg].to(device)
            featfiles = batch['featfile']
            file_src_trg_flag = batch['file_src_trg_flag']
            spk_cv = batch['spk_trg']
            n_batch_utt = feat.size(0)
            if spcidx:
                flens_full = batch['flen_src_full'].data.numpy()
                max_flen_full = np.max(flens_full) ## get max samples length
                feat_full = batch['h_src_full'][:,:max_flen_full].to(device)
                sc_full = batch['src_code_full'][:,:max_flen_full].to(device)
                sc_cv_full = batch['src_trg_code_full'][:,:max_flen_full].to(device)

            len_frm = max_flen
            f_ss = 0
            f_bs = batch_size
            delta_frm = batch_size
            flens_acc = np.array(flens)
            while True:
                del_index_utt = []
                idx_select = []
                idx_select_full = []
                for i in range(n_batch_utt):
                    if flens_acc[i] >= f_bs:
                        flens_acc[i] -= delta_frm
                    else:
                        del_index_utt.append(i)
                if len(del_index_utt) > 0:
                    flens = np.delete(flens, del_index_utt, axis=0)
                    flens_trg = np.delete(flens_trg, del_index_utt, axis=0)
                    flens_spc_src = np.delete(flens_spc_src, del_index_utt, axis=0)
                    flens_spc_src_trg = np.delete(flens_spc_src_trg, del_index_utt, axis=0)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_magsp = torch.FloatTensor(np.delete(feat_magsp.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_trg = torch.FloatTensor(np.delete(feat_trg.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc = torch.LongTensor(np.delete(sc.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc_cv = torch.LongTensor(np.delete(sc_cv.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_cv = torch.FloatTensor(np.delete(feat_cv.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    spcidx_src = torch.LongTensor(np.delete(spcidx_src.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    spcidx_src_trg = torch.LongTensor(np.delete(spcidx_src_trg.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    spk_cv = np.delete(spk_cv, del_index_utt, axis=0)
                    file_src_trg_flag = np.delete(file_src_trg_flag, del_index_utt, axis=0)
                    featfiles = np.delete(featfiles, del_index_utt, axis=0)
                    flens_acc = np.delete(flens_acc, del_index_utt, axis=0)
                    n_batch_utt -= len(del_index_utt)
                for i in range(n_batch_utt):
                    if flens_acc[i] < f_bs:
                        idx_select.append(i)
                if len(idx_select) > 0:
                    idx_select_full = torch.LongTensor(np.delete(np.arange(n_batch_utt), idx_select, axis=0)).to(device)
                    idx_select = torch.LongTensor(idx_select).to(device)
                if spcidx:
                    yield feat, feat_magsp, feat_trg, sc, sc_cv, feat_cv, c_idx, idx, featfiles, f_bs, f_ss, flens, \
                        n_batch_utt, del_index_utt, max_flen, spk_cv, file_src_trg_flag, spcidx_src, \
                            spcidx_src_trg, flens_spc_src, flens_spc_src_trg, feat_full, sc_full, sc_cv_full, \
                                idx_select, idx_select_full
                else:
                    yield feat, feat_magsp, feat_trg, sc, sc_cv, feat_cv, c_idx, idx, featfiles, f_bs, f_ss, flens, \
                        n_batch_utt, del_index_utt, max_flen, spk_cv, file_src_trg_flag, spcidx_src, \
                            spcidx_src_trg, flens_spc_src, flens_spc_src_trg, idx_select, idx_select_full
                len_frm -= delta_frm
                if len_frm >= f_bs:
                    f_ss += delta_frm
                else:
                    break

            count += 1
            if limit_count is not None and count >= limit_count:
                break
            c_idx += 1
            #if c_idx > 0:
            #if c_idx > 1:
            #if c_idx > 2:
            #    break

        if spcidx:
            yield [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        else:
            yield [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


def save_checkpoint(checkpoint_dir, model_encoder_melsp, model_decoder_melsp_fix, model_decoder_melsp, model_encoder_excit, model_decoder_excit,
        model_spk, model_post, model_classifier, min_eval_loss_melsp_dB, min_eval_loss_melsp_dB_std, min_eval_loss_melsp_cv,
        min_eval_loss_melsp, min_eval_loss_melsp_dB_src_trg, min_eval_loss_melsp_dB_src_trg_std, min_eval_loss_gv_src_trg,
        min_eval_loss_melsp_dB_post, min_eval_loss_melsp_dB_post_std, min_eval_loss_melsp_cv_post,
        min_eval_loss_melsp_post, min_eval_loss_melsp_dB_src_trg_post, min_eval_loss_melsp_dB_src_trg_post_std, min_eval_loss_gv_src_trg_post,
        iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, iterations, model_spkidtr=None):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    model_encoder_melsp.cpu()
    model_decoder_melsp_fix.cpu()
    model_decoder_melsp.cpu()
    model_encoder_excit.cpu()
    model_decoder_excit.cpu()
    model_spk.cpu()
    model_post.cpu()
    model_classifier.cpu()
    if model_spkidtr is not None:
        model_spkidtr.cpu()
    checkpoint = {
        "model_encoder_melsp": model_encoder_melsp.state_dict(),
        "model_decoder_melsp_fix": model_decoder_melsp_fix.state_dict(),
        "model_decoder_melsp": model_decoder_melsp.state_dict(),
        "model_encoder_excit": model_encoder_excit.state_dict(),
        "model_decoder_excit": model_decoder_excit.state_dict(),
        "model_spk": model_spk.state_dict(),
        "model_post": model_post.state_dict(),
        "model_classifier": model_classifier.state_dict(),
        "min_eval_loss_melsp_dB": min_eval_loss_melsp_dB,
        "min_eval_loss_melsp_dB_std": min_eval_loss_melsp_dB_std,
        "min_eval_loss_melsp_cv": min_eval_loss_melsp_cv,
        "min_eval_loss_melsp": min_eval_loss_melsp,
        "min_eval_loss_melsp_dB_src_trg": min_eval_loss_melsp_dB_src_trg,
        "min_eval_loss_melsp_dB_src_trg_std": min_eval_loss_melsp_dB_src_trg_std,
        "min_eval_loss_melsp_dB_post": min_eval_loss_melsp_dB_post,
        "min_eval_loss_melsp_dB_post_std": min_eval_loss_melsp_dB_post_std,
        "min_eval_loss_melsp_cv_post": min_eval_loss_melsp_cv_post,
        "min_eval_loss_melsp_post": min_eval_loss_melsp_post,
        "min_eval_loss_melsp_dB_src_trg_post": min_eval_loss_melsp_dB_src_trg_post,
        "min_eval_loss_melsp_dB_src_trg_post_std": min_eval_loss_melsp_dB_src_trg_post_std,
        "iter_idx": iter_idx,
        "min_idx": min_idx,
        "optimizer": optimizer.state_dict(),
        "numpy_random_state": numpy_random_state,
        "torch_random_state": torch_random_state,
        "iterations": iterations}
    if model_spkidtr is not None:
        checkpoint["model_spkidtr"] = model_spkidtr.state_dict()
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    model_encoder_melsp.cuda()
    model_decoder_melsp_fix.cuda()
    model_decoder_melsp.cuda()
    model_encoder_excit.cuda()
    model_decoder_excit.cuda()
    model_spk.cuda()
    model_post.cuda()
    model_classifier.cuda()
    if model_spkidtr is not None:
        model_spkidtr.cuda()
    logging.info("%d-iter checkpoint created." % iterations)


def write_to_tensorboard(writer, steps, loss):
    """Write to tensorboard."""
    for key, value in loss.items():
        writer.add_scalar(key, value, steps)


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--feats_eval_list", required=True,
                        type=str, help="directory or list of evaluation feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="directory or list of evaluation wav files")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    # network structure setting
    parser.add_argument("--hidden_units_enc", default=512,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_layers_enc", default=1,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_units_dec", default=640,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_layers_dec", default=1,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_units_lf0", default=128,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_layers_lf0", default=1,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_units_post", default=128,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_layers_post", default=1,
                        type=int, help="depth of dilation")
    parser.add_argument("--kernel_size_enc", default=5,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_enc", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--kernel_size_dec", default=5,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_dec", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--kernel_size_lf0", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_lf0", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--kernel_size_post", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_post", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--spk_list", required=True,
                        type=str, help="kernel size of dilated causal convolution")
    parser.add_argument("--stats_list", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument("--lat_dim", default=32,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--lat_dim_e", default=32,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--mel_dim", default=80,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--excit_dim", default=2,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--full_excit_dim", default=6,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--spkidtr_dim", default=0,
                        type=int, help="number of dimension of reduced one-hot spk-dim (if 0 not apply reduction)")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=30,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--epoch_count", default=200,
                        type=int, help="number of training epochs")
    parser.add_argument("--do_prob", default=0.5,
                        type=float, help="dropout probability")
    parser.add_argument("--batch_size_utt", default=8,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--batch_size_utt_eval", default=14,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--n_workers", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--n_half_cyc", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_enc", default=False,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_dec", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_lf0", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_post", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_enc", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_dec", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_lf0", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_post", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--fftl", default=2048,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--fs", default=24000,
                        type=int, help="kernel size of dilated causal convolution")
    # other setting
    parser.add_argument("--pad_len", default=3000,
                        type=int, help="seed number")
    #parser.add_argument("--save_interval_iter", default=5000,
    #                    type=int, help="interval steps to logr")
    #parser.add_argument("--save_interval_epoch", default=10,
    #                    type=int, help="interval steps to logr")
    parser.add_argument("--log_interval_steps", default=50,
                        type=int, help="interval steps to logr")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--gen_model", required=True,
                        type=str, help="model path to restart training")
    #parser.add_argument("--string_path", default=None,
    #                    type=str, help="model path to restart training")
    parser.add_argument("--GPU_device", default=None,
                        type=int, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    if args.GPU_device is not None:
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]  = str(args.GPU_device)

    # make experimental directory
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "cpu":
        raise ValueError('ERROR: Training by CPU is not acceptable.')

    torch.backends.cudnn.benchmark = True #faster

    spk_list = args.spk_list.split('@')
    n_spk = len(spk_list)
    feat_eval_src_list = args.feats_eval_list.split('@')
    assert(n_spk == len(feat_eval_src_list))

    mean_stats = torch.FloatTensor(np.r_[read_hdf5(args.stats, "/mean_feat_mceplf0cap")[:args.excit_dim], read_hdf5(args.stats, "/mean_melsp")])
    scale_stats = torch.FloatTensor(np.r_[read_hdf5(args.stats, "/scale_feat_mceplf0cap")[:args.excit_dim], read_hdf5(args.stats, "/scale_melsp")])
    mean_cap = torch.FloatTensor(read_hdf5(args.stats, "/mean_feat_mceplf0cap")[3:args.full_excit_dim])
    scale_cap = torch.FloatTensor(read_hdf5(args.stats, "/scale_feat_mceplf0cap")[3:args.full_excit_dim])
    mean_full = torch.FloatTensor(np.r_[read_hdf5(args.stats, "/mean_feat_mceplf0cap")[:args.full_excit_dim], read_hdf5(args.stats, "/mean_melsp")])
    scale_full = torch.FloatTensor(np.r_[read_hdf5(args.stats, "/scale_feat_mceplf0cap")[:args.full_excit_dim], read_hdf5(args.stats, "/scale_melsp")])
    args.cap_dim = mean_cap.shape[0]

    # save args as conf
    args.fftsize = 2 ** (len(bin(200)) - 2 + 1)
    args.string_path = "/log_1pmelmagsp"
    torch.save(args, args.expdir + "/model.conf")

    # define network
    model_encoder_melsp = GRU_VAE_ENCODER(
        in_dim=args.mel_dim,
        n_spk=n_spk,
        lat_dim=args.lat_dim,
        hidden_layers=args.hidden_layers_enc,
        hidden_units=args.hidden_units_enc,
        kernel_size=args.kernel_size_enc,
        dilation_size=args.dilation_size_enc,
        causal_conv=args.causal_conv_enc,
        pad_first=True,
        right_size=args.right_size_enc,
        do_prob=args.do_prob)
    logging.info(model_encoder_melsp)
    model_decoder_melsp_fix = GRU_SPEC_DECODER(
        feat_dim=args.lat_dim+args.lat_dim_e,
        excit_dim=args.excit_dim,
        out_dim=args.mel_dim,
        n_spk=n_spk,
        aux_dim=n_spk,
        hidden_layers=args.hidden_layers_dec,
        hidden_units=args.hidden_units_dec,
        kernel_size=args.kernel_size_dec,
        dilation_size=args.dilation_size_dec,
        causal_conv=args.causal_conv_dec,
        pad_first=True,
        right_size=args.right_size_dec,
        do_prob=args.do_prob)
    logging.info(model_decoder_melsp_fix)
    model_decoder_melsp = GRU_SPEC_DECODER(
        feat_dim=args.lat_dim+args.lat_dim_e,
        excit_dim=args.excit_dim,
        out_dim=args.mel_dim,
        n_spk=n_spk,
        aux_dim=n_spk,
        hidden_layers=args.hidden_layers_dec,
        hidden_units=args.hidden_units_dec,
        kernel_size=args.kernel_size_dec,
        dilation_size=args.dilation_size_dec,
        causal_conv=args.causal_conv_dec,
        pad_first=True,
        right_size=args.right_size_dec,
        do_prob=args.do_prob)
    logging.info(model_decoder_melsp)
    model_encoder_excit = GRU_VAE_ENCODER(
        in_dim=args.mel_dim,
        n_spk=n_spk,
        lat_dim=args.lat_dim_e,
        hidden_layers=args.hidden_layers_enc,
        hidden_units=args.hidden_units_enc,
        kernel_size=args.kernel_size_enc,
        dilation_size=args.dilation_size_enc,
        causal_conv=args.causal_conv_enc,
        pad_first=True,
        right_size=args.right_size_enc,
        do_prob=args.do_prob)
    logging.info(model_encoder_excit)
    model_decoder_excit = GRU_EXCIT_DECODER(
        feat_dim=args.lat_dim_e,
        cap_dim=args.cap_dim,
        n_spk=n_spk,
        aux_dim=n_spk,
        hidden_layers=args.hidden_layers_lf0,
        hidden_units=args.hidden_units_lf0,
        kernel_size=args.kernel_size_lf0,
        dilation_size=args.dilation_size_lf0,
        causal_conv=args.causal_conv_lf0,
        pad_first=True,
        right_size=args.right_size_lf0,
        do_prob=args.do_prob)
    logging.info(model_decoder_excit)
    if (args.spkidtr_dim > 0):
        model_spkidtr = SPKID_TRANSFORM_LAYER(
            n_spk=n_spk,
            spkidtr_dim=args.spkidtr_dim)
        logging.info(model_spkidtr)
    else:
        model_spkidtr = None
    model_classifier = GRU_LAT_FEAT_CLASSIFIER(
        feat_dim=args.mel_dim,
        feat_aux_dim=args.fftl//2+1,
        n_spk=n_spk,
        hidden_units=32,
        hidden_layers=1,
        do_prob=args.do_prob)
    logging.info(model_classifier) 
    model_post = GRU_POST_NET(
        spec_dim=args.mel_dim,
        excit_dim=args.excit_dim+args.cap_dim+1,
        n_spk=n_spk,
        aux_dim=n_spk,
        hidden_layers=args.hidden_layers_post,
        hidden_units=args.hidden_units_post,
        kernel_size=args.kernel_size_post,
        dilation_size=args.dilation_size_post,
        causal_conv=args.causal_conv_post,
        pad_first=True,
        right_size=args.right_size_post,
        res=True,
        laplace=True,
        do_prob=args.do_prob)
    logging.info(model_post)
    model_spk = GRU_SPK(
        n_spk=n_spk,
        feat_dim=args.lat_dim+args.lat_dim_e,
        hidden_units=32,
        do_prob=args.do_prob)
    logging.info(model_spk)
    criterion_ms = ModulationSpectrumLoss(args.fftsize)
    criterion_laplace = LaplaceLoss()
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_l1 = torch.nn.L1Loss(reduction='none')
    criterion_l2 = torch.nn.MSELoss(reduction='none')
    melfb_t = torch.FloatTensor(np.linalg.pinv(librosa.filters.mel(args.fs, args.fftl, n_mels=args.mel_dim)).T)

    p_spk = torch.ones(n_spk)/n_spk

    # send to gpu
    if torch.cuda.is_available():
        model_encoder_melsp.cuda()
        model_decoder_melsp_fix.cuda()
        model_decoder_melsp.cuda()
        model_encoder_excit.cuda()
        model_decoder_excit.cuda()
        model_spk.cuda()
        model_classifier.cuda()
        if (args.spkidtr_dim > 0):
            model_spkidtr.cuda()
        model_post.cuda()
        criterion_ms.cuda()
        criterion_laplace.cuda()
        criterion_ce.cuda()
        criterion_l1.cuda()
        criterion_l2.cuda()
        mean_stats = mean_stats.cuda()
        scale_stats = scale_stats.cuda()
        mean_cap = mean_cap.cuda()
        scale_cap = scale_cap.cuda()
        mean_full = mean_full.cuda()
        scale_full = scale_full.cuda()
        p_spk = p_spk.cuda()
        melfb_t = melfb_t.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)
    logits_p_spk = torch.log(p_spk)

    logging.info(p_spk)
    logging.info(logits_p_spk)

    model_encoder_melsp.train()
    model_decoder_melsp_fix.train()
    model_decoder_melsp.train()
    model_encoder_excit.train()
    model_decoder_excit.train()
    model_spk.train()
    model_classifier.train()
    if (args.spkidtr_dim > 0):
        model_spkidtr.train()
    model_post.train()

    if model_post.use_weight_norm:
        torch.nn.utils.remove_weight_norm(model_post.scale_in)
        torch.nn.utils.remove_weight_norm(model_post.scale_out)

    model_post.scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/scale_full.data),2))
    model_post.scale_in.bias = torch.nn.Parameter(-(mean_full.data/scale_full.data))
    model_post.scale_out.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(scale_stats[args.excit_dim:].data),2))
    model_post.scale_out.bias = torch.nn.Parameter(mean_stats[args.excit_dim:].data)

    if model_post.use_weight_norm:
        torch.nn.utils.weight_norm(model_post.scale_in)
        torch.nn.utils.weight_norm(model_post.scale_out)

    parameters = filter(lambda p: p.requires_grad, model_encoder_melsp.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder_melsp): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_decoder_melsp_fix.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder_melsp_fix): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_decoder_melsp.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder_melsp): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_encoder_excit.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder_excit): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_decoder_excit.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder_excit): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_spk.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (spk): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_classifier.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (classifier): %.3f million' % parameters)
    if (args.spkidtr_dim > 0):
        parameters = filter(lambda p: p.requires_grad, model_spkidtr.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        logging.info('Trainable Parameters (spkidtr): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_post.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (post): %.3f million' % parameters)

    for param in model_encoder_melsp.parameters():
        param.requires_grad = False
    for param in model_decoder_melsp_fix.parameters():
        param.requires_grad = False
    for param in model_decoder_melsp.parameters():
        param.requires_grad = True
    for param in model_decoder_melsp.scale_in.parameters():
        param.requires_grad = False
    for param in model_decoder_melsp.scale_out.parameters():
        param.requires_grad = False
    for param in model_encoder_excit.parameters():
        param.requires_grad = False
    for param in model_decoder_excit.parameters():
        param.requires_grad = False
    if (args.spkidtr_dim > 0):
        for param in model_spkidtr.parameters():
            param.requires_grad = False
    for param in model_spk.parameters():
        param.requires_grad = False
    for param in model_post.parameters():
        param.requires_grad = True
    for param in model_post.scale_in.parameters():
        param.requires_grad = False
    for param in model_post.scale_out.parameters():
        param.requires_grad = False

    module_list = list(model_decoder_melsp.conv.parameters())
    module_list += list(model_decoder_melsp.gru.parameters()) + list(model_decoder_melsp.out.parameters())

    module_list += list(model_post.gru.parameters()) + list(model_post.out.parameters())

    module_list += list(model_classifier.conv_feat.parameters()) + list(model_classifier.conv_feat_aux.parameters())
    module_list += list(model_classifier.gru.parameters()) + list(model_classifier.out.parameters())

    # model = ...
    optimizer = optim.RAdam(
        module_list,
        lr= args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    checkpoint = torch.load(args.gen_model)
    model_encoder_melsp.load_state_dict(checkpoint["model_encoder_melsp"])
    model_decoder_melsp_fix.load_state_dict(checkpoint["model_decoder_melsp"])
    model_decoder_melsp.load_state_dict(checkpoint["model_decoder_melsp"])
    model_encoder_excit.load_state_dict(checkpoint["model_encoder_excit"])
    model_decoder_excit.load_state_dict(checkpoint["model_decoder_excit"])
    model_spk.load_state_dict(checkpoint["model_spk"])
    if (args.spkidtr_dim > 0):
        model_spkidtr.load_state_dict(checkpoint["model_spkidtr"])
    epoch_idx = checkpoint["iterations"]
    logging.info("gen_model from %d-iter checkpoint." % epoch_idx)
    epoch_idx = 0

    # resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model_encoder_melsp.load_state_dict(checkpoint["model_encoder_melsp"])
        model_decoder_melsp_fix.load_state_dict(checkpoint["model_decoder_melsp_fix"])
        model_decoder_melsp.load_state_dict(checkpoint["model_decoder_melsp"])
        model_encoder_excit.load_state_dict(checkpoint["model_encoder_excit"])
        model_decoder_excit.load_state_dict(checkpoint["model_decoder_excit"])
        model_spk.load_state_dict(checkpoint["model_spk"])
        model_post.load_state_dict(checkpoint["model_post"])
        model_classifier.load_state_dict(checkpoint["model_classifier"])
        if (args.spkidtr_dim > 0):
            model_spkidtr.load_state_dict(checkpoint["model_spkidtr"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)

    def zero_feat_pad(x): return padding(x, args.pad_len, value=None)
    pad_feat_transform = transforms.Compose([zero_feat_pad])

    n_rec = args.n_half_cyc + args.n_half_cyc%2
    n_cv = int(args.n_half_cyc/2+args.n_half_cyc%2)

    stats_list = args.stats_list.split('@')
    assert(n_spk == len(stats_list))

    if os.path.isdir(args.feats):
        feat_list = [args.feats + "/" + filename for filename in filenames]
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)
    logging.info("number of training data = %d." % len(feat_list))
    dataset = FeatureDatasetCycMceplf0WavVAE(feat_list, pad_feat_transform, spk_list, stats_list,
                    args.n_half_cyc, args.string_path, excit_dim=args.full_excit_dim, magsp=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_utt, shuffle=True, num_workers=args.n_workers)
    #generator = train_generator(dataloader, device, args.batch_size, n_cv, limit_count=1)
    generator = train_generator(dataloader, device, args.batch_size, n_cv, limit_count=None)

    # define generator evaluation
    feat_list_eval_src_list = [None]*n_spk
    for i in range(n_spk):
        if os.path.isdir(feat_eval_src_list[i]):
            feat_list_eval_src_list[i] = sorted(find_files(feat_eval_src_list[i], "*.h5", use_dir_name=False))
        elif os.path.isfile(feat_eval_src_list[i]):
            feat_list_eval_src_list[i] = read_txt(feat_eval_src_list[i])
        else:
            logging.error("%s should be directory or list." % (feat_eval_src_list[i]))
            sys.exit(1)
    dataset_eval = FeatureDatasetEvalCycMceplf0WavVAE(feat_list_eval_src_list, pad_feat_transform, spk_list, \
                    stats_list, args.string_path, excit_dim=args.full_excit_dim, magsp=True)
    n_eval_data = len(dataset_eval.file_list_src)
    logging.info("number of evaluation data = %d." % n_eval_data)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size_utt_eval, shuffle=False, num_workers=args.n_workers)
    #generator_eval = eval_generator(dataloader_eval, device, args.batch_size, limit_count=1)
    generator_eval = eval_generator(dataloader_eval, device, args.batch_size, limit_count=None)

    writer = SummaryWriter(args.expdir)
    total_train_loss = defaultdict(list)
    total_eval_loss = defaultdict(list)

    gv_mean = [None]*n_spk
    for i in range(n_spk):
        gv_mean[i] = read_hdf5(stats_list[i], "/gv_melsp_mean")

    # train
    logging.info(args.fftsize)
    eps_1 = torch.finfo(mean_stats.dtype).eps-1
    logging.info(eps_1)
    logging.info(f'n_cyc: {args.n_half_cyc}')
    logging.info(f'n_rec: {n_rec}')
    logging.info(f'n_cv: {n_cv}')
    enc_pad_left = model_encoder_melsp.pad_left
    enc_pad_right = model_encoder_melsp.pad_right
    logging.info(f'enc_pad_left: {enc_pad_left}')
    logging.info(f'enc_pad_right: {enc_pad_right}')
    lf0_pad_left = model_decoder_excit.pad_left
    lf0_pad_right = model_decoder_excit.pad_right
    logging.info(f'lf0_pad_left: {lf0_pad_left}')
    logging.info(f'lf0_pad_right: {lf0_pad_right}')
    dec_pad_left = model_decoder_melsp.pad_left
    dec_pad_right = model_decoder_melsp.pad_right
    logging.info(f'dec_pad_left: {dec_pad_left}')
    logging.info(f'dec_pad_right: {dec_pad_right}')
    post_pad_left = model_post.pad_left
    post_pad_right = model_post.pad_right
    logging.info(f'post_pad_left: {post_pad_left}')
    logging.info(f'post_pad_right: {post_pad_right}')
    dec_post_pad_left = dec_pad_left + post_pad_left
    dec_post_pad_right = dec_pad_right + post_pad_right
    first_pad_left = (enc_pad_left + lf0_pad_left + dec_post_pad_left)*args.n_half_cyc
    first_pad_right = (enc_pad_right + lf0_pad_right + dec_post_pad_right)*args.n_half_cyc
    # paddings are added with 1 more lf0_pad because excit. flow part is also used in melsp flow part
    if args.n_half_cyc == 1:
        first_pad_left += enc_pad_left
        first_pad_right += enc_pad_right
    logging.info(f'first_pad_left: {first_pad_left}')
    logging.info(f'first_pad_right: {first_pad_right}')
    if args.n_half_cyc > 1:
        outpad_lefts = [None]*args.n_half_cyc*4
        outpad_rights = [None]*args.n_half_cyc*4
    else:
        outpad_lefts = [None]*(args.n_half_cyc*4+1)
        outpad_rights = [None]*(args.n_half_cyc*4+1)
    outpad_lefts[0] = first_pad_left-enc_pad_left
    outpad_rights[0] = first_pad_right-enc_pad_right
    for i in range(1,args.n_half_cyc*4):
        if i % 4 == 3:
            outpad_lefts[i] = outpad_lefts[i-1]-post_pad_left
            outpad_rights[i] = outpad_rights[i-1]-post_pad_right
        elif i % 4 == 2:
            outpad_lefts[i] = outpad_lefts[i-1]-dec_pad_left
            outpad_rights[i] = outpad_rights[i-1]-dec_pad_right
        elif i % 4 == 1:
            outpad_lefts[i] = outpad_lefts[i-1]-lf0_pad_left
            outpad_rights[i] = outpad_rights[i-1]-lf0_pad_right
        else:
            outpad_lefts[i] = outpad_lefts[i-1]-enc_pad_left
            outpad_rights[i] = outpad_rights[i-1]-enc_pad_right
    if args.n_half_cyc == 1:
        outpad_lefts[len(outpad_lefts)-1] = outpad_lefts[len(outpad_lefts)-2]-enc_pad_left
        outpad_rights[len(outpad_rights)-1] = outpad_rights[len(outpad_rights)-2]-enc_pad_right
    logging.info(outpad_lefts)
    logging.info(outpad_rights)
    batch_feat_in = [None]*args.n_half_cyc*4
    batch_sc_in = [None]*args.n_half_cyc*4
    batch_sc_cv_in = [None]*n_cv*3
    total = 0
    iter_count = 0
    batch_sc_cv = [None]*n_cv
    batch_excit_cv = [None]*n_cv
    qy_logits = [None]*n_rec
    qz_alpha = [None]*n_rec
    qy_logits_e = [None]*n_rec
    qz_alpha_e = [None]*n_rec
    z = [None]*n_rec
    z_e = [None]*n_rec
    batch_pdf = [None]*n_rec
    batch_melsp_rec = [None]*n_rec
    batch_melsp_rec_fix = [None]*n_rec
    batch_melsp_rec_post = [None]*n_rec
    batch_magsp_rec = [None]*n_rec
    batch_magsp_rec_post = [None]*n_rec
    batch_feat_rec_sc = [None]*n_rec
    batch_feat_magsp_rec_sc = [None]*n_rec
    batch_feat_rec_sc_post = [None]*n_rec
    batch_feat_magsp_rec_sc_post = [None]*n_rec
    batch_pdf_cv = [None]*n_cv
    batch_melsp_cv = [None]*n_cv
    batch_melsp_cv_fix = [None]*n_cv
    batch_magsp_cv = [None]*n_cv
    batch_melsp_cv_post = [None]*n_cv
    batch_magsp_cv_post = [None]*n_cv
    batch_feat_cv_sc = [None]*n_cv
    batch_feat_magsp_cv_sc = [None]*n_cv
    batch_feat_cv_sc_post = [None]*n_cv
    batch_feat_magsp_cv_sc_post = [None]*n_cv
    batch_lf0_rec = [None]*n_rec
    batch_lf0_cv = [None]*n_cv
    h_z = [None]*n_rec
    h_z_e = [None]*n_rec
    h_spk = [None]*n_rec
    h_spk_cv = [None]*n_cv
    h_melsp = [None]*n_rec
    h_melsp_fix = [None]*n_rec
    h_melsp_post = [None]*n_rec
    h_lf0 = [None]*n_rec
    h_feat_sc = [None]*n_rec
    h_feat_magsp_sc = [None]*n_rec
    h_feat_sc_post = [None]*n_rec
    h_feat_magsp_sc_post = [None]*n_rec
    h_melsp_cv = [None]*n_cv
    h_melsp_cv_fix = [None]*n_cv
    h_melsp_cv_post = [None]*n_cv
    h_lf0_cv = [None]*n_cv
    h_feat_cv_sc = [None]*n_rec
    h_feat_magsp_cv_sc = [None]*n_rec
    h_feat_cv_sc_post = [None]*n_rec
    h_feat_magsp_cv_sc_post = [None]*n_rec
    loss_elbo = [None]*n_rec
    loss_px = [None]*n_rec
    loss_sc_feat = [None]*n_rec
    loss_sc_feat_cv = [None]*n_rec
    loss_sc_feat_post = [None]*n_rec
    loss_sc_feat_cv_post = [None]*n_rec
    loss_sc_feat_magsp = [None]*n_rec
    loss_sc_feat_magsp_cv = [None]*n_rec
    loss_sc_feat_magsp_post = [None]*n_rec
    loss_sc_feat_magsp_cv_post = [None]*n_rec
    loss_ms_norm = [None]*n_rec
    loss_ms_err = [None]*n_rec
    loss_ms_norm_post = [None]*n_rec
    loss_ms_err_post = [None]*n_rec
    loss_ms_norm_magsp = [None]*n_rec
    loss_ms_err_magsp = [None]*n_rec
    loss_ms_norm_magsp_post = [None]*n_rec
    loss_ms_err_magsp_post = [None]*n_rec
    loss_laplace = [None]*n_rec
    loss_melsp = [None]*n_rec
    loss_melsp_dB = [None]*n_rec
    loss_melsp_post = [None]*n_rec
    loss_melsp_dB_post = [None]*n_rec
    loss_magsp = [None]*n_rec
    loss_magsp_dB = [None]*n_rec
    loss_magsp_post = [None]*n_rec
    loss_magsp_dB_post = [None]*n_rec
    loss_laplace_cv = [None]*n_rec
    loss_melsp_cv = [None]*n_rec
    loss_melsp_cv_post = [None]*n_rec
    loss_magsp_cv = [None]*n_rec
    loss_magsp_cv_post = [None]*n_rec
    loss_melsp_dB_src_trg = []
    loss_melsp_dB_src_trg_post = []
    loss_lat_dist_rmse = []
    loss_lat_dist_cossim = []
    loss_sc_feat_in = []
    loss_sc_feat_magsp_in = []
    for i in range(n_rec):
        loss_elbo[i] = []
        loss_px[i] = []
        loss_laplace[i] = []
        loss_laplace_cv[i] = []
        loss_sc_feat[i] = []
        loss_sc_feat_cv[i] = []
        loss_sc_feat_post[i] = []
        loss_sc_feat_cv_post[i] = []
        loss_sc_feat_magsp[i] = []
        loss_sc_feat_magsp_cv[i] = []
        loss_sc_feat_magsp_post[i] = []
        loss_sc_feat_magsp_cv_post[i] = []
        loss_ms_norm[i] = []
        loss_ms_err[i] = []
        loss_ms_norm_post[i] = []
        loss_ms_err_post[i] = []
        loss_ms_norm_magsp[i] = []
        loss_ms_err_magsp[i] = []
        loss_ms_norm_magsp_post[i] = []
        loss_ms_err_magsp_post[i] = []
        loss_melsp[i] = []
        loss_melsp_cv[i] = []
        loss_melsp_dB[i] = []
        loss_melsp_post[i] = []
        loss_melsp_cv_post[i] = []
        loss_melsp_dB_post[i] = []
        loss_magsp[i] = []
        loss_magsp_cv[i] = []
        loss_magsp_dB[i] = []
        loss_magsp_post[i] = []
        loss_magsp_cv_post[i] = []
        loss_magsp_dB_post[i] = []
    batch_loss_laplace = [None]*n_rec
    batch_loss_melsp = [None]*n_rec
    batch_loss_melsp_post = [None]*n_rec
    batch_loss_magsp = [None]*n_rec
    batch_loss_magsp_post = [None]*n_rec
    batch_loss_sc_feat = [None]*n_rec
    batch_loss_sc_feat_post = [None]*n_rec
    batch_loss_sc_feat_magsp = [None]*n_rec
    batch_loss_sc_feat_magsp_post = [None]*n_rec
    batch_loss_melsp_dB = [None]*n_rec
    batch_loss_melsp_dB_post = [None]*n_rec
    batch_loss_magsp_dB = [None]*n_rec
    batch_loss_magsp_dB_post = [None]*n_rec
    batch_loss_ms_norm = [None]*n_rec
    batch_loss_ms_err = [None]*n_rec
    batch_loss_ms_norm_post = [None]*n_rec
    batch_loss_ms_err_post = [None]*n_rec
    batch_loss_ms_norm_magsp = [None]*n_rec
    batch_loss_ms_err_magsp = [None]*n_rec
    batch_loss_ms_norm_magsp_post = [None]*n_rec
    batch_loss_ms_err_magsp_post = [None]*n_rec
    batch_loss_laplace_cv = [None]*n_cv
    batch_loss_melsp_cv = [None]*n_cv
    batch_loss_melsp_cv_post = [None]*n_cv
    batch_loss_magsp_cv = [None]*n_cv
    batch_loss_magsp_cv_post = [None]*n_cv
    batch_loss_sc_feat_cv = [None]*n_cv
    batch_loss_sc_feat_cv_post = [None]*n_cv
    batch_loss_sc_feat_magsp_cv = [None]*n_cv
    batch_loss_sc_feat_magsp_cv_post = [None]*n_cv
    batch_loss_px = [None]*args.n_half_cyc
    batch_loss_elbo = [None]*n_rec
    n_half_cyc_eval = min(2,args.n_half_cyc)
    n_rec_eval = n_half_cyc_eval + n_half_cyc_eval%2
    n_cv_eval = int(n_half_cyc_eval/2+n_half_cyc_eval%2)
    first_pad_left_eval = (enc_pad_left + lf0_pad_left + dec_post_pad_left)*n_half_cyc_eval
    first_pad_right_eval = (enc_pad_right + lf0_pad_right + dec_post_pad_right)*n_half_cyc_eval
    if n_half_cyc_eval == 1:
        first_pad_left_eval += enc_pad_left
        first_pad_right_eval += enc_pad_right
    logging.info(f'first_pad_left_eval: {first_pad_left_eval}')
    logging.info(f'first_pad_right_eval: {first_pad_right_eval}')
    if n_half_cyc_eval > 1:
        outpad_lefts_eval = [None]*n_half_cyc_eval*4
        outpad_rights_eval = [None]*n_half_cyc_eval*4
    else:
        outpad_lefts_eval = [None]*(n_half_cyc_eval*4+1)
        outpad_rights_eval = [None]*(n_half_cyc_eval*4+1)
    outpad_lefts_eval[0] = first_pad_left_eval-enc_pad_left
    outpad_rights_eval[0] = first_pad_right_eval-enc_pad_right
    for i in range(1,n_half_cyc_eval*4):
        if i % 4 == 3:
            outpad_lefts_eval[i] = outpad_lefts_eval[i-1]-post_pad_left
            outpad_rights_eval[i] = outpad_rights_eval[i-1]-post_pad_right
        elif i % 4 == 2:
            outpad_lefts_eval[i] = outpad_lefts_eval[i-1]-dec_pad_left
            outpad_rights_eval[i] = outpad_rights_eval[i-1]-dec_pad_right
        elif i % 4 == 1:
            outpad_lefts_eval[i] = outpad_lefts_eval[i-1]-lf0_pad_left
            outpad_rights_eval[i] = outpad_rights_eval[i-1]-lf0_pad_right
        else:
            outpad_lefts_eval[i] = outpad_lefts_eval[i-1]-enc_pad_left
            outpad_rights_eval[i] = outpad_rights_eval[i-1]-enc_pad_right
    if n_half_cyc_eval == 1:
        outpad_lefts_eval[len(outpad_lefts_eval)-1] = outpad_lefts_eval[len(outpad_lefts_eval)-2]-enc_pad_left
        outpad_rights_eval[len(outpad_rights_eval)-1] = outpad_rights_eval[len(outpad_rights_eval)-2]-enc_pad_right
    logging.info(outpad_lefts_eval)
    logging.info(outpad_rights_eval)
    first_pad_left_eval_utt_dec = lf0_pad_left + dec_pad_left + post_pad_left
    first_pad_right_eval_utt_dec = lf0_pad_right + dec_pad_right + post_pad_right
    logging.info(f'first_pad_left_eval_utt_dec: {first_pad_left_eval_utt_dec}')
    logging.info(f'first_pad_right_eval_utt_dec: {first_pad_right_eval_utt_dec}')
    first_pad_left_eval_utt = enc_pad_left + first_pad_left_eval_utt_dec
    first_pad_right_eval_utt = enc_pad_right + first_pad_right_eval_utt_dec
    logging.info(f'first_pad_left_eval_utt: {first_pad_left_eval_utt}')
    logging.info(f'first_pad_right_eval_utt: {first_pad_right_eval_utt}')
    gv_src_src = [None]*n_spk
    gv_src_trg = [None]*n_spk
    gv_src_src_post = [None]*n_spk
    gv_src_trg_post = [None]*n_spk
    eval_loss_elbo = [None]*n_half_cyc_eval
    eval_loss_elbo_std = [None]*n_half_cyc_eval
    eval_loss_laplace = [None]*n_half_cyc_eval
    eval_loss_laplace_std = [None]*n_half_cyc_eval
    eval_loss_laplace_cv = [None]*n_cv_eval
    eval_loss_laplace_cv_std = [None]*n_cv_eval
    eval_loss_px = [None]*n_half_cyc_eval
    eval_loss_px_std = [None]*n_half_cyc_eval
    eval_loss_sc_feat = [None]*n_rec_eval
    eval_loss_sc_feat_std = [None]*n_rec_eval
    eval_loss_sc_feat_cv = [None]*n_cv_eval
    eval_loss_sc_feat_cv_std = [None]*n_cv_eval
    eval_loss_sc_feat_post = [None]*n_rec_eval
    eval_loss_sc_feat_post_std = [None]*n_rec_eval
    eval_loss_sc_feat_cv_post = [None]*n_cv_eval
    eval_loss_sc_feat_cv_post_std = [None]*n_cv_eval
    eval_loss_sc_feat_magsp = [None]*n_rec_eval
    eval_loss_sc_feat_magsp_std = [None]*n_rec_eval
    eval_loss_sc_feat_magsp_cv = [None]*n_cv_eval
    eval_loss_sc_feat_magsp_cv_std = [None]*n_cv_eval
    eval_loss_sc_feat_magsp_post = [None]*n_rec_eval
    eval_loss_sc_feat_magsp_post_std = [None]*n_rec_eval
    eval_loss_sc_feat_magsp_cv_post = [None]*n_cv_eval
    eval_loss_sc_feat_magsp_cv_post_std = [None]*n_cv_eval
    eval_loss_ms_norm = [None]*n_rec_eval
    eval_loss_ms_norm_std = [None]*n_rec_eval
    eval_loss_ms_err = [None]*n_rec_eval
    eval_loss_ms_err_std = [None]*n_rec_eval
    eval_loss_ms_norm_post = [None]*n_rec_eval
    eval_loss_ms_norm_post_std = [None]*n_rec_eval
    eval_loss_ms_err_post = [None]*n_rec_eval
    eval_loss_ms_err_post_std = [None]*n_rec_eval
    eval_loss_ms_norm_magsp = [None]*n_rec_eval
    eval_loss_ms_norm_magsp_std = [None]*n_rec_eval
    eval_loss_ms_err_magsp = [None]*n_rec_eval
    eval_loss_ms_err_magsp_std = [None]*n_rec_eval
    eval_loss_ms_norm_magsp_post = [None]*n_rec_eval
    eval_loss_ms_norm_magsp_post_std = [None]*n_rec_eval
    eval_loss_ms_err_magsp_post = [None]*n_rec_eval
    eval_loss_ms_err_magsp_post_std = [None]*n_rec_eval
    eval_loss_melsp = [None]*n_half_cyc_eval
    eval_loss_melsp_std = [None]*n_half_cyc_eval
    eval_loss_melsp_dB = [None]*n_half_cyc_eval
    eval_loss_melsp_dB_std = [None]*n_half_cyc_eval
    eval_loss_melsp_cv = [None]*n_cv_eval
    eval_loss_melsp_cv_std = [None]*n_cv_eval
    eval_loss_melsp_post = [None]*n_half_cyc_eval
    eval_loss_melsp_post_std = [None]*n_half_cyc_eval
    eval_loss_melsp_dB_post = [None]*n_half_cyc_eval
    eval_loss_melsp_dB_post_std = [None]*n_half_cyc_eval
    eval_loss_melsp_cv_post = [None]*n_cv_eval
    eval_loss_melsp_cv_post_std = [None]*n_cv_eval
    eval_loss_magsp = [None]*n_half_cyc_eval
    eval_loss_magsp_std = [None]*n_half_cyc_eval
    eval_loss_magsp_dB = [None]*n_half_cyc_eval
    eval_loss_magsp_dB_std = [None]*n_half_cyc_eval
    eval_loss_magsp_cv = [None]*n_cv_eval
    eval_loss_magsp_cv_std = [None]*n_cv_eval
    eval_loss_magsp_post = [None]*n_half_cyc_eval
    eval_loss_magsp_post_std = [None]*n_half_cyc_eval
    eval_loss_magsp_dB_post = [None]*n_half_cyc_eval
    eval_loss_magsp_dB_post_std = [None]*n_half_cyc_eval
    eval_loss_magsp_cv_post = [None]*n_cv_eval
    eval_loss_magsp_cv_post_std = [None]*n_cv_eval
    min_eval_loss_elbo = [None]*n_half_cyc_eval
    min_eval_loss_elbo_std = [None]*n_half_cyc_eval
    min_eval_loss_laplace = [None]*n_half_cyc_eval
    min_eval_loss_laplace_std = [None]*n_half_cyc_eval
    min_eval_loss_laplace_cv = [None]*n_cv_eval
    min_eval_loss_laplace_cv_std = [None]*n_cv_eval
    min_eval_loss_px = [None]*n_half_cyc_eval
    min_eval_loss_px_std = [None]*n_half_cyc_eval
    min_eval_loss_sc_feat = [None]*n_rec_eval
    min_eval_loss_sc_feat_std = [None]*n_rec_eval
    min_eval_loss_sc_feat_cv = [None]*n_cv_eval
    min_eval_loss_sc_feat_cv_std = [None]*n_cv_eval
    min_eval_loss_sc_feat_post = [None]*n_rec_eval
    min_eval_loss_sc_feat_post_std = [None]*n_rec_eval
    min_eval_loss_sc_feat_cv_post = [None]*n_cv_eval
    min_eval_loss_sc_feat_cv_post_std = [None]*n_cv_eval
    min_eval_loss_sc_feat_magsp = [None]*n_rec_eval
    min_eval_loss_sc_feat_magsp_std = [None]*n_rec_eval
    min_eval_loss_sc_feat_magsp_cv = [None]*n_cv_eval
    min_eval_loss_sc_feat_magsp_cv_std = [None]*n_cv_eval
    min_eval_loss_sc_feat_magsp_post = [None]*n_rec_eval
    min_eval_loss_sc_feat_magsp_post_std = [None]*n_rec_eval
    min_eval_loss_sc_feat_magsp_cv_post = [None]*n_cv_eval
    min_eval_loss_sc_feat_magsp_cv_post_std = [None]*n_cv_eval
    min_eval_loss_ms_norm = [None]*n_rec_eval
    min_eval_loss_ms_norm_std = [None]*n_rec_eval
    min_eval_loss_ms_err = [None]*n_rec_eval
    min_eval_loss_ms_err_std = [None]*n_rec_eval
    min_eval_loss_ms_norm_post = [None]*n_rec_eval
    min_eval_loss_ms_norm_post_std = [None]*n_rec_eval
    min_eval_loss_ms_err_post = [None]*n_rec_eval
    min_eval_loss_ms_err_post_std = [None]*n_rec_eval
    min_eval_loss_ms_norm_magsp = [None]*n_rec_eval
    min_eval_loss_ms_norm_magsp_std = [None]*n_rec_eval
    min_eval_loss_ms_err_magsp = [None]*n_rec_eval
    min_eval_loss_ms_err_magsp_std = [None]*n_rec_eval
    min_eval_loss_ms_norm_magsp_post = [None]*n_rec_eval
    min_eval_loss_ms_norm_magsp_post_std = [None]*n_rec_eval
    min_eval_loss_ms_err_magsp_post = [None]*n_rec_eval
    min_eval_loss_ms_err_magsp_post_std = [None]*n_rec_eval
    min_eval_loss_melsp = [None]*n_half_cyc_eval
    min_eval_loss_melsp_std = [None]*n_half_cyc_eval
    min_eval_loss_melsp_dB = [None]*n_half_cyc_eval
    min_eval_loss_melsp_dB_std = [None]*n_half_cyc_eval
    min_eval_loss_melsp_cv = [None]*n_cv_eval
    min_eval_loss_melsp_cv_std = [None]*n_cv_eval
    min_eval_loss_melsp_post = [None]*n_half_cyc_eval
    min_eval_loss_melsp_post_std = [None]*n_half_cyc_eval
    min_eval_loss_melsp_dB_post = [None]*n_half_cyc_eval
    min_eval_loss_melsp_dB_post_std = [None]*n_half_cyc_eval
    min_eval_loss_melsp_cv_post = [None]*n_cv_eval
    min_eval_loss_melsp_cv_post_std = [None]*n_cv_eval
    min_eval_loss_magsp = [None]*n_half_cyc_eval
    min_eval_loss_magsp_std = [None]*n_half_cyc_eval
    min_eval_loss_magsp_dB = [None]*n_half_cyc_eval
    min_eval_loss_magsp_dB_std = [None]*n_half_cyc_eval
    min_eval_loss_magsp_cv = [None]*n_cv_eval
    min_eval_loss_magsp_cv_std = [None]*n_cv_eval
    min_eval_loss_magsp_post = [None]*n_half_cyc_eval
    min_eval_loss_magsp_post_std = [None]*n_half_cyc_eval
    min_eval_loss_magsp_dB_post = [None]*n_half_cyc_eval
    min_eval_loss_magsp_dB_post_std = [None]*n_half_cyc_eval
    min_eval_loss_magsp_cv_post = [None]*n_cv_eval
    min_eval_loss_magsp_cv_post_std = [None]*n_cv_eval
    min_eval_loss_melsp_dB[0] = 99999999.99
    min_eval_loss_melsp_dB_std[0] = 99999999.99
    min_eval_loss_melsp_cv[0] = 99999999.99
    min_eval_loss_melsp[0] = 99999999.99
    min_eval_loss_gv_src_trg = 99999999.99
    min_eval_loss_melsp_dB_src_trg = 99999999.99
    min_eval_loss_melsp_dB_src_trg_std = 99999999.99
    min_eval_loss_melsp_dB_post[0] = 99999999.99
    min_eval_loss_melsp_dB_post_std[0] = 99999999.99
    min_eval_loss_melsp_cv_post[0] = 99999999.99
    min_eval_loss_melsp_post[0] = 99999999.99
    min_eval_loss_melsp_dB_src_trg_post = 99999999.99
    min_eval_loss_melsp_dB_src_trg_post_std = 99999999.99
    min_eval_loss_gv_src_trg_post = 99999999.99
    iter_idx = 0
    min_idx = -1
    change_min_flag = False
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        min_eval_loss_melsp_dB[0] = checkpoint["min_eval_loss_melsp_dB"]
        min_eval_loss_melsp_dB_std[0] = checkpoint["min_eval_loss_melsp_dB_std"]
        min_eval_loss_melsp_cv[0] = checkpoint["min_eval_loss_melsp_cv"]
        min_eval_loss_melsp[0] = checkpoint["min_eval_loss_melsp"]
        min_eval_loss_melsp_dB_src_trg = checkpoint["min_eval_loss_melsp_dB_src_trg"]
        min_eval_loss_melsp_dB_src_trg_std = checkpoint["min_eval_loss_melsp_dB_src_trg_std"]
        min_eval_loss_gv_src_trg = checkpoint["min_eval_loss_gv_src_trg"]
        min_eval_loss_melsp_dB_post[0] = checkpoint["min_eval_loss_melsp_dB_post"]
        min_eval_loss_melsp_dB_post_std[0] = checkpoint["min_eval_loss_melsp_dB_post_std"]
        min_eval_loss_melsp_cv_post[0] = checkpoint["min_eval_loss_melsp_cv_post"]
        min_eval_loss_melsp_post[0] = checkpoint["min_eval_loss_melsp_post"]
        min_eval_loss_melsp_dB_src_trg_post = checkpoint["min_eval_loss_melsp_dB_src_trg_post"]
        min_eval_loss_melsp_dB_src_trg_post_std = checkpoint["min_eval_loss_melsp_dB_src_trg_post_std"]
        min_eval_loss_gv_src_trg_post = checkpoint["min_eval_loss_gv_src_trg_post"]
        iter_idx = checkpoint["iter_idx"]
        min_idx = checkpoint["min_idx"]
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        batch_feat, batch_feat_magsp, batch_sc, batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
            f_bs, f_ss, flens, n_batch_utt, del_index_utt, max_flen, spk_cv, idx_select, idx_select_full, flens_acc = next(generator)
        if c_idx < 0: # summarize epoch
            # save current epoch model
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # report current epoch
            text_log = "(EPOCH:%d) average optimization loss = %.6f (+- %.6f) ; " % (epoch_idx + 1, np.mean(loss_sc_feat_in), np.std(loss_sc_feat_in))
            for i in range(args.n_half_cyc):
                if i % 2 == 0:
                    text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB ;; " % (
                            i+1, np.mean(loss_elbo[i]), np.std(loss_elbo[i]), np.mean(loss_laplace[i]), np.std(loss_laplace[i]),
                                np.mean(loss_laplace_cv[i//2]), np.std(loss_laplace_cv[i//2]), np.mean(loss_px[i]), np.std(loss_px[i]),
                            np.mean(loss_ms_norm[i]), np.std(loss_ms_norm[i]), np.mean(loss_ms_err[i]), np.std(loss_ms_err[i]),
                            np.mean(loss_ms_norm_post[i]), np.std(loss_ms_norm_post[i]), np.mean(loss_ms_err_post[i]), np.std(loss_ms_err_post[i]),
                            np.mean(loss_ms_norm_magsp[i]), np.std(loss_ms_norm_magsp[i]), np.mean(loss_ms_err_magsp[i]), np.std(loss_ms_err_magsp[i]),
                            np.mean(loss_ms_norm_magsp_post[i]), np.std(loss_ms_norm_magsp_post[i]), np.mean(loss_ms_err_magsp_post[i]), np.std(loss_ms_err_magsp_post[i]),
                            np.mean(loss_sc_feat[i]), np.std(loss_sc_feat[i]), np.mean(loss_sc_feat_cv[i//2]), np.std(loss_sc_feat_cv[i//2]),
                            np.mean(loss_sc_feat_post[i]), np.std(loss_sc_feat_post[i]), np.mean(loss_sc_feat_cv_post[i//2]), np.std(loss_sc_feat_cv_post[i//2]),
                            np.mean(loss_sc_feat_magsp[i]), np.std(loss_sc_feat_magsp[i]), np.mean(loss_sc_feat_magsp_cv[i//2]), np.std(loss_sc_feat_magsp_cv[i//2]),
                            np.mean(loss_sc_feat_magsp_post[i]), np.std(loss_sc_feat_magsp_post[i]), np.mean(loss_sc_feat_magsp_cv_post[i//2]), np.std(loss_sc_feat_magsp_cv_post[i//2]),
                            np.mean(loss_melsp[i]), np.std(loss_melsp[i]), np.mean(loss_melsp_cv[i//2]), np.std(loss_melsp_cv[i//2]), np.mean(loss_melsp_dB[i]), np.std(loss_melsp_dB[i]),
                            np.mean(loss_melsp_post[i]), np.std(loss_melsp_post[i]), np.mean(loss_melsp_cv_post[i//2]), np.std(loss_melsp_cv_post[i//2]), np.mean(loss_melsp_dB_post[i]), np.std(loss_melsp_dB_post[i]),
                            np.mean(loss_magsp[i]), np.std(loss_magsp[i]), np.mean(loss_magsp_cv[i//2]), np.std(loss_magsp_cv[i//2]), np.mean(loss_magsp_dB[i]), np.std(loss_magsp_dB[i]),
                            np.mean(loss_magsp_post[i]), np.std(loss_magsp_post[i]), np.mean(loss_magsp_cv_post[i//2]), np.std(loss_magsp_cv_post[i//2]), np.mean(loss_magsp_dB_post[i]), np.std(loss_magsp_dB_post[i]))
                else:
                    text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ;; " % (i+1,
                            np.mean(loss_elbo[i]), np.std(loss_elbo[i]), np.mean(loss_laplace[i]), np.std(loss_laplace[i]), np.mean(loss_px[i]), np.std(loss_px[i]),
                            np.mean(loss_ms_norm[i]), np.std(loss_ms_norm[i]), np.mean(loss_ms_err[i]), np.std(loss_ms_err[i]),
                            np.mean(loss_ms_norm_post[i]), np.std(loss_ms_norm_post[i]), np.mean(loss_ms_err_post[i]), np.std(loss_ms_err_post[i]),
                            np.mean(loss_ms_norm_magsp[i]), np.std(loss_ms_norm_magsp[i]), np.mean(loss_ms_err_magsp[i]), np.std(loss_ms_err_magsp[i]),
                            np.mean(loss_ms_norm_magsp_post[i]), np.std(loss_ms_norm_magsp_post[i]), np.mean(loss_ms_err_magsp_post[i]), np.std(loss_ms_err_magsp_post[i]),
                            np.mean(loss_sc_feat[i]), np.std(loss_sc_feat[i]), np.mean(loss_sc_feat_post[i]), np.std(loss_sc_feat_post[i]), 
                            np.mean(loss_sc_feat_magsp[i]), np.std(loss_sc_feat_magsp[i]), np.mean(loss_sc_feat_magsp_post[i]), np.std(loss_sc_feat_magsp_post[i]),
                            np.mean(loss_melsp[i]), np.std(loss_melsp[i]), np.mean(loss_melsp_dB[i]), np.std(loss_melsp_dB[i]),
                            np.mean(loss_melsp_post[i]), np.std(loss_melsp_post[i]), np.mean(loss_melsp_dB_post[i]), np.std(loss_melsp_dB_post[i]),
                            np.mean(loss_magsp[i]), np.std(loss_magsp[i]), np.mean(loss_magsp_dB[i]), np.std(loss_magsp_dB[i]),
                            np.mean(loss_magsp_post[i]), np.std(loss_magsp_post[i]), np.mean(loss_magsp_dB_post[i]), np.std(loss_magsp_dB_post[i]))
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            logging.info("estimated time until max. epoch = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * total))))
            # compute loss in evaluation data
            total = 0
            iter_count = 0
            for i in range(n_spk):
                gv_src_src[i] = []
                gv_src_trg[i] = []
                gv_src_src_post[i] = []
                gv_src_trg_post[i] = []
            loss_melsp_dB_src_trg = []
            loss_melsp_dB_src_trg_post = []
            loss_lat_dist_rmse = []
            loss_lat_dist_cossim = []
            loss_sc_feat_in = []
            loss_sc_feat_magsp_in = []
            for i in range(args.n_half_cyc):
                loss_elbo[i] = []
                loss_px[i] = []
                loss_laplace[i] = []
                loss_laplace_cv[i] = []
                loss_sc_feat[i] = []
                loss_sc_feat_cv[i] = []
                loss_sc_feat_post[i] = []
                loss_sc_feat_cv_post[i] = []
                loss_sc_feat_magsp[i] = []
                loss_sc_feat_magsp_cv[i] = []
                loss_sc_feat_magsp_post[i] = []
                loss_sc_feat_magsp_cv_post[i] = []
                loss_ms_norm[i] = []
                loss_ms_err[i] = []
                loss_ms_norm_post[i] = []
                loss_ms_err_post[i] = []
                loss_ms_norm_magsp[i] = []
                loss_ms_err_magsp[i] = []
                loss_ms_norm_magsp_post[i] = []
                loss_ms_err_magsp_post[i] = []
                loss_melsp[i] = []
                loss_melsp_cv[i] = []
                loss_melsp_dB[i] = []
                loss_melsp_post[i] = []
                loss_melsp_cv_post[i] = []
                loss_melsp_dB_post[i] = []
                loss_magsp[i] = []
                loss_magsp_cv[i] = []
                loss_magsp_dB[i] = []
                loss_magsp_post[i] = []
                loss_magsp_cv_post[i] = []
                loss_magsp_dB_post[i] = []
            model_encoder_melsp.eval()
            model_decoder_melsp_fix.eval()
            model_decoder_melsp.eval()
            model_encoder_excit.eval()
            model_decoder_excit.eval()
            model_spk.eval()
            model_post.eval()
            model_classifier.eval()
            if args.spkidtr_dim > 0:
                model_spkidtr.eval()
            for param in model_decoder_melsp.parameters():
                param.requires_grad = False
            for param in model_post.parameters():
                param.requires_grad = False
            for param in model_classifier.parameters():
                param.requires_grad = False
            pair_exist = False
            logging.info("Evaluation data")
            while True:
                with torch.no_grad():
                    start = time.time()
                    batch_feat_data, batch_feat_magsp_data, batch_feat_trg_data, batch_sc_data, \
                        batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
                        f_bs, f_ss, flens, n_batch_utt, del_index_utt, max_flen, spk_cv, src_trg_flag, \
                            spcidx_src, spcidx_src_trg, flens_spc_src, flens_spc_src_trg, \
                                batch_feat_data_full, batch_sc_data_full, batch_sc_cv_data_full, \
                                    idx_select, idx_select_full = next(generator_eval)
                    if c_idx < 0:
                        break

                    f_es = f_ss+f_bs
                    logging.info(f'{f_ss} {f_bs} {f_es} {max_flen}')
                    # handle first pad for input on melsp flow
                    flag_cv = True
                    i_cv = 0
                    i_cv_in = 0
                    f_ss_first_pad_left = f_ss-first_pad_left_eval
                    f_es_first_pad_right = f_es+first_pad_right_eval
                    i_end = n_half_cyc_eval*4
                    if n_half_cyc_eval == 1:
                        i_end += 1
                    for i in range(i_end):
                        if i % 4 == 0: #enc
                            if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                                batch_feat_in[i] = batch_feat_data[:,f_ss_first_pad_left:f_es_first_pad_right,args.full_excit_dim:]
                            elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                                batch_feat_in[i] = F.pad(batch_feat_data[:,:f_es_first_pad_right,args.full_excit_dim:].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
                            elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                                batch_feat_in[i] = F.pad(batch_feat_data[:,f_ss_first_pad_left:max_flen,args.full_excit_dim:].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                            else: # pad left and right need additional replicate
                                batch_feat_in[i] = F.pad(batch_feat_data[:,:max_flen,args.full_excit_dim:].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                            f_ss_first_pad_left += enc_pad_left
                            f_es_first_pad_right -= enc_pad_right
                        else: #dec/post
                            if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                                batch_sc_in[i] = batch_sc_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                                if flag_cv:
                                    batch_sc_cv_in[i_cv_in] = batch_sc_cv_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                                    i_cv_in += 1
                                    if i % 4 == 3:
                                        i_cv += 1
                                        flag_cv = False
                                else:
                                    if (i + 1) % 8 == 0:
                                        flag_cv = True
                            elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                                batch_sc_in[i] = F.pad(batch_sc_data[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                                    i_cv_in += 1
                                    if i % 4 == 3:
                                        i_cv += 1
                                        flag_cv = False
                                else:
                                    if (i + 1) % 8 == 0:
                                        flag_cv = True
                            elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                                diff_pad = f_es_first_pad_right - max_flen
                                batch_sc_in[i] = F.pad(batch_sc_data[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                                    i_cv_in += 1
                                    if i % 4 == 3:
                                        i_cv += 1
                                        flag_cv = False
                                else:
                                    if (i + 1) % 8 == 0:
                                        flag_cv = True
                            else: # pad left and right need additional replicate
                                diff_pad = f_es_first_pad_right - max_flen
                                batch_sc_in[i] = F.pad(batch_sc_data[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                                    i_cv_in += 1
                                    if i % 4 == 3:
                                        i_cv += 1
                                        flag_cv = False
                                else:
                                    if (i + 1) % 8 == 0:
                                        flag_cv = True
                            if i % 4 == 1:
                                f_ss_first_pad_left += lf0_pad_left
                                f_es_first_pad_right -= lf0_pad_right
                            elif i % 4 == 2:
                                f_ss_first_pad_left += dec_pad_left
                                f_es_first_pad_right -= dec_pad_right
                            else:
                                f_ss_first_pad_left += post_pad_left
                                f_es_first_pad_right -= post_pad_right
                    batch_melsp = batch_feat_data[:,f_ss:f_es,args.full_excit_dim:]
                    batch_magsp = batch_feat_magsp_data[:,f_ss:f_es]
                    batch_excit = batch_feat_data[:,f_ss:f_es,:args.full_excit_dim]
                    batch_melsp_data_full = batch_feat_data_full[:,:,args.full_excit_dim:]
                    batch_melsp_trg_data = batch_feat_trg_data[:,:,args.full_excit_dim:]
                    batch_excit_trg_data = batch_feat_trg_data[:,:,:args.full_excit_dim]
                    batch_sc = batch_sc_data[:,f_ss:f_es]
                    batch_sc_cv[0] = batch_sc_cv_data[:,f_ss:f_es]
                    batch_excit_cv[0] = batch_feat_cv_data[:,f_ss:f_es]

                    if f_ss > 0:
                        idx_in = 0
                        i_cv_in = 0
                        for i in range(0,n_half_cyc_eval,2):
                            i_cv = i//2
                            j = i+1
                            if len(del_index_utt) > 0:
                                if i == 0:
                                    h_feat_in_sc = torch.FloatTensor(np.delete(h_feat_in_sc.cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_feat_magsp_in_sc = torch.FloatTensor(np.delete(h_feat_magsp_in_sc.cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                h_z[i] = torch.FloatTensor(np.delete(h_z[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z_e[i] = torch.FloatTensor(np.delete(h_z_e[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_spk[i] = torch.FloatTensor(np.delete(h_spk[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_spk_cv[i_cv] = torch.FloatTensor(np.delete(h_spk_cv[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_lf0[i] = torch.FloatTensor(np.delete(h_lf0[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_lf0_cv[i_cv] = torch.FloatTensor(np.delete(h_lf0_cv[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp[i] = torch.FloatTensor(np.delete(h_melsp[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp_cv[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp_cv_fix[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv_fix[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp_post[i] = torch.FloatTensor(np.delete(h_melsp_post[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp_cv_post[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv_post[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_sc[i] = torch.FloatTensor(np.delete(h_feat_sc[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_magsp_sc[i] = torch.FloatTensor(np.delete(h_feat_magsp_sc[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_feat_cv_sc[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_magsp_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_feat_magsp_cv_sc[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_sc_post[i] = torch.FloatTensor(np.delete(h_feat_sc_post[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_magsp_sc_post[i] = torch.FloatTensor(np.delete(h_feat_magsp_sc_post[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_cv_sc_post[i_cv] = torch.FloatTensor(np.delete(h_feat_cv_sc_post[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_magsp_cv_sc_post[i_cv] = torch.FloatTensor(np.delete(h_feat_magsp_cv_sc_post[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z[j] = torch.FloatTensor(np.delete(h_z[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z_e[j] = torch.FloatTensor(np.delete(h_z_e[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                if n_half_cyc_eval > 1:
                                    h_spk[j] = torch.FloatTensor(np.delete(h_spk[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_lf0[j] = torch.FloatTensor(np.delete(h_lf0[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_melsp[j] = torch.FloatTensor(np.delete(h_melsp[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_melsp_fix[j] = torch.FloatTensor(np.delete(h_melsp_fix[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_melsp_post[j] = torch.FloatTensor(np.delete(h_melsp_post[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_feat_sc[j] = torch.FloatTensor(np.delete(h_feat_sc[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_feat_magsp_sc[j] = torch.FloatTensor(np.delete(h_feat_magsp_sc[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_feat_sc_post[j] = torch.FloatTensor(np.delete(h_feat_sc_post[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_feat_magsp_sc_post[j] = torch.FloatTensor(np.delete(h_feat_magsp_sc_post[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                            if i > 0:
                                idx_in += 1
                                i_cv_in += 1
                                i_1 = i-1
                                if post_pad_right > 0:
                                    cyc_rec_feat = batch_melsp_rec_fix[i_1][:,post_pad_left:-post_pad_right]
                                else:
                                    cyc_rec_feat = batch_melsp_rec_fix[i_1][:,post_pad_left:]
                                qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(cyc_rec_feat, outpad_right=outpad_rights[idx_in], h=h_z[i], sampling=False)
                                qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(cyc_rec_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[i], sampling=False)
                                idx_in_1 = idx_in-1
                                batch_melsp_rec_post[i_1] = batch_melsp_rec_post[i_1][:,outpad_lefts[idx_in_1]:batch_melsp_rec_post[i_1].shape[1]-outpad_rights[idx_in_1]]
                                batch_magsp_rec_post[i_1] = torch.matmul((torch.exp(batch_melsp_rec_post[i_1])-1)/10000, melfb_t)
                                batch_feat_rec_sc_post[i_1], h_feat_sc_post[i_1] = model_classifier(feat=batch_melsp_rec_post[i_1], h=h_feat_sc_post[i_1])
                                batch_feat_magsp_rec_sc_post[i_1], h_feat_magsp_sc_post[i_1] = model_classifier(feat_aux=batch_magsp_rec_post[i_1], h=h_feat_magsp_sc_post[i_1])
                            else:
                                qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z[i], sampling=False)
                                qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_e[i], sampling=False)
                                batch_feat_in_sc, h_feat_in_sc = model_classifier(feat=batch_melsp, h=h_feat_in_sc)
                                batch_feat_magsp_in_sc, h_feat_magsp_in_sc = model_classifier(feat_aux=batch_magsp, h=h_feat_magsp_in_sc)
                            ## reconstruction and conversion
                            idx_in += 1
                            z_cat = torch.cat((z_e[i], z[i]), 2)
                            if args.spkidtr_dim > 0:
                                spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                                spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                                batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv])
                                batch_lf0_rec[i], h_lf0[i] \
                                        = model_decoder_excit(z_e[i], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[i])
                                batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                        = model_decoder_excit(z_e[i], y=spk_cv_code_in, aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], h=h_lf0_cv[i_cv])
                            else:
                                batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv])
                                batch_lf0_rec[i], h_lf0[i] \
                                        = model_decoder_excit(z_e[i], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[i])
                                batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                        = model_decoder_excit(z_e[i], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], h=h_lf0_cv[i_cv])
                            if lf0_pad_right > 0:
                                z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,lf0_pad_left:-lf0_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,lf0_pad_left:-lf0_pad_right]
                                batch_spk = batch_spk[:,lf0_pad_left:-lf0_pad_right]
                                batch_spk_cv = batch_spk_cv[:,lf0_pad_left:-lf0_pad_right]
                            else:
                                z_cat = z_cat[:,lf0_pad_left:]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,lf0_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,lf0_pad_left:]
                                batch_spk = batch_spk[:,lf0_pad_left:]
                                batch_spk_cv = batch_spk_cv[:,lf0_pad_left:]
                            idx_in += 1
                            i_cv_in += 1
                            if args.spkidtr_dim > 0:
                                batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                    e=batch_lf0_rec[i][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp[i])
                                batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                                    e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv])
                                batch_melsp_cv_fix[i_cv], h_melsp_cv_fix[i_cv] = model_decoder_melsp_fix(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                                    e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_cv_fix[i_cv])
                            else:
                                batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=batch_lf0_rec[i][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp[i])
                                batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv])
                                batch_melsp_cv_fix[i_cv], h_melsp_cv_fix[i_cv] = model_decoder_melsp_fix(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_cv_fix[i_cv])
                            if dec_pad_right > 0:
                                e_post = batch_lf0_rec[i][:,dec_pad_left:-dec_pad_right]
                                e_cv_post = batch_lf0_cv[i_cv][:,dec_pad_left:-dec_pad_right]
                            else:
                                e_post = batch_lf0_rec[i][:,dec_pad_left:]
                                e_cv_post = batch_lf0_cv[i_cv][:,dec_pad_left:]
                            idx_in_1 = idx_in-1
                            feat_len_e = batch_lf0_rec[i].shape[1]
                            batch_lf0_rec[i] = batch_lf0_rec[i][:,outpad_lefts[idx_in_1]:feat_len_e-outpad_rights[idx_in_1]]
                            batch_lf0_cv[i_cv] = batch_lf0_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len_e-outpad_rights[idx_in_1]]
                            ## post
                            idx_in += 1
                            i_cv_in += 1
                            if dec_pad_right > 0:
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,dec_pad_left:-dec_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,dec_pad_left:-dec_pad_right]
                                batch_spk = batch_spk[:,dec_pad_left:-dec_pad_right]
                                batch_spk_cv = batch_spk_cv[:,dec_pad_left:-dec_pad_right]
                            else:
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,dec_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,dec_pad_left:]
                                batch_spk = batch_spk[:,dec_pad_left:]
                                batch_spk_cv = batch_spk_cv[:,dec_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_pdf[i], batch_melsp_rec_post[i], h_melsp_post[i] \
                                                = model_post(batch_melsp_rec[i], y=spk_code_in, aux=batch_spk,
                                                    e=e_post, outpad_right=outpad_rights[idx_in], h=h_melsp_post[i])
                                batch_pdf_cv[i_cv], batch_melsp_cv_post[i_cv], h_melsp_cv_post[i_cv] \
                                                = model_post(batch_melsp_cv[i_cv], y=spk_cv_code_in, aux=batch_spk_cv,
                                                    e=e_cv_post, outpad_right=outpad_rights[idx_in], h=h_melsp_cv_post[i_cv])
                            else:
                                batch_pdf[i], batch_melsp_rec_post[i], h_melsp_post[i] \
                                                = model_post(batch_melsp_rec[i], y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=e_post, outpad_right=outpad_rights[idx_in], h=h_melsp_post[i])
                                batch_pdf_cv[i_cv], batch_melsp_cv_post[i_cv], h_melsp_cv_post[i_cv] \
                                                = model_post(batch_melsp_cv[i_cv], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    e=e_cv_post, outpad_right=outpad_rights[idx_in], h=h_melsp_cv_post[i_cv])
                            idx_in_1 = idx_in-1
                            feat_len = batch_melsp_rec[i].shape[1]
                            batch_melsp_rec[i] = batch_melsp_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_rec[i] = torch.matmul((torch.exp(batch_melsp_rec[i])-1)/10000, melfb_t)
                            batch_melsp_cv[i_cv] = batch_melsp_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_cv[i_cv] = torch.matmul((torch.exp(batch_melsp_cv[i_cv])-1)/10000, melfb_t)
                            batch_feat_rec_sc[i], h_feat_sc[i] = model_classifier(feat=batch_melsp_rec[i], h=h_feat_sc[i])
                            batch_feat_magsp_rec_sc[i], h_feat_magsp_sc[i] = model_classifier(feat_aux=batch_magsp_rec[i], h=h_feat_magsp_sc[i])
                            batch_feat_cv_sc[i_cv], h_feat_cv_sc[i_cv] = model_classifier(feat=batch_melsp_cv[i_cv], h=h_feat_cv_sc[i_cv])
                            batch_feat_magsp_cv_sc[i_cv], h_feat_magsp_cv_sc[i_cv] = model_classifier(feat_aux=batch_magsp_cv[i_cv], h=h_feat_magsp_cv_sc[i_cv])
                            ## cyclic reconstruction
                            idx_in += 1
                            if post_pad_right > 0:
                                cv_feat = batch_melsp_cv_fix[i_cv][:,post_pad_left:-post_pad_right]
                            else:
                                cv_feat = batch_melsp_cv_fix[i_cv][:,post_pad_left:]
                            qy_logits[j], qz_alpha[j], z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z[j], sampling=False)
                            qy_logits_e[j], qz_alpha_e[j], z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[j], sampling=False)
                            feat_len = batch_melsp_rec[i].shape[1]
                            idx_in_1 = idx_in-1
                            feat_len = batch_melsp_rec_post[i].shape[1]
                            batch_pdf[i] = batch_pdf[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_rec_post[i] = batch_melsp_rec_post[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_rec_post[i] = torch.matmul((torch.exp(batch_melsp_rec_post[i])-1)/10000, melfb_t)
                            batch_pdf_cv[i_cv] = batch_pdf_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_cv_post[i_cv] = batch_melsp_cv_post[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_cv_post[i_cv] = torch.matmul((torch.exp(batch_melsp_cv_post[i_cv])-1)/10000, melfb_t)
                            batch_feat_rec_sc_post[i], h_feat_sc_post[i] = model_classifier(feat=batch_melsp_rec_post[i], h=h_feat_sc_post[i])
                            batch_feat_magsp_rec_sc_post[i], h_feat_magsp_sc_post[i] = model_classifier(feat_aux=batch_magsp_rec_post[i], h=h_feat_magsp_sc_post[i])
                            batch_feat_cv_sc_post[i_cv], h_feat_cv_sc_post[i_cv] = model_classifier(feat=batch_melsp_cv_post[i_cv], h=h_feat_cv_sc_post[i_cv])
                            batch_feat_magsp_cv_sc_post[i_cv], h_feat_magsp_cv_sc_post[i_cv] = model_classifier(feat_aux=batch_magsp_cv_post[i_cv], h=h_feat_magsp_cv_sc_post[i_cv])
                            if n_half_cyc_eval > 1:
                                idx_in += 1
                                z_cat = torch.cat((z_e[j], z[j]), 2)
                                if args.spkidtr_dim > 0:
                                    if enc_pad_right > 0:
                                        spk_code_in = spk_code_in[:,enc_pad_left:-enc_pad_right]
                                    else:
                                        spk_code_in = spk_code_in[:,enc_pad_left:]
                                    batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j])
                                    batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[j])
                                else:
                                    batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j])
                                    batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[j])
                                if lf0_pad_right > 0:
                                    z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,lf0_pad_left:-lf0_pad_right]
                                    batch_spk = batch_spk[:,lf0_pad_left:-lf0_pad_right]
                                else:
                                    z_cat = z_cat[:,lf0_pad_left:]
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,lf0_pad_left:]
                                    batch_spk = batch_spk[:,lf0_pad_left:]
                                idx_in += 1
                                if args.spkidtr_dim > 0:
                                    batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                            e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp[j])
                                    batch_melsp_rec_fix[j], h_melsp_fix[j] = model_decoder_melsp_fix(z_cat, y=spk_code_in, aux=batch_spk,
                                                            e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_fix[j])
                                else:
                                    batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                            e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp[j])
                                    batch_melsp_rec_fix[j], h_melsp_fix[j] = model_decoder_melsp_fix(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                            e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_fix[j])
                                if dec_pad_right > 0:
                                    e_post = batch_lf0_rec[j][:,dec_pad_left:-dec_pad_right]
                                else:
                                    e_post = batch_lf0_rec[j][:,dec_pad_left:]
                                idx_in_1 = idx_in-1
                                batch_lf0_rec[j] = batch_lf0_rec[j][:,outpad_lefts[idx_in_1]:batch_lf0_rec[j].shape[1]-outpad_rights[idx_in_1]]
                                ## post
                                idx_in += 1
                                if dec_pad_right > 0:
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,dec_pad_left:-dec_pad_right]
                                    batch_spk = batch_spk[:,dec_pad_left:-dec_pad_right]
                                else:
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,dec_pad_left:]
                                    batch_spk = batch_spk[:,dec_pad_left:]
                                if args.spkidtr_dim > 0:
                                    batch_pdf[j], batch_melsp_rec_post[j], h_melsp_post[j] \
                                                    = model_post(batch_melsp_rec[j], y=spk_code_in, aux=batch_spk,
                                                        e=e_post, outpad_right=outpad_rights[idx_in], h=h_melsp_post[j])
                                else:
                                    batch_pdf[j], batch_melsp_rec_post[j], h_melsp_post[j] \
                                                    = model_post(batch_melsp_rec[j], y=batch_sc_in[idx_in], aux=batch_spk,
                                                        e=e_post, outpad_right=outpad_rights[idx_in], h=h_melsp_post[j])
                                idx_in_1 = idx_in-1
                                batch_melsp_rec[j] = batch_melsp_rec[j][:,outpad_lefts[idx_in_1]:batch_melsp_rec[j].shape[1]-outpad_rights[idx_in_1]]
                                batch_magsp_rec[j] = torch.matmul((torch.exp(batch_melsp_rec[j])-1)/10000, melfb_t)
                                batch_feat_rec_sc[j], h_feat_sc[j] = model_classifier(feat=batch_melsp_rec[j], h=h_feat_sc[j])
                                batch_feat_magsp_rec_sc[j], h_feat_magsp_sc[j] = model_classifier(feat_aux=batch_magsp_rec[j], h=h_feat_magsp_sc[j])
                                feat_len = batch_pdf[j].shape[1]
                                batch_pdf[j] = batch_pdf[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                                batch_melsp_rec_post[j] = batch_melsp_rec_post[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                                batch_magsp_rec_post[j] = torch.matmul((torch.exp(batch_melsp_rec_post[j])-1)/10000, melfb_t)
                                batch_feat_rec_sc_post[j], h_feat_sc_post[j] = model_classifier(feat=batch_melsp_rec_post[j], h=h_feat_sc_post[j])
                                batch_feat_magsp_rec_sc_post[j], h_feat_magsp_sc_post[j] = model_classifier(feat_aux=batch_magsp_rec_post[j], h=h_feat_magsp_sc_post[j])
                    else:
                        pair_flag = False
                        for k in range(n_batch_utt):
                            if src_trg_flag[k]:
                                pair_flag = True
                                pair_exist = True
                                break
                        batch_melsp_data_full = F.pad(batch_melsp_data_full.transpose(1,2), (first_pad_left_eval_utt,first_pad_right_eval_utt), "replicate").transpose(1,2)
                        _, _, trj_lat_src, _ = model_encoder_melsp(batch_melsp_data_full, sampling=False)
                        _, _, trj_lat_src_e, _ = model_encoder_excit(batch_melsp_data_full, sampling=False)
                        batch_sc_data_full = F.pad(batch_sc_data_full.unsqueeze(1).float(), (first_pad_left_eval_utt_dec,first_pad_right_eval_utt_dec), "replicate").squeeze(1).long()
                        batch_sc_cv_data_full = F.pad(batch_sc_cv_data_full.unsqueeze(1).float(), (first_pad_left_eval_utt_dec,first_pad_right_eval_utt_dec), "replicate").squeeze(1).long()
                        z_cat = torch.cat((trj_lat_src_e, trj_lat_src), 2)
                        if args.spkidtr_dim > 0:
                            trj_spk_code = model_spkidtr(batch_sc_data_full)
                            trj_spk_cv_code = model_spkidtr(batch_sc_cv_data_full)
                            trj_spk, _ = model_spk(trj_spk_code, z=z_cat)
                            trj_spk_cv, _ = model_spk(trj_spk_cv_code, z=z_cat)
                        else:
                            trj_spk_code = batch_sc_data_full
                            trj_spk_cv_code = batch_sc_cv_data_full
                            trj_spk, _ = model_spk(batch_sc_data_full, z=z_cat)
                            trj_spk_cv, _ = model_spk(batch_sc_cv_data_full, z=z_cat)
                        trj_src_src_uvlf0, _ = model_decoder_excit(trj_lat_src_e, y=trj_spk_code, aux=trj_spk)
                        trj_src_trg_uvlf0, _ = model_decoder_excit(trj_lat_src_e, y=trj_spk_cv_code, aux=trj_spk_cv)
                        if lf0_pad_right > 0:
                            z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                            trj_spk_code = trj_spk_code[:,lf0_pad_left:-lf0_pad_right]
                            trj_spk_cv_code = trj_spk_cv_code[:,lf0_pad_left:-lf0_pad_right]
                            trj_spk = trj_spk[:,lf0_pad_left:-lf0_pad_right]
                            trj_spk_cv = trj_spk_cv[:,lf0_pad_left:-lf0_pad_right]
                        else:
                            z_cat = z_cat[:,lf0_pad_left:]
                            trj_spk_code = trj_spk_code[:,lf0_pad_left:]
                            trj_spk_cv_code = trj_spk_cv_code[:,lf0_pad_left:]
                            trj_spk = trj_spk[:,lf0_pad_left:]
                            trj_spk_cv = trj_spk_cv[:,lf0_pad_left:]
                        trj_src_src, _ = model_decoder_melsp(z_cat, y=trj_spk_code, aux=trj_spk, e=trj_src_src_uvlf0[:,:,:args.excit_dim])
                        trj_src_trg, _ = model_decoder_melsp(z_cat, y=trj_spk_cv_code, aux=trj_spk_cv, e=trj_src_trg_uvlf0[:,:,:args.excit_dim])
                        if dec_pad_right > 0:
                            trj_spk_code = trj_spk_code[:,dec_pad_left:-dec_pad_right]
                            trj_spk_cv_code = trj_spk_cv_code[:,dec_pad_left:-dec_pad_right]
                            trj_spk = trj_spk[:,dec_pad_left:-dec_pad_right]
                            trj_spk_cv = trj_spk_cv[:,dec_pad_left:-dec_pad_right]
                            e_post = trj_src_src_uvlf0[:,dec_pad_left:-dec_pad_right]
                            e_cv_post = trj_src_trg_uvlf0[:,dec_pad_left:-dec_pad_right]
                        else:
                            trj_spk_code = trj_spk_code[:,dec_pad_left:]
                            trj_spk_cv_code = trj_spk_cv_code[:,dec_pad_left:]
                            trj_spk = trj_spk[:,dec_pad_left:]
                            trj_spk_cv = trj_spk_cv[:,dec_pad_left:]
                            e_post = trj_src_src_uvlf0[:,dec_pad_left:]
                            e_cv_post = trj_src_trg_uvlf0[:,dec_pad_left:]
                        _, trj_src_src_post, _  = model_post(trj_src_src, y=trj_spk_code, aux=trj_spk, e=e_post)
                        _, trj_src_trg_post, _  = model_post(trj_src_trg, y=trj_spk_cv_code, aux=trj_spk_cv, e=e_cv_post)
                        if dec_post_pad_right > 0:
                            trj_lat_src = trj_lat_src[:,dec_post_pad_left:-dec_post_pad_right]
                            trj_lat_src_e = trj_lat_src_e[:,dec_post_pad_left:-dec_post_pad_right]
                        else:
                            trj_lat_src = trj_lat_src[:,dec_post_pad_left:]
                            trj_lat_src_e = trj_lat_src_e[:,dec_post_pad_left:]
                        if post_pad_right > 0:
                            trj_src_src = trj_src_src[:,post_pad_left:-post_pad_right]
                            trj_src_trg = trj_src_trg[:,post_pad_left:-post_pad_right]
                        else:
                            trj_src_src = trj_src_src[:,post_pad_left:]
                            trj_src_trg = trj_src_trg[:,post_pad_left:]
                        for k in range(n_batch_utt):
                            spk_src = os.path.basename(os.path.dirname(featfile[k]))
                            spk_src_trg = spk_cv[k] # find target pair
                            #GV stat of reconstructed
                            gv_src_src[spk_list.index(spk_src)].append(torch.var(\
                                (torch.exp(trj_src_src[k,:flens[k]])-1)/10000, 0).cpu().data.numpy())
                            #GV stat of converted
                            gv_src_trg[spk_list.index(spk_src_trg)].append(torch.var(\
                                (torch.exp(trj_src_trg[k,:flens[k]])-1)/10000, 0).cpu().data.numpy())
                            #GV stat of reconstructed post
                            gv_src_src_post[spk_list.index(spk_src)].append(torch.var(\
                                (torch.exp(trj_src_src_post[k,:flens[k]])-1)/10000, 0).cpu().data.numpy())
                            #GV stat of converted post
                            gv_src_trg_post[spk_list.index(spk_src_trg)].append(torch.var(\
                                (torch.exp(trj_src_trg_post[k,:flens[k]])-1)/10000, 0).cpu().data.numpy())
                        if pair_flag:
                            trj_lat_src = torch.cat((trj_lat_src_e, trj_lat_src), 2)
                            batch_melsp_trg_data_in = F.pad(batch_melsp_trg_data.transpose(1,2), (enc_pad_left,enc_pad_right), "replicate").transpose(1,2)
                            _, _, trj_lat_trg, _ = model_encoder_melsp(batch_melsp_trg_data_in,sampling=False)
                            _, _, trj_lat_trg_e, _ = model_encoder_excit(batch_melsp_trg_data_in, sampling=False)
                            trj_lat_trg = torch.cat((trj_lat_trg_e, trj_lat_trg), 2)

                            for k in range(n_batch_utt):
                                if src_trg_flag[k]:
                                    # spcidx lat
                                    trj_lat_src_ = np.array(torch.index_select(trj_lat_src[k],0,spcidx_src[k,:flens_spc_src[k]]).cpu().data.numpy(), dtype=np.float64)
                                    trj_lat_trg_ = np.array(torch.index_select(trj_lat_trg[k],0,spcidx_src_trg[k,:flens_spc_src_trg[k]]).cpu().data.numpy(), dtype=np.float64)
                                    # spcidx melsp, excit, trg
                                    trj_src_trg_ = (torch.exp(torch.index_select(trj_src_trg[k],0,spcidx_src[k,:flens_spc_src[k]]))-1)/10000
                                    trj_src_trg_post_ = (torch.exp(torch.index_select(trj_src_trg_post[k],0,spcidx_src[k,:flens_spc_src[k]]))-1)/10000
                                    trj_trg_ = (torch.exp(torch.index_select(batch_melsp_trg_data[k],0,spcidx_src_trg[k,:flens_spc_src_trg[k]]))-1)/10000
                                    # spec dtw
                                    # MCD of spectral
                                    _, twf_melsp, _, _ = dtw.dtw_org_to_trg(\
                                        np.array(trj_src_trg_.cpu().data.numpy(), dtype=np.float64), \
                                        np.array(trj_trg_.cpu().data.numpy(), dtype=np.float64), mcd=-1)
                                    twf_melsp = torch.LongTensor(twf_melsp[:,0]).cuda()
                                    batch_melsp_dB_src_trg = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(torch.index_select(trj_src_trg_,0,twf_melsp), min=1e-16))
                                                                -torch.log10(torch.clamp(trj_trg_, min=1e-16))))**2, -1))).item()
                                    # MCD of spectral post
                                    _, twf_melsp_post, _, _ = dtw.dtw_org_to_trg(\
                                        np.array(trj_src_trg_post_.cpu().data.numpy(), dtype=np.float64), \
                                        np.array(trj_trg_.cpu().data.numpy(), dtype=np.float64), mcd=-1)
                                    twf_melsp_post = torch.LongTensor(twf_melsp_post[:,0]).cuda()
                                    batch_melsp_dB_src_trg_post = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(torch.index_select(trj_src_trg_post_,0,twf_melsp_post), min=1e-16))
                                                                -torch.log10(torch.clamp(trj_trg_, min=1e-16))))**2, -1))).item()
                                    # time-warping of latent source-to-target for RMSE
                                    aligned_lat_srctrg1, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src_, trj_lat_trg_)
                                    batch_lat_dist_srctrg1 = np.mean(np.sqrt(np.mean((\
                                        aligned_lat_srctrg1-trj_lat_trg_)**2, axis=0)))
                                    # Cos-sim of latent source-to-target
                                    _, _, batch_lat_cdist_srctrg1, _ = dtw.dtw_org_to_trg(\
                                        trj_lat_trg_, trj_lat_src_, mcd=0)
                                    # time-warping of latent target-to-source for RMSE
                                    aligned_lat_srctrg2, _, _, _ = dtw.dtw_org_to_trg(trj_lat_trg_, trj_lat_src_)
                                    batch_lat_dist_srctrg2 = np.mean(np.sqrt(np.mean((\
                                        aligned_lat_srctrg2-trj_lat_src_)**2, axis=0)))
                                    # Cos-sim of latent target-to-source
                                    _, _, batch_lat_cdist_srctrg2, _ = dtw.dtw_org_to_trg(\
                                        trj_lat_src_, trj_lat_trg_, mcd=0)
                                    # RMSE
                                    batch_lat_dist_rmse = (batch_lat_dist_srctrg1+batch_lat_dist_srctrg2)/2
                                    # Cos-sim
                                    batch_lat_dist_cossim = (batch_lat_cdist_srctrg1+batch_lat_cdist_srctrg2)/2
                                    loss_melsp_dB_src_trg.append(batch_melsp_dB_src_trg)
                                    loss_melsp_dB_src_trg_post.append(batch_melsp_dB_src_trg_post)
                                    loss_lat_dist_rmse.append(batch_lat_dist_rmse)
                                    loss_lat_dist_cossim.append(batch_lat_dist_cossim)
                                    total_eval_loss["eval/loss_melsp_dB_src_trg"].append(batch_melsp_dB_src_trg)
                                    total_eval_loss["eval/loss_melsp_dB_src_trg_post"].append(batch_melsp_dB_src_trg_post)
                                    total_eval_loss["eval/loss_lat_dist_rmse"].append(batch_lat_dist_rmse)
                                    total_eval_loss["eval/loss_lat_dist_cossim"].append(batch_lat_dist_cossim)
                                    logging.info('acc cv %s %s %.3f dB %.3f dB %.3f %.3f' % (featfile[k],
                                        spk_cv[k], batch_melsp_dB_src_trg, batch_melsp_dB_src_trg_post,
                                            batch_lat_dist_rmse, batch_lat_dist_cossim))
                        idx_in = 0
                        i_cv_in = 0
                        for i in range(0,n_half_cyc_eval,2):
                            i_cv = i//2
                            j = i+1
                            if i > 0:
                                idx_in += 1
                                i_cv_in += 1
                                i_1 = i-1
                                if post_pad_right > 0:
                                    cyc_rec_feat = batch_melsp_rec_fix[i_1][:,post_pad_left:-post_pad_right]
                                else:
                                    cyc_rec_feat = batch_melsp_rec_fix[i_1][:,post_pad_left:]
                                qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(cyc_rec_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                                qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(cyc_rec_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                            else:
                                qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], sampling=False)
                                qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], sampling=False)
                                batch_feat_in_sc, h_feat_in_sc = model_classifier(feat=batch_melsp)
                                batch_feat_magsp_in_sc, h_feat_magsp_in_sc = model_classifier(feat_aux=batch_magsp)
                            ## reconstruction and conversion
                            idx_in += 1
                            z_cat = torch.cat((z_e[i], z[i]), 2)
                            if args.spkidtr_dim > 0:
                                spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                                spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                                batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                                batch_lf0_rec[i], h_lf0[i] = model_decoder_excit(z_e[i], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in])
                                batch_lf0_cv[i_cv], h_lf0_cv[i_cv] = model_decoder_excit(z_e[i], y=spk_cv_code_in, aux=batch_spk_cv, outpad_right=outpad_rights[idx_in])
                            else:
                                batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                                batch_lf0_rec[i], h_lf0[i] = model_decoder_excit(z_e[i], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in])
                                batch_lf0_cv[i_cv], h_lf0_cv[i_cv] = model_decoder_excit(z_e[i], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv, outpad_right=outpad_rights[idx_in])
                            if lf0_pad_right > 0:
                                z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,lf0_pad_left:-lf0_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,lf0_pad_left:-lf0_pad_right]
                                batch_spk = batch_spk[:,lf0_pad_left:-lf0_pad_right]
                                batch_spk_cv = batch_spk_cv[:,lf0_pad_left:-lf0_pad_right]
                            else:
                                z_cat = z_cat[:,lf0_pad_left:]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,lf0_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,lf0_pad_left:]
                                batch_spk = batch_spk[:,lf0_pad_left:]
                                batch_spk_cv = batch_spk_cv[:,lf0_pad_left:]
                            idx_in += 1
                            i_cv_in += 1
                            if args.spkidtr_dim > 0:
                                batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk,
                                                    e=batch_lf0_rec[i][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                                batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                                    e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                                batch_melsp_cv_fix[i_cv], h_melsp_cv_fix[i_cv] = model_decoder_melsp_fix(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                                    e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                            else:
                                batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=batch_lf0_rec[i][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                                batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                                batch_melsp_cv_fix[i_cv], h_melsp_cv_fix[i_cv] = model_decoder_melsp_fix(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                            if dec_pad_right > 0:
                                e_post = batch_lf0_rec[i][:,dec_pad_left:-dec_pad_right]
                                e_cv_post = batch_lf0_cv[i_cv][:,dec_pad_left:-dec_pad_right]
                            else:
                                e_post = batch_lf0_rec[i][:,dec_pad_left:]
                                e_cv_post = batch_lf0_cv[i_cv][:,dec_pad_left:]
                            idx_in_1 = idx_in-1
                            feat_len_e = batch_lf0_rec[i].shape[1]
                            batch_lf0_rec[i] = batch_lf0_rec[i][:,outpad_lefts[idx_in_1]:feat_len_e-outpad_rights[idx_in_1]]
                            batch_lf0_cv[i_cv] = batch_lf0_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len_e-outpad_rights[idx_in_1]]
                            ## post
                            idx_in += 1
                            i_cv_in += 1
                            if dec_pad_right > 0:
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,dec_pad_left:-dec_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,dec_pad_left:-dec_pad_right]
                                batch_spk = batch_spk[:,dec_pad_left:-dec_pad_right]
                                batch_spk_cv = batch_spk_cv[:,dec_pad_left:-dec_pad_right]
                            else:
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,dec_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,dec_pad_left:]
                                batch_spk = batch_spk[:,dec_pad_left:]
                                batch_spk_cv = batch_spk_cv[:,dec_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_pdf[i], batch_melsp_rec_post[i], h_melsp_post[i] \
                                                = model_post(batch_melsp_rec[i], y=spk_code_in, aux=batch_spk,
                                                    e=e_post, outpad_right=outpad_rights[idx_in])
                                batch_pdf_cv[i_cv], batch_melsp_cv_post[i_cv], h_melsp_cv_post[i_cv] \
                                                = model_post(batch_melsp_cv[i_cv], y=spk_cv_code_in, aux=batch_spk_cv,
                                                    e=e_cv_post, outpad_right=outpad_rights[idx_in])
                            else:
                                batch_pdf[i], batch_melsp_rec_post[i], h_melsp_post[i] \
                                                = model_post(batch_melsp_rec[i], y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=e_post, outpad_right=outpad_rights[idx_in])
                                batch_pdf_cv[i_cv], batch_melsp_cv_post[i_cv], h_melsp_cv_post[i_cv] \
                                                = model_post(batch_melsp_cv[i_cv], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    e=e_cv_post, outpad_right=outpad_rights[idx_in])
                            idx_in_1 = idx_in-1
                            feat_len = batch_melsp_rec[i].shape[1]
                            batch_melsp_rec[i] = batch_melsp_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_rec[i] = torch.matmul((torch.exp(batch_melsp_rec[i])-1)/10000, melfb_t)
                            batch_melsp_cv[i_cv] = batch_melsp_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_cv[i_cv] = torch.matmul((torch.exp(batch_melsp_cv[i_cv])-1)/10000, melfb_t)
                            batch_feat_rec_sc[i], h_feat_sc[i] = model_classifier(feat=batch_melsp_rec[i])
                            batch_feat_magsp_rec_sc[i], h_feat_magsp_sc[i] = model_classifier(feat_aux=batch_magsp_rec[i])
                            batch_feat_cv_sc[i_cv], h_feat_cv_sc[i_cv] = model_classifier(feat=batch_melsp_cv[i_cv])
                            batch_feat_magsp_cv_sc[i_cv], h_feat_magsp_cv_sc[i_cv] = model_classifier(feat_aux=batch_magsp_cv[i_cv])
                            ## cyclic reconstruction
                            idx_in += 1
                            if post_pad_right > 0:
                                cv_feat = batch_melsp_cv_fix[i_cv][:,post_pad_left:-post_pad_right]
                            else:
                                cv_feat = batch_melsp_cv_fix[i_cv][:,post_pad_left:]
                            qy_logits[j], qz_alpha[j], z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                            qy_logits_e[j], qz_alpha_e[j], z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                            feat_len = batch_melsp_rec[i].shape[1]
                            idx_in_1 = idx_in-1
                            feat_len = batch_melsp_rec_post[i].shape[1]
                            batch_pdf[i] = batch_pdf[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_rec_post[i] = batch_melsp_rec_post[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_rec_post[i] = torch.matmul((torch.exp(batch_melsp_rec_post[i])-1)/10000, melfb_t)
                            batch_pdf_cv[i_cv] = batch_pdf_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_cv_post[i_cv] = batch_melsp_cv_post[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_cv_post[i_cv] = torch.matmul((torch.exp(batch_melsp_cv_post[i_cv])-1)/10000, melfb_t)
                            batch_feat_rec_sc_post[i], h_feat_sc_post[i] = model_classifier(feat=batch_melsp_rec_post[i])
                            batch_feat_magsp_rec_sc_post[i], h_feat_magsp_sc_post[i] = model_classifier(feat_aux=batch_magsp_rec_post[i])
                            batch_feat_cv_sc_post[i_cv], h_feat_cv_sc_post[i_cv] = model_classifier(feat=batch_melsp_cv_post[i_cv])
                            batch_feat_magsp_cv_sc_post[i_cv], h_feat_magsp_cv_sc_post[i_cv] = model_classifier(feat_aux=batch_magsp_cv_post[i_cv])
                            if n_half_cyc_eval > 1:
                                idx_in += 1
                                z_cat = torch.cat((z_e[j], z[j]), 2)
                                if args.spkidtr_dim > 0:
                                    if enc_pad_right > 0:
                                        spk_code_in = spk_code_in[:,enc_pad_left:-enc_pad_right]
                                    else:
                                        spk_code_in = spk_code_in[:,enc_pad_left:]
                                    batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                                    batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                                    batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in])
                                if lf0_pad_right > 0:
                                    z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,lf0_pad_left:-lf0_pad_right]
                                    batch_spk = batch_spk[:,lf0_pad_left:-lf0_pad_right]
                                else:
                                    z_cat = z_cat[:,lf0_pad_left:]
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,lf0_pad_left:]
                                    batch_spk = batch_spk[:,lf0_pad_left:]
                                idx_in += 1
                                if args.spkidtr_dim > 0:
                                    batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                            e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                                    batch_melsp_rec_fix[j], h_melsp_fix[j] = model_decoder_melsp_fix(z_cat, y=spk_code_in, aux=batch_spk,
                                                            e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                            e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                                    batch_melsp_rec_fix[j], h_melsp_fix[j] = model_decoder_melsp_fix(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                            e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in])
                                if dec_pad_right > 0:
                                    e_post = batch_lf0_rec[j][:,dec_pad_left:-dec_pad_right]
                                else:
                                    e_post = batch_lf0_rec[j][:,dec_pad_left:]
                                idx_in_1 = idx_in-1
                                batch_lf0_rec[j] = batch_lf0_rec[j][:,outpad_lefts[idx_in_1]:batch_lf0_rec[j].shape[1]-outpad_rights[idx_in_1]]
                                ## post
                                idx_in += 1
                                if dec_pad_right > 0:
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,dec_pad_left:-dec_pad_right]
                                    batch_spk = batch_spk[:,dec_pad_left:-dec_pad_right]
                                else:
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,dec_pad_left:]
                                    batch_spk = batch_spk[:,dec_pad_left:]
                                if args.spkidtr_dim > 0:
                                    batch_pdf[j], batch_melsp_rec_post[j], h_melsp_post[j] \
                                                    = model_post(batch_melsp_rec[j], y=spk_code_in, aux=batch_spk,
                                                        e=e_post, outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_pdf[j], batch_melsp_rec_post[j], h_melsp_post[j] \
                                                    = model_post(batch_melsp_rec[j], y=batch_sc_in[idx_in], aux=batch_spk,
                                                        e=e_post, outpad_right=outpad_rights[idx_in])
                                idx_in_1 = idx_in-1                                
                                batch_melsp_rec[j] = batch_melsp_rec[j][:,outpad_lefts[idx_in_1]:batch_melsp_rec[j].shape[1]-outpad_rights[idx_in_1]]
                                batch_magsp_rec[j] = torch.matmul((torch.exp(batch_melsp_rec[j])-1)/10000, melfb_t)
                                batch_feat_rec_sc[j], h_feat_sc[j] = model_classifier(feat=batch_melsp_rec[j])
                                batch_feat_magsp_rec_sc[j], h_feat_magsp_sc[j] = model_classifier(feat_aux=batch_magsp_rec[j])
                                feat_len = batch_pdf[j].shape[1]
                                batch_pdf[j] = batch_pdf[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                                batch_melsp_rec_post[j] = batch_melsp_rec_post[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                                batch_magsp_rec_post[j] = torch.matmul((torch.exp(batch_melsp_rec_post[j])-1)/10000, melfb_t)
                                batch_feat_rec_sc_post[j], h_feat_sc_post[j] = model_classifier(feat=batch_melsp_rec_post[j])
                                batch_feat_magsp_rec_sc_post[j], h_feat_magsp_sc_post[j] = model_classifier(feat_aux=batch_magsp_rec_post[j])

                    # samples check
                    i = np.random.randint(0, batch_melsp_rec[0].shape[0])
                    logging.info("%d %s %d %d %d %d %s" % (i, \
                        os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                            f_ss, f_es, flens[i], max_flen, spk_cv[i]))
                    logging.info(batch_melsp_rec[0][i,:2,:4])
                    logging.info(batch_melsp_rec_post[0][i,:2,:4])
                    if n_half_cyc_eval > 1: 
                        logging.info(batch_melsp_rec[1][i,:2,:4])
                        logging.info(batch_melsp_rec_post[1][i,:2,:4])
                    logging.info(batch_melsp[i,:2,:4])
                    logging.info(batch_melsp_cv[0][i,:2,:4])
                    logging.info(batch_melsp_cv_post[0][i,:2,:4])
                    logging.info(batch_lf0_rec[0][i,:2,0])
                    if n_half_cyc_eval > 1:
                        logging.info(batch_lf0_rec[1][i,:2,0])
                    logging.info(batch_excit[i,:2,0])
                    logging.info(batch_lf0_cv[0][i,:2,0])
                    logging.info(torch.exp(batch_lf0_rec[0][i,:2,1]))
                    if n_half_cyc_eval > 1:
                        logging.info(torch.exp(batch_lf0_rec[1][i,:2,1]))
                    logging.info(torch.exp(batch_excit[i,:2,1]))
                    logging.info(torch.exp(batch_lf0_cv[0][i,:2,1]))
                    logging.info(torch.exp(batch_excit_cv[0][i,:2,1]))
                    logging.info(batch_lf0_rec[0][i,:2,2])
                    if n_half_cyc_eval > 1:
                        logging.info(batch_lf0_rec[1][i,:2,2])
                    logging.info(batch_excit[i,:2,2])
                    logging.info(batch_lf0_cv[0][i,:2,2])
                    logging.info(-torch.exp(batch_lf0_rec[0][i,:2,3:]))
                    if n_half_cyc_eval > 1:
                        logging.info(-torch.exp(batch_lf0_rec[1][i,:2,3:]))
                    logging.info(-torch.exp(batch_excit[i,:2,3:]))
                    logging.info(-torch.exp(batch_lf0_cv[0][i,:2,3:]))
                    #logging.info(qy_logits[0][i,:2])
                    #logging.info(batch_sc[i,0])
                    #logging.info(qy_logits[1][i,:2])
                    #logging.info(batch_sc_cv[0][i,0])
                    #logging.info(torch.max(z[0][i,5:10], -1))
                    #unique, counts = np.unique(torch.max(z[0][i], -1)[1].cpu().data.numpy(), return_counts=True)
                    #logging.info(dict(zip(unique, counts)))

                    # handle short ending
                    if len(idx_select) > 0:
                        if len(idx_select_full) > 0:
                            logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                            batch_melsp = torch.index_select(batch_melsp,0,idx_select_full)
                            batch_magsp = torch.index_select(batch_magsp,0,idx_select_full)
                            batch_sc = torch.index_select(batch_sc,0,idx_select_full)
                            batch_feat_in_sc = torch.index_select(batch_feat_in_sc,0,idx_select_full)
                            batch_feat_magsp_in_sc = torch.index_select(batch_feat_magsp_in_sc,0,idx_select_full)
                            n_batch_utt = batch_melsp.shape[0]
                            for i in range(n_half_cyc_eval):
                                batch_melsp_rec[i] = torch.index_select(batch_melsp_rec[i],0,idx_select_full)
                                batch_magsp_rec[i] = torch.index_select(batch_magsp_rec[i],0,idx_select_full)
                                batch_pdf[i] = torch.index_select(batch_pdf[i],0,idx_select_full)
                                batch_melsp_rec_post[i] = torch.index_select(batch_melsp_rec_post[i],0,idx_select_full)
                                batch_magsp_rec_post[i] = torch.index_select(batch_magsp_rec_post[i],0,idx_select_full)
                                batch_feat_rec_sc[i] = torch.index_select(batch_feat_rec_sc[i],0,idx_select_full)
                                batch_feat_rec_sc_post[i] = torch.index_select(batch_feat_rec_sc_post[i],0,idx_select_full)
                                batch_feat_magsp_rec_sc[i] = torch.index_select(batch_feat_magsp_rec_sc[i],0,idx_select_full)
                                batch_feat_magsp_rec_sc_post[i] = torch.index_select(batch_feat_magsp_rec_sc_post[i],0,idx_select_full)
                                if i % 2 == 0:
                                    batch_melsp_cv[i//2] = torch.index_select(batch_melsp_cv[i//2],0,idx_select_full)
                                    batch_magsp_cv[i//2] = torch.index_select(batch_magsp_cv[i//2],0,idx_select_full)
                                    batch_pdf_cv[i//2] = torch.index_select(batch_pdf_cv[i//2],0,idx_select_full)
                                    batch_melsp_cv_post[i//2] = torch.index_select(batch_melsp_cv_post[i//2],0,idx_select_full)
                                    batch_magsp_cv_post[i//2] = torch.index_select(batch_magsp_cv_post[i//2],0,idx_select_full)
                                    batch_feat_cv_sc[i//2] = torch.index_select(batch_feat_cv_sc[i//2],0,idx_select_full)
                                    batch_feat_cv_sc_post[i//2] = torch.index_select(batch_feat_cv_sc_post[i//2],0,idx_select_full)
                                    batch_feat_magsp_cv_sc[i//2] = torch.index_select(batch_feat_magsp_cv_sc[i//2],0,idx_select_full)
                                    batch_feat_magsp_cv_sc_post[i//2] = torch.index_select(batch_feat_magsp_cv_sc_post[i//2],0,idx_select_full)
                                    batch_sc_cv[i//2] = torch.index_select(batch_sc_cv[i//2],0,idx_select_full)
                        else:
                            iter_count += 1
                            total += time.time() - start
                            continue

                    # loss_compute
                    melsp = batch_melsp
                    melsp_rest = (torch.exp(melsp)-1)/10000
                    melsp_rest_log = torch.log10(torch.clamp(melsp_rest, min=1e-16))
                    magsp = magsp_rest = batch_magsp
                    batch_sc_ = batch_sc.reshape(-1)
                    batch_loss_sc_feat_in_ = torch.mean(criterion_ce(batch_feat_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                    batch_loss_sc_feat_in = batch_loss_sc_feat_in_.mean()
                    batch_loss_sc_feat_magsp_in_ = torch.mean(criterion_ce(batch_feat_magsp_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                    batch_loss_sc_feat_magsp_in = batch_loss_sc_feat_magsp_in_.mean()
                    for i in range(n_half_cyc_eval):
                        ## reconst. [i % 2 == 0] / cyclic reconst. [i % 2 == 1]
                        melsp_est = batch_melsp_rec[i]
                        melsp_est_rest = (torch.exp(melsp_est)-1)/10000
                        melsp_est_post = batch_melsp_rec_post[i]
                        melsp_est_rest_post = (torch.exp(melsp_est_post)-1)/10000
                        magsp_est = magsp_est_rest = batch_magsp_rec[i]
                        magsp_est_post = magsp_est_rest_post = batch_magsp_rec_post[i]
                        ## conversion
                        if i % 2 == 0:
                            melsp_cv = batch_melsp_cv[i//2]
                            magsp_cv = batch_magsp_cv[i//2]
                            melsp_cv_post = batch_melsp_cv_post[i//2]
                            magsp_cv_post = batch_magsp_cv_post[i//2]

                        ## melsp acc.
                        batch_loss_melsp_ = torch.mean(torch.sum(criterion_l1(melsp_est, melsp), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_est, melsp), -1), -1)) \
                                            + torch.mean(torch.mean(criterion_l1(melsp_est, melsp), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_est, melsp), -1), -1))
                        batch_loss_px_sum = batch_loss_melsp_.sum()
                        batch_loss_melsp[i] = batch_loss_melsp_.mean()
                        batch_loss_px[i] = batch_loss_melsp_.mean()
                        batch_loss_melsp_dB[i] = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_est_rest, min=1e-16))-melsp_rest_log))**2, -1)))

                        batch_loss_magsp_ = torch.mean(torch.sum(criterion_l1(magsp_est, magsp), -1), -1) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_est, magsp), -1), -1) \
                                            + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_est, magsp), -1), -1)) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_est, magsp), -1), -1))
                        batch_loss_px_sum += batch_loss_magsp_.sum()
                        batch_loss_magsp[i] = batch_loss_magsp_.mean()
                        batch_loss_px[i] += batch_loss_magsp[i]
                        batch_loss_magsp_dB[i] = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_est_rest, min=1e-16))
                                                                -torch.log10(torch.clamp(magsp_rest, min=1e-16))))**2, -1)), -1))

                        batch_loss_melsp_post_ = torch.mean(torch.sum(criterion_l1(melsp_est_post, melsp), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_est_post, melsp), -1), -1)) \
                                            + torch.mean(torch.mean(criterion_l1(melsp_est_post, melsp), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_est_post, melsp), -1), -1))
                        batch_loss_px_sum += batch_loss_melsp_post_.sum()
                        batch_loss_melsp_post[i] = batch_loss_melsp_post_.mean()
                        batch_loss_px[i] += batch_loss_melsp_post[i]
                        batch_loss_melsp_dB_post[i] = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_est_rest_post, min=1e-16))-melsp_rest_log))**2, -1)))

                        batch_loss_magsp_post_ = torch.mean(torch.sum(criterion_l1(magsp_est_post, magsp), -1), -1) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_est_post, magsp), -1), -1) \
                                            + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_est_post, magsp), -1), -1)) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_est_post, magsp), -1), -1))
                        batch_loss_px_sum += batch_loss_magsp_post_.sum()
                        batch_loss_magsp_post[i] = batch_loss_magsp_post_.mean()
                        batch_loss_px[i] += batch_loss_magsp_post[i]
                        batch_loss_magsp_dB_post[i] = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_est_rest_post, min=1e-16))
                                                                -torch.log10(torch.clamp(magsp_rest, min=1e-16))))**2, -1)), -1))

                        batch_loss_px_ms_norm_, batch_loss_px_ms_err_ = criterion_ms(melsp_est_rest, melsp_rest)
                        batch_loss_ms_norm[i] = batch_loss_px_ms_norm_.mean()
                        batch_loss_ms_err[i] = batch_loss_px_ms_err_.mean()
                        if  not torch.isinf(batch_loss_ms_norm[i]) and not torch.isnan(batch_loss_ms_norm[i]):
                            batch_loss_px_sum += batch_loss_px_ms_norm_.sum()
                        if not torch.isinf(batch_loss_ms_err[i]) and not torch.isnan(batch_loss_ms_err[i]):
                            batch_loss_px_sum += batch_loss_px_ms_err_.sum()

                        batch_loss_px_ms_magsp_norm_, batch_loss_px_ms_magsp_err_ = criterion_ms(magsp_est_rest, magsp_rest)
                        batch_loss_ms_norm_magsp[i] = batch_loss_px_ms_magsp_norm_.mean()
                        batch_loss_ms_err_magsp[i] = batch_loss_px_ms_magsp_err_.mean()
                        if not torch.isinf(batch_loss_ms_norm_magsp[i]) and not torch.isnan(batch_loss_ms_norm_magsp[i]):
                            batch_loss_px_sum += batch_loss_px_ms_magsp_norm_.sum()
                        if not torch.isinf(batch_loss_ms_err_magsp[i]) and not torch.isnan(batch_loss_ms_err_magsp[i]):
                            batch_loss_px_sum += batch_loss_px_ms_magsp_err_.sum()

                        batch_loss_px_ms_norm_post_, batch_loss_px_ms_err_post_ = criterion_ms(melsp_est_rest_post, melsp_rest)
                        batch_loss_ms_norm_post[i] = batch_loss_px_ms_norm_post_.mean()
                        batch_loss_ms_err_post[i] = batch_loss_px_ms_err_post_.mean()
                        if  not torch.isinf(batch_loss_ms_norm_post[i]) and not torch.isnan(batch_loss_ms_norm_post[i]):
                            batch_loss_px_sum += batch_loss_px_ms_norm_post_.sum()
                        if not torch.isinf(batch_loss_ms_err_post[i]) and not torch.isnan(batch_loss_ms_err_post[i]):
                            batch_loss_px_sum += batch_loss_px_ms_err_post_.sum()

                        batch_loss_px_ms_magsp_norm_post_, batch_loss_px_ms_magsp_err_post_ = criterion_ms(magsp_est_rest_post, magsp_rest)
                        batch_loss_ms_norm_magsp_post[i] = batch_loss_px_ms_magsp_norm_post_.mean()
                        batch_loss_ms_err_magsp_post[i] = batch_loss_px_ms_magsp_err_post_.mean()
                        if not torch.isinf(batch_loss_ms_norm_magsp_post[i]) and not torch.isnan(batch_loss_ms_norm_magsp_post[i]):
                            batch_loss_px_sum += batch_loss_px_ms_magsp_norm_post_.sum()
                        if not torch.isinf(batch_loss_ms_err_magsp_post[i]) and not torch.isnan(batch_loss_ms_err_magsp_post[i]):
                            batch_loss_px_sum += batch_loss_px_ms_magsp_err_post_.sum()

                        batch_loss_laplace_ = criterion_laplace(batch_pdf[i][:,:,:args.mel_dim], batch_pdf[i][:,:,args.mel_dim:], melsp)
                        batch_loss_laplace[i] = batch_loss_laplace_.mean()
                        batch_loss_px_sum += batch_loss_laplace_.sum()

                        ## conversion
                        if i % 2 == 0:
                            batch_loss_melsp_cv[i//2] = torch.mean(torch.sum(criterion_l1(melsp_cv, melsp), -1)) \
                                                            + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_cv, melsp), -1))) \
                                                        + torch.mean(criterion_l1(melsp_cv, melsp)) \
                                                            + torch.sqrt(torch.mean(criterion_l2(melsp_cv, melsp)))
                            batch_loss_magsp_cv[i//2] = torch.mean(torch.sum(criterion_l1(magsp_cv, magsp), -1)) \
                                                            + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_cv, magsp), -1))) \
                                                        + torch.mean(torch.mean(criterion_l1(magsp_cv, magsp), -1)) \
                                                            + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_cv, magsp), -1)))
                            batch_loss_melsp_cv_post[i//2] = torch.mean(torch.sum(criterion_l1(melsp_cv_post, melsp), -1)) \
                                                            + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_cv_post, melsp), -1))) \
                                                        + torch.mean(criterion_l1(melsp_cv_post, melsp)) \
                                                            + torch.sqrt(torch.mean(criterion_l2(melsp_cv_post, melsp)))
                            batch_loss_magsp_cv_post[i//2] = torch.mean(torch.sum(criterion_l1(magsp_cv_post, magsp), -1)) \
                                                            + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_cv_post, magsp), -1))) \
                                                        + torch.mean(torch.mean(criterion_l1(magsp_cv_post, magsp), -1)) \
                                                            + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_cv_post, magsp), -1)))
                            batch_loss_laplace_cv[i//2] = torch.mean(criterion_laplace(batch_pdf_cv[i//2][:,:,:args.mel_dim], batch_pdf_cv[i//2][:,:,args.mel_dim:], melsp))

                        # KL-div lat., CE and error-percentage spk.
                        batch_sc_cv_ = batch_sc_cv[i//2].reshape(-1)
                        batch_loss_sc_feat_ = torch.mean(criterion_ce(batch_feat_rec_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                        batch_loss_sc_feat[i] = batch_loss_sc_feat_.mean()
                        batch_loss_sc_feat_magsp_ = torch.mean(criterion_ce(batch_feat_magsp_rec_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                        batch_loss_sc_feat_magsp[i] = batch_loss_sc_feat_magsp_.mean()
                        batch_loss_sc_feat_post_ = torch.mean(criterion_ce(batch_feat_rec_sc_post[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                        batch_loss_sc_feat_post[i] = batch_loss_sc_feat_post_.mean()
                        batch_loss_sc_feat_magsp_post_ = torch.mean(criterion_ce(batch_feat_magsp_rec_sc_post[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                        batch_loss_sc_feat_magsp_post[i] = batch_loss_sc_feat_magsp_post_.mean()
                        batch_loss_sc_feat_kl = batch_loss_sc_feat_.sum() + batch_loss_sc_feat_magsp_.sum() \
                                                + batch_loss_sc_feat_post_.sum() + batch_loss_sc_feat_magsp_post_.sum()
                        if i % 2 == 0:
                            batch_loss_sc_feat_cv_ = torch.mean(criterion_ce(batch_feat_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                            batch_loss_sc_feat_cv[i//2] = batch_loss_sc_feat_cv_.mean()
                            batch_loss_sc_feat_magsp_cv_ = torch.mean(criterion_ce(batch_feat_magsp_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                            batch_loss_sc_feat_magsp_cv[i//2] = batch_loss_sc_feat_magsp_cv_.mean()
                            batch_loss_sc_feat_cv_post_ = torch.mean(criterion_ce(batch_feat_cv_sc_post[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                            batch_loss_sc_feat_cv_post[i//2] = batch_loss_sc_feat_cv_post_.mean()
                            batch_loss_sc_feat_magsp_cv_post_ = torch.mean(criterion_ce(batch_feat_magsp_cv_sc_post[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                            batch_loss_sc_feat_magsp_cv_post[i//2] = batch_loss_sc_feat_magsp_cv_post_.mean()
                            batch_loss_sc_feat_kl += batch_loss_sc_feat_cv_.sum() + batch_loss_sc_feat_magsp_cv_.sum() \
                                                    + batch_loss_sc_feat_cv_post_.sum() + batch_loss_sc_feat_magsp_cv_post_.sum()

                        # elbo
                        batch_loss_elbo[i] = batch_loss_px_sum + batch_loss_sc_feat_kl

                        total_eval_loss["eval/loss_elbo-%d"%(i+1)].append(batch_loss_elbo[i].item())
                        total_eval_loss["eval/loss_px-%d"%(i+1)].append(batch_loss_px[i].item())
                        total_eval_loss["eval/loss_laplace-%d"%(i+1)].append(batch_loss_laplace[i].item())
                        total_eval_loss["eval/loss_sc_feat-%d"%(i+1)].append(batch_loss_sc_feat[i].item())
                        total_eval_loss["eval/loss_sc_feat_post-%d"%(i+1)].append(batch_loss_sc_feat_post[i].item())
                        total_eval_loss["eval/loss_sc_feat_magsp-%d"%(i+1)].append(batch_loss_sc_feat_magsp[i].item())
                        total_eval_loss["eval/loss_sc_feat_magsp_post-%d"%(i+1)].append(batch_loss_sc_feat_magsp_post[i].item())
                        if i == 0:
                            total_eval_loss["eval/loss_sc_feat_in"].append(batch_loss_sc_feat_in.item())
                            total_eval_loss["eval/loss_sc_feat_magsp_in"].append(batch_loss_sc_feat_magsp_in.item())
                            loss_sc_feat_in.append(batch_loss_sc_feat_in.item())
                            loss_sc_feat_magsp_in.append(batch_loss_sc_feat_magsp_in.item())
                        total_eval_loss["eval/loss_ms_norm-%d"%(i+1)].append(batch_loss_ms_norm[i].item())
                        total_eval_loss["eval/loss_ms_err-%d"%(i+1)].append(batch_loss_ms_err[i].item())
                        total_eval_loss["eval/loss_ms_norm_post-%d"%(i+1)].append(batch_loss_ms_norm_post[i].item())
                        total_eval_loss["eval/loss_ms_err_post-%d"%(i+1)].append(batch_loss_ms_err_post[i].item())
                        total_eval_loss["eval/loss_ms_norm_magsp-%d"%(i+1)].append(batch_loss_ms_norm_magsp[i].item())
                        total_eval_loss["eval/loss_ms_err_magsp-%d"%(i+1)].append(batch_loss_ms_err_magsp[i].item())
                        total_eval_loss["eval/loss_ms_norm_magsp_post-%d"%(i+1)].append(batch_loss_ms_norm_magsp_post[i].item())
                        total_eval_loss["eval/loss_ms_err_magsp_post-%d"%(i+1)].append(batch_loss_ms_err_magsp_post[i].item())
                        loss_elbo[i].append(batch_loss_elbo[i].item())
                        loss_px[i].append(batch_loss_px[i].item())
                        loss_laplace[i].append(batch_loss_laplace[i].item())
                        loss_sc_feat[i].append(batch_loss_sc_feat[i].item())
                        loss_sc_feat_post[i].append(batch_loss_sc_feat_post[i].item())
                        loss_sc_feat_magsp[i].append(batch_loss_sc_feat_magsp[i].item())
                        loss_sc_feat_magsp_post[i].append(batch_loss_sc_feat_magsp_post[i].item())
                        loss_ms_norm[i].append(batch_loss_ms_norm[i].item())
                        loss_ms_err[i].append(batch_loss_ms_err[i].item())
                        loss_ms_norm_post[i].append(batch_loss_ms_norm_post[i].item())
                        loss_ms_err_post[i].append(batch_loss_ms_err_post[i].item())
                        loss_ms_norm_magsp[i].append(batch_loss_ms_norm_magsp[i].item())
                        loss_ms_err_magsp[i].append(batch_loss_ms_err_magsp[i].item())
                        loss_ms_norm_magsp_post[i].append(batch_loss_ms_norm_magsp_post[i].item())
                        loss_ms_err_magsp_post[i].append(batch_loss_ms_err_magsp_post[i].item())
                        ## in-domain reconst.
                        total_eval_loss["eval/loss_melsp-%d"%(i+1)].append(batch_loss_melsp[i].item())
                        total_eval_loss["eval/loss_melsp_dB-%d"%(i+1)].append(batch_loss_melsp_dB[i].item())
                        total_eval_loss["eval/loss_melsp_post-%d"%(i+1)].append(batch_loss_melsp_post[i].item())
                        total_eval_loss["eval/loss_melsp_dB_post-%d"%(i+1)].append(batch_loss_melsp_dB_post[i].item())
                        total_eval_loss["eval/loss_magsp-%d"%(i+1)].append(batch_loss_magsp[i].item())
                        total_eval_loss["eval/loss_magsp_dB-%d"%(i+1)].append(batch_loss_magsp_dB[i].item())
                        total_eval_loss["eval/loss_magsp_post-%d"%(i+1)].append(batch_loss_magsp_post[i].item())
                        total_eval_loss["eval/loss_magsp_dB_post-%d"%(i+1)].append(batch_loss_magsp_dB_post[i].item())
                        loss_melsp[i].append(batch_loss_melsp[i].item())
                        loss_melsp_dB[i].append(batch_loss_melsp_dB[i].item())
                        loss_melsp_post[i].append(batch_loss_melsp_post[i].item())
                        loss_melsp_dB_post[i].append(batch_loss_melsp_dB_post[i].item())
                        loss_magsp[i].append(batch_loss_magsp[i].item())
                        loss_magsp_dB[i].append(batch_loss_magsp_dB[i].item())
                        loss_magsp_post[i].append(batch_loss_magsp_post[i].item())
                        loss_magsp_dB_post[i].append(batch_loss_magsp_dB_post[i].item())
                        ## conversion
                        if i % 2 == 0:
                            total_eval_loss["eval/loss_laplace_cv-%d"%(i+1)].append(batch_loss_laplace_cv[i//2].item())
                            total_eval_loss["eval/loss_sc_feat_cv-%d"%(i+1)].append(batch_loss_sc_feat_cv[i//2].item())
                            total_eval_loss["eval/loss_sc_feat_cv_post-%d"%(i+1)].append(batch_loss_sc_feat_cv_post[i//2].item())
                            total_eval_loss["eval/loss_sc_feat_magsp_cv-%d"%(i+1)].append(batch_loss_sc_feat_magsp_cv[i//2].item())
                            total_eval_loss["eval/loss_sc_feat_magsp_cv_post-%d"%(i+1)].append(batch_loss_sc_feat_magsp_cv_post[i//2].item())
                            total_eval_loss["eval/loss_melsp_cv-%d"%(i+1)].append(batch_loss_melsp_cv[i//2].item())
                            total_eval_loss["eval/loss_melsp_cv_post-%d"%(i+1)].append(batch_loss_melsp_cv_post[i//2].item())
                            total_eval_loss["eval/loss_magsp_cv-%d"%(i+1)].append(batch_loss_magsp_cv[i//2].item())
                            total_eval_loss["eval/loss_magsp_cv_post-%d"%(i+1)].append(batch_loss_magsp_cv_post[i//2].item())
                            loss_laplace_cv[i//2].append(batch_loss_laplace_cv[i//2].item())
                            loss_sc_feat_cv[i//2].append(batch_loss_sc_feat_cv[i//2].item())
                            loss_sc_feat_cv_post[i//2].append(batch_loss_sc_feat_cv_post[i//2].item())
                            loss_sc_feat_magsp_cv[i//2].append(batch_loss_sc_feat_magsp_cv[i//2].item())
                            loss_sc_feat_magsp_cv_post[i//2].append(batch_loss_sc_feat_magsp_cv_post[i//2].item())
                            loss_melsp_cv[i//2].append(batch_loss_melsp_cv[i//2].item())
                            loss_melsp_cv_post[i//2].append(batch_loss_melsp_cv_post[i//2].item())
                            loss_magsp_cv[i//2].append(batch_loss_magsp_cv[i//2].item())
                            loss_magsp_cv_post[i//2].append(batch_loss_magsp_cv_post[i//2].item())

                    text_log = "batch eval loss [%d] %d %d %.3f %.3f " % (c_idx+1, f_ss, f_bs, batch_loss_sc_feat_in.item(), batch_loss_sc_feat_magsp_in.item())
                    for i in range(n_half_cyc_eval):
                        if i % 2 == 0:
                            text_log += "[%ld] %.3f , %.3f %.3f , %.3f ; %.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; "\
                                "%.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; " \
                                "%.3f %.3f %.3f dB , %.3f %.3f %.3f dB , %.3f %.3f %.3f dB , %.3f %.3f %.3f dB ;; " % (i+1,
                                batch_loss_elbo[i].item(), batch_loss_laplace[i].item(), batch_loss_laplace_cv[i//2].item(), batch_loss_px[i].item(),
                                    batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                                    batch_loss_ms_norm_post[i].item(), batch_loss_ms_err_post[i].item(),
                                    batch_loss_ms_norm_magsp[i].item(), batch_loss_ms_err_magsp[i].item(),
                                    batch_loss_ms_norm_magsp_post[i].item(), batch_loss_ms_err_magsp_post[i].item(),
                                        batch_loss_sc_feat[i].item(), batch_loss_sc_feat_cv[i//2].item(),
                                        batch_loss_sc_feat_post[i].item(), batch_loss_sc_feat_cv_post[i//2].item(),
                                        batch_loss_sc_feat_magsp[i].item(), batch_loss_sc_feat_magsp_cv[i//2].item(),
                                        batch_loss_sc_feat_magsp_post[i].item(), batch_loss_sc_feat_magsp_cv_post[i//2].item(),
                                            batch_loss_melsp[i].item(), batch_loss_melsp_cv[i//2].item(), batch_loss_melsp_dB[i].item(),
                                            batch_loss_melsp_post[i].item(), batch_loss_melsp_cv_post[i//2].item(), batch_loss_melsp_dB_post[i].item(),
                                            batch_loss_magsp[i].item(), batch_loss_magsp_cv[i//2].item(), batch_loss_magsp_dB[i].item(),
                                            batch_loss_magsp_post[i].item(), batch_loss_magsp_cv_post[i//2].item(), batch_loss_magsp_dB_post[i].item())
                        else:
                            text_log += "[%ld] %.3f , %.3f , %.3f ; %.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; "\
                                "%.3f , %.3f , %.3f , %.3f ; %.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB ;; " % (i+1,
                                batch_loss_elbo[i].item(), batch_loss_laplace[i].item(), batch_loss_px[i].item(),
                                        batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                                        batch_loss_ms_norm_post[i].item(), batch_loss_ms_err_post[i].item(),
                                        batch_loss_ms_norm_magsp[i].item(), batch_loss_ms_err_magsp[i].item(),
                                        batch_loss_ms_norm_magsp_post[i].item(), batch_loss_ms_err_magsp_post[i].item(),
                                            batch_loss_sc_feat[i].item(), batch_loss_sc_feat_post[i].item(),
                                            batch_loss_sc_feat_magsp[i].item(), batch_loss_sc_feat_magsp_post[i].item(),
                                                batch_loss_melsp[i].item(), batch_loss_melsp_dB[i].item(),
                                                batch_loss_melsp_post[i].item(), batch_loss_melsp_dB_post[i].item(),
                                                batch_loss_magsp[i].item(), batch_loss_magsp_dB[i].item(),
                                                batch_loss_magsp_post[i].item(), batch_loss_magsp_dB_post[i].item())
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            tmp_gv_1 = []
            tmp_gv_2 = []
            tmp_gv_3 = []
            tmp_gv_4 = []
            for j in range(n_spk):
                if len(gv_src_src[j]) > 0:
                    tmp_gv_1.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_src[j],
                                        axis=0))-np.log(gv_mean[j])))))
                if len(gv_src_trg[j]) > 0:
                    tmp_gv_2.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg[j],
                                        axis=0))-np.log(gv_mean[j])))))
                if len(gv_src_src_post[j]) > 0:
                    tmp_gv_3.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_src_post[j],
                                        axis=0))-np.log(gv_mean[j])))))
                if len(gv_src_trg_post[j]) > 0:
                    tmp_gv_4.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg_post[j],
                                        axis=0))-np.log(gv_mean[j])))))
            eval_loss_gv_src_src = np.mean(tmp_gv_1)
            eval_loss_gv_src_trg = np.mean(tmp_gv_2)
            eval_loss_gv_src_src_post = np.mean(tmp_gv_3)
            eval_loss_gv_src_trg_post = np.mean(tmp_gv_4)
            total_eval_loss["eval/loss_gv_src_src"].append(eval_loss_gv_src_src)
            total_eval_loss["eval/loss_gv_src_trg"].append(eval_loss_gv_src_trg)
            total_eval_loss["eval/loss_gv_src_src_post"].append(eval_loss_gv_src_src_post)
            total_eval_loss["eval/loss_gv_src_trg_post"].append(eval_loss_gv_src_trg_post)
            logging.info('sme %d' % (epoch_idx + 1))
            for key in total_eval_loss.keys():
                total_eval_loss[key] = np.mean(total_eval_loss[key])
                logging.info(f"(Steps: {iter_idx}) {key} = {total_eval_loss[key]:.4f}.")
            write_to_tensorboard(writer, iter_idx, total_eval_loss)
            total_eval_loss = defaultdict(list)
            eval_loss_sc_feat_in = np.mean(loss_sc_feat_in)
            eval_loss_sc_feat_in_std = np.std(loss_sc_feat_in)
            eval_loss_sc_feat_magsp_in = np.mean(loss_sc_feat_magsp_in)
            eval_loss_sc_feat_magsp_in_std = np.std(loss_sc_feat_magsp_in)
            if pair_exist:
                eval_loss_melsp_dB_src_trg = np.mean(loss_melsp_dB_src_trg)
                eval_loss_melsp_dB_src_trg_std = np.std(loss_melsp_dB_src_trg)
                eval_loss_melsp_dB_src_trg_post = np.mean(loss_melsp_dB_src_trg_post)
                eval_loss_melsp_dB_src_trg_post_std = np.std(loss_melsp_dB_src_trg_post)
                eval_loss_lat_dist_rmse = np.mean(loss_lat_dist_rmse)
                eval_loss_lat_dist_rmse_std = np.std(loss_lat_dist_rmse)
                eval_loss_lat_dist_cossim = np.mean(loss_lat_dist_cossim)
                eval_loss_lat_dist_cossim_std = np.std(loss_lat_dist_cossim)
            for i in range(n_half_cyc_eval):
                eval_loss_elbo[i] = np.mean(loss_elbo[i])
                eval_loss_elbo_std[i] = np.std(loss_elbo[i])
                eval_loss_laplace[i] = np.mean(loss_laplace[i])
                eval_loss_laplace_std[i] = np.std(loss_laplace[i])
                eval_loss_px[i] = np.mean(loss_px[i])
                eval_loss_px_std[i] = np.std(loss_px[i])
                eval_loss_sc_feat[i] = np.mean(loss_sc_feat[i])
                eval_loss_sc_feat_std[i] = np.std(loss_sc_feat[i])
                eval_loss_sc_feat_post[i] = np.mean(loss_sc_feat_post[i])
                eval_loss_sc_feat_post_std[i] = np.std(loss_sc_feat_post[i])
                eval_loss_sc_feat_magsp[i] = np.mean(loss_sc_feat_magsp[i])
                eval_loss_sc_feat_magsp_std[i] = np.std(loss_sc_feat_magsp[i])
                eval_loss_sc_feat_magsp_post[i] = np.mean(loss_sc_feat_magsp_post[i])
                eval_loss_sc_feat_magsp_post_std[i] = np.std(loss_sc_feat_magsp_post[i])
                eval_loss_ms_norm[i] = np.mean(loss_ms_norm[i])
                eval_loss_ms_norm_std[i] = np.std(loss_ms_norm[i])
                eval_loss_ms_err[i] = np.mean(loss_ms_err[i])
                eval_loss_ms_err_std[i] = np.std(loss_ms_err[i])
                eval_loss_ms_norm_post[i] = np.mean(loss_ms_norm_post[i])
                eval_loss_ms_norm_post_std[i] = np.std(loss_ms_norm_post[i])
                eval_loss_ms_err_post[i] = np.mean(loss_ms_err_post[i])
                eval_loss_ms_err_post_std[i] = np.std(loss_ms_err_post[i])
                eval_loss_ms_norm_magsp[i] = np.mean(loss_ms_norm_magsp[i])
                eval_loss_ms_norm_magsp_std[i] = np.std(loss_ms_norm_magsp[i])
                eval_loss_ms_err_magsp[i] = np.mean(loss_ms_err_magsp[i])
                eval_loss_ms_err_magsp_std[i] = np.std(loss_ms_err_magsp[i])
                eval_loss_ms_norm_magsp_post[i] = np.mean(loss_ms_norm_magsp_post[i])
                eval_loss_ms_norm_magsp_post_std[i] = np.std(loss_ms_norm_magsp_post[i])
                eval_loss_ms_err_magsp_post[i] = np.mean(loss_ms_err_magsp_post[i])
                eval_loss_ms_err_magsp_post_std[i] = np.std(loss_ms_err_magsp_post[i])
                eval_loss_melsp[i] = np.mean(loss_melsp[i])
                eval_loss_melsp_std[i] = np.std(loss_melsp[i])
                eval_loss_melsp_dB[i] = np.mean(loss_melsp_dB[i])
                eval_loss_melsp_dB_std[i] = np.std(loss_melsp_dB[i])
                eval_loss_melsp_post[i] = np.mean(loss_melsp_post[i])
                eval_loss_melsp_post_std[i] = np.std(loss_melsp_post[i])
                eval_loss_melsp_dB_post[i] = np.mean(loss_melsp_dB_post[i])
                eval_loss_melsp_dB_post_std[i] = np.std(loss_melsp_dB_post[i])
                eval_loss_magsp[i] = np.mean(loss_magsp[i])
                eval_loss_magsp_std[i] = np.std(loss_magsp[i])
                eval_loss_magsp_dB[i] = np.mean(loss_magsp_dB[i])
                eval_loss_magsp_dB_std[i] = np.std(loss_magsp_dB[i])
                eval_loss_magsp_post[i] = np.mean(loss_magsp_post[i])
                eval_loss_magsp_post_std[i] = np.std(loss_magsp_post[i])
                eval_loss_magsp_dB_post[i] = np.mean(loss_magsp_dB_post[i])
                eval_loss_magsp_dB_post_std[i] = np.std(loss_magsp_dB_post[i])
                if i % 2 == 0:
                    eval_loss_laplace_cv[i//2] = np.mean(loss_laplace_cv[i//2])
                    eval_loss_laplace_cv_std[i//2] = np.std(loss_laplace_cv[i//2])
                    eval_loss_sc_feat_cv[i//2] = np.mean(loss_sc_feat_cv[i//2])
                    eval_loss_sc_feat_cv_std[i//2] = np.std(loss_sc_feat_cv[i//2])
                    eval_loss_sc_feat_cv_post[i//2] = np.mean(loss_sc_feat_cv_post[i//2])
                    eval_loss_sc_feat_cv_post_std[i//2] = np.std(loss_sc_feat_cv_post[i//2])
                    eval_loss_sc_feat_magsp_cv[i//2] = np.mean(loss_sc_feat_magsp_cv[i//2])
                    eval_loss_sc_feat_magsp_cv_std[i//2] = np.std(loss_sc_feat_magsp_cv[i//2])
                    eval_loss_sc_feat_magsp_cv_post[i//2] = np.mean(loss_sc_feat_magsp_cv_post[i//2])
                    eval_loss_sc_feat_magsp_cv_post_std[i//2] = np.std(loss_sc_feat_magsp_cv_post[i//2])
                    eval_loss_melsp_cv[i//2] = np.mean(loss_melsp_cv[i//2])
                    eval_loss_melsp_cv_std[i//2] = np.std(loss_melsp_cv[i//2])
                    eval_loss_melsp_cv_post[i//2] = np.mean(loss_melsp_cv_post[i//2])
                    eval_loss_melsp_cv_post_std[i//2] = np.std(loss_melsp_cv_post[i//2])
                    eval_loss_magsp_cv[i//2] = np.mean(loss_magsp_cv[i//2])
                    eval_loss_magsp_cv_std[i//2] = np.std(loss_magsp_cv[i//2])
                    eval_loss_magsp_cv_post[i//2] = np.mean(loss_magsp_cv_post[i//2])
                    eval_loss_magsp_cv_post_std[i//2] = np.std(loss_magsp_cv_post[i//2])
            text_log = "(EPOCH:%d) average evaluation loss = %.6f (+- %.6f) %.6f (+- %.6f) ; " % (
                epoch_idx + 1, eval_loss_sc_feat_in, eval_loss_sc_feat_in_std, eval_loss_sc_feat_magsp_in, eval_loss_sc_feat_magsp_in_std)
            for i in range(n_half_cyc_eval):
                if i % 2 == 0:
                    text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                        "%.6f %.6f %.6f %.6f " % (i+1,
                        eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_laplace[i], eval_loss_laplace_std[i],
                        eval_loss_laplace_cv[i//2], eval_loss_laplace_cv_std[i//2], eval_loss_px[i], eval_loss_px_std[i],
                        eval_loss_ms_norm[i], eval_loss_ms_norm_std[i], eval_loss_ms_err[i], eval_loss_ms_err_std[i],
                        eval_loss_ms_norm_post[i], eval_loss_ms_norm_post_std[i], eval_loss_ms_err_post[i], eval_loss_ms_err_post_std[i],
                        eval_loss_ms_norm_magsp[i], eval_loss_ms_norm_magsp_std[i], eval_loss_ms_err_magsp[i], eval_loss_ms_err_magsp_std[i],
                        eval_loss_ms_norm_magsp_post[i], eval_loss_ms_norm_magsp_post_std[i], eval_loss_ms_err_magsp_post[i], eval_loss_ms_err_magsp_post_std[i],
                        eval_loss_sc_feat[i], eval_loss_sc_feat_std[i], eval_loss_sc_feat_cv[i//2], eval_loss_sc_feat_cv_std[i//2],
                        eval_loss_sc_feat_post[i], eval_loss_sc_feat_post_std[i], eval_loss_sc_feat_cv_post[i//2], eval_loss_sc_feat_cv_post_std[i//2],
                        eval_loss_sc_feat_magsp[i], eval_loss_sc_feat_magsp_std[i], eval_loss_sc_feat_magsp_cv[i//2], eval_loss_sc_feat_magsp_cv_std[i//2],
                        eval_loss_sc_feat_magsp_post[i], eval_loss_sc_feat_magsp_post_std[i], eval_loss_sc_feat_magsp_cv_post[i//2], eval_loss_sc_feat_magsp_cv_post_std[i//2],
                        eval_loss_melsp[i], eval_loss_melsp_std[i], eval_loss_melsp_cv[i//2], eval_loss_melsp_cv_std[i//2], eval_loss_melsp_dB[i], eval_loss_melsp_dB_std[i],
                        eval_loss_melsp_post[i], eval_loss_melsp_post_std[i], eval_loss_melsp_cv_post[i//2], eval_loss_melsp_cv_post_std[i//2], eval_loss_melsp_dB_post[i], eval_loss_melsp_dB_post_std[i],
                        eval_loss_magsp[i], eval_loss_magsp_std[i], eval_loss_magsp_cv[i//2], eval_loss_magsp_cv_std[i//2], eval_loss_magsp_dB[i], eval_loss_magsp_dB_std[i],
                        eval_loss_magsp_post[i], eval_loss_magsp_post_std[i], eval_loss_magsp_cv_post[i//2], eval_loss_magsp_cv_post_std[i//2], eval_loss_magsp_dB_post[i], eval_loss_magsp_dB_post_std[i],
                        eval_loss_gv_src_src, eval_loss_gv_src_trg, eval_loss_gv_src_src_post, eval_loss_gv_src_trg_post)
                    if pair_exist:
                        text_log += "%.6f (+- %.6f) dB %.6f (+- %.6f) dB %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (
                            eval_loss_melsp_dB_src_trg, eval_loss_melsp_dB_src_trg_std, eval_loss_melsp_dB_src_trg_post, eval_loss_melsp_dB_src_trg_post_std,
                            eval_loss_lat_dist_rmse, eval_loss_lat_dist_rmse_std, eval_loss_lat_dist_cossim, eval_loss_lat_dist_cossim_std)
                    else:
                        text_log += "n/a (+- n/a) dB n/a (+- n/a) dB n/a (+- n/a) n/a (+- n/a) ;; "
                else:
                    text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                        "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ;; " % (i+1,
                        eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_laplace[i], eval_loss_laplace_std[i], eval_loss_px[i], eval_loss_px_std[i],
                        eval_loss_ms_norm[i], eval_loss_ms_norm_std[i], eval_loss_ms_err[i], eval_loss_ms_err_std[i],
                        eval_loss_ms_norm_post[i], eval_loss_ms_norm_post_std[i], eval_loss_ms_err_post[i], eval_loss_ms_err_post_std[i],
                        eval_loss_ms_norm_magsp[i], eval_loss_ms_norm_magsp_std[i], eval_loss_ms_err_magsp[i], eval_loss_ms_err_magsp_std[i],
                        eval_loss_ms_norm_magsp_post[i], eval_loss_ms_norm_magsp_post_std[i], eval_loss_ms_err_magsp_post[i], eval_loss_ms_err_magsp_post_std[i],
                        eval_loss_sc_feat[i], eval_loss_sc_feat_std[i], eval_loss_sc_feat_post[i], eval_loss_sc_feat_post_std[i],
                        eval_loss_sc_feat_magsp[i], eval_loss_sc_feat_magsp_std[i], eval_loss_sc_feat_magsp_post[i], eval_loss_sc_feat_magsp_post_std[i],
                        eval_loss_melsp[i], eval_loss_melsp_std[i], eval_loss_melsp_dB[i], eval_loss_melsp_dB_std[i],
                        eval_loss_melsp_post[i], eval_loss_melsp_post_std[i], eval_loss_melsp_dB_post[i], eval_loss_melsp_dB_post_std[i],
                        eval_loss_magsp[i], eval_loss_magsp_std[i], eval_loss_magsp_dB[i], eval_loss_magsp_dB_std[i],
                        eval_loss_magsp_post[i], eval_loss_magsp_post_std[i], eval_loss_magsp_dB_post[i], eval_loss_magsp_dB_post_std[i])
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            if (pair_exist and \
                    eval_loss_melsp_dB_src_trg_post <= min_eval_loss_melsp_dB_src_trg_post \
                    or (eval_loss_melsp_dB_src_trg_post+eval_loss_melsp_dB_src_trg_post_std) <= (min_eval_loss_melsp_dB_src_trg_post+min_eval_loss_melsp_dB_src_trg_post_std) \
                    or round(eval_loss_melsp_dB_src_trg_post,2) <= round(min_eval_loss_melsp_dB_src_trg_post,2) \
                    or ((round(eval_loss_melsp_dB_src_trg_post,2)-0.1) <= round(min_eval_loss_melsp_dB_src_trg_post,2) and round(eval_loss_gv_src_trg_post,2) <= round(min_eval_loss_gv_src_trg_post,2)) \
                    or round(eval_loss_melsp_dB_src_trg_post+eval_loss_melsp_dB_src_trg_post_std,2) <= round(min_eval_loss_melsp_dB_src_trg_post+min_eval_loss_melsp_dB_src_trg_post_std,2) \
                    or ((round(eval_loss_melsp_dB_src_trg_post+eval_loss_melsp_dB_src_trg_post_std,2)-0.1) <= round(min_eval_loss_melsp_dB_src_trg_post+min_eval_loss_melsp_dB_src_trg_post_std,2) \
                        and round(eval_loss_gv_src_trg_post,2) <= round(min_eval_loss_gv_src_trg_post,2))) \
                or (not pair_exist and \
                    (eval_loss_melsp_cv_post[0]-eval_loss_melsp_post[0]) >= (min_eval_loss_melsp_cv_post[0]-min_eval_loss_melsp_post[0]) \
                    and eval_loss_melsp_dB_post[0] <= min_eval_loss_melsp_dB_post[0]):
                min_eval_loss_gv_src_src = eval_loss_gv_src_src
                min_eval_loss_gv_src_trg = eval_loss_gv_src_trg
                min_eval_loss_gv_src_src_post = eval_loss_gv_src_src_post
                min_eval_loss_gv_src_trg_post = eval_loss_gv_src_trg_post
                min_eval_loss_sc_feat_in = eval_loss_sc_feat_in
                min_eval_loss_sc_feat_in_std = eval_loss_sc_feat_in_std
                min_eval_loss_sc_feat_magsp_in = eval_loss_sc_feat_magsp_in
                min_eval_loss_sc_feat_magsp_in_std = eval_loss_sc_feat_magsp_in_std
                if pair_exist:
                    min_eval_loss_melsp_dB_src_trg = eval_loss_melsp_dB_src_trg
                    min_eval_loss_melsp_dB_src_trg_std = eval_loss_melsp_dB_src_trg_std
                    min_eval_loss_melsp_dB_src_trg_post = eval_loss_melsp_dB_src_trg_post
                    min_eval_loss_melsp_dB_src_trg_post_std = eval_loss_melsp_dB_src_trg_post_std
                    min_eval_loss_lat_dist_rmse = eval_loss_lat_dist_rmse
                    min_eval_loss_lat_dist_rmse_std = eval_loss_lat_dist_rmse_std
                    min_eval_loss_lat_dist_cossim = eval_loss_lat_dist_cossim
                    min_eval_loss_lat_dist_cossim_std = eval_loss_lat_dist_cossim_std
                for i in range(n_half_cyc_eval):
                    min_eval_loss_elbo[i] = eval_loss_elbo[i]
                    min_eval_loss_elbo_std[i] = eval_loss_elbo_std[i]
                    min_eval_loss_laplace[i] = eval_loss_laplace[i]
                    min_eval_loss_laplace_std[i] = eval_loss_laplace_std[i]
                    min_eval_loss_px[i] = eval_loss_px[i]
                    min_eval_loss_px_std[i] = eval_loss_px_std[i]
                    min_eval_loss_sc_feat[i] = eval_loss_sc_feat[i]
                    min_eval_loss_sc_feat_std[i] = eval_loss_sc_feat_std[i]
                    min_eval_loss_sc_feat_post[i] = eval_loss_sc_feat_post[i]
                    min_eval_loss_sc_feat_post_std[i] = eval_loss_sc_feat_post_std[i]
                    min_eval_loss_sc_feat_magsp[i] = eval_loss_sc_feat_magsp[i]
                    min_eval_loss_sc_feat_magsp_std[i] = eval_loss_sc_feat_magsp_std[i]
                    min_eval_loss_sc_feat_magsp_post[i] = eval_loss_sc_feat_magsp_post[i]
                    min_eval_loss_sc_feat_magsp_post_std[i] = eval_loss_sc_feat_magsp_post_std[i]
                    min_eval_loss_ms_norm[i] = eval_loss_ms_norm[i]
                    min_eval_loss_ms_norm_std[i] = eval_loss_ms_norm_std[i]
                    min_eval_loss_ms_err[i] = eval_loss_ms_err[i]
                    min_eval_loss_ms_err_std[i] = eval_loss_ms_err_std[i]
                    min_eval_loss_ms_norm_post[i] = eval_loss_ms_norm_post[i]
                    min_eval_loss_ms_norm_post_std[i] = eval_loss_ms_norm_post_std[i]
                    min_eval_loss_ms_err_post[i] = eval_loss_ms_err_post[i]
                    min_eval_loss_ms_err_post_std[i] = eval_loss_ms_err_post_std[i]
                    min_eval_loss_ms_norm_magsp[i] = eval_loss_ms_norm_magsp[i]
                    min_eval_loss_ms_norm_magsp_std[i] = eval_loss_ms_norm_magsp_std[i]
                    min_eval_loss_ms_err_magsp[i] = eval_loss_ms_err_magsp[i]
                    min_eval_loss_ms_err_magsp_std[i] = eval_loss_ms_err_magsp_std[i]
                    min_eval_loss_ms_norm_magsp_post[i] = eval_loss_ms_norm_magsp_post[i]
                    min_eval_loss_ms_norm_magsp_post_std[i] = eval_loss_ms_norm_magsp_post_std[i]
                    min_eval_loss_ms_err_magsp_post[i] = eval_loss_ms_err_magsp_post[i]
                    min_eval_loss_ms_err_magsp_post_std[i] = eval_loss_ms_err_magsp_post_std[i]
                    min_eval_loss_melsp[i] = eval_loss_melsp[i]
                    min_eval_loss_melsp_std[i] = eval_loss_melsp_std[i]
                    min_eval_loss_melsp_dB[i] = eval_loss_melsp_dB[i]
                    min_eval_loss_melsp_dB_std[i] = eval_loss_melsp_dB_std[i]
                    min_eval_loss_melsp_post[i] = eval_loss_melsp_post[i]
                    min_eval_loss_melsp_post_std[i] = eval_loss_melsp_post_std[i]
                    min_eval_loss_melsp_dB_post[i] = eval_loss_melsp_dB_post[i]
                    min_eval_loss_melsp_dB_post_std[i] = eval_loss_melsp_dB_post_std[i]
                    min_eval_loss_magsp[i] = eval_loss_magsp[i]
                    min_eval_loss_magsp_std[i] = eval_loss_magsp_std[i]
                    min_eval_loss_magsp_dB[i] = eval_loss_magsp_dB[i]
                    min_eval_loss_magsp_dB_std[i] = eval_loss_magsp_dB_std[i]
                    min_eval_loss_magsp_post[i] = eval_loss_magsp_post[i]
                    min_eval_loss_magsp_post_std[i] = eval_loss_magsp_post_std[i]
                    min_eval_loss_magsp_dB_post[i] = eval_loss_magsp_dB_post[i]
                    min_eval_loss_magsp_dB_post_std[i] = eval_loss_magsp_dB_post_std[i]
                    if i % 2 == 0:
                        min_eval_loss_laplace_cv[i//2] = eval_loss_laplace_cv[i//2]
                        min_eval_loss_laplace_cv_std[i//2] = eval_loss_laplace_cv_std[i//2]
                        min_eval_loss_sc_feat_cv[i//2] = eval_loss_sc_feat_cv[i//2]
                        min_eval_loss_sc_feat_cv_std[i//2] = eval_loss_sc_feat_cv_std[i//2]
                        min_eval_loss_sc_feat_cv_post[i//2] = eval_loss_sc_feat_cv_post[i//2]
                        min_eval_loss_sc_feat_cv_post_std[i//2] = eval_loss_sc_feat_cv_post_std[i//2]
                        min_eval_loss_sc_feat_magsp_cv[i//2] = eval_loss_sc_feat_magsp_cv[i//2]
                        min_eval_loss_sc_feat_magsp_cv_std[i//2] = eval_loss_sc_feat_magsp_cv_std[i//2]
                        min_eval_loss_sc_feat_magsp_cv_post[i//2] = eval_loss_sc_feat_magsp_cv_post[i//2]
                        min_eval_loss_sc_feat_magsp_cv_post_std[i//2] = eval_loss_sc_feat_magsp_cv_post_std[i//2]
                        min_eval_loss_melsp_cv[i//2] = eval_loss_melsp_cv[i//2]
                        min_eval_loss_melsp_cv_std[i//2] = eval_loss_melsp_cv_std[i//2]
                        min_eval_loss_melsp_cv_post[i//2] = eval_loss_melsp_cv_post[i//2]
                        min_eval_loss_melsp_cv_post_std[i//2] = eval_loss_melsp_cv_post_std[i//2]
                        min_eval_loss_magsp_cv[i//2] = eval_loss_magsp_cv[i//2]
                        min_eval_loss_magsp_cv_std[i//2] = eval_loss_magsp_cv_std[i//2]
                        min_eval_loss_magsp_cv_post[i//2] = eval_loss_magsp_cv_post[i//2]
                        min_eval_loss_magsp_cv_post_std[i//2] = eval_loss_magsp_cv_post_std[i//2]
                min_idx = epoch_idx
                #epoch_min_flag = True
                change_min_flag = True
            if change_min_flag:
                text_log = "min_eval_loss = %.6f (+- %.6f) %.6f (+- %.6f) ; " % (
                    min_eval_loss_sc_feat_in, min_eval_loss_sc_feat_in_std,
                        min_eval_loss_sc_feat_magsp_in, min_eval_loss_sc_feat_magsp_in_std)
                for i in range(n_half_cyc_eval):
                    if i % 2 == 0:
                        text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                            "%.6f %.6f %.6f %.6f " % (i+1,
                            min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_laplace[i], min_eval_loss_laplace_std[i],
                            min_eval_loss_laplace_cv[i//2], min_eval_loss_laplace_cv_std[i//2], min_eval_loss_px[i], min_eval_loss_px_std[i],
                            min_eval_loss_ms_norm[i], min_eval_loss_ms_norm_std[i], min_eval_loss_ms_err[i], min_eval_loss_ms_err_std[i],
                            min_eval_loss_ms_norm_post[i], min_eval_loss_ms_norm_post_std[i], min_eval_loss_ms_err_post[i], min_eval_loss_ms_err_post_std[i],
                            min_eval_loss_ms_norm_magsp[i], min_eval_loss_ms_norm_magsp_std[i], min_eval_loss_ms_err_magsp[i], min_eval_loss_ms_err_magsp_std[i],
                            min_eval_loss_ms_norm_magsp_post[i], min_eval_loss_ms_norm_magsp_post_std[i], min_eval_loss_ms_err_magsp_post[i], min_eval_loss_ms_err_magsp_post_std[i],
                            min_eval_loss_sc_feat[i], min_eval_loss_sc_feat_std[i], min_eval_loss_sc_feat_cv[i//2], min_eval_loss_sc_feat_cv_std[i//2],
                            min_eval_loss_sc_feat_post[i], min_eval_loss_sc_feat_post_std[i], min_eval_loss_sc_feat_cv_post[i//2], min_eval_loss_sc_feat_cv_post_std[i//2],
                            min_eval_loss_sc_feat_magsp[i], min_eval_loss_sc_feat_magsp_std[i], min_eval_loss_sc_feat_magsp_cv[i//2], min_eval_loss_sc_feat_magsp_cv_std[i//2],
                            min_eval_loss_sc_feat_magsp_post[i], min_eval_loss_sc_feat_magsp_post_std[i], min_eval_loss_sc_feat_magsp_cv_post[i//2], min_eval_loss_sc_feat_magsp_cv_post_std[i//2],
                            min_eval_loss_melsp[i], min_eval_loss_melsp_std[i], min_eval_loss_melsp_cv[i//2], min_eval_loss_melsp_cv_std[i//2], min_eval_loss_melsp_dB[i], min_eval_loss_melsp_dB_std[i],
                            min_eval_loss_melsp_post[i], min_eval_loss_melsp_post_std[i], min_eval_loss_melsp_cv_post[i//2], min_eval_loss_melsp_cv_post_std[i//2], min_eval_loss_melsp_dB_post[i], min_eval_loss_melsp_dB_post_std[i],
                            min_eval_loss_magsp[i], min_eval_loss_magsp_std[i], min_eval_loss_magsp_cv[i//2], min_eval_loss_magsp_cv_std[i//2], min_eval_loss_magsp_dB[i], min_eval_loss_magsp_dB_std[i],
                            min_eval_loss_magsp_post[i], min_eval_loss_magsp_post_std[i], min_eval_loss_magsp_cv_post[i//2], min_eval_loss_magsp_cv_post_std[i//2], min_eval_loss_magsp_dB_post[i], min_eval_loss_magsp_dB_post_std[i],
                            min_eval_loss_gv_src_src, min_eval_loss_gv_src_trg, min_eval_loss_gv_src_src_post, min_eval_loss_gv_src_trg_post)
                        if pair_exist:
                            text_log += "%.6f (+- %.6f) dB %.6f (+- %.6f) dB %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (
                                min_eval_loss_melsp_dB_src_trg, min_eval_loss_melsp_dB_src_trg_std, min_eval_loss_melsp_dB_src_trg_post, min_eval_loss_melsp_dB_src_trg_post_std,
                                min_eval_loss_lat_dist_rmse, min_eval_loss_lat_dist_rmse_std, min_eval_loss_lat_dist_cossim, min_eval_loss_lat_dist_cossim_std)
                        else:
                            text_log += "n/a (+- n/a) dB n/a (+- n/a) dB n/a (+- n/a) n/a (+- n/a) ;; "
                    else:
                        text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ;; " % (i+1,
                            min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_laplace[i], min_eval_loss_laplace_std[i], min_eval_loss_px[i], min_eval_loss_px_std[i],
                            min_eval_loss_ms_norm[i], min_eval_loss_ms_norm_std[i], min_eval_loss_ms_err[i], min_eval_loss_ms_err_std[i],
                            min_eval_loss_ms_norm_post[i], min_eval_loss_ms_norm_post_std[i], min_eval_loss_ms_err_post[i], min_eval_loss_ms_err_post_std[i],
                            min_eval_loss_ms_norm_magsp[i], min_eval_loss_ms_norm_magsp_std[i], min_eval_loss_ms_err_magsp[i], min_eval_loss_ms_err_magsp_std[i],
                            min_eval_loss_ms_norm_magsp_post[i], min_eval_loss_ms_norm_magsp_post_std[i], min_eval_loss_ms_err_magsp_post[i], min_eval_loss_ms_err_magsp_post_std[i],
                            min_eval_loss_sc_feat[i], min_eval_loss_sc_feat_std[i], min_eval_loss_sc_feat_post[i], min_eval_loss_sc_feat_post_std[i],
                            min_eval_loss_sc_feat_magsp[i], min_eval_loss_sc_feat_magsp_std[i], min_eval_loss_sc_feat_magsp_post[i], min_eval_loss_sc_feat_magsp_post_std[i],
                            min_eval_loss_melsp[i], min_eval_loss_melsp_std[i], min_eval_loss_melsp_dB[i], min_eval_loss_melsp_dB_std[i],
                            min_eval_loss_melsp_post[i], min_eval_loss_melsp_post_std[i], min_eval_loss_melsp_dB_post[i], min_eval_loss_melsp_dB_post_std[i],
                            min_eval_loss_magsp[i], min_eval_loss_magsp_std[i], min_eval_loss_magsp_dB[i], min_eval_loss_magsp_dB_std[i],
                            min_eval_loss_magsp_post[i], min_eval_loss_magsp_post_std[i], min_eval_loss_magsp_dB_post[i], min_eval_loss_magsp_dB_post_std[i])
                logging.info("%s min_idx=%d" % (text_log, min_idx+1))
            #if ((epoch_idx + 1) % args.save_interval_epoch == 0) or (epoch_min_flag):
            if True:
                logging.info('save epoch:%d' % (epoch_idx+1))
                save_checkpoint(args.expdir, model_encoder_melsp, model_decoder_melsp_fix, model_decoder_melsp, model_encoder_excit, model_decoder_excit,
                    model_spk, model_post, model_classifier, min_eval_loss_melsp_dB[0], min_eval_loss_melsp_dB_std[0], min_eval_loss_melsp_cv[0],
                    min_eval_loss_melsp[0], min_eval_loss_melsp_dB_src_trg, min_eval_loss_melsp_dB_src_trg_std, min_eval_loss_gv_src_trg,
                    min_eval_loss_melsp_dB_post[0], min_eval_loss_melsp_dB_post_std[0], min_eval_loss_melsp_cv_post[0],
                    min_eval_loss_melsp_post[0], min_eval_loss_melsp_dB_src_trg_post, min_eval_loss_melsp_dB_src_trg_post_std, min_eval_loss_gv_src_trg_post,
                    iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1, model_spkidtr=model_spkidtr)
            total = 0
            iter_count = 0
            loss_sc_feat_in = []
            loss_sc_feat_magsp_in = []
            for i in range(args.n_half_cyc):
                loss_elbo[i] = []
                loss_px[i] = []
                loss_laplace[i] = []
                loss_laplace_cv[i] = []
                loss_sc_feat[i] = []
                loss_sc_feat_cv[i] = []
                loss_sc_feat_post[i] = []
                loss_sc_feat_cv_post[i] = []
                loss_sc_feat_magsp[i] = []
                loss_sc_feat_magsp_cv[i] = []
                loss_sc_feat_magsp_post[i] = []
                loss_sc_feat_magsp_cv_post[i] = []
                loss_ms_norm[i] = []
                loss_ms_err[i] = []
                loss_ms_norm_post[i] = []
                loss_ms_err_post[i] = []
                loss_ms_norm_magsp[i] = []
                loss_ms_err_magsp[i] = []
                loss_ms_norm_magsp_post[i] = []
                loss_ms_err_magsp_post[i] = []
                loss_melsp[i] = []
                loss_melsp_cv[i] = []
                loss_melsp_dB[i] = []
                loss_melsp_post[i] = []
                loss_melsp_cv_post[i] = []
                loss_melsp_dB_post[i] = []
                loss_magsp[i] = []
                loss_magsp_cv[i] = []
                loss_magsp_dB[i] = []
                loss_magsp_post[i] = []
                loss_magsp_cv_post[i] = []
                loss_magsp_dB_post[i] = []
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model_encoder_melsp.train()
            model_decoder_melsp_fix.train()
            model_decoder_melsp.train()
            model_encoder_excit.train()
            model_decoder_excit.train()
            model_spk.train()
            model_post.train()
            model_classifier.train()
            if args.spkidtr_dim > 0:
                model_spkidtr.train()
            for param in model_decoder_melsp.parameters():
                param.requires_grad = True
            for param in model_decoder_melsp.scale_in.parameters():
                param.requires_grad = False
            for param in model_decoder_melsp.scale_out.parameters():
                param.requires_grad = False
            for param in model_post.parameters():
                param.requires_grad = True
            for param in model_post.scale_in.parameters():
                param.requires_grad = False
            for param in model_post.scale_out.parameters():
                param.requires_grad = False
            for param in model_classifier.parameters():
                param.requires_grad = True
            # start next epoch
            if epoch_idx < args.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_feat, batch_feat_magsp, batch_sc, batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
                    f_bs, f_ss, flens, n_batch_utt, del_index_utt, max_flen, spk_cv, idx_select, idx_select_full, flens_acc = next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

            f_es = f_ss+f_bs
            logging.info(f'{f_ss} {f_bs} {f_es} {max_flen}')
            # handle first pad for input on melsp flow
            flag_cv = True
            i_cv = 0
            i_cv_in = 0
            f_ss_first_pad_left = f_ss-first_pad_left
            f_es_first_pad_right = f_es+first_pad_right
            i_end = args.n_half_cyc*4
            if args.n_half_cyc == 1:
                i_end += 1
            for i in range(i_end):
                if i % 4 == 0: #enc
                    if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                        batch_feat_in[i] = batch_feat[:,f_ss_first_pad_left:f_es_first_pad_right,args.full_excit_dim:]
                    elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                        batch_feat_in[i] = F.pad(batch_feat[:,:f_es_first_pad_right,args.full_excit_dim:].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
                    elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                        batch_feat_in[i] = F.pad(batch_feat[:,f_ss_first_pad_left:max_flen,args.full_excit_dim:].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                    else: # pad left and right need additional replicate
                        batch_feat_in[i] = F.pad(batch_feat[:,:max_flen,args.full_excit_dim:].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                    f_ss_first_pad_left += enc_pad_left
                    f_es_first_pad_right -= enc_pad_right
                else: #dec/post
                    if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                        batch_sc_in[i] = batch_sc[:,f_ss_first_pad_left:f_es_first_pad_right]
                        if flag_cv:
                            batch_sc_cv_in[i_cv_in] = batch_sc_cv_data[i_cv][:,f_ss_first_pad_left:f_es_first_pad_right]
                            i_cv_in += 1
                            if i % 4 == 3:
                                i_cv += 1
                                flag_cv = False
                        else:
                            if (i + 1) % 8 == 0:
                                flag_cv = True
                    elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                        batch_sc_in[i] = F.pad(batch_sc[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[i_cv][:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                            i_cv_in += 1
                            if i % 4 == 3:
                                i_cv += 1
                                flag_cv = False
                        else:
                            if (i + 1) % 8 == 0:
                                flag_cv = True
                    elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                        diff_pad = f_es_first_pad_right - max_flen
                        batch_sc_in[i] = F.pad(batch_sc[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[i_cv][:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                            i_cv_in += 1
                            if i % 4 == 3:
                                i_cv += 1
                                flag_cv = False
                        else:
                            if (i + 1) % 8 == 0:
                                flag_cv = True
                    else: # pad left and right need additional replicate
                        diff_pad = f_es_first_pad_right - max_flen
                        batch_sc_in[i] = F.pad(batch_sc[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[i_cv][:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                            i_cv_in += 1
                            if i % 4 == 3:
                                i_cv += 1
                                flag_cv = False
                        else:
                            if (i + 1) % 8 == 0:
                                flag_cv = True
                    if i % 4 == 1:
                        f_ss_first_pad_left += lf0_pad_left
                        f_es_first_pad_right -= lf0_pad_right
                    elif i % 4 == 2:
                        f_ss_first_pad_left += dec_pad_left
                        f_es_first_pad_right -= dec_pad_right
                    else:
                        f_ss_first_pad_left += post_pad_left
                        f_es_first_pad_right -= post_pad_right
            batch_melsp = batch_feat[:,f_ss:f_es,args.full_excit_dim:]
            batch_magsp = batch_feat_magsp[:,f_ss:f_es]
            batch_excit = batch_feat[:,f_ss:f_es,:args.full_excit_dim]
            batch_sc = batch_sc[:,f_ss:f_es]
            for i in range(n_cv):
                batch_sc_cv[i] = batch_sc_cv_data[i][:,f_ss:f_es]
                batch_excit_cv[i] = batch_feat_cv_data[i][:,f_ss:f_es]

            if f_ss > 0:
                idx_in = 0
                i_cv_in = 0
                for i in range(0,args.n_half_cyc,2):
                    i_cv = i//2
                    j = i+1
                    if len(del_index_utt) > 0:
                        if i == 0:
                            h_feat_in_sc = torch.FloatTensor(np.delete(h_feat_in_sc.cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_feat_magsp_in_sc = torch.FloatTensor(np.delete(h_feat_magsp_in_sc.cpu().data.numpy(), \
                                                            del_index_utt, axis=1)).to(device)
                        h_z[i] = torch.FloatTensor(np.delete(h_z[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_z_e[i] = torch.FloatTensor(np.delete(h_z_e[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_spk[i] = torch.FloatTensor(np.delete(h_spk[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_spk_cv[i_cv] = torch.FloatTensor(np.delete(h_spk_cv[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_lf0[i] = torch.FloatTensor(np.delete(h_lf0[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_lf0_cv[i_cv] = torch.FloatTensor(np.delete(h_lf0_cv[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_melsp[i] = torch.FloatTensor(np.delete(h_melsp[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_melsp_cv[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_melsp_cv_fix[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv_fix[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_melsp_post[i] = torch.FloatTensor(np.delete(h_melsp_post[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_melsp_cv_post[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv_post[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_feat_sc[i] = torch.FloatTensor(np.delete(h_feat_sc[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_feat_magsp_sc[i] = torch.FloatTensor(np.delete(h_feat_magsp_sc[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_feat_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_feat_cv_sc[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_feat_magsp_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_feat_magsp_cv_sc[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_feat_sc_post[i] = torch.FloatTensor(np.delete(h_feat_sc_post[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_feat_magsp_sc_post[i] = torch.FloatTensor(np.delete(h_feat_magsp_sc_post[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_feat_cv_sc_post[i_cv] = torch.FloatTensor(np.delete(h_feat_cv_sc_post[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_feat_magsp_cv_sc_post[i_cv] = torch.FloatTensor(np.delete(h_feat_magsp_cv_sc_post[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_z[j] = torch.FloatTensor(np.delete(h_z[j].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_z_e[j] = torch.FloatTensor(np.delete(h_z_e[j].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        if args.n_half_cyc > 1:
                            h_spk[j] = torch.FloatTensor(np.delete(h_spk[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_lf0[j] = torch.FloatTensor(np.delete(h_lf0[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_melsp[j] = torch.FloatTensor(np.delete(h_melsp[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_melsp_fix[j] = torch.FloatTensor(np.delete(h_melsp_fix[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_melsp_post[j] = torch.FloatTensor(np.delete(h_melsp_post[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_feat_sc[j] = torch.FloatTensor(np.delete(h_feat_sc[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_feat_magsp_sc[j] = torch.FloatTensor(np.delete(h_feat_magsp_sc[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_feat_sc_post[j] = torch.FloatTensor(np.delete(h_feat_sc_post[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_feat_magsp_sc_post[j] = torch.FloatTensor(np.delete(h_feat_magsp_sc_post[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                    if i > 0:
                        idx_in += 1
                        i_cv_in += 1
                        i_1 = i-1
                        if post_pad_right > 0:
                            cyc_rec_feat = batch_melsp_rec_fix[i_1][:,post_pad_left:-post_pad_right].detach()
                        else:
                            cyc_rec_feat = batch_melsp_rec_fix[i_1][:,post_pad_left:].detach()
                        qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(cyc_rec_feat, outpad_right=outpad_rights[idx_in], h=h_z[i], do=True)
                        qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(cyc_rec_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[i], do=True)
                    else:
                        qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z[i], do=True)
                        qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_e[i], do=True)
                        batch_feat_in_sc, h_feat_in_sc = model_classifier(feat=batch_melsp, h=h_feat_in_sc, do=True)
                        batch_feat_magsp_in_sc, h_feat_magsp_in_sc = model_classifier(feat_aux=batch_magsp, h=h_feat_magsp_in_sc, do=True)
                    ## reconstruction and conversion
                    idx_in += 1
                    z_cat = torch.cat((z_e[i], z[i]), 2)
                    if args.spkidtr_dim > 0:
                        spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                        spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                        batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i], do=True)
                        batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv], do=True)
                        batch_lf0_rec[i], h_lf0[i] \
                                = model_decoder_excit(z_e[i], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[i], do=True)
                        batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                = model_decoder_excit(z_e[i], y=spk_cv_code_in, aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], h=h_lf0_cv[i_cv], do=True)
                    else:
                        batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i], do=True)
                        batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv], do=True)
                        batch_lf0_rec[i], h_lf0[i] \
                                = model_decoder_excit(z_e[i], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[i], do=True)
                        batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                = model_decoder_excit(z_e[i], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], h=h_lf0_cv[i_cv], do=True)
                    if lf0_pad_right > 0:
                        z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,lf0_pad_left:-lf0_pad_right]
                            spk_cv_code_in = spk_cv_code_in[:,lf0_pad_left:-lf0_pad_right]
                        batch_spk = batch_spk[:,lf0_pad_left:-lf0_pad_right]
                        batch_spk_cv = batch_spk_cv[:,lf0_pad_left:-lf0_pad_right]
                    else:
                        z_cat = z_cat[:,lf0_pad_left:]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,lf0_pad_left:]
                            spk_cv_code_in = spk_cv_code_in[:,lf0_pad_left:]
                        batch_spk = batch_spk[:,lf0_pad_left:]
                        batch_spk_cv = batch_spk_cv[:,lf0_pad_left:]
                    idx_in += 1
                    i_cv_in += 1
                    if args.spkidtr_dim > 0:
                        batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                            e=batch_lf0_rec[i][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp[i], do=True)
                        batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                            e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv], do=True)
                        batch_melsp_cv_fix[i_cv], h_melsp_cv_fix[i_cv] = model_decoder_melsp_fix(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                            e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_cv_fix[i_cv], do=True)
                    else:
                        batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                            e=batch_lf0_rec[i][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp[i], do=True)
                        batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                            e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv], do=True)
                        batch_melsp_cv_fix[i_cv], h_melsp_cv_fix[i_cv] = model_decoder_melsp_fix(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                            e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_cv_fix[i_cv], do=True)
                    if dec_pad_right > 0:
                        e_post = batch_lf0_rec[i][:,dec_pad_left:-dec_pad_right]
                        e_cv_post = batch_lf0_cv[i_cv][:,dec_pad_left:-dec_pad_right]
                    else:
                        e_post = batch_lf0_rec[i][:,dec_pad_left:]
                        e_cv_post = batch_lf0_cv[i_cv][:,dec_pad_left:]
                    idx_in_1 = idx_in-1
                    feat_len_e = batch_lf0_rec[i].shape[1]
                    batch_lf0_rec[i] = batch_lf0_rec[i][:,outpad_lefts[idx_in_1]:feat_len_e-outpad_rights[idx_in_1]]
                    batch_lf0_cv[i_cv] = batch_lf0_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len_e-outpad_rights[idx_in_1]]
                    ## post
                    idx_in += 1
                    i_cv_in += 1
                    if dec_pad_right > 0:
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,dec_pad_left:-dec_pad_right]
                            spk_cv_code_in = spk_cv_code_in[:,dec_pad_left:-dec_pad_right]
                        batch_spk = batch_spk[:,dec_pad_left:-dec_pad_right]
                        batch_spk_cv = batch_spk_cv[:,dec_pad_left:-dec_pad_right]
                    else:
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,dec_pad_left:]
                            spk_cv_code_in = spk_cv_code_in[:,dec_pad_left:]
                        batch_spk = batch_spk[:,dec_pad_left:]
                        batch_spk_cv = batch_spk_cv[:,dec_pad_left:]
                    if args.spkidtr_dim > 0:
                        batch_pdf[i], batch_melsp_rec_post[i], h_melsp_post[i] \
                                        = model_post(batch_melsp_rec[i], y=spk_code_in, aux=batch_spk,
                                            e=e_post, outpad_right=outpad_rights[idx_in], h=h_melsp_post[i], do=True)
                        batch_pdf_cv[i_cv], batch_melsp_cv_post[i_cv], h_melsp_cv_post[i_cv] \
                                        = model_post(batch_melsp_cv[i_cv], y=spk_cv_code_in, aux=batch_spk_cv,
                                            e=e_cv_post, outpad_right=outpad_rights[idx_in], h=h_melsp_cv_post[i_cv], do=True)
                    else:
                        batch_pdf[i], batch_melsp_rec_post[i], h_melsp_post[i] \
                                        = model_post(batch_melsp_rec[i], y=batch_sc_in[idx_in], aux=batch_spk,
                                            e=e_post, outpad_right=outpad_rights[idx_in], h=h_melsp_post[i], do=True)
                        batch_pdf_cv[i_cv], batch_melsp_cv_post[i_cv], h_melsp_cv_post[i_cv] \
                                        = model_post(batch_melsp_cv[i_cv], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                            e=e_cv_post, outpad_right=outpad_rights[idx_in], h=h_melsp_cv_post[i_cv], do=True)
                    idx_in_1 = idx_in-1
                    feat_len = batch_melsp_rec[i].shape[1]
                    batch_melsp_rec[i] = batch_melsp_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_magsp_rec[i] = torch.matmul((torch.exp(batch_melsp_rec[i])-1)/10000, melfb_t)
                    batch_melsp_cv[i_cv] = batch_melsp_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_magsp_cv[i_cv] = torch.matmul((torch.exp(batch_melsp_cv[i_cv])-1)/10000, melfb_t)
                    batch_feat_rec_sc[i], h_feat_sc[i] = model_classifier(feat=batch_melsp_rec[i], h=h_feat_sc[i], do=True)
                    batch_feat_magsp_rec_sc[i], h_feat_magsp_sc[i] = model_classifier(feat_aux=batch_magsp_rec[i], h=h_feat_magsp_sc[i], do=True)
                    batch_feat_cv_sc[i_cv], h_feat_cv_sc[i_cv] = model_classifier(feat=batch_melsp_cv[i_cv], h=h_feat_cv_sc[i_cv], do=True)
                    batch_feat_magsp_cv_sc[i_cv], h_feat_magsp_cv_sc[i_cv] = model_classifier(feat_aux=batch_magsp_cv[i_cv], h=h_feat_magsp_cv_sc[i_cv], do=True)
                    ## cyclic reconstruction
                    idx_in += 1
                    if post_pad_right > 0:
                        cv_feat = batch_melsp_cv_fix[i_cv][:,post_pad_left:-post_pad_right].detach()
                    else:
                        cv_feat = batch_melsp_cv_fix[i_cv][:,post_pad_left:].detach()
                    qy_logits[j], qz_alpha[j], z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z[j], do=True)
                    qy_logits_e[j], qz_alpha_e[j], z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[j], do=True)
                    feat_len = batch_melsp_rec[i].shape[1]
                    idx_in_1 = idx_in-1
                    feat_len = batch_melsp_rec_post[i].shape[1]
                    batch_pdf[i] = batch_pdf[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_melsp_rec_post[i] = batch_melsp_rec_post[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_magsp_rec_post[i] = torch.matmul((torch.exp(batch_melsp_rec_post[i])-1)/10000, melfb_t)
                    batch_pdf_cv[i_cv] = batch_pdf_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_melsp_cv_post[i_cv] = batch_melsp_cv_post[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_magsp_cv_post[i_cv] = torch.matmul((torch.exp(batch_melsp_cv_post[i_cv])-1)/10000, melfb_t)
                    batch_feat_rec_sc_post[i], h_feat_sc_post[i] = model_classifier(feat=batch_melsp_rec_post[i], h=h_feat_sc_post[i], do=True)
                    batch_feat_magsp_rec_sc_post[i], h_feat_magsp_sc_post[i] = model_classifier(feat_aux=batch_magsp_rec_post[i], h=h_feat_magsp_sc_post[i], do=True)
                    batch_feat_cv_sc_post[i_cv], h_feat_cv_sc_post[i_cv] = model_classifier(feat=batch_melsp_cv_post[i_cv], h=h_feat_cv_sc_post[i_cv], do=True)
                    batch_feat_magsp_cv_sc_post[i_cv], h_feat_magsp_cv_sc_post[i_cv] = model_classifier(feat_aux=batch_magsp_cv_post[i_cv], h=h_feat_magsp_cv_sc_post[i_cv], do=True)
                    if args.n_half_cyc > 1:
                        idx_in += 1
                        z_cat = torch.cat((z_e[j], z[j]), 2)
                        if args.spkidtr_dim > 0:
                            if enc_pad_right > 0:
                                spk_code_in = spk_code_in[:,enc_pad_left:-enc_pad_right]
                            else:
                                spk_code_in = spk_code_in[:,enc_pad_left:]
                            batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j], do=True)
                            batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[j], do=True)
                        else:
                            batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j], do=True)
                            batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[j], do=True)
                        if lf0_pad_right > 0:
                            z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,lf0_pad_left:-lf0_pad_right]
                            batch_spk = batch_spk[:,lf0_pad_left:-lf0_pad_right]
                        else:
                            z_cat = z_cat[:,lf0_pad_left:]
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,lf0_pad_left:]
                            batch_spk = batch_spk[:,lf0_pad_left:]
                        idx_in += 1
                        if args.spkidtr_dim > 0:
                            batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                    e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp[j], do=True)
                            batch_melsp_rec_fix[j], h_melsp_fix[j] = model_decoder_melsp_fix(z_cat, y=spk_code_in, aux=batch_spk,
                                                    e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_fix[j], do=True)
                        else:
                            batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp[j], do=True)
                            batch_melsp_rec_fix[j], h_melsp_fix[j] = model_decoder_melsp_fix(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], h=h_melsp_fix[j], do=True)
                        if dec_pad_right > 0:
                            e_post = batch_lf0_rec[j][:,dec_pad_left:-dec_pad_right]
                        else:
                            e_post = batch_lf0_rec[j][:,dec_pad_left:]
                        idx_in_1 = idx_in-1
                        batch_lf0_rec[j] = batch_lf0_rec[j][:,outpad_lefts[idx_in_1]:batch_lf0_rec[j].shape[1]-outpad_rights[idx_in_1]]
                        ## post
                        idx_in += 1
                        if dec_pad_right > 0:
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,dec_pad_left:-dec_pad_right]
                            batch_spk = batch_spk[:,dec_pad_left:-dec_pad_right]
                        else:
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,dec_pad_left:]
                            batch_spk = batch_spk[:,dec_pad_left:]
                        if args.spkidtr_dim > 0:
                            batch_pdf[j], batch_melsp_rec_post[j], h_melsp_post[j] \
                                            = model_post(batch_melsp_rec[j], y=spk_code_in, aux=batch_spk,
                                                e=e_post, outpad_right=outpad_rights[idx_in], h=h_melsp_post[j], do=True)
                        else:
                            batch_pdf[j], batch_melsp_rec_post[j], h_melsp_post[j] \
                                            = model_post(batch_melsp_rec[j], y=batch_sc_in[idx_in], aux=batch_spk,
                                                e=e_post, outpad_right=outpad_rights[idx_in], h=h_melsp_post[j], do=True)
                        idx_in_1 = idx_in-1
                        batch_melsp_rec[j] = batch_melsp_rec[j][:,outpad_lefts[idx_in_1]:batch_melsp_rec[j].shape[1]-outpad_rights[idx_in_1]]
                        batch_magsp_rec[j] = torch.matmul((torch.exp(batch_melsp_rec[j])-1)/10000, melfb_t)
                        batch_feat_rec_sc[j], h_feat_sc[j] = model_classifier(feat=batch_melsp_rec[j], h=h_feat_sc[j], do=True)
                        batch_feat_magsp_rec_sc[j], h_feat_magsp_sc[j] = model_classifier(feat_aux=batch_magsp_rec[j], h=h_feat_magsp_sc[j], do=True)
                        feat_len = batch_pdf[j].shape[1]
                        batch_pdf[j] = batch_pdf[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                        batch_melsp_rec_post[j] = batch_melsp_rec_post[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                        batch_magsp_rec_post[j] = torch.matmul((torch.exp(batch_melsp_rec_post[j])-1)/10000, melfb_t)
                        batch_feat_rec_sc_post[j], h_feat_sc_post[j] = model_classifier(feat=batch_melsp_rec_post[j], h=h_feat_sc_post[j], do=True)
                        batch_feat_magsp_rec_sc_post[j], h_feat_magsp_sc_post[j] = model_classifier(feat_aux=batch_magsp_rec_post[j], h=h_feat_magsp_sc_post[j], do=True)
            else:
                idx_in = 0
                i_cv_in = 0
                for i in range(0,args.n_half_cyc,2):
                    i_cv = i//2
                    j = i+1
                    if i > 0:
                        idx_in += 1
                        i_cv_in += 1
                        i_1 = i-1
                        if post_pad_right > 0:
                            cyc_rec_feat = batch_melsp_rec_fix[i_1][:,post_pad_left:-post_pad_right].detach()
                        else:
                            cyc_rec_feat = batch_melsp_rec_fix[i_1][:,post_pad_left:].detach()
                        qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(cyc_rec_feat, outpad_right=outpad_rights[idx_in], do=True)
                        qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(cyc_rec_feat, outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        batch_feat_in_sc, h_feat_in_sc = model_classifier(feat=batch_melsp, do=True)
                        batch_feat_magsp_in_sc, h_feat_magsp_in_sc = model_classifier(feat_aux=batch_magsp, do=True)
                    ## reconstruction and conversion
                    idx_in += 1
                    z_cat = torch.cat((z_e[i], z[i]), 2)
                    if args.spkidtr_dim > 0:
                        spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                        spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                        batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                        batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                        batch_lf0_rec[i], h_lf0[i] = model_decoder_excit(z_e[i], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], do=True)
                        batch_lf0_cv[i_cv], h_lf0_cv[i_cv] = model_decoder_excit(z_e[i], y=spk_cv_code_in, aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                        batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                        batch_lf0_rec[i], h_lf0[i] = model_decoder_excit(z_e[i], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], do=True)
                        batch_lf0_cv[i_cv], h_lf0_cv[i_cv] = model_decoder_excit(z_e[i], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], do=True)
                    if lf0_pad_right > 0:
                        z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,lf0_pad_left:-lf0_pad_right]
                            spk_cv_code_in = spk_cv_code_in[:,lf0_pad_left:-lf0_pad_right]
                        batch_spk = batch_spk[:,lf0_pad_left:-lf0_pad_right]
                        batch_spk_cv = batch_spk_cv[:,lf0_pad_left:-lf0_pad_right]
                    else:
                        z_cat = z_cat[:,lf0_pad_left:]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,lf0_pad_left:]
                            spk_cv_code_in = spk_cv_code_in[:,lf0_pad_left:]
                        batch_spk = batch_spk[:,lf0_pad_left:]
                        batch_spk_cv = batch_spk_cv[:,lf0_pad_left:]
                    idx_in += 1
                    i_cv_in += 1
                    if args.spkidtr_dim > 0:
                        batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk,
                                            e=batch_lf0_rec[i][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                        batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                            e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                        batch_melsp_cv_fix[i_cv], h_melsp_cv_fix[i_cv] = model_decoder_melsp_fix(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                            e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                            e=batch_lf0_rec[i][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                        batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                            e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                        batch_melsp_cv_fix[i_cv], h_melsp_cv_fix[i_cv] = model_decoder_melsp_fix(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                            e=batch_lf0_cv[i_cv][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                    if dec_pad_right > 0:
                        e_post = batch_lf0_rec[i][:,dec_pad_left:-dec_pad_right]
                        e_cv_post = batch_lf0_cv[i_cv][:,dec_pad_left:-dec_pad_right]
                    else:
                        e_post = batch_lf0_rec[i][:,dec_pad_left:]
                        e_cv_post = batch_lf0_cv[i_cv][:,dec_pad_left:]
                    idx_in_1 = idx_in-1
                    feat_len_e = batch_lf0_rec[i].shape[1]
                    batch_lf0_rec[i] = batch_lf0_rec[i][:,outpad_lefts[idx_in_1]:feat_len_e-outpad_rights[idx_in_1]]
                    batch_lf0_cv[i_cv] = batch_lf0_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len_e-outpad_rights[idx_in_1]]
                    ## post
                    idx_in += 1
                    i_cv_in += 1
                    if dec_pad_right > 0:
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,dec_pad_left:-dec_pad_right]
                            spk_cv_code_in = spk_cv_code_in[:,dec_pad_left:-dec_pad_right]
                        batch_spk = batch_spk[:,dec_pad_left:-dec_pad_right]
                        batch_spk_cv = batch_spk_cv[:,dec_pad_left:-dec_pad_right]
                    else:
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,dec_pad_left:]
                            spk_cv_code_in = spk_cv_code_in[:,dec_pad_left:]
                        batch_spk = batch_spk[:,dec_pad_left:]
                        batch_spk_cv = batch_spk_cv[:,dec_pad_left:]
                    if args.spkidtr_dim > 0:
                        batch_pdf[i], batch_melsp_rec_post[i], h_melsp_post[i] \
                                        = model_post(batch_melsp_rec[i], y=spk_code_in, aux=batch_spk,
                                            e=e_post, outpad_right=outpad_rights[idx_in], do=True)
                        batch_pdf_cv[i_cv], batch_melsp_cv_post[i_cv], h_melsp_cv_post[i_cv] \
                                        = model_post(batch_melsp_cv[i_cv], y=spk_cv_code_in, aux=batch_spk_cv,
                                            e=e_cv_post, outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        batch_pdf[i], batch_melsp_rec_post[i], h_melsp_post[i] \
                                        = model_post(batch_melsp_rec[i], y=batch_sc_in[idx_in], aux=batch_spk,
                                            e=e_post, outpad_right=outpad_rights[idx_in], do=True)
                        batch_pdf_cv[i_cv], batch_melsp_cv_post[i_cv], h_melsp_cv_post[i_cv] \
                                        = model_post(batch_melsp_cv[i_cv], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                            e=e_cv_post, outpad_right=outpad_rights[idx_in], do=True)
                    idx_in_1 = idx_in-1
                    feat_len = batch_melsp_rec[i].shape[1]
                    batch_melsp_rec[i] = batch_melsp_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_magsp_rec[i] = torch.matmul((torch.exp(batch_melsp_rec[i])-1)/10000, melfb_t)
                    batch_melsp_cv[i_cv] = batch_melsp_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_magsp_cv[i_cv] = torch.matmul((torch.exp(batch_melsp_cv[i_cv])-1)/10000, melfb_t)
                    batch_feat_rec_sc[i], h_feat_sc[i] = model_classifier(feat=batch_melsp_rec[i], do=True)
                    batch_feat_magsp_rec_sc[i], h_feat_magsp_sc[i] = model_classifier(feat_aux=batch_magsp_rec[i], do=True)
                    batch_feat_cv_sc[i_cv], h_feat_cv_sc[i_cv] = model_classifier(feat=batch_melsp_cv[i_cv], do=True)
                    batch_feat_magsp_cv_sc[i_cv], h_feat_magsp_cv_sc[i_cv] = model_classifier(feat_aux=batch_magsp_cv[i_cv], do=True)
                    ## cyclic reconstruction
                    idx_in += 1
                    if post_pad_right > 0:
                        cv_feat = batch_melsp_cv_fix[i_cv][:,post_pad_left:-post_pad_right].detach()
                    else:
                        cv_feat = batch_melsp_cv_fix[i_cv][:,post_pad_left:].detach()
                    qy_logits[j], qz_alpha[j], z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], do=True)
                    qy_logits_e[j], qz_alpha_e[j], z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], do=True)
                    feat_len = batch_melsp_rec[i].shape[1]
                    idx_in_1 = idx_in-1
                    feat_len = batch_melsp_rec_post[i].shape[1]
                    batch_pdf[i] = batch_pdf[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_melsp_rec_post[i] = batch_melsp_rec_post[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_magsp_rec_post[i] = torch.matmul((torch.exp(batch_melsp_rec_post[i])-1)/10000, melfb_t)
                    batch_pdf_cv[i_cv] = batch_pdf_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_melsp_cv_post[i_cv] = batch_melsp_cv_post[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_magsp_cv_post[i_cv] = torch.matmul((torch.exp(batch_melsp_cv_post[i_cv])-1)/10000, melfb_t)
                    batch_feat_rec_sc_post[i], h_feat_sc_post[i] = model_classifier(feat=batch_melsp_rec_post[i], do=True)
                    batch_feat_magsp_rec_sc_post[i], h_feat_magsp_sc_post[i] = model_classifier(feat_aux=batch_magsp_rec_post[i], do=True)
                    batch_feat_cv_sc_post[i_cv], h_feat_cv_sc_post[i_cv] = model_classifier(feat=batch_melsp_cv_post[i_cv], do=True)
                    batch_feat_magsp_cv_sc_post[i_cv], h_feat_magsp_cv_sc_post[i_cv] = model_classifier(feat_aux=batch_magsp_cv_post[i_cv], do=True)
                    if args.n_half_cyc > 1:
                        idx_in += 1
                        z_cat = torch.cat((z_e[j], z[j]), 2)
                        if args.spkidtr_dim > 0:
                            if enc_pad_right > 0:
                                spk_code_in = spk_code_in[:,enc_pad_left:-enc_pad_right]
                            else:
                                spk_code_in = spk_code_in[:,enc_pad_left:]
                            batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                            batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                            batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], do=True)
                        if lf0_pad_right > 0:
                            z_cat = z_cat[:,lf0_pad_left:-lf0_pad_right]
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,lf0_pad_left:-lf0_pad_right]
                            batch_spk = batch_spk[:,lf0_pad_left:-lf0_pad_right]
                        else:
                            z_cat = z_cat[:,lf0_pad_left:]
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,lf0_pad_left:]
                            batch_spk = batch_spk[:,lf0_pad_left:]
                        idx_in += 1
                        if args.spkidtr_dim > 0:
                            batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                    e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                            batch_melsp_rec_fix[j], h_melsp_fix[j] = model_decoder_melsp_fix(z_cat, y=spk_code_in, aux=batch_spk,
                                                    e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                            batch_melsp_rec_fix[j], h_melsp_fix[j] = model_decoder_melsp_fix(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=batch_lf0_rec[j][:,:,:args.excit_dim], outpad_right=outpad_rights[idx_in], do=True)
                        if dec_pad_right > 0:
                            e_post = batch_lf0_rec[j][:,dec_pad_left:-dec_pad_right]
                        else:
                            e_post = batch_lf0_rec[j][:,dec_pad_left:]
                        idx_in_1 = idx_in-1
                        batch_lf0_rec[j] = batch_lf0_rec[j][:,outpad_lefts[idx_in_1]:batch_lf0_rec[j].shape[1]-outpad_rights[idx_in_1]]
                        ## post
                        idx_in += 1
                        if dec_pad_right > 0:
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,dec_pad_left:-dec_pad_right]
                            batch_spk = batch_spk[:,dec_pad_left:-dec_pad_right]
                        else:
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,dec_pad_left:]
                            batch_spk = batch_spk[:,dec_pad_left:]
                        if args.spkidtr_dim > 0:
                            batch_pdf[j], batch_melsp_rec_post[j], h_melsp_post[j] \
                                            = model_post(batch_melsp_rec[j], y=spk_code_in, aux=batch_spk,
                                                e=e_post, outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_pdf[j], batch_melsp_rec_post[j], h_melsp_post[j] \
                                            = model_post(batch_melsp_rec[j], y=batch_sc_in[idx_in], aux=batch_spk,
                                                e=e_post, outpad_right=outpad_rights[idx_in], do=True)
                        idx_in_1 = idx_in-1
                        batch_melsp_rec[j] = batch_melsp_rec[j][:,outpad_lefts[idx_in_1]:batch_melsp_rec[j].shape[1]-outpad_rights[idx_in_1]]
                        batch_magsp_rec[j] = torch.matmul((torch.exp(batch_melsp_rec[j])-1)/10000, melfb_t)
                        batch_feat_rec_sc[j], h_feat_sc[j] = model_classifier(feat=batch_melsp_rec[j], do=True)
                        batch_feat_magsp_rec_sc[j], h_feat_magsp_sc[j] = model_classifier(feat_aux=batch_magsp_rec[j], do=True)
                        feat_len = batch_pdf[j].shape[1]
                        batch_pdf[j] = batch_pdf[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                        batch_melsp_rec_post[j] = batch_melsp_rec_post[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                        batch_magsp_rec_post[j] = torch.matmul((torch.exp(batch_melsp_rec_post[j])-1)/10000, melfb_t)
                        batch_feat_rec_sc_post[j], h_feat_sc_post[j] = model_classifier(feat=batch_melsp_rec_post[j], do=True)
                        batch_feat_magsp_rec_sc_post[j], h_feat_magsp_sc_post[j] = model_classifier(feat_aux=batch_magsp_rec_post[j], do=True)

            # samples check
            with torch.no_grad():
                i = np.random.randint(0, batch_melsp_rec[0].shape[0])
                logging.info("%d %s %d %d %d %d %s" % (i, \
                    os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                        f_ss, f_es, flens[i], max_flen, spk_cv[0][i]))
                logging.info(batch_melsp_rec[0][i,:2,:4])
                logging.info(batch_melsp_rec_post[0][i,:2,:4])
                if args.n_half_cyc > 1:
                    logging.info(batch_melsp_rec[1][i,:2,:4])
                    logging.info(batch_melsp_rec_post[1][i,:2,:4])
                logging.info(batch_melsp[i,:2,:4])
                logging.info(batch_melsp_cv[0][i,:2,:4])
                logging.info(batch_melsp_cv_post[0][i,:2,:4])
                logging.info(batch_lf0_rec[0][i,:2,0])
                if args.n_half_cyc > 1:
                    logging.info(batch_lf0_rec[1][i,:2,0])
                logging.info(batch_excit[i,:2,0])
                logging.info(batch_lf0_cv[0][i,:2,0])
                logging.info(torch.exp(batch_lf0_rec[0][i,:2,1]))
                if args.n_half_cyc > 1:
                    logging.info(torch.exp(batch_lf0_rec[1][i,:2,1]))
                logging.info(torch.exp(batch_excit[i,:2,1]))
                logging.info(torch.exp(batch_lf0_cv[0][i,:2,1]))
                logging.info(torch.exp(batch_excit_cv[0][i,:2,1]))
                logging.info(batch_lf0_rec[0][i,:2,2])
                if args.n_half_cyc > 1:
                    logging.info(batch_lf0_rec[1][i,:2,2])
                logging.info(batch_excit[i,:2,2])
                logging.info(batch_lf0_cv[0][i,:2,2])
                logging.info(-torch.exp(batch_lf0_rec[0][i,:2,3:]))
                if args.n_half_cyc > 1:
                    logging.info(-torch.exp(batch_lf0_rec[1][i,:2,3:]))
                logging.info(-torch.exp(batch_excit[i,:2,3:]))
                logging.info(-torch.exp(batch_lf0_cv[0][i,:2,3:]))
                #logging.info(qy_logits[0][i,:2])
                #logging.info(batch_sc[i,0])
                #logging.info(qy_logits[1][i,:2])
                #logging.info(batch_sc_cv[0][i,0])
                #logging.info(torch.max(z[0][i,5:10], -1))
                #unique, counts = np.unique(torch.max(z[0][i], -1)[1].cpu().data.numpy(), return_counts=True)
                #logging.info(dict(zip(unique, counts)))

            # Losses computation
            batch_loss = 0

            # handle short ending
            if len(idx_select) > 0:
                logging.info('len_idx_select: '+str(len(idx_select)))
                batch_loss_px_select = 0
                batch_loss_px_ms_norm_select = 0
                batch_loss_px_ms_err_select = 0
                batch_loss_sc_feat_in_kl_select = 0
                batch_loss_sc_feat_kl_select = 0
                for j in range(len(idx_select)):
                    k = idx_select[j]
                    flens_utt = flens_acc[k]
                    logging.info('%s %d' % (featfile[k], flens_utt))
                    melsp = batch_melsp[k,:flens_utt]
                    melsp_rest = (torch.exp(melsp)-1)/10000
                    magsp = batch_magsp[k,:flens_utt]
                    batch_excit_select = batch_excit[k,:flens_utt]

                    batch_sc_ = batch_sc[k,:flens_utt]
                    batch_loss_sc_feat_in_kl_select += torch.mean(criterion_ce(batch_feat_in_sc[k,:flens_utt], batch_sc_)) \
                                                    + torch.mean(criterion_ce(batch_feat_magsp_in_sc[k,:flens_utt], batch_sc_))
                    for i in range(args.n_half_cyc):
                        ## reconst. [i % 2 == 0] / cyclic reconst. [i % 2 == 1]
                        melsp_est = batch_melsp_rec[i][k,:flens_utt]
                        melsp_est_rest = (torch.exp(melsp_est)-1)/10000
                        melsp_est_post = batch_melsp_rec_post[i][k,:flens_utt]
                        melsp_est_rest_post = (torch.exp(melsp_est_post)-1)/10000
                        magsp_est = batch_magsp_rec[i][k,:flens_utt]
                        magsp_est_post = batch_magsp_rec_post[i][k,:flens_utt]
                        batch_pdf_ = batch_pdf[i][k,:flens_utt]

                        batch_loss_px_select += criterion_laplace(batch_pdf_[:,:args.mel_dim], batch_pdf_[:,args.mel_dim:], melsp)
                        ## U/V, lf0, codeap, melsp acc.
                        if flens_utt > 1:
                            batch_loss_px_select += torch.mean(torch.sum(criterion_l1(melsp_est, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_est, melsp), -1))) \
                                                        + torch.mean(criterion_l1(melsp_est, melsp)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(melsp_est, melsp))) \
                                                        + torch.mean(torch.sum(criterion_l1(melsp_est_post, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_est_post, melsp), -1))) \
                                                        + torch.mean(criterion_l1(melsp_est_post, melsp)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(melsp_est_post, melsp)))
                            if iter_idx >= 50: #prevent early large losses
                                batch_loss_px_select += torch.mean(torch.sum(criterion_l1(magsp_est, magsp), -1)) \
                                                            + torch.mean(torch.mean(criterion_l1(magsp_est, magsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_est, magsp), -1))) \
                                                            + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_est, magsp), -1))) \
                                                        + torch.mean(torch.sum(criterion_l1(magsp_est_post, magsp), -1)) \
                                                            + torch.mean(torch.mean(criterion_l1(magsp_est_post, magsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_est_post, magsp), -1))) \
                                                            + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_est_post, magsp), -1)))
                        else:
                            batch_loss_px_select += torch.mean(torch.sum(criterion_l1(melsp_est, melsp), -1)) \
                                                        + torch.mean(criterion_l1(melsp_est, melsp)) \
                                                        + torch.mean(torch.sum(criterion_l1(melsp_est_post, melsp), -1)) \
                                                        + torch.mean(criterion_l1(melsp_est_post, melsp))
                            if iter_idx >= 50: #prevent early large losses
                                batch_loss_px_select += torch.mean(torch.sum(criterion_l1(magsp_est, magsp), -1)) \
                                                            + torch.mean(torch.mean(criterion_l1(magsp_est, magsp), -1)) \
                                                            + torch.mean(torch.sum(criterion_l1(magsp_est_post, magsp), -1)) \
                                                            + torch.mean(torch.mean(criterion_l1(magsp_est_post, magsp), -1))

                        if iter_idx >= 50:
                            batch_loss_px_ms_norm_, batch_loss_px_ms_err_ = criterion_ms(melsp_est_rest, melsp_rest)
                            if not torch.isinf(batch_loss_px_ms_norm_) and not torch.isnan(batch_loss_px_ms_norm_):
                                batch_loss_px_ms_norm_select += batch_loss_px_ms_norm_
                            if not torch.isinf(batch_loss_px_ms_err_) and not torch.isnan(batch_loss_px_ms_err_):
                                batch_loss_px_ms_err_select += batch_loss_px_ms_err_

                            batch_loss_px_ms_magsp_norm_, batch_loss_px_ms_magsp_err_ = criterion_ms(magsp_est, magsp)
                            if not torch.isinf(batch_loss_px_ms_magsp_norm_) and not torch.isnan(batch_loss_px_ms_magsp_norm_):
                                batch_loss_px_ms_norm_select += batch_loss_px_ms_magsp_norm_
                            if not torch.isinf(batch_loss_px_ms_magsp_err_) and not torch.isnan(batch_loss_px_ms_magsp_err_):
                                batch_loss_px_ms_err_select += batch_loss_px_ms_magsp_err_

                            batch_loss_px_ms_norm_, batch_loss_px_ms_err_ = criterion_ms(melsp_est_rest_post, melsp_rest)
                            if not torch.isinf(batch_loss_px_ms_norm_) and not torch.isnan(batch_loss_px_ms_norm_):
                                batch_loss_px_ms_norm_select += batch_loss_px_ms_norm_
                            if not torch.isinf(batch_loss_px_ms_err_) and not torch.isnan(batch_loss_px_ms_err_):
                                batch_loss_px_ms_err_select += batch_loss_px_ms_err_

                            batch_loss_px_ms_magsp_norm_, batch_loss_px_ms_magsp_err_ = criterion_ms(magsp_est_post, magsp)
                            if not torch.isinf(batch_loss_px_ms_magsp_norm_) and not torch.isnan(batch_loss_px_ms_magsp_norm_):
                                batch_loss_px_ms_norm_select += batch_loss_px_ms_magsp_norm_
                            if not torch.isinf(batch_loss_px_ms_magsp_err_) and not torch.isnan(batch_loss_px_ms_magsp_err_):
                                batch_loss_px_ms_err_select += batch_loss_px_ms_magsp_err_

                        batch_loss_sc_feat_kl_select += torch.mean(criterion_ce(batch_feat_rec_sc[i][k,:flens_utt], batch_sc_)) \
                                                        + torch.mean(criterion_ce(batch_feat_magsp_rec_sc[i][k,:flens_utt], batch_sc_)) \
                                                        + torch.mean(criterion_ce(batch_feat_rec_sc_post[i][k,:flens_utt], batch_sc_)) \
                                                        + torch.mean(criterion_ce(batch_feat_magsp_rec_sc_post[i][k,:flens_utt], batch_sc_))
                        if i % 2 == 0:
                            batch_sc_cv_ = batch_sc_cv[i//2][k,:flens_utt]
                            ## conversion
                            batch_loss_sc_feat_kl_select += torch.mean(criterion_ce(batch_feat_cv_sc[i//2][k,:flens_utt], batch_sc_cv_)) \
                                                            + torch.mean(criterion_ce(batch_feat_magsp_cv_sc[i//2][k,:flens_utt], batch_sc_cv_)) \
                                                            + torch.mean(criterion_ce(batch_feat_cv_sc_post[i//2][k,:flens_utt], batch_sc_cv_)) \
                                                            + torch.mean(criterion_ce(batch_feat_magsp_cv_sc_post[i//2][k,:flens_utt], batch_sc_cv_))

                batch_loss += batch_loss_px_select + batch_loss_px_ms_norm_select + batch_loss_px_ms_err_select \
                                + batch_loss_sc_feat_kl_select + batch_loss_sc_feat_in_kl_select
                if len(idx_select_full) > 0:
                    logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                    batch_melsp = torch.index_select(batch_melsp,0,idx_select_full)
                    batch_magsp = torch.index_select(batch_magsp,0,idx_select_full)
                    batch_sc = torch.index_select(batch_sc,0,idx_select_full)
                    batch_feat_in_sc = torch.index_select(batch_feat_in_sc,0,idx_select_full)
                    batch_feat_magsp_in_sc = torch.index_select(batch_feat_magsp_in_sc,0,idx_select_full)
                    n_batch_utt = batch_melsp.shape[0]
                    for i in range(args.n_half_cyc):
                        batch_melsp_rec[i] = torch.index_select(batch_melsp_rec[i],0,idx_select_full)
                        batch_magsp_rec[i] = torch.index_select(batch_magsp_rec[i],0,idx_select_full)
                        batch_pdf[i] = torch.index_select(batch_pdf[i],0,idx_select_full)
                        batch_melsp_rec_post[i] = torch.index_select(batch_melsp_rec_post[i],0,idx_select_full)
                        batch_magsp_rec_post[i] = torch.index_select(batch_magsp_rec_post[i],0,idx_select_full)
                        batch_feat_rec_sc[i] = torch.index_select(batch_feat_rec_sc[i],0,idx_select_full)
                        batch_feat_rec_sc_post[i] = torch.index_select(batch_feat_rec_sc_post[i],0,idx_select_full)
                        batch_feat_magsp_rec_sc[i] = torch.index_select(batch_feat_magsp_rec_sc[i],0,idx_select_full)
                        batch_feat_magsp_rec_sc_post[i] = torch.index_select(batch_feat_magsp_rec_sc_post[i],0,idx_select_full)
                        if i % 2 == 0:
                            batch_melsp_cv[i//2] = torch.index_select(batch_melsp_cv[i//2],0,idx_select_full)
                            batch_magsp_cv[i//2] = torch.index_select(batch_magsp_cv[i//2],0,idx_select_full)
                            batch_pdf_cv[i//2] = torch.index_select(batch_pdf_cv[i//2],0,idx_select_full)
                            batch_melsp_cv_post[i//2] = torch.index_select(batch_melsp_cv_post[i//2],0,idx_select_full)
                            batch_magsp_cv_post[i//2] = torch.index_select(batch_magsp_cv_post[i//2],0,idx_select_full)
                            batch_feat_cv_sc[i//2] = torch.index_select(batch_feat_cv_sc[i//2],0,idx_select_full)
                            batch_feat_cv_sc_post[i//2] = torch.index_select(batch_feat_cv_sc_post[i//2],0,idx_select_full)
                            batch_feat_magsp_cv_sc[i//2] = torch.index_select(batch_feat_magsp_cv_sc[i//2],0,idx_select_full)
                            batch_feat_magsp_cv_sc_post[i//2] = torch.index_select(batch_feat_magsp_cv_sc_post[i//2],0,idx_select_full)
                            batch_sc_cv[i//2] = torch.index_select(batch_sc_cv[i//2],0,idx_select_full)
                else:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    text_log = "batch loss_select %lf " % (batch_loss.item())
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    iter_idx += 1
                    #if iter_idx % args.save_interval_iter == 0:
                    #    logging.info('save iter:%d' % (iter_idx))
                    #    save_checkpoint(args.expdir, model_encoder, model_decoder, model_lf0, \
                    #        optimizer, np.random.get_state(), torch.get_rng_state(), iter_idx)
                    iter_count += 1
                    if iter_idx % args.log_interval_steps == 0:
                        logging.info('smt')
                        for key in total_train_loss.keys():
                            total_train_loss[key] = np.mean(total_train_loss[key])
                            logging.info(f"(Steps: {iter_idx}) {key} = {total_train_loss[key]:.4f}.")
                        write_to_tensorboard(writer, iter_idx, total_train_loss)
                        total_train_loss = defaultdict(list)
                    total += time.time() - start
                    continue

            # loss_compute
            melsp = batch_melsp
            melsp_rest = (torch.exp(melsp)-1)/10000
            melsp_rest_log = torch.log10(torch.clamp(melsp_rest, min=1e-16))
            magsp = magsp_rest = batch_magsp
            batch_sc_ = batch_sc.reshape(-1)
            batch_loss_sc_feat_in_ = torch.mean(criterion_ce(batch_feat_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_feat_in = batch_loss_sc_feat_in_.mean()
            batch_loss_sc_feat_magsp_in_ = torch.mean(criterion_ce(batch_feat_magsp_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_feat_magsp_in = batch_loss_sc_feat_magsp_in_.mean()
            batch_loss += batch_loss_sc_feat_in_.sum() + batch_loss_sc_feat_magsp_in_.sum()
            for i in range(args.n_half_cyc):
                ## reconst. [i % 2 == 0] / cyclic reconst. [i % 2 == 1]
                melsp_est = batch_melsp_rec[i]
                melsp_est_rest = (torch.exp(melsp_est)-1)/10000
                melsp_est_post = batch_melsp_rec_post[i]
                melsp_est_rest_post = (torch.exp(melsp_est_post)-1)/10000
                magsp_est = magsp_est_rest = batch_magsp_rec[i]
                magsp_est_post = magsp_est_rest_post = batch_magsp_rec_post[i]
                ## conversion
                if i % 2 == 0:
                    melsp_cv = batch_melsp_cv[i//2]
                    magsp_cv = batch_magsp_cv[i//2]
                    melsp_cv_post = batch_melsp_cv_post[i//2]
                    magsp_cv_post = batch_magsp_cv_post[i//2]

                ## melsp acc.
                batch_loss_melsp_ = torch.mean(torch.sum(criterion_l1(melsp_est, melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_est, melsp), -1), -1)) \
                                    + torch.mean(torch.mean(criterion_l1(melsp_est, melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_est, melsp), -1), -1))
                batch_loss_px_sum = batch_loss_melsp_.sum()
                batch_loss_melsp[i] = batch_loss_melsp_.mean()
                batch_loss_px[i] = batch_loss_melsp_.mean()
                batch_loss_melsp_dB[i] = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_est_rest, min=1e-16))-melsp_rest_log))**2, -1)))

                batch_loss_magsp_ = torch.mean(torch.sum(criterion_l1(magsp_est, magsp), -1), -1) \
                                        + torch.mean(torch.mean(criterion_l1(magsp_est, magsp), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_est, magsp), -1), -1)) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_est, magsp), -1), -1))
                if iter_idx >= 50:
                    batch_loss_px_sum += batch_loss_magsp_.sum()
                batch_loss_magsp[i] = batch_loss_magsp_.mean()
                batch_loss_px[i] += batch_loss_magsp[i]
                batch_loss_magsp_dB[i] = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_est_rest, min=1e-16))
                                                        -torch.log10(torch.clamp(magsp_rest, min=1e-16))))**2, -1)), -1))

                batch_loss_melsp_post_ = torch.mean(torch.sum(criterion_l1(melsp_est_post, melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_est_post, melsp), -1), -1)) \
                                    + torch.mean(torch.mean(criterion_l1(melsp_est_post, melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_est_post, melsp), -1), -1))
                batch_loss_px_sum += batch_loss_melsp_post_.sum()
                batch_loss_melsp_post[i] = batch_loss_melsp_post_.mean()
                batch_loss_px[i] += batch_loss_melsp_post[i]
                batch_loss_melsp_dB_post[i] = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_est_rest_post, min=1e-16))-melsp_rest_log))**2, -1)))

                batch_loss_magsp_post_ = torch.mean(torch.sum(criterion_l1(magsp_est_post, magsp), -1), -1) \
                                        + torch.mean(torch.mean(criterion_l1(magsp_est_post, magsp), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_est_post, magsp), -1), -1)) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_est_post, magsp), -1), -1))
                if iter_idx >= 50:
                    batch_loss_px_sum += batch_loss_magsp_post_.sum()
                batch_loss_magsp_post[i] = batch_loss_magsp_post_.mean()
                batch_loss_px[i] += batch_loss_magsp_post[i]
                batch_loss_magsp_dB_post[i] = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_est_rest_post, min=1e-16))
                                                        -torch.log10(torch.clamp(magsp_rest, min=1e-16))))**2, -1)), -1))

                batch_loss_px_ms_norm_, batch_loss_px_ms_err_ = criterion_ms(melsp_est_rest, melsp_rest)
                batch_loss_ms_norm[i] = batch_loss_px_ms_norm_.mean()
                batch_loss_ms_err[i] = batch_loss_px_ms_err_.mean()
                if iter_idx >= 50:
                    if  not torch.isinf(batch_loss_ms_norm[i]) and not torch.isnan(batch_loss_ms_norm[i]):
                        batch_loss_px_sum += batch_loss_px_ms_norm_.sum()
                    if not torch.isinf(batch_loss_ms_err[i]) and not torch.isnan(batch_loss_ms_err[i]):
                        batch_loss_px_sum += batch_loss_px_ms_err_.sum()

                batch_loss_px_ms_magsp_norm_, batch_loss_px_ms_magsp_err_ = criterion_ms(magsp_est_rest, magsp_rest)
                batch_loss_ms_norm_magsp[i] = batch_loss_px_ms_magsp_norm_.mean()
                batch_loss_ms_err_magsp[i] = batch_loss_px_ms_magsp_err_.mean()
                if iter_idx >= 50:
                    if not torch.isinf(batch_loss_ms_norm_magsp[i]) and not torch.isnan(batch_loss_ms_norm_magsp[i]):
                        batch_loss_px_sum += batch_loss_px_ms_magsp_norm_.sum()
                    if not torch.isinf(batch_loss_ms_err_magsp[i]) and not torch.isnan(batch_loss_ms_err_magsp[i]):
                        batch_loss_px_sum += batch_loss_px_ms_magsp_err_.sum()

                batch_loss_px_ms_norm_post_, batch_loss_px_ms_err_post_ = criterion_ms(melsp_est_rest_post, melsp_rest)
                batch_loss_ms_norm_post[i] = batch_loss_px_ms_norm_post_.mean()
                batch_loss_ms_err_post[i] = batch_loss_px_ms_err_post_.mean()
                if iter_idx >= 50:
                    if  not torch.isinf(batch_loss_ms_norm_post[i]) and not torch.isnan(batch_loss_ms_norm_post[i]):
                        batch_loss_px_sum += batch_loss_px_ms_norm_post_.sum()
                    if not torch.isinf(batch_loss_ms_err_post[i]) and not torch.isnan(batch_loss_ms_err_post[i]):
                        batch_loss_px_sum += batch_loss_px_ms_err_post_.sum()

                batch_loss_px_ms_magsp_norm_post_, batch_loss_px_ms_magsp_err_post_ = criterion_ms(magsp_est_rest_post, magsp_rest)
                batch_loss_ms_norm_magsp_post[i] = batch_loss_px_ms_magsp_norm_post_.mean()
                batch_loss_ms_err_magsp_post[i] = batch_loss_px_ms_magsp_err_post_.mean()
                if iter_idx >= 50:
                    if not torch.isinf(batch_loss_ms_norm_magsp_post[i]) and not torch.isnan(batch_loss_ms_norm_magsp_post[i]):
                        batch_loss_px_sum += batch_loss_px_ms_magsp_norm_post_.sum()
                    if not torch.isinf(batch_loss_ms_err_magsp_post[i]) and not torch.isnan(batch_loss_ms_err_magsp_post[i]):
                        batch_loss_px_sum += batch_loss_px_ms_magsp_err_post_.sum()

                batch_loss_laplace_ = criterion_laplace(batch_pdf[i][:,:,:args.mel_dim], batch_pdf[i][:,:,args.mel_dim:], melsp)
                batch_loss_laplace[i] = batch_loss_laplace_.mean()
                batch_loss_px_sum += batch_loss_laplace_.sum()

                ## conversion
                if i % 2 == 0:
                    batch_loss_melsp_cv[i//2] = torch.mean(torch.sum(criterion_l1(melsp_cv, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_cv, melsp), -1))) \
                                                + torch.mean(criterion_l1(melsp_cv, melsp)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(melsp_cv, melsp)))
                    batch_loss_magsp_cv[i//2] = torch.mean(torch.sum(criterion_l1(magsp_cv, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_cv, magsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_cv, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_cv, magsp), -1)))
                    batch_loss_melsp_cv_post[i//2] = torch.mean(torch.sum(criterion_l1(melsp_cv_post, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_cv_post, melsp), -1))) \
                                                + torch.mean(criterion_l1(melsp_cv_post, melsp)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(melsp_cv_post, melsp)))
                    batch_loss_magsp_cv_post[i//2] = torch.mean(torch.sum(criterion_l1(magsp_cv_post, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_cv_post, magsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_cv_post, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_cv_post, magsp), -1)))
                    batch_loss_laplace_cv[i//2] = torch.mean(criterion_laplace(batch_pdf_cv[i//2][:,:,:args.mel_dim], batch_pdf_cv[i//2][:,:,args.mel_dim:], melsp))

                # KL-div lat., CE and error-percentage spk.
                batch_sc_cv_ = batch_sc_cv[i//2].reshape(-1)
                batch_loss_sc_feat_ = torch.mean(criterion_ce(batch_feat_rec_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                batch_loss_sc_feat[i] = batch_loss_sc_feat_.mean()
                batch_loss_sc_feat_magsp_ = torch.mean(criterion_ce(batch_feat_magsp_rec_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                batch_loss_sc_feat_magsp[i] = batch_loss_sc_feat_magsp_.mean()
                batch_loss_sc_feat_post_ = torch.mean(criterion_ce(batch_feat_rec_sc_post[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                batch_loss_sc_feat_post[i] = batch_loss_sc_feat_post_.mean()
                batch_loss_sc_feat_magsp_post_ = torch.mean(criterion_ce(batch_feat_magsp_rec_sc_post[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                batch_loss_sc_feat_magsp_post[i] = batch_loss_sc_feat_magsp_post_.mean()
                batch_loss_sc_feat_kl = batch_loss_sc_feat_.sum() + batch_loss_sc_feat_magsp_.sum() \
                                        + batch_loss_sc_feat_post_.sum() + batch_loss_sc_feat_magsp_post_.sum()
                if i % 2 == 0:
                    batch_loss_sc_feat_cv_ = torch.mean(criterion_ce(batch_feat_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                    batch_loss_sc_feat_cv[i//2] = batch_loss_sc_feat_cv_.mean()
                    batch_loss_sc_feat_magsp_cv_ = torch.mean(criterion_ce(batch_feat_magsp_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                    batch_loss_sc_feat_magsp_cv[i//2] = batch_loss_sc_feat_magsp_cv_.mean()
                    batch_loss_sc_feat_cv_post_ = torch.mean(criterion_ce(batch_feat_cv_sc_post[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                    batch_loss_sc_feat_cv_post[i//2] = batch_loss_sc_feat_cv_post_.mean()
                    batch_loss_sc_feat_magsp_cv_post_ = torch.mean(criterion_ce(batch_feat_magsp_cv_sc_post[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                    batch_loss_sc_feat_magsp_cv_post[i//2] = batch_loss_sc_feat_magsp_cv_post_.mean()
                    batch_loss_sc_feat_kl += batch_loss_sc_feat_cv_.sum() + batch_loss_sc_feat_magsp_cv_.sum() \
                                            + batch_loss_sc_feat_cv_post_.sum() + batch_loss_sc_feat_magsp_cv_post_.sum()

                # elbo
                batch_loss_elbo[i] = batch_loss_px_sum + batch_loss_sc_feat_kl
                batch_loss += batch_loss_elbo[i]

                total_train_loss["train/loss_elbo-%d"%(i+1)].append(batch_loss_elbo[i].item())
                total_train_loss["train/loss_px-%d"%(i+1)].append(batch_loss_px[i].item())
                total_train_loss["train/loss_laplace-%d"%(i+1)].append(batch_loss_laplace[i].item())
                total_train_loss["train/loss_sc_feat-%d"%(i+1)].append(batch_loss_sc_feat[i].item())
                total_train_loss["train/loss_sc_feat_post-%d"%(i+1)].append(batch_loss_sc_feat_post[i].item())
                total_train_loss["train/loss_sc_feat_magsp-%d"%(i+1)].append(batch_loss_sc_feat_magsp[i].item())
                total_train_loss["train/loss_sc_feat_magsp_post-%d"%(i+1)].append(batch_loss_sc_feat_magsp_post[i].item())
                if i == 0:
                    total_train_loss["train/loss_sc_feat_in"].append(batch_loss_sc_feat_in.item())
                    total_train_loss["train/loss_sc_feat_magsp_in"].append(batch_loss_sc_feat_magsp_in.item())
                    loss_sc_feat_in.append(batch_loss_sc_feat_in.item())
                    loss_sc_feat_magsp_in.append(batch_loss_sc_feat_magsp_in.item())
                total_train_loss["train/loss_ms_norm-%d"%(i+1)].append(batch_loss_ms_norm[i].item())
                total_train_loss["train/loss_ms_err-%d"%(i+1)].append(batch_loss_ms_err[i].item())
                total_train_loss["train/loss_ms_norm_post-%d"%(i+1)].append(batch_loss_ms_norm_post[i].item())
                total_train_loss["train/loss_ms_err_post-%d"%(i+1)].append(batch_loss_ms_err_post[i].item())
                total_train_loss["train/loss_ms_norm_magsp-%d"%(i+1)].append(batch_loss_ms_norm_magsp[i].item())
                total_train_loss["train/loss_ms_err_magsp-%d"%(i+1)].append(batch_loss_ms_err_magsp[i].item())
                total_train_loss["train/loss_ms_norm_magsp_post-%d"%(i+1)].append(batch_loss_ms_norm_magsp_post[i].item())
                total_train_loss["train/loss_ms_err_magsp_post-%d"%(i+1)].append(batch_loss_ms_err_magsp_post[i].item())
                loss_elbo[i].append(batch_loss_elbo[i].item())
                loss_px[i].append(batch_loss_px[i].item())
                loss_laplace[i].append(batch_loss_laplace[i].item())
                loss_sc_feat[i].append(batch_loss_sc_feat[i].item())
                loss_sc_feat_post[i].append(batch_loss_sc_feat_post[i].item())
                loss_sc_feat_magsp[i].append(batch_loss_sc_feat_magsp[i].item())
                loss_sc_feat_magsp_post[i].append(batch_loss_sc_feat_magsp_post[i].item())
                loss_ms_norm[i].append(batch_loss_ms_norm[i].item())
                loss_ms_err[i].append(batch_loss_ms_err[i].item())
                loss_ms_norm_post[i].append(batch_loss_ms_norm_post[i].item())
                loss_ms_err_post[i].append(batch_loss_ms_err_post[i].item())
                loss_ms_norm_magsp[i].append(batch_loss_ms_norm_magsp[i].item())
                loss_ms_err_magsp[i].append(batch_loss_ms_err_magsp[i].item())
                loss_ms_norm_magsp_post[i].append(batch_loss_ms_norm_magsp_post[i].item())
                loss_ms_err_magsp_post[i].append(batch_loss_ms_err_magsp_post[i].item())
                ## in-domain reconst.
                total_train_loss["train/loss_melsp-%d"%(i+1)].append(batch_loss_melsp[i].item())
                total_train_loss["train/loss_melsp_dB-%d"%(i+1)].append(batch_loss_melsp_dB[i].item())
                total_train_loss["train/loss_melsp_post-%d"%(i+1)].append(batch_loss_melsp_post[i].item())
                total_train_loss["train/loss_melsp_dB_post-%d"%(i+1)].append(batch_loss_melsp_dB_post[i].item())
                total_train_loss["train/loss_magsp-%d"%(i+1)].append(batch_loss_magsp[i].item())
                total_train_loss["train/loss_magsp_dB-%d"%(i+1)].append(batch_loss_magsp_dB[i].item())
                total_train_loss["train/loss_magsp_post-%d"%(i+1)].append(batch_loss_magsp_post[i].item())
                total_train_loss["train/loss_magsp_dB_post-%d"%(i+1)].append(batch_loss_magsp_dB_post[i].item())
                loss_melsp[i].append(batch_loss_melsp[i].item())
                loss_melsp_dB[i].append(batch_loss_melsp_dB[i].item())
                loss_melsp_post[i].append(batch_loss_melsp_post[i].item())
                loss_melsp_dB_post[i].append(batch_loss_melsp_dB_post[i].item())
                loss_magsp[i].append(batch_loss_magsp[i].item())
                loss_magsp_dB[i].append(batch_loss_magsp_dB[i].item())
                loss_magsp_post[i].append(batch_loss_magsp_post[i].item())
                loss_magsp_dB_post[i].append(batch_loss_magsp_dB_post[i].item())
                ## conversion
                if i % 2 == 0:
                    total_train_loss["train/loss_laplace_cv-%d"%(i+1)].append(batch_loss_laplace_cv[i//2].item())
                    total_train_loss["train/loss_sc_feat_cv-%d"%(i+1)].append(batch_loss_sc_feat_cv[i//2].item())
                    total_train_loss["train/loss_sc_feat_cv_post-%d"%(i+1)].append(batch_loss_sc_feat_cv_post[i//2].item())
                    total_train_loss["train/loss_sc_feat_magsp_cv-%d"%(i+1)].append(batch_loss_sc_feat_magsp_cv[i//2].item())
                    total_train_loss["train/loss_sc_feat_magsp_cv_post-%d"%(i+1)].append(batch_loss_sc_feat_magsp_cv_post[i//2].item())
                    total_train_loss["train/loss_melsp_cv-%d"%(i+1)].append(batch_loss_melsp_cv[i//2].item())
                    total_train_loss["train/loss_melsp_cv_post-%d"%(i+1)].append(batch_loss_melsp_cv_post[i//2].item())
                    total_train_loss["train/loss_magsp_cv-%d"%(i+1)].append(batch_loss_magsp_cv[i//2].item())
                    total_train_loss["train/loss_magsp_cv_post-%d"%(i+1)].append(batch_loss_magsp_cv_post[i//2].item())
                    loss_laplace_cv[i//2].append(batch_loss_laplace_cv[i//2].item())
                    loss_sc_feat_cv[i//2].append(batch_loss_sc_feat_cv[i//2].item())
                    loss_sc_feat_cv_post[i//2].append(batch_loss_sc_feat_cv_post[i//2].item())
                    loss_sc_feat_magsp_cv[i//2].append(batch_loss_sc_feat_magsp_cv[i//2].item())
                    loss_sc_feat_magsp_cv_post[i//2].append(batch_loss_sc_feat_magsp_cv_post[i//2].item())
                    loss_melsp_cv[i//2].append(batch_loss_melsp_cv[i//2].item())
                    loss_melsp_cv_post[i//2].append(batch_loss_melsp_cv_post[i//2].item())
                    loss_magsp_cv[i//2].append(batch_loss_magsp_cv[i//2].item())
                    loss_magsp_cv_post[i//2].append(batch_loss_magsp_cv_post[i//2].item())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            text_log = "batch loss [%d] %d %d %.3f %.3f " % (c_idx+1, f_ss, f_bs, batch_loss_sc_feat_in.item(), batch_loss_sc_feat_magsp_in.item())
            for i in range(args.n_half_cyc):
                if i % 2 == 0:
                    text_log += "[%ld] %.3f , %.3f %.3f , %.3f ; %.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; "\
                        "%.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; " \
                        "%.3f %.3f %.3f dB , %.3f %.3f %.3f dB , %.3f %.3f %.3f dB , %.3f %.3f %.3f dB ;; " % (i+1,
                        batch_loss_elbo[i].item(), batch_loss_laplace[i].item(), batch_loss_laplace_cv[i//2].item(), batch_loss_px[i].item(),
                            batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                            batch_loss_ms_norm_post[i].item(), batch_loss_ms_err_post[i].item(),
                            batch_loss_ms_norm_magsp[i].item(), batch_loss_ms_err_magsp[i].item(),
                            batch_loss_ms_norm_magsp_post[i].item(), batch_loss_ms_err_magsp_post[i].item(),
                                batch_loss_sc_feat[i].item(), batch_loss_sc_feat_cv[i//2].item(),
                                batch_loss_sc_feat_post[i].item(), batch_loss_sc_feat_cv_post[i//2].item(),
                                batch_loss_sc_feat_magsp[i].item(), batch_loss_sc_feat_magsp_cv[i//2].item(),
                                batch_loss_sc_feat_magsp_post[i].item(), batch_loss_sc_feat_magsp_cv_post[i//2].item(),
                                    batch_loss_melsp[i].item(), batch_loss_melsp_cv[i//2].item(), batch_loss_melsp_dB[i].item(),
                                    batch_loss_melsp_post[i].item(), batch_loss_melsp_cv_post[i//2].item(), batch_loss_melsp_dB_post[i].item(),
                                    batch_loss_magsp[i].item(), batch_loss_magsp_cv[i//2].item(), batch_loss_magsp_dB[i].item(),
                                    batch_loss_magsp_post[i].item(), batch_loss_magsp_cv_post[i//2].item(), batch_loss_magsp_dB_post[i].item())
                else:
                    text_log += "[%ld] %.3f , %.3f , %.3f ; %.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; "\
                        "%.3f , %.3f , %.3f , %.3f ; %.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB ;; " % (i+1,
                        batch_loss_elbo[i].item(), batch_loss_laplace[i].item(), batch_loss_px[i].item(),
                                batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                                batch_loss_ms_norm_post[i].item(), batch_loss_ms_err_post[i].item(),
                                batch_loss_ms_norm_magsp[i].item(), batch_loss_ms_err_magsp[i].item(),
                                batch_loss_ms_norm_magsp_post[i].item(), batch_loss_ms_err_magsp_post[i].item(),
                                    batch_loss_sc_feat[i].item(), batch_loss_sc_feat_post[i].item(),
                                    batch_loss_sc_feat_magsp[i].item(), batch_loss_sc_feat_magsp_post[i].item(),
                                        batch_loss_melsp[i].item(), batch_loss_melsp_dB[i].item(),
                                        batch_loss_melsp_post[i].item(), batch_loss_melsp_dB_post[i].item(),
                                        batch_loss_magsp[i].item(), batch_loss_magsp_dB[i].item(),
                                        batch_loss_magsp_post[i].item(), batch_loss_magsp_dB_post[i].item())
            logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
            iter_idx += 1
            #if iter_idx % args.save_interval_iter == 0:
            #    logging.info('save iter:%d' % (iter_idx))
            #    save_checkpoint(args.expdir, model_encoder, model_decoder, model_lf0, \
            #        optimizer, np.random.get_state(), torch.get_rng_state(), iter_idx)
            #        optimizer, optimizer_excit, np.random.get_state(), torch.get_rng_state(), iter_idx)
            iter_count += 1
            if iter_idx % args.log_interval_steps == 0:
                logging.info('smt')
                for key in total_train_loss.keys():
                    total_train_loss[key] = np.mean(total_train_loss[key])
                    logging.info(f"(Steps: {iter_idx}) {key} = {total_train_loss[key]:.4f}.")
                write_to_tensorboard(writer, iter_idx, total_train_loss)
                total_train_loss = defaultdict(list)
            total += time.time() - start


    logging.info("Maximum epoch is reached, please check the development optimum index, or continue training by increasing maximum epoch.")


if __name__ == "__main__":
    main()
