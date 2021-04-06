#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
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
import math

from collections import defaultdict
from tensorboardX import SummaryWriter

import numpy as np
import six
import torch

from pqmf import PQMF

from torchvision import transforms
from torch.utils.data import DataLoader

from decimal import Decimal

import torch.nn.functional as F

from utils import find_files
from utils import read_hdf5
from utils import read_txt
from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER, GRU_LAT_FEAT_CLASSIFIER
from vcneuvoco import SPKID_TRANSFORM_LAYER, GRU_SPK
from vcneuvoco import GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF, encode_mu_law
from vcneuvoco import kl_laplace_laplace, kl_categorical_categorical_logits, LaplaceLoss
from vcneuvoco import decode_mu_law_torch, MultiResolutionSTFTLoss

import torch_optimizer as optim

from dataset import FeatureDatasetCycMceplf0WavVAE, FeatureDatasetEvalCycMceplf0WavVAE, padding

import librosa
from dtw_c import dtw_c as dtw

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)

MIN_CLAMP = -34.65728569
MAX_CLAMP = 34.65728569


def train_generator(dataloader, device, batch_size, n_cv, upsampling_factor, limit_count=None, n_bands=4):
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
        upsampling_factor_bands = upsampling_factor // n_bands
        for idx, batch in enumerate(dataloader):
            slens = batch['slen'].data.numpy()
            flens = batch['flen'].data.numpy()
            max_slen = np.max(slens) ## get max samples length
            max_flen = np.max(flens) ## get max samples length
            x = batch['x_org'][:,:max_slen*n_bands].to(device)
            xs = batch['x_org_band'][:,:max_slen].to(device)
            xs_c = batch['x_c'][:,:max_slen].to(device)
            xs_f = batch['x_f'][:,:max_slen].to(device)
            feat = batch['feat'][:,:max_flen].to(device)
            feat_magsp = batch['feat_magsp'][:,:max_flen].to(device)
            sc = batch['src_codes'][:,:max_flen].to(device)
            sc_cv = [None]*n_cv
            for i in range(n_cv):
                sc_cv[i] = batch['src_trg_codes_list'][i][:,:max_flen].to(device)
            featfiles = batch['featfile']
            spk_cv = batch['pair_spk_list']
            n_batch_utt = feat.size(0)

            len_frm = max_flen
            x_ss = 0
            f_ss = 0
            x_bs = batch_size*upsampling_factor_bands
            f_bs = batch_size
            delta = batch_size*upsampling_factor_bands
            delta_frm = batch_size
            slens_acc = np.array(slens)
            flens_acc = np.array(flens)
            while True:
                del_index_utt = []
                idx_select = []
                idx_select_full = []
                for i in range(n_batch_utt):
                    if flens_acc[i] <= 0:
                        del_index_utt.append(i)
                if len(del_index_utt) > 0:
                    slens = np.delete(slens, del_index_utt, axis=0)
                    flens = np.delete(flens, del_index_utt, axis=0)
                    x = torch.FloatTensor(np.delete(x.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    xs = torch.FloatTensor(np.delete(xs.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    xs_c = torch.LongTensor(np.delete(xs_c.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    xs_f = torch.LongTensor(np.delete(xs_f.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_magsp = torch.FloatTensor(np.delete(feat_magsp.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc = torch.LongTensor(np.delete(sc.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    for j in range(n_cv):
                        sc_cv[j] = torch.LongTensor(np.delete(sc_cv[j].cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                        spk_cv[j] = np.delete(spk_cv[j], del_index_utt, axis=0)
                    featfiles = np.delete(featfiles, del_index_utt, axis=0)
                    slens_acc = np.delete(slens_acc, del_index_utt, axis=0)
                    flens_acc = np.delete(flens_acc, del_index_utt, axis=0)
                    n_batch_utt -= len(del_index_utt)
                for i in range(n_batch_utt):
                    if flens_acc[i] < f_bs:
                        idx_select.append(i)
                if len(idx_select) > 0:
                    idx_select_full = torch.LongTensor(np.delete(np.arange(n_batch_utt), idx_select, axis=0)).to(device)
                    idx_select = torch.LongTensor(idx_select).to(device)
                yield x, xs, xs_c, xs_f, feat, feat_magsp, sc, sc_cv, c_idx, idx, featfiles, x_bs, x_ss, f_bs, f_ss, slens, flens, \
                    n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, idx_select, idx_select_full, slens_acc, flens_acc
                for i in range(n_batch_utt):
                    slens_acc[i] -= delta
                    flens_acc[i] -= delta_frm

                count += 1
                if limit_count is not None and count > limit_count:
                    break
                len_frm -= delta_frm
                if len_frm > 0:
                    x_ss += delta
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

        yield [], [], [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


def eval_generator(dataloader, device, batch_size, upsampling_factor, limit_count=None, spcidx=True, n_bands=4):
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
        upsampling_factor_bands = upsampling_factor // n_bands
        for idx, batch in enumerate(dataloader):
            slens = batch['slen_src'].data.numpy()
            flens = batch['flen_src'].data.numpy()
            flens_trg = batch['flen_src_trg'].data.numpy()
            flens_spc_src = batch['flen_spc_src'].data.numpy()
            flens_spc_src_trg = batch['flen_spc_src_trg'].data.numpy()
            max_slen = np.max(slens) ## get max samples length
            max_flen = np.max(flens) ## get max samples length
            max_flen_trg = np.max(flens_trg) ## get max samples length
            max_flen_spc_src = np.max(flens_spc_src) ## get max samples length
            max_flen_spc_src_trg = np.max(flens_spc_src_trg) ## get max samples length
            x = batch['x_org'][:,:max_slen*n_bands].to(device)
            xs = batch['x_org_band'][:,:max_slen].to(device)
            xs_c = batch['x_c'][:,:max_slen].to(device)
            xs_f = batch['x_f'][:,:max_slen].to(device)
            feat = batch['h_src'][:,:max_flen].to(device)
            feat_magsp = batch['h_src_magsp'][:,:max_flen].to(device)
            feat_trg = batch['h_src_trg'][:,:max_flen_trg].to(device)
            sc = batch['src_code'][:,:max_flen].to(device)
            sc_cv = batch['src_trg_code'][:,:max_flen].to(device)
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
            x_ss = 0
            f_ss = 0
            x_bs = batch_size*upsampling_factor_bands
            f_bs = batch_size
            delta = batch_size*upsampling_factor_bands
            delta_frm = batch_size
            slens_acc = np.array(slens)
            flens_acc = np.array(flens)
            while True:
                del_index_utt = []
                idx_select = []
                idx_select_full = []
                for i in range(n_batch_utt):
                    if flens_acc[i] <= 0:
                        del_index_utt.append(i)
                if len(del_index_utt) > 0:
                    slens = np.delete(slens, del_index_utt, axis=0)
                    flens = np.delete(flens, del_index_utt, axis=0)
                    flens_trg = np.delete(flens_trg, del_index_utt, axis=0)
                    flens_spc_src = np.delete(flens_spc_src, del_index_utt, axis=0)
                    flens_spc_src_trg = np.delete(flens_spc_src_trg, del_index_utt, axis=0)
                    x = torch.FloatTensor(np.delete(x.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    xs = torch.FloatTensor(np.delete(xs.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    xs_c = torch.LongTensor(np.delete(xs_c.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    xs_f = torch.LongTensor(np.delete(xs_f.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_magsp = torch.FloatTensor(np.delete(feat_magsp.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_trg = torch.FloatTensor(np.delete(feat_trg.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc = torch.LongTensor(np.delete(sc.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc_cv = torch.LongTensor(np.delete(sc_cv.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    spcidx_src = torch.LongTensor(np.delete(spcidx_src.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    spcidx_src_trg = torch.LongTensor(np.delete(spcidx_src_trg.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    spk_cv = np.delete(spk_cv, del_index_utt, axis=0)
                    file_src_trg_flag = np.delete(file_src_trg_flag, del_index_utt, axis=0)
                    featfiles = np.delete(featfiles, del_index_utt, axis=0)
                    slens_acc = np.delete(slens_acc, del_index_utt, axis=0)
                    flens_acc = np.delete(flens_acc, del_index_utt, axis=0)
                    n_batch_utt -= len(del_index_utt)
                for i in range(n_batch_utt):
                    if flens_acc[i] < f_bs:
                        idx_select.append(i)
                if len(idx_select) > 0:
                    idx_select_full = torch.LongTensor(np.delete(np.arange(n_batch_utt), idx_select, axis=0)).to(device)
                    idx_select = torch.LongTensor(idx_select).to(device)
                if spcidx:
                    yield x, xs, xs_c, xs_f, feat, feat_magsp, feat_trg, sc, sc_cv, c_idx, idx, featfiles, x_bs, x_ss, f_bs, f_ss, slens, flens, \
                        n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, file_src_trg_flag, spcidx_src, \
                            spcidx_src_trg, flens_spc_src, flens_spc_src_trg, feat_full, sc_full, sc_cv_full, \
                                idx_select, idx_select_full, slens_acc, flens_acc
                else:
                    yield x, xs, xs_c, xs_f, feat, feat_magsp, feat_trg, sc, sc_cv, c_idx, idx, featfiles, x_bs, x_ss, f_bs, f_ss, slens, flens, \
                        n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, file_src_trg_flag, spcidx_src, \
                            spcidx_src_trg, flens_spc_src, flens_spc_src_trg, idx_select, idx_select_full, slens_acc, flens_acc
                for i in range(n_batch_utt):
                    slens_acc[i] -= delta
                    flens_acc[i] -= delta_frm

                count += 1
                if limit_count is not None and count > limit_count:
                    break
                len_frm -= delta_frm
                if len_frm > 0:
                    x_ss += delta
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

        if spcidx:
            yield [], [], [], [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        else:
            yield [], [], [], [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


def save_checkpoint(checkpoint_dir, model_encoder_melsp_fix, model_encoder_melsp, model_decoder_melsp,
        model_encoder_excit_fix, model_encoder_excit, model_spk, model_classifier,
        model_waveform, min_eval_loss_melsp_dB, min_eval_loss_melsp_dB_std, min_eval_loss_melsp_cv,
        min_eval_loss_melsp, min_eval_loss_laplace_cv, min_eval_loss_laplace,
        min_eval_loss_melsp_dB_src_trg, min_eval_loss_melsp_dB_src_trg_std, min_eval_loss_gv_src_trg,
        min_eval_loss_ce_avg, min_eval_loss_ce_avg_std, min_eval_loss_err_avg, min_eval_loss_err_avg_std,
        min_eval_loss_l1_avg, min_eval_loss_l1_fb, err_flag,
        iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, iterations, model_spkidtr=None):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    model_encoder_melsp_fix.cpu()
    model_encoder_melsp.cpu()
    model_decoder_melsp.cpu()
    model_encoder_excit_fix.cpu()
    model_encoder_excit.cpu()
    model_spk.cpu()
    model_classifier.cpu()
    if model_spkidtr is not None:
        model_spkidtr.cpu()
    model_waveform.cpu()
    checkpoint = {
        "model_encoder_melsp_fix": model_encoder_melsp_fix.state_dict(),
        "model_encoder_melsp": model_encoder_melsp.state_dict(),
        "model_decoder_melsp": model_decoder_melsp.state_dict(),
        "model_encoder_excit_fix": model_encoder_excit_fix.state_dict(),
        "model_encoder_excit": model_encoder_excit.state_dict(),
        "model_spk": model_spk.state_dict(),
        "model_classifier": model_classifier.state_dict(),
        "model_waveform": model_waveform.state_dict(),
        "min_eval_loss_melsp_dB": min_eval_loss_melsp_dB,
        "min_eval_loss_melsp_dB_std": min_eval_loss_melsp_dB_std,
        "min_eval_loss_melsp_cv": min_eval_loss_melsp_cv,
        "min_eval_loss_melsp": min_eval_loss_melsp,
        "min_eval_loss_laplace_cv": min_eval_loss_laplace_cv,
        "min_eval_loss_laplace": min_eval_loss_laplace,
        "min_eval_loss_melsp_dB_src_trg": min_eval_loss_melsp_dB_src_trg,
        "min_eval_loss_melsp_dB_src_trg_std": min_eval_loss_melsp_dB_src_trg_std,
        "min_eval_loss_gv_src_trg": min_eval_loss_gv_src_trg,
        "min_eval_loss_ce_avg": min_eval_loss_ce_avg,
        "min_eval_loss_ce_avg_std": min_eval_loss_ce_avg_std,
        "min_eval_loss_err_avg": min_eval_loss_err_avg,
        "min_eval_loss_err_avg_std": min_eval_loss_err_avg_std,
        "min_eval_loss_l1_avg": min_eval_loss_l1_avg,
        "min_eval_loss_l1_fb": min_eval_loss_l1_fb,
        "err_flag": err_flag,
        "iter_idx": iter_idx,
        "last_epoch": iterations,
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
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-last.pkl")
    model_encoder_melsp_fix.cuda()
    model_encoder_melsp.cuda()
    model_decoder_melsp.cuda()
    model_encoder_excit_fix.cuda()
    model_encoder_excit.cuda()
    model_spk.cuda()
    model_classifier.cuda()
    if model_spkidtr is not None:
        model_spkidtr.cuda()
    model_waveform.cuda()
    logging.info("%d-iter and last checkpoints created." % iterations)


def write_to_tensorboard(writer, steps, loss):
    """Write to tensorboard."""
    for key, value in loss.items():
        writer.add_scalar(key, value, steps)


## Based on lpcnet.py [https://github.com/mozilla/LPCNet/blob/master/src/lpcnet.py]
## Modified to accomodate PyTorch model and n-stages of sparsification
def sparsify(model, iter_idx, t_start, t_end, interval, densities, densities_p=None):
    if iter_idx < t_start or ((iter_idx-t_start) % interval != 0 and iter_idx < t_end):
        pass
    else:
        logging.info('sparsify: %ld %ld %ld %ld' % (iter_idx, t_start, t_end, interval))
        p = model.gru.weight_hh_l0 #recurrent
        nb = p.shape[0] // p.shape[1]
        N = p.shape[0] // nb
        N_16 = N // 16
        ones = torch.diag(torch.ones(N, device=p.device))
        if densities_p is not None:
            for k in range(nb):
                density = densities[k]
                if iter_idx < t_end:
                    r = 1 - (iter_idx-t_start)/(t_end - t_start)
                    density = densities_p[k] - (densities_p[k]-densities[k])*(1 - r)**5
                logging.info('%ld: %lf %lf %lf' % (k+1, densities_p[k], densities[k], density))
                #recurrent weight
                A = p[k*N:(k+1)*N, :]
                #horizontal block structure (16) in input part, in real-time simultaneously computed for each 16 output using 2 registers (256x2 bits)
                L = (A - torch.diag(torch.diag(A))).transpose(1, 0).reshape(N, N_16, 16)
                S = torch.sum(L*L, -1)
                SS, _ = torch.sort(S.reshape(-1))
                thresh = SS[round(N*N_16*(1-density))]
                mask = torch.clamp(torch.repeat_interleave((S>=thresh).float(), 16, dim=1) + ones, max=1).transpose(1,0)
                p[k*N:(k+1)*N, :] = p[k*N:(k+1)*N, :]*mask # outputxinput
        else:
            for k in range(nb):
                density = densities[k]
                if iter_idx < t_end:
                    r = 1 - (iter_idx-t_start)/(t_end - t_start)
                    density = 1 - (1-densities[k])*(1 - r)**5
                logging.info('%ld: 1 %lf %lf' % (k+1, densities[k], density))
                #recurrent weight
                A = p[k*N:(k+1)*N, :]
                L = (A - torch.diag(torch.diag(A))).transpose(1, 0).reshape(N, N_16, 16)
                S = torch.sum(L*L, -1)
                SS, _ = torch.sort(S.reshape(-1))
                thresh = SS[round(N*N_16*(1-density))]
                mask = torch.clamp(torch.repeat_interleave((S>=thresh).float(), 16, dim=1) + ones, max=1).transpose(1,0)
                p[k*N:(k+1)*N, :] = p[k*N:(k+1)*N, :]*mask


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--feats_eval_list", required=True,
                        type=str, help="directory or list of evaluation feat files")
    parser.add_argument("--waveforms",
                        type=str, help="directory or list of wav files")
    parser.add_argument("--waveforms_eval_list",
                        type=str, help="directory or list of evaluation wav files")
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
    parser.add_argument("--kernel_size_enc", default=5,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_enc", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--kernel_size_spk", default=5,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_spk", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--kernel_size_dec", default=5,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_dec", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--upsampling_factor", default=120,
                        type=int, help="number of dimension of aux feats")
    parser.add_argument("--n_bands", default=10,
                        type=int, help="number of bands")
    parser.add_argument("--hidden_units_wave", default=896,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_units_wave_2", default=32,
                        type=int, help="depth of dilation")
    parser.add_argument("--kernel_size_wave", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_wave", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--lpc", default=12,
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
    parser.add_argument("--spkidtr_dim", default=0,
                        type=int, help="number of dimension of reduced one-hot spk-dim (if 0 not apply reduction)")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=30,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--step_count", default=1155000,
                        type=int, help="number of training steps")
    parser.add_argument("--do_prob", default=0.5,
                        type=float, help="dropout probability")
    parser.add_argument("--n_workers", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--n_half_cyc", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_enc", default=False,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_spk", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_dec", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_wave", default=False,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_enc", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_spk", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_dec", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_wave", default=0,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--mid_dim", default=32,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--n_stage", default=4,
                        type=int, help="number of sparsification stages")
    parser.add_argument("--t_start", default=6000,
                        type=int, help="iter idx to start sparsify")
    parser.add_argument("--t_end", default=346000,
                        type=int, help="iter idx to finish densitiy sparsify")
    parser.add_argument("--interval", default=10,
                        type=int, help="interval in finishing densitiy sparsify")
    parser.add_argument("--densities", default="0.018-0.018-0.24",
                        type=str, help="final densitiy of reset, update, new hidden gate matrices")
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
    parser.add_argument("--gen_model_waveform", required=True,
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
    wav_eval_src_list = args.waveforms_eval_list.split('@')
    n_spk_data = len(feat_eval_src_list)
    assert(n_spk_data == len(feat_eval_src_list))
    assert(n_spk_data == len(wav_eval_src_list))
    logging.info(f'{n_spk} {n_spk_data}')

    # save args as conf
    args.string_path = "/log_1pmelmagsp"
    args.n_quantize = 1024
    args.cf_dim = int(np.sqrt(args.n_quantize))
    args.half_n_quantize = args.n_quantize // 2
    args.c_pad = args.half_n_quantize // args.cf_dim
    args.f_pad = args.half_n_quantize % args.cf_dim
    logging.info(f'{args.t_start} {args.t_end} {args.interval} {args.step_count}')
    args.n_half_cyc = 2
    args.t_start = args.t_start // args.n_half_cyc
    args.t_end = args.t_end // args.n_half_cyc
    args.interval = args.interval // args.n_half_cyc
    args.step_count = args.step_count // args.n_half_cyc
    logging.info(f'{args.t_start} {args.t_end} {args.interval} {args.step_count}')
    #args.factor = 0.727272727
    #args.factor = 0.714932127
    #args.factor = 0.7
    #args.factor = 0.686956522
    #args.factor = 0.6
    #args.factor = 0.4424
    args.factor = 0.33
    args.t_start = max(math.ceil(args.t_start * args.factor),1)
    args.t_end = max(math.ceil(args.t_end * args.factor),1)
    args.interval = max(math.ceil(args.interval * args.factor),1)
    args.step_count = max(math.ceil(args.t_end * 3.36),1)
    logging.info(f'{args.t_start} {args.t_end} {args.interval} {args.step_count}')
    torch.save(args, args.expdir + "/model.conf")

    # define network
    model_encoder_melsp_fix = GRU_VAE_ENCODER(
        in_dim=args.mel_dim,
        n_spk=n_spk,
        lat_dim=args.lat_dim,
        hidden_layers=args.hidden_layers_enc,
        hidden_units=args.hidden_units_enc,
        kernel_size=args.kernel_size_enc,
        dilation_size=args.dilation_size_enc,
        causal_conv=args.causal_conv_enc,
        pad_first=True,
        right_size=args.right_size_enc)
    logging.info(model_encoder_melsp_fix)
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
        right_size=args.right_size_enc)
    logging.info(model_encoder_melsp)
    model_decoder_melsp = GRU_SPEC_DECODER(
        feat_dim=args.lat_dim+args.lat_dim_e,
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
        red_dim_upd=args.mel_dim,
        pdf=True)
    logging.info(model_decoder_melsp)
    model_encoder_excit_fix = GRU_VAE_ENCODER(
        in_dim=args.mel_dim,
        n_spk=n_spk,
        lat_dim=args.lat_dim_e,
        hidden_layers=args.hidden_layers_enc,
        hidden_units=args.hidden_units_enc,
        kernel_size=args.kernel_size_enc,
        dilation_size=args.dilation_size_enc,
        causal_conv=args.causal_conv_enc,
        pad_first=True,
        right_size=args.right_size_enc)
    logging.info(model_encoder_excit_fix)
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
        right_size=args.right_size_enc)
    logging.info(model_encoder_excit)
    if args.spkidtr_dim > 0:
        model_spkidtr = SPKID_TRANSFORM_LAYER(
            n_spk=n_spk,
            spkidtr_dim=args.spkidtr_dim)
        logging.info(model_spkidtr)
    else:
        model_spkidtr = None
    model_classifier = GRU_LAT_FEAT_CLASSIFIER(
        lat_dim=args.lat_dim+args.lat_dim_e,
        feat_dim=args.mel_dim,
        feat_aux_dim=args.fftl//2+1,
        n_spk=n_spk,
        hidden_units=32,
        hidden_layers=1)
    logging.info(model_classifier) 
    model_spk = GRU_SPK(
        n_spk=n_spk,
        feat_dim=args.lat_dim+args.lat_dim_e,
        hidden_units=32,
        kernel_size=args.kernel_size_spk,
        dilation_size=args.dilation_size_spk,
        causal_conv=args.causal_conv_spk,
        pad_first=True,
        right_size=args.right_size_spk,
        red_dim=args.mel_dim)
    logging.info(model_spk)
    model_waveform = GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF(
        feat_dim=args.mel_dim,
        upsampling_factor=args.upsampling_factor,
        hidden_units=args.hidden_units_wave,
        hidden_units_2=args.hidden_units_wave_2,
        kernel_size=args.kernel_size_wave,
        dilation_size=args.dilation_size_wave,
        n_quantize=args.n_quantize,
        causal_conv=args.causal_conv_wave,
        lpc=args.lpc,
        right_size=args.right_size_wave,
        n_bands=args.n_bands,
        pad_first=True,
        mid_dim=args.mid_dim)
    logging.info(model_waveform)
    criterion_laplace = LaplaceLoss(sum=False)
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_l1 = torch.nn.L1Loss(reduction='none')
    criterion_l2 = torch.nn.MSELoss(reduction='none')
    pqmf = PQMF(args.n_bands)
    fft_sizes = [256, 128, 64, 32, 16]
    if args.fs == 22050 or args.fs == 44100:
        hop_sizes = [88, 44, 22, 11, 8]
    else:
        hop_sizes = [80, 40, 20, 10, 8]
    win_lengths = [elmt*2 for elmt in hop_sizes]
    if args.fs == 8000:
        fft_sizes_fb = [512, 256, 128, 64, 32]
        hop_sizes_fb = [160, 80, 40, 20, 16]
    elif args.fs <= 24000:
        if args.fs == 16000:
            fft_sizes_fb = [1024, 512, 256, 128, 64]
            hop_sizes_fb = [320, 160, 80, 40, 32]
        elif args.fs == 22050:
            fft_sizes_fb = [1024, 512, 256, 128, 128]
            hop_sizes_fb = [440, 220, 110, 55, 44]
        else:
            fft_sizes_fb = [1024, 512, 256, 128, 128]
            hop_sizes_fb = [480, 240, 120, 60, 48]
    else:
        fft_sizes_fb = [2048, 1024, 512, 256, 256]
        if args.fs == 44100:
            hop_sizes_fb = [880, 440, 220, 110, 88]
        else:
            hop_sizes_fb = [960, 480, 240, 120, 96]
    win_lengths_fb = [elmt*2 for elmt in hop_sizes_fb]
    criterion_stft = MultiResolutionSTFTLoss(
        fft_sizes = fft_sizes,
        hop_sizes = hop_sizes,
        win_lengths = win_lengths,
    )
    criterion_stft_fb = MultiResolutionSTFTLoss(
        fft_sizes = fft_sizes_fb,
        hop_sizes = hop_sizes_fb,
        win_lengths = win_lengths_fb,
    )
    indices_1hot = torch.FloatTensor(np.arange(args.cf_dim))
    p_spk = torch.ones(n_spk)/n_spk
    melfb_t = torch.FloatTensor(np.linalg.pinv(librosa.filters.mel(args.fs, args.fftl, n_mels=args.mel_dim)).T)

    # send to gpu
    if torch.cuda.is_available():
        model_encoder_melsp_fix.cuda()
        model_encoder_melsp.cuda()
        model_decoder_melsp.cuda()
        model_encoder_excit_fix.cuda()
        model_encoder_excit.cuda()
        model_spk.cuda()
        model_classifier.cuda()
        if args.spkidtr_dim > 0:
            model_spkidtr.cuda()
        model_waveform.cuda()
        pqmf.cuda()
        criterion_laplace.cuda()
        criterion_ce.cuda()
        criterion_l1.cuda()
        criterion_l2.cuda()
        criterion_stft.cuda()
        criterion_stft_fb.cuda()
        indices_1hot = indices_1hot.cuda()
        melfb_t = melfb_t.cuda()
        p_spk = p_spk.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)
    logits_p_spk = torch.log(p_spk)

    logging.info(p_spk)
    logging.info(logits_p_spk)

    logging.info(indices_1hot)
    logging.info(criterion_stft.fft_sizes)
    logging.info(criterion_stft.hop_sizes)
    logging.info(criterion_stft.win_lengths)
    logging.info(criterion_stft_fb.fft_sizes)
    logging.info(criterion_stft_fb.hop_sizes)
    logging.info(criterion_stft_fb.win_lengths)
    logging.info(f'{pqmf.subbands} {pqmf.A} {pqmf.taps} {pqmf.cutoff_ratio} {pqmf.beta}')

    model_encoder_melsp_fix.train()
    model_encoder_melsp.train()
    model_decoder_melsp.train()
    model_encoder_excit_fix.train()
    model_encoder_excit.train()
    model_spk.train()
    model_classifier.train()
    if args.spkidtr_dim > 0:
        model_spkidtr.train()
    model_waveform.train()

    parameters = filter(lambda p: p.requires_grad, model_encoder_melsp_fix.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder_melsp_fix): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_encoder_melsp.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder_melsp): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_decoder_melsp.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder_melsp): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_encoder_excit_fix.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder_excit_fix): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_encoder_excit.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder_excit): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_spk.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (spk): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_classifier.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (classifier): %.3f million' % parameters)
    if args.spkidtr_dim > 0:
        parameters = filter(lambda p: p.requires_grad, model_spkidtr.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        logging.info('Trainable Parameters (spkidtr): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_waveform.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (waveform): %.3f million' % parameters)

    if args.resume is None:
        checkpoint = torch.load(args.gen_model)
        model_encoder_melsp_fix.load_state_dict(checkpoint["model_encoder_melsp"])
        model_encoder_melsp.load_state_dict(checkpoint["model_encoder_melsp"])
        model_decoder_melsp.load_state_dict(checkpoint["model_decoder_melsp"], strict=False)
        model_encoder_excit_fix.load_state_dict(checkpoint["model_encoder_excit"])
        model_encoder_excit.load_state_dict(checkpoint["model_encoder_excit"])
        model_classifier.load_state_dict(checkpoint["model_classifier"], strict=False)
        model_spk.load_state_dict(checkpoint["model_spk"])
        if args.spkidtr_dim > 0:
            model_spkidtr.load_state_dict(checkpoint["model_spkidtr"])
        epoch_idx = checkpoint["iterations"]
        logging.info("gen_model from %d-iter checkpoint." % epoch_idx)
        checkpoint = torch.load(args.gen_model_waveform)
        model_waveform.load_state_dict(checkpoint["model_waveform"])
        epoch_idx = checkpoint["iterations"]
        logging.info("gen_model_waveform from %d-iter checkpoint." % epoch_idx)
        epoch_idx = 0

    if model_waveform.use_weight_norm:
        torch.nn.utils.weight_norm(model_waveform.scale_in)

    for param in model_encoder_melsp_fix.parameters():
        param.requires_grad = False
    for param in model_encoder_melsp.parameters():
        param.requires_grad = True
    for param in model_encoder_melsp.scale_in.parameters():
        param.requires_grad = False
    for param in model_decoder_melsp.parameters():
        param.requires_grad = True
    for param in model_decoder_melsp.scale_out.parameters():
        param.requires_grad = False
    for param in model_encoder_excit_fix.parameters():
        param.requires_grad = False
    for param in model_encoder_excit.parameters():
        param.requires_grad = True
    for param in model_encoder_excit.scale_in.parameters():
        param.requires_grad = False
    for param in model_spk.parameters():
        param.requires_grad = True
    if args.spkidtr_dim > 0:
        for param in model_spkidtr.parameters():
            param.requires_grad = True
    for param in model_waveform.parameters():
        param.requires_grad = False

    module_list = list(model_encoder_melsp.conv.parameters())
    module_list += list(model_encoder_melsp.gru.parameters()) + list(model_encoder_melsp.out.parameters())

    module_list += list(model_decoder_melsp.in_red_upd.parameters()) + list(model_decoder_melsp.conv.parameters())
    module_list += list(model_decoder_melsp.gru.parameters()) + list(model_decoder_melsp.out.parameters())

    module_list += list(model_encoder_excit.conv.parameters())
    module_list += list(model_encoder_excit.gru.parameters()) + list(model_encoder_excit.out.parameters())

    module_list += list(model_spk.in_red.parameters()) + list(model_spk.conv.parameters())
    module_list += list(model_spk.gru.parameters()) + list(model_spk.out.parameters())

    module_list += list(model_classifier.conv_lat.parameters()) + list(model_classifier.conv_feat.parameters())
    module_list += list(model_classifier.conv_feat_aux.parameters())
    module_list += list(model_classifier.gru.parameters()) + list(model_classifier.out.parameters())

    if args.spkidtr_dim > 0:
        module_list += list(model_spkidtr.conv.parameters()) + list(model_spkidtr.deconv.parameters())

    # model = ...
    optimizer = optim.RAdam(
        module_list,
        lr= args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    # resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model_encoder_melsp_fix.load_state_dict(checkpoint["model_encoder_melsp_fix"])
        model_encoder_melsp.load_state_dict(checkpoint["model_encoder_melsp"])
        model_decoder_melsp.load_state_dict(checkpoint["model_decoder_melsp"])
        model_encoder_excit_fix.load_state_dict(checkpoint["model_encoder_excit_fix"])
        model_encoder_excit.load_state_dict(checkpoint["model_encoder_excit"])
        model_classifier.load_state_dict(checkpoint["model_classifier"])
        model_spk.load_state_dict(checkpoint["model_spk"])
        if args.spkidtr_dim > 0:
            model_spkidtr.load_state_dict(checkpoint["model_spkidtr"])
        if model_waveform.use_weight_norm:
            torch.nn.utils.remove_weight_norm(model_waveform.scale_in)
        model_waveform.load_state_dict(checkpoint["model_waveform"])
        if model_waveform.use_weight_norm:
            torch.nn.utils.weight_norm(model_waveform.scale_in)
        for param in model_waveform.scale_in.parameters():
            param.requires_grad = False
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)

    def zero_wav_pad(x): return padding(x, args.pad_len*(args.upsampling_factor // args.n_bands), value=args.half_n_quantize)
    def zero_wav_org_pad(x): return padding(x, args.pad_len*args.upsampling_factor, value=args.half_n_quantize)
    def zero_feat_pad(x): return padding(x, args.pad_len, value=None)
    pad_wav_transform = transforms.Compose([zero_wav_pad])
    pad_wav_org_transform = transforms.Compose([zero_wav_org_pad])
    pad_feat_transform = transforms.Compose([zero_feat_pad])

    wav_transform = transforms.Compose([lambda x: encode_mu_law(x, args.n_quantize)])

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
    if os.path.isdir(args.waveforms):
        wav_list = [args.waveforms + "/" + filename for filename in filenames]
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)
    assert(len(feat_list) == len(wav_list))
    n_data = len(feat_list)
    if n_data >= 225:
        batch_size_utt = round(n_data/150)
        if batch_size_utt > 30:
            batch_size_utt = 30
    else:
        batch_size_utt = 1
    logging.info("number of training_data -- batch_size = %d -- %d " % (n_data, batch_size_utt))
    dataset = FeatureDatasetCycMceplf0WavVAE(feat_list, pad_feat_transform, spk_list, stats_list,
                args.n_half_cyc, args.string_path, magsp=True, worgx_flag=True,
                    wav_list=wav_list, pad_wav_transform=pad_wav_transform, wav_transform=wav_transform, pad_wav_org_transform=pad_wav_org_transform,
                        cf_dim=args.cf_dim, upsampling_factor=args.upsampling_factor, n_bands=args.n_bands)
    dataloader = DataLoader(dataset, batch_size=batch_size_utt, shuffle=True, num_workers=args.n_workers)
    #generator = train_generator(dataloader, device, args.batch_size, n_cv, args.upsampling_factor, limit_count=1, n_bands=args.n_bands)
    #generator = train_generator(dataloader, device, args.batch_size, n_cv, args.upsampling_factor, limit_count=20, n_bands=args.n_bands)
    generator = train_generator(dataloader, device, args.batch_size, n_cv, args.upsampling_factor, limit_count=None, n_bands=args.n_bands)

    # define generator evaluation
    feat_list_eval_src_list = [None]*n_spk_data
    for i in range(n_spk_data):
        if os.path.isdir(feat_eval_src_list[i]):
            feat_list_eval_src_list[i] = sorted(find_files(feat_eval_src_list[i], "*.h5", use_dir_name=False))
        elif os.path.isfile(feat_eval_src_list[i]):
            feat_list_eval_src_list[i] = read_txt(feat_eval_src_list[i])
        else:
            logging.error("%s should be directory or list." % (feat_eval_src_list[i]))
            sys.exit(1)
    wav_list_eval_src_list = [None]*n_spk_data
    for i in range(n_spk_data):
        if os.path.isdir(wav_eval_src_list[i]):
            wav_list_eval_src_list[i] = sorted(find_files(wav_eval_src_list[i], "*.h5", use_dir_name=False))
        elif os.path.isfile(wav_eval_src_list[i]):
            wav_list_eval_src_list[i] = read_txt(wav_eval_src_list[i])
        else:
            logging.error("%s should be directory or list." % (wav_eval_src_list[i]))
            sys.exit(1)
        assert(len(feat_list_eval_src_list[i]) == len(wav_list_eval_src_list[i]))
    dataset_eval = FeatureDatasetEvalCycMceplf0WavVAE(feat_list_eval_src_list, pad_feat_transform, spk_list,
                    stats_list, args.string_path, magsp=True, worgx_flag=True, n_spk_data=n_spk_data,
                        wav_list=wav_list_eval_src_list, pad_wav_transform=pad_wav_transform, wav_transform=wav_transform, pad_wav_org_transform=pad_wav_org_transform,
                            cf_dim=args.cf_dim, upsampling_factor=args.upsampling_factor, n_bands=args.n_bands)
    n_eval_data = len(dataset_eval.file_list_src)
    if n_eval_data >= 15:
        batch_size_utt_eval = round(n_eval_data/10)
        if batch_size_utt_eval > 30:
            batch_size_utt_eval = 30
    else:
        batch_size_utt_eval = 1
    logging.info("number of evaluation_data -- batch_size_eval = %d -- %d" % (n_eval_data, batch_size_utt_eval))
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size_utt_eval, shuffle=False, num_workers=args.n_workers)
    #generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=1, n_bands=args.n_bands)
    generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=None, n_bands=args.n_bands)

    writer = SummaryWriter(args.expdir)
    total_train_loss = defaultdict(list)
    total_eval_loss = defaultdict(list)

    gv_mean = [None]*n_spk
    for i in range(n_spk):
        gv_mean[i] = read_hdf5(stats_list[i], "/gv_melsp_mean")

    density_deltas_ = args.densities.split('-')
    logging.info(density_deltas_)
    density_deltas = [None]*len(density_deltas_)
    for i in range(len(density_deltas_)):
        density_deltas[i] = (1-float(density_deltas_[i]))/args.n_stage
    t_deltas = [None]*args.n_stage
    t_starts = [None]*args.n_stage
    t_ends = [None]*args.n_stage
    densities = [None]*args.n_stage
    t_delta = args.t_end - args.t_start + 1
    if args.n_stage > 3:
        t_deltas[0] = round((1/2)*0.2*t_delta)
    else:
        t_deltas[0] = round(0.2*t_delta)
    t_starts[0] = args.t_start
    t_ends[0] = args.t_start + t_deltas[0] - 1
    densities[0] = [None]*len(density_deltas)
    for j in range(len(density_deltas)):
        densities[0][j] = 1-density_deltas[j]
    for i in range(1,args.n_stage):
        if i < args.n_stage-1:
            if args.n_stage > 3:
                if i < 2:
                    t_deltas[i] = round((1/2)*0.2*t_delta)
                else:
                    if args.n_stage > 4:
                        t_deltas[i] = round((1/2)*0.3*t_delta)
                    else:
                        t_deltas[i] = round(0.3*t_delta)
            else:
                t_deltas[i] = round(0.3*t_delta)
        else:
            t_deltas[i] = round(0.5*t_delta)
        t_starts[i] = t_ends[i-1] + 1
        t_ends[i] = t_starts[i] + t_deltas[i] - 1
        densities[i] = [None]*len(density_deltas)
        if i < args.n_stage-1:
            for j in range(len(density_deltas)):
                densities[i][j] = densities[i-1][j]-density_deltas[j]
        else:
            for j in range(len(density_deltas)):
                densities[i][j] = float(density_deltas_[j])
    logging.info(t_delta)
    logging.info(t_deltas)
    logging.info(t_starts)
    logging.info(t_ends)
    logging.info(args.interval)
    logging.info(densities)
    #idx_stage = args.n_stage - 1
    #logging.info(idx_stage)
    idx_stage = 0

    # train
    logging.info(f'n_cyc: {args.n_half_cyc}')
    logging.info(f'n_rec: {n_rec}')
    logging.info(f'n_cv: {n_cv}')
    enc_pad_left = model_encoder_melsp.pad_left
    enc_pad_right = model_encoder_melsp.pad_right
    logging.info(f'enc_pad_left: {enc_pad_left}')
    logging.info(f'enc_pad_right: {enc_pad_right}')
    spk_pad_left = model_spk.pad_left
    spk_pad_right = model_spk.pad_right
    logging.info(f'spk_pad_left: {spk_pad_left}')
    logging.info(f'spk_pad_right: {spk_pad_right}')
    dec_pad_left = model_decoder_melsp.pad_left
    dec_pad_right = model_decoder_melsp.pad_right
    logging.info(f'dec_pad_left: {dec_pad_left}')
    logging.info(f'dec_pad_right: {dec_pad_right}')
    wav_pad_left = model_waveform.pad_left
    wav_pad_right = model_waveform.pad_right
    logging.info(f'wav_pad_left: {wav_pad_left}')
    logging.info(f'wav_pad_right: {wav_pad_right}')
    dec_enc_pad_left = dec_pad_left + wav_pad_left + enc_pad_left
    dec_enc_pad_right = dec_pad_right + wav_pad_right + enc_pad_right
    first_pad_left = (enc_pad_left + spk_pad_left + dec_pad_left + wav_pad_left)*args.n_half_cyc
    first_pad_right = (enc_pad_right + spk_pad_right + dec_pad_right + wav_pad_right)*args.n_half_cyc
    logging.info(f'first_pad_left: {first_pad_left}')
    logging.info(f'first_pad_right: {first_pad_right}')
    outpad_lefts = [None]*args.n_half_cyc*4
    outpad_rights = [None]*args.n_half_cyc*4
    outpad_lefts[0] = first_pad_left-enc_pad_left
    outpad_rights[0] = first_pad_right-enc_pad_right
    for i in range(1,args.n_half_cyc*4):
        if i % 4 == 3:
            outpad_lefts[i] = outpad_lefts[i-1]-wav_pad_left
            outpad_rights[i] = outpad_rights[i-1]-wav_pad_right
        elif i % 4 == 2:
            outpad_lefts[i] = outpad_lefts[i-1]-dec_pad_left
            outpad_rights[i] = outpad_rights[i-1]-dec_pad_right
        elif i % 4 == 1:
            outpad_lefts[i] = outpad_lefts[i-1]-spk_pad_left
            outpad_rights[i] = outpad_rights[i-1]-spk_pad_right
        else:
            outpad_lefts[i] = outpad_lefts[i-1]-enc_pad_left
            outpad_rights[i] = outpad_rights[i-1]-enc_pad_right
    logging.info(outpad_lefts)
    logging.info(outpad_rights)
    batch_feat_in = [None]*args.n_half_cyc*4
    batch_sc_in = [None]*args.n_half_cyc*4
    batch_sc_cv_in = [None]*n_cv*3
    total = 0
    iter_count = 0
    batch_sc_cv = [None]*n_cv
    z = [None]*n_rec
    batch_z_sc = [None]*n_rec
    z_e = [None]*n_rec
    qy_logits = [None]*n_rec
    qy_logits_e = [None]*n_rec
    qz_alpha = [None]*n_rec
    qz_alpha_e = [None]*n_rec
    batch_pdf_rec = [None]*n_rec
    batch_melsp_rec = [None]*n_rec
    batch_magsp_rec = [None]*n_rec
    batch_feat_rec_sc = [None]*n_rec
    batch_feat_magsp_rec_sc = [None]*n_rec
    batch_x_c_output_noclamp = [None]*n_rec
    batch_x_f_output_noclamp = [None]*n_rec
    batch_x_c_output = [None]*n_rec
    batch_x_f_output = [None]*n_rec
    batch_x_output = [None]*n_rec
    batch_x_output_fb = [None]*n_rec
    batch_seg_conv = [None]*n_rec
    batch_conv_sc = [None]*n_rec
    batch_out = [None]*n_rec
    batch_out_2 = [None]*n_rec
    batch_out_f = [None]*n_rec
    batch_signs_c = [None]*n_rec
    batch_scales_c = [None]*n_rec
    batch_logits_c = [None]*n_rec
    batch_signs_f = [None]*n_rec
    batch_scales_f = [None]*n_rec
    batch_logits_f = [None]*n_rec
    batch_pdf_cv = [None]*n_cv
    batch_melsp_cv = [None]*n_cv
    batch_magsp_cv = [None]*n_cv
    batch_feat_cv_sc = [None]*n_cv
    batch_feat_magsp_cv_sc = [None]*n_cv
    h_x = [None]*n_rec
    h_x_2 = [None]*n_rec
    h_f = [None]*n_rec
    h_z = [None]*n_rec
    h_z_sc = [None]*n_rec
    h_z_e = [None]*n_rec
    h_spk = [None]*n_rec
    h_spk_cv = [None]*n_cv
    h_melsp = [None]*n_rec
    h_feat_sc = [None]*n_rec
    h_feat_magsp_sc = [None]*n_rec
    h_melsp_cv = [None]*n_cv
    h_feat_cv_sc = [None]*n_rec
    h_feat_magsp_cv_sc = [None]*n_rec
    loss_elbo = [None]*n_rec
    loss_px = [None]*n_rec
    loss_qy_py = [None]*n_rec
    loss_qy_py_err = [None]*n_rec
    loss_qz_pz = [None]*n_rec
    loss_qy_py_e = [None]*n_rec
    loss_qy_py_err_e = [None]*n_rec
    loss_qz_pz_e = [None]*n_rec
    loss_lat_cossim = [None]*n_rec
    loss_lat_rmse = [None]*n_rec
    loss_sc_z = [None]*n_rec
    loss_sc_feat = [None]*n_rec
    loss_sc_feat_cv = [None]*n_rec
    loss_sc_feat_magsp = [None]*n_rec
    loss_sc_feat_magsp_cv = [None]*n_rec
    loss_laplace = [None]*n_rec
    loss_melsp = [None]*n_rec
    loss_melsp_dB = [None]*n_rec
    loss_laplace_cv = [None]*n_rec
    loss_melsp_cv = [None]*n_rec
    loss_magsp = [None]*n_rec
    loss_magsp_dB = [None]*n_rec
    loss_magsp_cv = [None]*n_rec
    loss_ce_avg = [None]*n_rec
    loss_err_avg = [None]*n_rec
    loss_ce_c_avg = [None]*n_rec
    loss_err_c_avg = [None]*n_rec
    loss_ce_f_avg = [None]*n_rec
    loss_err_f_avg = [None]*n_rec
    loss_seg_conv = [None]*n_rec
    loss_conv_sc = [None]*n_rec
    loss_h = [None]*n_rec
    loss_mid_smpl = [None]*n_rec
    loss_ce = [None]*n_rec
    loss_err = [None]*n_rec
    loss_ce_f = [None]*n_rec
    loss_err_f = [None]*n_rec
    loss_fro_avg = [None]*n_rec
    loss_l1_avg = [None]*n_rec
    loss_fro_fb = [None]*n_rec
    loss_l1_fb = [None]*n_rec
    loss_fro = [None]*n_rec
    loss_l1 = [None]*n_rec
    loss_melsp_dB_src_trg = []
    loss_lat_dist_rmse = []
    loss_lat_dist_cossim = []
    loss_sc_feat_in = []
    loss_sc_feat_magsp_in = []
    for i in range(args.n_half_cyc):
        loss_elbo[i] = []
        loss_px[i] = []
        loss_qy_py[i] = []
        loss_qy_py_err[i] = []
        loss_qz_pz[i] = []
        loss_qy_py_e[i] = []
        loss_qy_py_err_e[i] = []
        loss_qz_pz_e[i] = []
        loss_lat_cossim[i] = []
        loss_lat_rmse[i] = []
        loss_sc_z[i] = []
        loss_sc_feat[i] = []
        loss_sc_feat_cv[i] = []
        loss_sc_feat_magsp[i] = []
        loss_sc_feat_magsp_cv[i] = []
        loss_laplace[i] = []
        loss_melsp[i] = []
        loss_laplace_cv[i] = []
        loss_melsp_cv[i] = []
        loss_melsp_dB[i] = []
        loss_magsp[i] = []
        loss_magsp_cv[i] = []
        loss_magsp_dB[i] = []
        loss_ce_avg[i] = []
        loss_err_avg[i] = []
        loss_ce_c_avg[i] = []
        loss_err_c_avg[i] = []
        loss_ce_f_avg[i] = []
        loss_err_f_avg[i] = []
        loss_seg_conv[i] = []
        loss_conv_sc[i] = []
        loss_h[i] = []
        loss_mid_smpl[i] = []
        loss_fro_avg[i] = []
        loss_l1_avg[i] = []
        loss_fro_fb[i] = []
        loss_l1_fb[i] = []
        loss_ce[i] = [None]*args.n_bands
        loss_err[i] = [None]*args.n_bands
        loss_ce_f[i] = [None]*args.n_bands
        loss_err_f[i] = [None]*args.n_bands
        loss_fro[i] = [None]*args.n_bands
        loss_l1[i] = [None]*args.n_bands
        for j in range(args.n_bands):
            loss_ce[i][j] = []
            loss_err[i][j] = []
            loss_ce_f[i][j] = []
            loss_err_f[i][j] = []
            loss_fro[i][j] = []
            loss_l1[i][j] = []
    batch_loss_laplace = [None]*n_rec
    batch_loss_melsp = [None]*n_rec
    batch_loss_magsp = [None]*n_rec
    batch_loss_sc_feat = [None]*n_rec
    batch_loss_sc_feat_magsp = [None]*n_rec
    batch_loss_melsp_dB = [None]*n_rec
    batch_loss_magsp_dB = [None]*n_rec
    batch_loss_laplace_cv = [None]*n_cv
    batch_loss_melsp_cv = [None]*n_cv
    batch_loss_sc_feat_cv = [None]*n_cv
    batch_loss_magsp_cv = [None]*n_cv
    batch_loss_sc_feat_magsp_cv = [None]*n_cv
    batch_loss_px = [None]*args.n_half_cyc
    batch_loss_qy_py = [None]*n_rec
    batch_loss_qy_py_err = [None]*n_rec
    batch_loss_qz_pz = [None]*n_rec
    batch_loss_qy_py_e = [None]*n_rec
    batch_loss_qy_py_err_e = [None]*n_rec
    batch_loss_qz_pz_e = [None]*n_rec
    batch_loss_lat_cossim = [None]*n_rec
    batch_loss_lat_rmse = [None]*n_rec
    batch_loss_sc_z = [None]*n_rec
    batch_loss_elbo = [None]*n_rec
    batch_loss_seg_conv = [None]*n_rec
    batch_loss_conv_sc = [None]*n_rec
    batch_loss_h = [None]*n_rec
    batch_loss_mid_smpl = [None]*n_rec
    batch_loss_ce_c_avg = [None]*n_rec
    batch_loss_err_c_avg = [None]*n_rec
    batch_loss_ce_f_avg = [None]*n_rec
    batch_loss_err_f_avg = [None]*n_rec
    batch_loss_ce_avg = [None]*n_rec
    batch_loss_err_avg = [None]*n_rec
    batch_loss_ce = [None]*n_rec
    batch_loss_err = [None]*n_rec
    batch_loss_ce_f = [None]*n_rec
    batch_loss_err_f = [None]*n_rec
    batch_loss_fro_avg = [None]*n_rec
    batch_loss_l1_avg = [None]*n_rec
    batch_loss_fro_fb = [None]*n_rec
    batch_loss_l1_fb = [None]*n_rec
    batch_loss_fro = [None]*n_rec
    batch_loss_l1 = [None]*n_rec
    batch_loss_ce_select = [None]*n_rec
    batch_loss_err_select = [None]*n_rec
    batch_loss_ce_f_select = [None]*n_rec
    batch_loss_err_f_select = [None]*n_rec
    batch_loss_fro_select = [None]*n_rec
    batch_loss_l1_select = [None]*n_rec
    for i in range(args.n_half_cyc):
        batch_loss_ce[i] = [None]*args.n_bands
        batch_loss_err[i] = [None]*args.n_bands
        batch_loss_ce_f[i] = [None]*args.n_bands
        batch_loss_err_f[i] = [None]*args.n_bands
        batch_loss_fro[i] = [None]*args.n_bands
        batch_loss_l1[i] = [None]*args.n_bands
    n_half_cyc_eval = min(2,args.n_half_cyc)
    n_rec_eval = n_half_cyc_eval + n_half_cyc_eval%2
    n_cv_eval = int(n_half_cyc_eval/2+n_half_cyc_eval%2)
    first_pad_left_eval_utt_dec = spk_pad_left + dec_pad_left
    first_pad_right_eval_utt_dec = spk_pad_right + dec_pad_right
    logging.info(f'first_pad_left_eval_utt_dec: {first_pad_left_eval_utt_dec}')
    logging.info(f'first_pad_right_eval_utt_dec: {first_pad_right_eval_utt_dec}')
    first_pad_left_eval_utt = enc_pad_left + first_pad_left_eval_utt_dec
    first_pad_right_eval_utt = enc_pad_right + first_pad_right_eval_utt_dec
    logging.info(f'first_pad_left_eval_utt: {first_pad_left_eval_utt}')
    logging.info(f'first_pad_right_eval_utt: {first_pad_right_eval_utt}')
    gv_src_src = [None]*n_spk
    gv_src_trg = [None]*n_spk
    for i in range(n_spk):
        gv_src_src[i] = []
        gv_src_trg[i] = []
    eval_loss_elbo = [None]*n_half_cyc_eval
    eval_loss_elbo_std = [None]*n_half_cyc_eval
    eval_loss_px = [None]*n_half_cyc_eval
    eval_loss_px_std = [None]*n_half_cyc_eval
    eval_loss_qy_py = [None]*n_rec_eval
    eval_loss_qy_py_std = [None]*n_rec_eval
    eval_loss_qy_py_err = [None]*n_rec_eval
    eval_loss_qy_py_err_std = [None]*n_rec_eval
    eval_loss_qz_pz = [None]*n_rec_eval
    eval_loss_qz_pz_std = [None]*n_rec_eval
    eval_loss_qy_py_e = [None]*n_rec_eval
    eval_loss_qy_py_e_std = [None]*n_rec_eval
    eval_loss_qy_py_err_e = [None]*n_rec_eval
    eval_loss_qy_py_err_e_std = [None]*n_rec_eval
    eval_loss_qz_pz_e = [None]*n_rec_eval
    eval_loss_qz_pz_e_std = [None]*n_rec_eval
    eval_loss_lat_cossim = [None]*n_rec_eval
    eval_loss_lat_cossim_std = [None]*n_rec_eval
    eval_loss_lat_rmse = [None]*n_rec_eval
    eval_loss_lat_rmse_std = [None]*n_rec_eval
    eval_loss_sc_z = [None]*n_rec_eval
    eval_loss_sc_z_std = [None]*n_rec_eval
    eval_loss_sc_feat = [None]*n_rec_eval
    eval_loss_sc_feat_std = [None]*n_rec_eval
    eval_loss_sc_feat_cv = [None]*n_cv_eval
    eval_loss_sc_feat_cv_std = [None]*n_cv_eval
    eval_loss_sc_feat_magsp = [None]*n_rec_eval
    eval_loss_sc_feat_magsp_std = [None]*n_rec_eval
    eval_loss_sc_feat_magsp_cv = [None]*n_cv_eval
    eval_loss_sc_feat_magsp_cv_std = [None]*n_cv_eval
    eval_loss_laplace = [None]*n_half_cyc_eval
    eval_loss_laplace_std = [None]*n_half_cyc_eval
    eval_loss_melsp = [None]*n_half_cyc_eval
    eval_loss_melsp_std = [None]*n_half_cyc_eval
    eval_loss_melsp_dB = [None]*n_half_cyc_eval
    eval_loss_melsp_dB_std = [None]*n_half_cyc_eval
    eval_loss_laplace_cv = [None]*n_cv_eval
    eval_loss_laplace_cv_std = [None]*n_cv_eval
    eval_loss_melsp_cv = [None]*n_cv_eval
    eval_loss_melsp_cv_std = [None]*n_cv_eval
    eval_loss_magsp = [None]*n_half_cyc_eval
    eval_loss_magsp_std = [None]*n_half_cyc_eval
    eval_loss_magsp_dB = [None]*n_half_cyc_eval
    eval_loss_magsp_dB_std = [None]*n_half_cyc_eval
    eval_loss_magsp_cv = [None]*n_cv_eval
    eval_loss_magsp_cv_std = [None]*n_cv_eval
    eval_loss_seg_conv = [None]*n_half_cyc_eval
    eval_loss_seg_conv_std = [None]*n_half_cyc_eval
    eval_loss_conv_sc = [None]*n_half_cyc_eval
    eval_loss_conv_sc_std = [None]*n_half_cyc_eval
    eval_loss_h = [None]*n_half_cyc_eval
    eval_loss_h_std = [None]*n_half_cyc_eval
    eval_loss_mid_smpl = [None]*n_half_cyc_eval
    eval_loss_mid_smpl_std = [None]*n_half_cyc_eval
    eval_loss_ce_c_avg = [None]*n_half_cyc_eval
    eval_loss_ce_c_avg_std = [None]*n_half_cyc_eval
    eval_loss_err_c_avg = [None]*n_half_cyc_eval
    eval_loss_err_c_avg_std = [None]*n_half_cyc_eval
    eval_loss_ce_f_avg = [None]*n_half_cyc_eval
    eval_loss_ce_f_avg_std = [None]*n_half_cyc_eval
    eval_loss_err_f_avg = [None]*n_half_cyc_eval
    eval_loss_err_f_avg_std = [None]*n_half_cyc_eval
    eval_loss_ce_avg = [None]*n_half_cyc_eval
    eval_loss_ce_avg_std = [None]*n_half_cyc_eval
    eval_loss_err_avg = [None]*n_half_cyc_eval
    eval_loss_err_avg_std = [None]*n_half_cyc_eval
    eval_loss_ce = [None]*n_half_cyc_eval
    eval_loss_ce_std = [None]*n_half_cyc_eval
    eval_loss_err = [None]*n_half_cyc_eval
    eval_loss_err_std = [None]*n_half_cyc_eval
    eval_loss_ce_f = [None]*n_half_cyc_eval
    eval_loss_ce_f_std = [None]*n_half_cyc_eval
    eval_loss_err_f = [None]*n_half_cyc_eval
    eval_loss_err_f_std = [None]*n_half_cyc_eval
    eval_loss_fro_avg = [None]*n_half_cyc_eval
    eval_loss_fro_avg_std = [None]*n_half_cyc_eval
    eval_loss_l1_avg = [None]*n_half_cyc_eval
    eval_loss_l1_avg_std = [None]*n_half_cyc_eval
    eval_loss_fro_fb = [None]*n_half_cyc_eval
    eval_loss_fro_fb_std = [None]*n_half_cyc_eval
    eval_loss_l1_fb = [None]*n_half_cyc_eval
    eval_loss_l1_fb_std = [None]*n_half_cyc_eval
    eval_loss_fro = [None]*n_half_cyc_eval
    eval_loss_fro_std = [None]*n_half_cyc_eval
    eval_loss_l1 = [None]*n_half_cyc_eval
    eval_loss_l1_std = [None]*n_half_cyc_eval
    for i in range(n_half_cyc_eval):
        eval_loss_ce[i] = [None]*args.n_bands
        eval_loss_ce_std[i] = [None]*args.n_bands
        eval_loss_err[i] = [None]*args.n_bands
        eval_loss_err_std[i] = [None]*args.n_bands
        eval_loss_ce_f[i] = [None]*args.n_bands
        eval_loss_ce_f_std[i] = [None]*args.n_bands
        eval_loss_err_f[i] = [None]*args.n_bands
        eval_loss_err_f_std[i] = [None]*args.n_bands
        eval_loss_fro[i] = [None]*args.n_bands
        eval_loss_fro_std[i] = [None]*args.n_bands
        eval_loss_l1[i] = [None]*args.n_bands
        eval_loss_l1_std[i] = [None]*args.n_bands
    min_eval_loss_elbo = [None]*n_half_cyc_eval
    min_eval_loss_elbo_std = [None]*n_half_cyc_eval
    min_eval_loss_px = [None]*n_half_cyc_eval
    min_eval_loss_px_std = [None]*n_half_cyc_eval
    min_eval_loss_qy_py = [None]*n_rec_eval
    min_eval_loss_qy_py_std = [None]*n_rec_eval
    min_eval_loss_qy_py_err = [None]*n_rec_eval
    min_eval_loss_qy_py_err_std = [None]*n_rec_eval
    min_eval_loss_qz_pz = [None]*n_rec_eval
    min_eval_loss_qz_pz_std = [None]*n_rec_eval
    min_eval_loss_qy_py_e = [None]*n_rec_eval
    min_eval_loss_qy_py_e_std = [None]*n_rec_eval
    min_eval_loss_qy_py_err_e = [None]*n_rec_eval
    min_eval_loss_qy_py_err_e_std = [None]*n_rec_eval
    min_eval_loss_qz_pz_e = [None]*n_rec_eval
    min_eval_loss_qz_pz_e_std = [None]*n_rec_eval
    min_eval_loss_lat_cossim = [None]*n_rec_eval
    min_eval_loss_lat_cossim_std = [None]*n_rec_eval
    min_eval_loss_lat_rmse = [None]*n_rec_eval
    min_eval_loss_lat_rmse_std = [None]*n_rec_eval
    min_eval_loss_sc_z = [None]*n_rec_eval
    min_eval_loss_sc_z_std = [None]*n_rec_eval
    min_eval_loss_sc_feat = [None]*n_rec_eval
    min_eval_loss_sc_feat_std = [None]*n_rec_eval
    min_eval_loss_sc_feat_cv = [None]*n_cv_eval
    min_eval_loss_sc_feat_cv_std = [None]*n_cv_eval
    min_eval_loss_sc_feat_magsp = [None]*n_rec_eval
    min_eval_loss_sc_feat_magsp_std = [None]*n_rec_eval
    min_eval_loss_sc_feat_magsp_cv = [None]*n_cv_eval
    min_eval_loss_sc_feat_magsp_cv_std = [None]*n_cv_eval
    min_eval_loss_laplace = [None]*n_half_cyc_eval
    min_eval_loss_laplace_std = [None]*n_half_cyc_eval
    min_eval_loss_melsp = [None]*n_half_cyc_eval
    min_eval_loss_melsp_std = [None]*n_half_cyc_eval
    min_eval_loss_melsp_dB = [None]*n_half_cyc_eval
    min_eval_loss_melsp_dB_std = [None]*n_half_cyc_eval
    min_eval_loss_laplace_cv = [None]*n_cv_eval
    min_eval_loss_laplace_cv_std = [None]*n_cv_eval
    min_eval_loss_melsp_cv = [None]*n_cv_eval
    min_eval_loss_melsp_cv_std = [None]*n_cv_eval
    min_eval_loss_magsp = [None]*n_half_cyc_eval
    min_eval_loss_magsp_std = [None]*n_half_cyc_eval
    min_eval_loss_magsp_dB = [None]*n_half_cyc_eval
    min_eval_loss_magsp_dB_std = [None]*n_half_cyc_eval
    min_eval_loss_magsp_cv = [None]*n_cv_eval
    min_eval_loss_magsp_cv_std = [None]*n_cv_eval
    min_eval_loss_seg_conv = [None]*n_half_cyc_eval
    min_eval_loss_seg_conv_std = [None]*n_half_cyc_eval
    min_eval_loss_conv_sc = [None]*n_half_cyc_eval
    min_eval_loss_conv_sc_std = [None]*n_half_cyc_eval
    min_eval_loss_h = [None]*n_half_cyc_eval
    min_eval_loss_h_std = [None]*n_half_cyc_eval
    min_eval_loss_mid_smpl = [None]*n_half_cyc_eval
    min_eval_loss_mid_smpl_std = [None]*n_half_cyc_eval
    min_eval_loss_ce_c_avg = [None]*n_half_cyc_eval
    min_eval_loss_ce_c_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_err_c_avg = [None]*n_half_cyc_eval
    min_eval_loss_err_c_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_ce_f_avg = [None]*n_half_cyc_eval
    min_eval_loss_ce_f_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_err_f_avg = [None]*n_half_cyc_eval
    min_eval_loss_err_f_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_ce_avg = [None]*n_half_cyc_eval
    min_eval_loss_ce_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_err_avg = [None]*n_half_cyc_eval
    min_eval_loss_err_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_ce = [None]*n_half_cyc_eval
    min_eval_loss_ce_std = [None]*n_half_cyc_eval
    min_eval_loss_err = [None]*n_half_cyc_eval
    min_eval_loss_err_std = [None]*n_half_cyc_eval
    min_eval_loss_ce_f = [None]*n_half_cyc_eval
    min_eval_loss_ce_f_std = [None]*n_half_cyc_eval
    min_eval_loss_err_f = [None]*n_half_cyc_eval
    min_eval_loss_err_f_std = [None]*n_half_cyc_eval
    min_eval_loss_fro_avg = [None]*n_half_cyc_eval
    min_eval_loss_fro_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_l1_avg = [None]*n_half_cyc_eval
    min_eval_loss_l1_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_fro_fb = [None]*n_half_cyc_eval
    min_eval_loss_fro_fb_std = [None]*n_half_cyc_eval
    min_eval_loss_l1_fb = [None]*n_half_cyc_eval
    min_eval_loss_l1_fb_std = [None]*n_half_cyc_eval
    min_eval_loss_fro = [None]*n_half_cyc_eval
    min_eval_loss_fro_std = [None]*n_half_cyc_eval
    min_eval_loss_l1 = [None]*n_half_cyc_eval
    min_eval_loss_l1_std = [None]*n_half_cyc_eval
    for i in range(n_half_cyc_eval):
        min_eval_loss_ce[i] = [None]*args.n_bands
        min_eval_loss_ce_std[i] = [None]*args.n_bands
        min_eval_loss_err[i] = [None]*args.n_bands
        min_eval_loss_err_std[i] = [None]*args.n_bands
        min_eval_loss_ce_f[i] = [None]*args.n_bands
        min_eval_loss_ce_f_std[i] = [None]*args.n_bands
        min_eval_loss_err_f[i] = [None]*args.n_bands
        min_eval_loss_err_f_std[i] = [None]*args.n_bands
        min_eval_loss_fro[i] = [None]*args.n_bands
        min_eval_loss_fro_std[i] = [None]*args.n_bands
        min_eval_loss_l1[i] = [None]*args.n_bands
        min_eval_loss_l1_std[i] = [None]*args.n_bands
    min_eval_loss_melsp_dB[0] = 99999999.99
    min_eval_loss_melsp_dB_std[0] = 99999999.99
    min_eval_loss_melsp_cv[0] = 99999999.99
    min_eval_loss_melsp[0] = 99999999.99
    min_eval_loss_laplace_cv[0] = 99999999.99
    min_eval_loss_laplace[0] = 99999999.99
    min_eval_loss_melsp_dB_src_trg = 99999999.99
    min_eval_loss_melsp_dB_src_trg_std = 99999999.99
    min_eval_loss_gv_src_trg = 99999999.99
    min_eval_loss_ce_avg[0] = 99999999.99
    min_eval_loss_ce_avg_std[0] = 99999999.99
    min_eval_loss_err_avg[0] = 99999999.99
    min_eval_loss_err_avg_std[0] = 99999999.99
    min_eval_loss_l1_avg[0] = 99999999.99
    min_eval_loss_l1_fb[0] = 99999999.99
    iter_idx = 0
    min_idx = -1
    err_flag = False
    change_min_flag = False
    sparse_min_flag = False
    sparse_check_flag = False
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        min_eval_loss_melsp_dB[0] = checkpoint["min_eval_loss_melsp_dB"]
        min_eval_loss_melsp_dB_std[0] = checkpoint["min_eval_loss_melsp_dB_std"]
        min_eval_loss_melsp_cv[0] = checkpoint["min_eval_loss_melsp_cv"]
        min_eval_loss_melsp[0] = checkpoint["min_eval_loss_melsp"]
        min_eval_loss_laplace_cv[0] = checkpoint["min_eval_loss_laplace_cv"]
        min_eval_loss_laplace[0] = checkpoint["min_eval_loss_laplace"]
        min_eval_loss_melsp_dB_src_trg = checkpoint["min_eval_loss_melsp_dB_src_trg"]
        min_eval_loss_melsp_dB_src_trg_std = checkpoint["min_eval_loss_melsp_dB_src_trg_std"]
        min_eval_loss_gv_src_trg = checkpoint["min_eval_loss_gv_src_trg"]
        min_eval_loss_ce_avg[0] = checkpoint["min_eval_loss_ce_avg"]
        min_eval_loss_ce_avg_std[0] = checkpoint["min_eval_loss_ce_avg_std"]
        min_eval_loss_err_avg[0] = checkpoint["min_eval_loss_err_avg"]
        min_eval_loss_err_avg_std[0] = checkpoint["min_eval_loss_err_avg_std"]
        min_eval_loss_l1_avg[0] = checkpoint["min_eval_loss_l1_avg"]
        min_eval_loss_l1_fb[0] = checkpoint["min_eval_loss_l1_fb"]
        err_flag = checkpoint["err_flag"]
        iter_idx = checkpoint["iter_idx"]
        min_idx = checkpoint["min_idx"]
    while idx_stage < args.n_stage-1 and iter_idx + 1 >= t_starts[idx_stage+1]:
        idx_stage += 1
        logging.info(idx_stage)
    if (not sparse_min_flag) and (iter_idx + 1 >= t_ends[idx_stage]):
        sparse_check_flag = True
        sparse_min_flag = True
    #idx_stage = args.n_stage-1
    factors = args.n_bands / 2
    logging.info(factors)
    eps = torch.finfo(indices_1hot.dtype).eps
    eps_1 = 1-eps
    logging.info(f"eps: {eps}\neps_1: {eps_1}")
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while True:
        start = time.time()
        batch_x_fb, batch_x, batch_x_c, batch_x_f, batch_feat, batch_feat_magsp, batch_sc, batch_sc_cv_data, c_idx, utt_idx, featfile, \
            x_bs, x_ss, f_bs, f_ss, slens, flens, n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, idx_select, idx_select_full, slens_acc, flens_acc = next(generator)
        if c_idx < 0: # summarize epoch
            # save current epoch model
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # report current epoch
            text_log = "(EPOCH:%d) average optimization loss = %.6f (+- %.6f) ; " % (epoch_idx + 1, np.mean(loss_sc_feat_in), np.std(loss_sc_feat_in))
            for i in range(args.n_half_cyc):
                if i % 2 == 0:
                    text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                            "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% ; %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " % (i+1,
                        np.mean(loss_elbo[i]), np.std(loss_elbo[i]), np.mean(loss_px[i]), np.std(loss_px[i]),
                        np.mean(loss_qy_py[i]), np.std(loss_qy_py[i]), np.mean(loss_qy_py_err[i]), np.std(loss_qy_py_err[i]), np.mean(loss_qz_pz[i]), np.std(loss_qz_pz[i]),
                        np.mean(loss_qy_py_e[i]), np.std(loss_qy_py_e[i]), np.mean(loss_qy_py_err_e[i]), np.std(loss_qy_py_err_e[i]), np.mean(loss_qz_pz_e[i]), np.std(loss_qz_pz_e[i]),
                        np.mean(loss_sc_z[i]), np.std(loss_sc_z[i]),
                        np.mean(loss_sc_feat[i]), np.std(loss_sc_feat[i]), np.mean(loss_sc_feat_cv[i//2]), np.std(loss_sc_feat_cv[i//2]),
                        np.mean(loss_sc_feat_magsp[i]), np.std(loss_sc_feat_magsp[i]), np.mean(loss_sc_feat_magsp_cv[i//2]), np.std(loss_sc_feat_magsp_cv[i//2]),
                        np.mean(loss_laplace[i]), np.std(loss_laplace[i]), np.mean(loss_laplace_cv[i//2]), np.std(loss_laplace_cv[i//2]),
                        np.mean(loss_melsp[i]), np.std(loss_melsp[i]), np.mean(loss_melsp_cv[i//2]), np.std(loss_melsp_cv[i//2]), np.mean(loss_melsp_dB[i]), np.std(loss_melsp_dB[i]),
                        np.mean(loss_magsp[i]), np.std(loss_magsp[i]), np.mean(loss_magsp_cv[i//2]), np.std(loss_magsp_cv[i//2]), np.mean(loss_magsp_dB[i]), np.std(loss_magsp_dB[i]),
                        np.mean(loss_seg_conv[i]), np.std(loss_seg_conv[i]), np.mean(loss_conv_sc[i]), np.std(loss_conv_sc[i]),
                        np.mean(loss_h[i]), np.std(loss_h[i]), np.mean(loss_mid_smpl[i]), np.std(loss_mid_smpl[i]),
                        np.mean(loss_ce_avg[i]), np.std(loss_ce_avg[i]), np.mean(loss_err_avg[i]), np.std(loss_err_avg[i]),
                            np.mean(loss_ce_c_avg[i]), np.std(loss_ce_c_avg[i]), np.mean(loss_err_c_avg[i]), np.std(loss_err_c_avg[i]),
                                np.mean(loss_ce_f_avg[i]), np.std(loss_ce_f_avg[i]), np.mean(loss_err_f_avg[i]), np.std(loss_err_f_avg[i]),
                                    np.mean(loss_fro_avg[i]), np.std(loss_fro_avg[i]), np.mean(loss_l1_avg[i]), np.std(loss_l1_avg[i]),
                                        np.mean(loss_fro_fb[i]), np.std(loss_fro_fb[i]), np.mean(loss_l1_fb[i]), np.std(loss_l1_fb[i]))
                else:
                    text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                            "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " % (i+1,
                        np.mean(loss_elbo[i]), np.std(loss_elbo[i]), np.mean(loss_px[i]), np.std(loss_px[i]),
                        np.mean(loss_lat_cossim[i]), np.std(loss_lat_cossim[i]), np.mean(loss_lat_rmse[i]), np.std(loss_lat_rmse[i]),
                        np.mean(loss_qy_py[i]), np.std(loss_qy_py[i]), np.mean(loss_qy_py_err[i]), np.std(loss_qy_py_err[i]), np.mean(loss_qz_pz[i]), np.std(loss_qz_pz[i]),
                        np.mean(loss_qy_py_e[i]), np.std(loss_qy_py_e[i]), np.mean(loss_qy_py_err_e[i]), np.std(loss_qy_py_err_e[i]), np.mean(loss_qz_pz_e[i]), np.std(loss_qz_pz_e[i]), 
                        np.mean(loss_sc_z[i]), np.std(loss_sc_z[i]),
                        np.mean(loss_sc_feat[i]), np.std(loss_sc_feat[i]), np.mean(loss_sc_feat_magsp[i]), np.std(loss_sc_feat_magsp[i]),
                        np.mean(loss_laplace[i]), np.std(loss_laplace[i]), np.mean(loss_melsp[i]), np.std(loss_melsp[i]), np.mean(loss_melsp_dB[i]), np.std(loss_melsp_dB[i]),
                        np.mean(loss_magsp[i]), np.std(loss_magsp[i]), np.mean(loss_magsp_dB[i]), np.std(loss_magsp_dB[i]),
                        np.mean(loss_seg_conv[i]), np.std(loss_seg_conv[i]), np.mean(loss_conv_sc[i]), np.std(loss_conv_sc[i]),
                        np.mean(loss_h[i]), np.std(loss_h[i]), np.mean(loss_mid_smpl[i]), np.std(loss_mid_smpl[i]),
                        np.mean(loss_ce_avg[i]), np.std(loss_ce_avg[i]), np.mean(loss_err_avg[i]), np.std(loss_err_avg[i]),
                            np.mean(loss_ce_c_avg[i]), np.std(loss_ce_c_avg[i]), np.mean(loss_err_c_avg[i]), np.std(loss_err_c_avg[i]),
                                np.mean(loss_ce_f_avg[i]), np.std(loss_ce_f_avg[i]), np.mean(loss_err_f_avg[i]), np.std(loss_err_f_avg[i]),
                                    np.mean(loss_fro_avg[i]), np.std(loss_fro_avg[i]), np.mean(loss_l1_avg[i]), np.std(loss_l1_avg[i]),
                                        np.mean(loss_fro_fb[i]), np.std(loss_fro_fb[i]), np.mean(loss_l1_fb[i]), np.std(loss_l1_fb[i]))
                for j in range(args.n_bands):
                    text_log += "[%d-%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% , %.6f (+- %.6f) %.6f (+- %.6f) " % (i+1, j+1,
                            np.mean(loss_ce[i][j]), np.std(loss_ce[i][j]), np.mean(loss_err[i][j]), np.std(loss_err[i][j]),
                                np.mean(loss_ce_f[i][j]), np.std(loss_ce_f[i][j]), np.mean(loss_err_f[i][j]), np.std(loss_err_f[i][j]),
                                    np.mean(loss_fro[i][j]), np.std(loss_fro[i][j]), np.mean(loss_l1[i][j]), np.std(loss_l1[i][j]))
                text_log += ";; "
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            logging.info("estimated time until max. steps = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((args.step_count - (iter_idx + 1)) * total))))
            # compute loss in evaluation data
            total = 0
            iter_count = 0
            loss_melsp_dB_src_trg = []
            loss_lat_dist_rmse = []
            loss_lat_dist_cossim = []
            for i in range(n_spk):
                gv_src_src[i] = []
                gv_src_trg[i] = []
            loss_sc_feat_in = []
            loss_sc_feat_magsp_in = []
            for i in range(args.n_half_cyc):
                loss_elbo[i] = []
                loss_px[i] = []
                loss_qy_py[i] = []
                loss_qy_py_err[i] = []
                loss_qz_pz[i] = []
                loss_qy_py_e[i] = []
                loss_qy_py_err_e[i] = []
                loss_qz_pz_e[i] = []
                loss_lat_cossim[i] = []
                loss_lat_rmse[i] = []
                loss_sc_z[i] = []
                loss_sc_feat[i] = []
                loss_sc_feat_cv[i] = []
                loss_sc_feat_magsp[i] = []
                loss_sc_feat_magsp_cv[i] = []
                loss_laplace[i] = []
                loss_melsp[i] = []
                loss_laplace_cv[i] = []
                loss_melsp_cv[i] = []
                loss_melsp_dB[i] = []
                loss_magsp[i] = []
                loss_magsp_cv[i] = []
                loss_magsp_dB[i] = []
                loss_ce_avg[i] = []
                loss_err_avg[i] = []
                loss_ce_c_avg[i] = []
                loss_err_c_avg[i] = []
                loss_ce_f_avg[i] = []
                loss_err_f_avg[i] = []
                loss_fro_avg[i] = []
                loss_l1_avg[i] = []
                loss_fro_fb[i] = []
                loss_l1_fb[i] = []
                loss_seg_conv[i] = []
                loss_conv_sc[i] = []
                loss_h[i] = []
                loss_mid_smpl[i] = []
                for j in range(args.n_bands):
                    loss_ce[i][j] = []
                    loss_err[i][j] = []
                    loss_ce_f[i][j] = []
                    loss_err_f[i][j] = []
                    loss_fro[i][j] = []
                    loss_l1[i][j] = []
            model_encoder_melsp_fix.eval()
            model_encoder_melsp.eval()
            model_decoder_melsp.eval()
            model_encoder_excit_fix.eval()
            model_encoder_excit.eval()
            model_classifier.eval()
            model_spk.eval()
            if args.spkidtr_dim > 0:
                model_spkidtr.eval()
            model_waveform.eval()
            for param in model_encoder_melsp.parameters():
                param.requires_grad = False
            for param in model_decoder_melsp.parameters():
                param.requires_grad = False
            for param in model_encoder_excit.parameters():
                param.requires_grad = False
            for param in model_classifier.parameters():
                param.requires_grad = False
            for param in model_spk.parameters():
                param.requires_grad = False
            if args.spkidtr_dim > 0:
                for param in model_spkidtr.parameters():
                    param.requires_grad = False
            pair_exist = False
            logging.info("Evaluation data")
            while True:
                with torch.no_grad():
                    start = time.time()
                    batch_x_fb, batch_x, batch_x_c, batch_x_f, batch_feat_data, batch_feat_magsp_data, batch_feat_trg_data, batch_sc_data, \
                        batch_sc_cv_data, c_idx, utt_idx, featfile, \
                        x_bs, x_ss, f_bs, f_ss, slens, flens, n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, src_trg_flag, \
                            spcidx_src, spcidx_src_trg, flens_spc_src, flens_spc_src_trg, \
                                batch_feat_data_full, batch_sc_data_full, batch_sc_cv_data_full, \
                                    idx_select, idx_select_full, slens_acc, flens_acc = next(generator_eval)
                    if c_idx < 0:
                        break

                    x_es = x_ss+x_bs
                    f_es = f_ss+f_bs
                    logging.info(f'{x_ss} {x_bs} {x_es} {f_ss} {f_bs} {f_es} {max_slen} {max_flen}')

                    # handle waveformb batch padding
                    f_ss_pad_left = f_ss-wav_pad_left
                    if f_es <= max_flen:
                        f_es_pad_right = f_es+wav_pad_right
                    else:
                        f_es_pad_right = max_flen+wav_pad_right
                    if f_ss_pad_left >= 0 and f_es_pad_right <= max_flen: # pad left and right available
                        batch_feat_org_in = batch_feat_data[:,f_ss_pad_left:f_es_pad_right]
                    elif f_es_pad_right <= max_flen: # pad right available, left need additional replicate
                        batch_feat_org_in = F.pad(batch_feat_data[:,:f_es_pad_right].transpose(1,2), (-f_ss_pad_left,0), "replicate").transpose(1,2)
                    elif f_ss_pad_left >= 0: # pad left available, right need additional replicate
                        batch_feat_org_in = F.pad(batch_feat_data[:,f_ss_pad_left:max_flen].transpose(1,2), (0,f_es_pad_right-max_flen), "replicate").transpose(1,2)
                    else: # pad left and right need additional replicate
                        batch_feat_org_in = F.pad(batch_feat_data[:,:max_flen].transpose(1,2), (-f_ss_pad_left,f_es_pad_right-max_flen), "replicate").transpose(1,2)
                    if x_ss > 0:
                        if x_es <= max_slen:
                            batch_x_c_prev = batch_x_c[:,x_ss-1:x_es-1]
                            batch_x_f_prev = batch_x_f[:,x_ss-1:x_es-1]
                            if args.lpc > 0:
                                if x_ss-args.lpc >= 0:
                                    batch_x_c_lpc = batch_x_c[:,x_ss-args.lpc:x_es-1]
                                    batch_x_f_lpc = batch_x_f[:,x_ss-args.lpc:x_es-1]
                                else:
                                    batch_x_c_lpc = F.pad(batch_x_c[:,:x_es-1], (0, 0, -(x_ss-args.lpc), 0), "constant", args.c_pad)
                                    batch_x_f_lpc = F.pad(batch_x_f[:,:x_es-1], (0, 0, -(x_ss-args.lpc), 0), "constant", args.f_pad)
                            batch_x_c = batch_x_c[:,x_ss:x_es]
                            batch_x_f = batch_x_f[:,x_ss:x_es]
                            batch_x = batch_x[:,x_ss:x_es]
                            batch_x_fb = batch_x_fb[:,x_ss*args.n_bands:x_es*args.n_bands]
                        else:
                            batch_x_c_prev = batch_x_c[:,x_ss-1:-1]
                            batch_x_f_prev = batch_x_f[:,x_ss-1:-1]
                            if args.lpc > 0:
                                if x_ss-args.lpc >= 0:
                                    batch_x_c_lpc = batch_x_c[:,x_ss-args.lpc:-1]
                                    batch_x_f_lpc = batch_x_f[:,x_ss-args.lpc:-1]
                                else:
                                    batch_x_c_lpc = F.pad(batch_x_c[:,:-1], (0, 0, -(x_ss-args.lpc), 0), "constant", args.c_pad)
                                    batch_x_f_lpc = F.pad(batch_x_f[:,:-1], (0, 0, -(x_ss-args.lpc), 0), "constant", args.f_pad)
                            batch_x_c = batch_x_c[:,x_ss:]
                            batch_x_f = batch_x_f[:,x_ss:]
                            batch_x = batch_x[:,x_ss:]
                            batch_x_fb = batch_x_fb[:,x_ss*args.n_bands:]
                    else:
                        batch_x_c_prev = F.pad(batch_x_c[:,:x_es-1], (0, 0, 1, 0), "constant", args.c_pad)
                        batch_x_f_prev = F.pad(batch_x_f[:,:x_es-1], (0, 0, 1, 0), "constant", args.f_pad)
                        if args.lpc > 0:
                            batch_x_c_lpc = F.pad(batch_x_c[:,:x_es-1], (0, 0, args.lpc, 0), "constant", args.c_pad)
                            batch_x_f_lpc = F.pad(batch_x_f[:,:x_es-1], (0, 0, args.lpc, 0), "constant", args.f_pad)
                        batch_x_c = batch_x_c[:,:x_es]
                        batch_x_f = batch_x_f[:,:x_es]
                        batch_x = batch_x[:,:x_es]
                        batch_x_fb = batch_x_fb[:,:x_es*args.n_bands]

                    # handle first pad for input on melsp flow
                    flag_cv = True
                    i_cv = 0
                    i_cv_in = 0
                    f_ss_first_pad_left = f_ss-first_pad_left
                    f_es_first_pad_right = f_es+first_pad_right
                    i_end = n_half_cyc_eval*4
                    for i in range(i_end):
                        if i % 4 == 0: #enc
                            if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                                batch_feat_in[i] = batch_feat_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                            elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                                batch_feat_in[i] = F.pad(batch_feat_data[:,:f_es_first_pad_right].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
                            elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                                batch_feat_in[i] = F.pad(batch_feat_data[:,f_ss_first_pad_left:max_flen].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                            else: # pad left and right need additional replicate
                                batch_feat_in[i] = F.pad(batch_feat_data[:,:max_flen].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                            f_ss_first_pad_left += enc_pad_left
                            f_es_first_pad_right -= enc_pad_right
                        else: #spk/dec/wav
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
                                f_ss_first_pad_left += spk_pad_left
                                f_es_first_pad_right -= spk_pad_right
                            elif i % 4 == 2:
                                f_ss_first_pad_left += dec_pad_left
                                f_es_first_pad_right -= dec_pad_right
                            elif i % 4 == 3:
                                f_ss_first_pad_left += wav_pad_left
                                f_es_first_pad_right -= wav_pad_right
                    batch_melsp = batch_feat_data[:,f_ss:f_es]
                    batch_magsp = batch_feat_magsp_data[:,f_ss:f_es]
                    batch_melsp_data_full = batch_feat_data_full
                    batch_melsp_trg_data = batch_feat_trg_data
                    batch_sc = batch_sc_data[:,f_ss:f_es]
                    batch_sc_cv[0] = batch_sc_cv_data[:,f_ss:f_es]

                    if f_ss > 0:
                        idx_in = 0
                        i_cv_in = 0
                        for i in range(0,n_half_cyc_eval,2):
                            i_cv = i//2
                            j = i+1
                            if len(del_index_utt) > 0:
                                h_feat_in_sc = torch.FloatTensor(np.delete(h_feat_in_sc.cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_magsp_in_sc = torch.FloatTensor(np.delete(h_feat_magsp_in_sc.cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_x_org = torch.FloatTensor(np.delete(h_x_org.cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                                h_x_2_org = torch.FloatTensor(np.delete(h_x_2_org.cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                                h_f_org = torch.FloatTensor(np.delete(h_f_org.cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                                h_z[i] = torch.FloatTensor(np.delete(h_z[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z_e[i] = torch.FloatTensor(np.delete(h_z_e[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z_fix = torch.FloatTensor(np.delete(h_z_fix.cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z_e_fix = torch.FloatTensor(np.delete(h_z_e_fix.cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_spk[i] = torch.FloatTensor(np.delete(h_spk[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_spk_cv[i_cv] = torch.FloatTensor(np.delete(h_spk_cv[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp[i] = torch.FloatTensor(np.delete(h_melsp[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp_cv[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z_sc[i] = torch.FloatTensor(np.delete(h_z_sc[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_sc[i] = torch.FloatTensor(np.delete(h_feat_sc[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_feat_cv_sc[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_magsp_sc[i] = torch.FloatTensor(np.delete(h_feat_magsp_sc[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_magsp_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_feat_magsp_cv_sc[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_x[i] = torch.FloatTensor(np.delete(h_x[i].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                                h_x_2[i] = torch.FloatTensor(np.delete(h_x_2[i].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                                h_f[i] = torch.FloatTensor(np.delete(h_f[i].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                                h_z[j] = torch.FloatTensor(np.delete(h_z[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z_e[j] = torch.FloatTensor(np.delete(h_z_e[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_spk[j] = torch.FloatTensor(np.delete(h_spk[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp[j] = torch.FloatTensor(np.delete(h_melsp[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_z_sc[j] = torch.FloatTensor(np.delete(h_z_sc[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_sc[j] = torch.FloatTensor(np.delete(h_feat_sc[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_feat_magsp_sc[j] = torch.FloatTensor(np.delete(h_feat_magsp_sc[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_x[j] = torch.FloatTensor(np.delete(h_x[j].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                                h_x_2[j] = torch.FloatTensor(np.delete(h_x_2[j].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                                h_f[j] = torch.FloatTensor(np.delete(h_f[j].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                            qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z[i], sampling=False)
                            qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_e[i], sampling=False)
                            _, qz_alpha_fix, z_fix, h_z_fix = model_encoder_melsp_fix(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_fix)
                            _, qz_alpha_e_fix, z_e_fix, h_z_e_fix = model_encoder_excit_fix(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_e_fix)
                            batch_feat_in_sc, h_feat_in_sc = model_classifier(feat=batch_melsp, h=h_feat_in_sc)
                            batch_feat_magsp_in_sc, h_feat_magsp_in_sc = model_classifier(feat_aux=batch_magsp, h=h_feat_magsp_in_sc)
                            seg_conv, conv_sc, out, out_2, out_f, signs_c, scales_c, logits_c, signs_f, scales_f, logits_f, x_c_output, x_f_output, h_x_org, h_x_2_org, h_f_org \
                                = model_waveform.gen_mid_feat_smpl(batch_feat_org_in, batch_x_c_prev, batch_x_f_prev, batch_x_c, h=h_x_org, h_2=h_x_2_org, h_f=h_f_org, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc)
                            ## time-varying speaker conditionings
                            z_cat = torch.cat((z_e[i], z[i]), 2)
                            feat_len = qy_logits[i].shape[1]
                            z[i] = z[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            z_e[i] = z_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            batch_z_sc[i], h_z_sc[i] = model_classifier(lat=torch.cat((z[i], z_e[i]), 2), h=h_z_sc[i])
                            qy_logits[i] = qy_logits[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha[i] = qz_alpha[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qy_logits_e[i] = qy_logits_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha_e[i] = qz_alpha_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha_fix = qz_alpha_fix[:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha_e_fix = qz_alpha_e_fix[:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            idx_in += 1
                            if args.spkidtr_dim > 0:
                                spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                                spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                                batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv])
                            else:
                                batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv])
                            ## melsp reconstruction & conversion
                            idx_in += 1
                            i_cv_in += 1
                            if spk_pad_right > 0:
                                z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:-spk_pad_right]
                            else:
                                z_cat = z_cat[:,spk_pad_left:]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_pdf_rec[i], batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                    outpad_right=outpad_rights[idx_in], h=h_melsp[i])
                                batch_pdf_cv[i_cv], batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                                    outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv])
                            else:
                                batch_pdf_rec[i], batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    outpad_right=outpad_rights[idx_in], h=h_melsp[i])
                                batch_pdf_cv[i_cv], batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv])
                            ## waveform reconstruction
                            idx_in += 1
                            batch_x_c_output_noclamp[i], batch_x_f_output_noclamp[i], batch_seg_conv[i], batch_conv_sc[i], \
                                batch_out[i], batch_out_2[i], batch_out_f[i], batch_signs_c[i], batch_scales_c[i], batch_logits_c[i], batch_signs_f[i], batch_scales_f[i], batch_logits_f[i], h_x[i], h_x_2[i], h_f[i] \
                                    = model_waveform(batch_melsp_rec[i], batch_x_c_prev, batch_x_f_prev, batch_x_c, h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in],
                                            x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc, ret_mid_feat=True, ret_mid_smpl=True)
                            batch_x_c_output[i] = torch.clamp(batch_x_c_output_noclamp[i], min=MIN_CLAMP, max=MAX_CLAMP)
                            batch_x_f_output[i] = torch.clamp(batch_x_f_output_noclamp[i], min=MIN_CLAMP, max=MAX_CLAMP)
                            u = torch.empty_like(batch_x_c_output[i])
                            logits_gumbel = F.softmax(batch_x_c_output[i] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                            logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                            sample_indices_c = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                            logits_gumbel = F.softmax(batch_x_f_output[i] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                            logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                            sample_indices_f = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                            batch_x_output[i] = decode_mu_law_torch(sample_indices_c*args.cf_dim+sample_indices_f)
                            batch_x_output_fb[i] = pqmf.synthesis(batch_x_output[i].transpose(1,2))[:,0]
                            if wav_pad_right > 0:
                                cv_feat = batch_melsp_cv[i_cv][:,wav_pad_left:-wav_pad_right]
                            else:
                                cv_feat = batch_melsp_cv[i_cv][:,wav_pad_left:]
                            idx_in_1 = idx_in-1
                            feat_len = batch_melsp_rec[i].shape[1]
                            batch_pdf_rec[i] = batch_pdf_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_rec[i] = batch_melsp_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_pdf_cv[i_cv] = batch_pdf_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_cv[i_cv] = batch_melsp_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_rec[i] = torch.matmul((torch.exp(batch_melsp_rec[i])-1)/10000, melfb_t)
                            batch_magsp_cv[i_cv] = torch.matmul((torch.exp(batch_melsp_cv[i_cv])-1)/10000, melfb_t)
                            batch_feat_rec_sc[i], h_feat_sc[i] = model_classifier(feat=batch_melsp_rec[i], h=h_feat_sc[i])
                            batch_feat_cv_sc[i_cv], h_feat_cv_sc[i_cv] = model_classifier(feat=batch_melsp_cv[i_cv], h=h_feat_cv_sc[i_cv])
                            batch_feat_magsp_rec_sc[i], h_feat_magsp_sc[i] = model_classifier(feat_aux=batch_magsp_rec[i], h=h_feat_magsp_sc[i])
                            batch_feat_magsp_cv_sc[i_cv], h_feat_magsp_cv_sc[i_cv] = model_classifier(feat_aux=batch_magsp_cv[i_cv], h=h_feat_magsp_cv_sc[i_cv])
                            ## cyclic reconstruction
                            idx_in += 1
                            qy_logits[j], qz_alpha[j], z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z[j], sampling=False)
                            qy_logits_e[j], qz_alpha_e[j], z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[j], sampling=False)
                            ## time-varying speaker conditionings
                            z_cat = torch.cat((z_e[j], z[j]), 2)
                            feat_len = qy_logits[j].shape[1]
                            z[j] = z[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            z_e[j] = z_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            batch_z_sc[j], h_z_sc[j] = model_classifier(lat=torch.cat((z[j], z_e[j]), 2), h=h_z_sc[j])
                            qy_logits[j] = qy_logits[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha[j] = qz_alpha[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qy_logits_e[j] = qy_logits_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha_e[j] = qz_alpha_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            idx_in += 1
                            if args.spkidtr_dim > 0:
                                if dec_enc_pad_right > 0:
                                    spk_code_in = spk_code_in[:,dec_enc_pad_left:-dec_enc_pad_right]
                                else:
                                    spk_code_in = spk_code_in[:,dec_enc_pad_left:]
                                batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j])
                            else:
                                batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j])
                            ## melsp reconstruction
                            idx_in += 1
                            if spk_pad_right > 0:
                                z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                            else:
                                z_cat = z_cat[:,spk_pad_left:]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_pdf_rec[j], batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                        outpad_right=outpad_rights[idx_in], h=h_melsp[j])
                            else:
                                batch_pdf_rec[j], batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                        outpad_right=outpad_rights[idx_in], h=h_melsp[j])
                            ## waveform reconstruction
                            idx_in += 1
                            batch_x_c_output_noclamp[j], batch_x_f_output_noclamp[j], batch_seg_conv[j], batch_conv_sc[j], \
                                batch_out[j], batch_out_2[j], batch_out_f[j], batch_signs_c[j], batch_scales_c[j], batch_logits_c[j], batch_signs_f[j], batch_scales_f[j], batch_logits_f[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(batch_melsp_rec[j], batch_x_c_prev, batch_x_f_prev, batch_x_c, h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in],
                                            x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc, ret_mid_feat=True, ret_mid_smpl=True)
                            batch_x_c_output[j] = torch.clamp(batch_x_c_output_noclamp[j], min=MIN_CLAMP, max=MAX_CLAMP)
                            batch_x_f_output[j] = torch.clamp(batch_x_f_output_noclamp[j], min=MIN_CLAMP, max=MAX_CLAMP)
                            logits_gumbel = F.softmax(batch_x_c_output[j] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                            logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                            sample_indices_c = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                            logits_gumbel = F.softmax(batch_x_f_output[j] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                            logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                            sample_indices_f = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                            batch_x_output[j] = decode_mu_law_torch(sample_indices_c*args.cf_dim+sample_indices_f)
                            batch_x_output_fb[j] = pqmf.synthesis(batch_x_output[j].transpose(1,2))[:,0]
                            idx_in_1 = idx_in-1
                            feat_len = batch_melsp_rec[j].shape[1]
                            batch_pdf_rec[j] = batch_pdf_rec[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_rec[j] = batch_melsp_rec[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_rec[j] = torch.matmul((torch.exp(batch_melsp_rec[j])-1)/10000, melfb_t)
                            batch_feat_rec_sc[j], h_feat_sc[j] = model_classifier(feat=batch_melsp_rec[j], h=h_feat_sc[j])
                            batch_feat_magsp_rec_sc[j], h_feat_magsp_sc[j] = model_classifier(feat_aux=batch_magsp_rec[j], h=h_feat_magsp_sc[j])
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
                        if spk_pad_right > 0:
                            z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                            trj_spk_code = trj_spk_code[:,spk_pad_left:-spk_pad_right]
                            trj_spk_cv_code = trj_spk_cv_code[:,spk_pad_left:-spk_pad_right]
                        else:
                            z_cat = z_cat[:,spk_pad_left:]
                            trj_spk_code = trj_spk_code[:,spk_pad_left:]
                            trj_spk_cv_code = trj_spk_cv_code[:,spk_pad_left:]
                        _, trj_src_src, _ = model_decoder_melsp(z_cat, y=trj_spk_code, aux=trj_spk)
                        _, trj_src_trg, _ = model_decoder_melsp(z_cat, y=trj_spk_cv_code, aux=trj_spk_cv)

                        for k in range(n_batch_utt):
                            spk_src = os.path.basename(os.path.dirname(featfile[k]))
                            spk_src_trg = spk_cv[k] # find target pair
                            #GV stat of reconstructed
                            gv_src_src[spk_list.index(spk_src)].append(torch.var(\
                                (torch.exp(trj_src_src[k,:flens[k]])-1)/10000, 0).cpu().data.numpy())
                            #GV stat of converted
                            gv_src_trg[spk_list.index(spk_src_trg)].append(torch.var(\
                                (torch.exp(trj_src_trg[k,:flens[k]])-1)/10000, 0).cpu().data.numpy())
                        if pair_flag:
                            if dec_pad_right > 0:
                                trj_lat_src = z_cat[:,dec_pad_left:-dec_pad_right]
                            else:
                                trj_lat_src = z_cat[:,dec_pad_left:]
                            batch_melsp_trg_data_in = F.pad(batch_melsp_trg_data.transpose(1,2), (enc_pad_left,enc_pad_right), "replicate").transpose(1,2)
                            _, _, trj_lat_trg, _ = model_encoder_melsp(batch_melsp_trg_data_in, sampling=False)
                            _, _, trj_lat_trg_e, _ = model_encoder_excit(batch_melsp_trg_data_in, sampling=False)
                            trj_lat_trg = torch.cat((trj_lat_trg_e, trj_lat_trg), 2)

                            for k in range(n_batch_utt):
                                if src_trg_flag[k]:
                                    # spcidx lat
                                    trj_lat_src_ = np.array(torch.index_select(trj_lat_src[k],0,spcidx_src[k,:flens_spc_src[k]]).cpu().data.numpy(), dtype=np.float64)
                                    trj_lat_trg_ = np.array(torch.index_select(trj_lat_trg[k],0,spcidx_src_trg[k,:flens_spc_src_trg[k]]).cpu().data.numpy(), dtype=np.float64)
                                    # spcidx melsp, excit, trg
                                    trj_src_trg_ = (torch.exp(torch.index_select(trj_src_trg[k],0,spcidx_src[k,:flens_spc_src[k]]))-1)/10000
                                    trj_trg_ = (torch.exp(torch.index_select(batch_melsp_trg_data[k],0,spcidx_src_trg[k,:flens_spc_src_trg[k]]))-1)/10000
                                    # spec dtw
                                    # MCD of spectral
                                    _, twf_melsp, _, _ = dtw.dtw_org_to_trg(\
                                        np.array(trj_src_trg_.cpu().data.numpy(), dtype=np.float64), \
                                        np.array(trj_trg_.cpu().data.numpy(), dtype=np.float64), mcd=-1)
                                    twf_melsp = torch.LongTensor(twf_melsp[:,0]).cuda()
                                    batch_melsp_dB_src_trg = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(torch.index_select(trj_src_trg_,0,twf_melsp), min=1e-16))
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
                                    loss_lat_dist_rmse.append(batch_lat_dist_rmse)
                                    loss_lat_dist_cossim.append(batch_lat_dist_cossim)
                                    total_eval_loss["eval/loss_melsp_dB_src_trg"].append(batch_melsp_dB_src_trg)
                                    total_eval_loss["eval/loss_lat_dist_rmse"].append(batch_lat_dist_rmse)
                                    total_eval_loss["eval/loss_lat_dist_cossim"].append(batch_lat_dist_cossim)
                                    logging.info('acc cv %s %s %.3f dB %.3f %.3f' % (featfile[k], spk_cv[k],
                                        batch_melsp_dB_src_trg, batch_lat_dist_rmse, batch_lat_dist_cossim))
                        idx_in = 0
                        i_cv_in = 0
                        for i in range(0,n_half_cyc_eval,2):
                            i_cv = i//2
                            j = i+1
                            qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], sampling=False)
                            qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], sampling=False)
                            _, qz_alpha_fix, z_fix, h_z_fix = model_encoder_melsp_fix(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in])
                            _, qz_alpha_e_fix, z_e_fix, h_z_e_fix = model_encoder_excit_fix(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in])
                            batch_feat_in_sc, h_feat_in_sc = model_classifier(feat=batch_melsp)
                            batch_feat_magsp_in_sc, h_feat_magsp_in_sc = model_classifier(feat_aux=batch_magsp)
                            seg_conv, conv_sc, out, out_2, out_f, signs_c, scales_c, logits_c, signs_f, scales_f, logits_f, x_c_output, x_f_output, h_x_org, h_x_2_org, h_f_org \
                                = model_waveform.gen_mid_feat_smpl(batch_feat_org_in, batch_x_c_prev, batch_x_f_prev, batch_x_c, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc)
                            ## time-varying speaker conditionings
                            z_cat = torch.cat((z_e[i], z[i]), 2)
                            feat_len = qy_logits[i].shape[1]
                            z[i] = z[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            z_e[i] = z_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            batch_z_sc[i], h_z_sc[i] = model_classifier(lat=torch.cat((z[i], z_e[i]), 2))
                            qy_logits[i] = qy_logits[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha[i] = qz_alpha[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qy_logits_e[i] = qy_logits_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha_e[i] = qz_alpha_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha_fix = qz_alpha_fix[:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha_e_fix = qz_alpha_e_fix[:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            idx_in += 1
                            if args.spkidtr_dim > 0:
                                spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                                spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                                batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                            else:
                                batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                            ## melsp reconstruction & conversion
                            idx_in += 1
                            i_cv_in += 1
                            if spk_pad_right > 0:
                                z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:-spk_pad_right]
                            else:
                                z_cat = z_cat[:,spk_pad_left:]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_pdf_rec[i], batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                    outpad_right=outpad_rights[idx_in])
                                batch_pdf_cv[i_cv], batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                                    outpad_right=outpad_rights[idx_in])
                            else:
                                batch_pdf_rec[i], batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    outpad_right=outpad_rights[idx_in])
                                batch_pdf_cv[i_cv], batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    outpad_right=outpad_rights[idx_in])
                            ## waveform reconstruction
                            idx_in += 1
                            batch_x_c_output_noclamp[i], batch_x_f_output_noclamp[i], batch_seg_conv[i], batch_conv_sc[i], \
                                batch_out[i], batch_out_2[i], batch_out_f[i], batch_signs_c[i], batch_scales_c[i], batch_logits_c[i], batch_signs_f[i], batch_scales_f[i], batch_logits_f[i], h_x[i], h_x_2[i], h_f[i] \
                                    = model_waveform(batch_melsp_rec[i], batch_x_c_prev, batch_x_f_prev, batch_x_c, outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in],
                                            x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc, ret_mid_feat=True, ret_mid_smpl=True)
                            batch_x_c_output[i] = torch.clamp(batch_x_c_output_noclamp[i], min=MIN_CLAMP, max=MAX_CLAMP)
                            batch_x_f_output[i] = torch.clamp(batch_x_f_output_noclamp[i], min=MIN_CLAMP, max=MAX_CLAMP)
                            u = torch.empty_like(batch_x_c_output[i])
                            logits_gumbel = F.softmax(batch_x_c_output[i] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                            logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                            sample_indices_c = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                            logits_gumbel = F.softmax(batch_x_f_output[i] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                            logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                            sample_indices_f = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                            batch_x_output[i] = decode_mu_law_torch(sample_indices_c*args.cf_dim+sample_indices_f)
                            batch_x_output_fb[i] = pqmf.synthesis(batch_x_output[i].transpose(1,2))[:,0]
                            if wav_pad_right > 0:
                                cv_feat = batch_melsp_cv[i_cv][:,wav_pad_left:-wav_pad_right]
                            else:
                                cv_feat = batch_melsp_cv[i_cv][:,wav_pad_left:]
                            idx_in_1 = idx_in-1
                            feat_len = batch_melsp_rec[i].shape[1]
                            batch_pdf_rec[i] = batch_pdf_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_rec[i] = batch_melsp_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_pdf_cv[i_cv] = batch_pdf_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_cv[i_cv] = batch_melsp_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_rec[i] = torch.matmul((torch.exp(batch_melsp_rec[i])-1)/10000, melfb_t)
                            batch_magsp_cv[i_cv] = torch.matmul((torch.exp(batch_melsp_cv[i_cv])-1)/10000, melfb_t)
                            batch_feat_rec_sc[i], h_feat_sc[i] = model_classifier(feat=batch_melsp_rec[i])
                            batch_feat_cv_sc[i_cv], h_feat_cv_sc[i_cv] = model_classifier(feat=batch_melsp_cv[i_cv])
                            batch_feat_magsp_rec_sc[i], h_feat_magsp_sc[i] = model_classifier(feat_aux=batch_magsp_rec[i])
                            batch_feat_magsp_cv_sc[i_cv], h_feat_magsp_cv_sc[i_cv] = model_classifier(feat_aux=batch_magsp_cv[i_cv])
                            ## cyclic reconstruction
                            idx_in += 1
                            qy_logits[j], qz_alpha[j], z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                            qy_logits_e[j], qz_alpha_e[j], z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                            ## time-varying speaker conditionings
                            z_cat = torch.cat((z_e[j], z[j]), 2)
                            feat_len = qy_logits[j].shape[1]
                            z[j] = z[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            z_e[j] = z_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            batch_z_sc[j], h_z_sc[j] = model_classifier(lat=torch.cat((z[j], z_e[j]), 2))
                            qy_logits[j] = qy_logits[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha[j] = qz_alpha[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qy_logits_e[j] = qy_logits_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            qz_alpha_e[j] = qz_alpha_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                            idx_in += 1
                            if args.spkidtr_dim > 0:
                                if dec_enc_pad_right > 0:
                                    spk_code_in = spk_code_in[:,dec_enc_pad_left:-dec_enc_pad_right]
                                else:
                                    spk_code_in = spk_code_in[:,dec_enc_pad_left:]
                                batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                            else:
                                batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                            ## melsp reconstruction
                            idx_in += 1
                            if spk_pad_right > 0:
                                z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                            else:
                                z_cat = z_cat[:,spk_pad_left:]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_pdf_rec[j], batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                        outpad_right=outpad_rights[idx_in])
                            else:
                                batch_pdf_rec[j], batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                        outpad_right=outpad_rights[idx_in])
                            ## waveform reconstruction
                            idx_in += 1
                            batch_x_c_output_noclamp[j], batch_x_f_output_noclamp[j], batch_seg_conv[j], batch_conv_sc[j], \
                                batch_out[j], batch_out_2[j], batch_out_f[j], batch_signs_c[j], batch_scales_c[j], batch_logits_c[j], batch_signs_f[j], batch_scales_f[j], batch_logits_f[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(batch_melsp_rec[j], batch_x_c_prev, batch_x_f_prev, batch_x_c, outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in],
                                            x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc, ret_mid_feat=True, ret_mid_smpl=True)
                            batch_x_c_output[j] = torch.clamp(batch_x_c_output_noclamp[j], min=MIN_CLAMP, max=MAX_CLAMP)
                            batch_x_f_output[j] = torch.clamp(batch_x_f_output_noclamp[j], min=MIN_CLAMP, max=MAX_CLAMP)
                            logits_gumbel = F.softmax(batch_x_c_output[j] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                            logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                            sample_indices_c = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                            logits_gumbel = F.softmax(batch_x_f_output[j] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                            logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                            sample_indices_f = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                            batch_x_output[j] = decode_mu_law_torch(sample_indices_c*args.cf_dim+sample_indices_f)
                            batch_x_output_fb[j] = pqmf.synthesis(batch_x_output[j].transpose(1,2))[:,0]
                            idx_in_1 = idx_in-1
                            feat_len = batch_melsp_rec[j].shape[1]
                            batch_pdf_rec[j] = batch_pdf_rec[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_melsp_rec[j] = batch_melsp_rec[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                            batch_magsp_rec[j] = torch.matmul((torch.exp(batch_melsp_rec[j])-1)/10000, melfb_t)
                            batch_feat_rec_sc[j], h_feat_sc[j] = model_classifier(feat=batch_melsp_rec[j])
                            batch_feat_magsp_rec_sc[j], h_feat_magsp_sc[j] = model_classifier(feat_aux=batch_magsp_rec[j])


                    # handle short ending
                    if len(idx_select) > 0:
                        len_idx_select = len(idx_select)
                        logging.info('len_idx_select: '+str(len_idx_select))
                        for i in range(n_half_cyc_eval):
                            batch_loss_laplace[i] = 0
                            batch_loss_melsp[i] = 0
                            batch_loss_magsp[i] = 0
                            batch_loss_melsp_dB[i] = 0
                            batch_loss_magsp_dB[i] = 0
                            batch_loss_seg_conv[i] = 0
                            batch_loss_conv_sc[i] = 0
                            batch_loss_h[i] = 0
                            batch_loss_mid_smpl[i] = 0
                            batch_loss_ce_select[i] = 0
                            batch_loss_ce_f_select[i] = 0
                            batch_loss_err_select[i] = 0
                            batch_loss_err_f_select[i] = 0
                            batch_loss_fro_select[i] = 0
                            batch_loss_l1_select[i] = 0
                            batch_loss_fro_fb[i] = 0
                            batch_loss_l1_fb[i] = 0
                            if i % 2 == 0:
                                batch_loss_laplace_cv[i//2] = 0
                                batch_loss_melsp_cv[i//2] = 0
                                batch_loss_magsp_cv[i//2] = 0
                        for j in range(len(idx_select)):
                            k = idx_select[j]
                            slens_utt = slens_acc[k]
                            slens_utt_fb = slens_utt*args.n_bands
                            flens_utt = flens_acc[k]
                            logging.info('%s %d %d %d' % (featfile[k], slens_utt, slens_utt_fb, flens_utt))
                            batch_x_c_ = batch_x_c[k,:slens_utt]
                            batch_x_f_ = batch_x_f[k,:slens_utt]
                            batch_x_ = batch_x[k,:slens_utt].transpose(1,0)
                            batch_x_fb_ = batch_x_fb[k,:slens_utt_fb]
                            seg_conv_ = seg_conv[k,:flens_utt]
                            conv_sc_ = conv_sc[k,:flens_utt]
                            scales_c_ = scales_c[k,:slens_utt]
                            signs_c_ = signs_c[k,:slens_utt]
                            logits_c_ = logits_c[k,:slens_utt]
                            scales_f_ = scales_f[k,:slens_utt]
                            signs_f_ = signs_f[k,:slens_utt]
                            logits_f_ = logits_f[k,:slens_utt]
                            x_c_output_ = x_c_output[k,:slens_utt]
                            x_f_output_ = x_f_output[k,:slens_utt]
                            out_ = out[k,:slens_utt]
                            out_2_ = out_2[k,:slens_utt]
                            out_f_ = out_f[k,:slens_utt]
                            melsp = batch_melsp[k,:flens_utt]
                            melsp_rest = (torch.exp(melsp)-1)/10000
                            magsp = magsp_rest = batch_magsp[k,:flens_utt]
                            qz_alpha_fix_ = qz_alpha_fix[k,:flens_utt]
                            qz_alpha_e_fix_ = qz_alpha_e_fix[k,:flens_utt]
                            melsp_rest_log = torch.log10(torch.clamp(melsp_rest, min=1e-16))
                            magsp_rest_log = torch.log10(torch.clamp(magsp_rest, min=1e-16))

                            for i in range(n_half_cyc_eval):
                                batch_x_c_output_ = batch_x_c_output[i][k,:slens_utt]
                                batch_x_f_output_ = batch_x_f_output[i][k,:slens_utt]
                                batch_seg_conv_ = batch_seg_conv[i][k,:flens_utt]
                                batch_conv_sc_ = batch_conv_sc[i][k,:flens_utt]
                                batch_scales_c_ = batch_scales_c[i][k,:slens_utt]
                                batch_signs_c_ = batch_signs_c[i][k,:slens_utt]
                                batch_logits_c_ = batch_logits_c[i][k,:slens_utt]
                                batch_scales_f_ = batch_scales_f[i][k,:slens_utt]
                                batch_signs_f_ = batch_signs_f[i][k,:slens_utt]
                                batch_logits_f_ = batch_logits_f[i][k,:slens_utt]
                                batch_x_c_output_noclamp_ = batch_x_c_output_noclamp[i][k,:slens_utt]
                                batch_x_f_output_noclamp_ = batch_x_f_output_noclamp[i][k,:slens_utt]
                                batch_out_ = batch_out[i][k,:slens_utt]
                                batch_out_2_ = batch_out_2[i][k,:slens_utt]
                                batch_out_f_ = batch_out_f[i][k,:slens_utt]

                                qy_logits_select_ = qy_logits[i][k,:flens_utt]
                                qy_logits_e_select_ = qy_logits_e[i][k,:flens_utt]

                                pdf = batch_pdf_rec[i][k,:flens_utt]
                                melsp_est = batch_melsp_rec[i][k,:flens_utt]
                                melsp_est_rest = (torch.exp(melsp_est)-1)/10000
                                magsp_est = magsp_est_rest = batch_magsp_rec[i][k,:flens_utt]

                                batch_loss_laplace[i] += criterion_laplace(pdf[:,:args.mel_dim], pdf[:,args.mel_dim:], melsp)
                                batch_loss_h[i] += torch.mean(criterion_l1(batch_out_, out_)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(batch_out_, out_))) \
                                                + torch.mean(criterion_l1(batch_out_2_, out_2_)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(batch_out_2_, out_2_))) \
                                                + torch.mean(criterion_l1(batch_out_f_, out_f_)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(batch_out_f_, out_f_)))
                                batch_loss_mid_smpl[i] += torch.mean(criterion_l1(batch_signs_c_, signs_c_)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(batch_signs_c_, signs_c_))) \
                                                    + torch.mean(criterion_l1(batch_scales_c_, scales_c_)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(batch_scales_c_, scales_c_))) \
                                                    + torch.mean(criterion_l1(batch_logits_c_, logits_c_)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(batch_logits_c_, logits_c_))) \
                                                    + torch.mean(criterion_l1(batch_signs_f_, signs_f_)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(batch_signs_f_, signs_f_))) \
                                                    + torch.mean(criterion_l1(batch_scales_f_, scales_f_)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(batch_scales_f_, scales_f_))) \
                                                    + torch.mean(criterion_l1(batch_logits_f_, logits_f_)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(batch_logits_f_, logits_f_))) \
                                                    + torch.mean(criterion_l1(batch_x_c_output_noclamp_, x_c_output_)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(batch_x_c_output_noclamp_, x_c_output_))) \
                                                    + torch.mean(criterion_l1(batch_x_f_output_noclamp_, x_f_output_)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(batch_x_f_output_noclamp_, x_f_output_)))

                                if flens_utt > 1:
                                    batch_loss_seg_conv[i] += torch.mean(criterion_l1(batch_seg_conv_, seg_conv_)) \
                                                                + torch.sqrt(torch.mean(criterion_l2(batch_seg_conv_, seg_conv_)))
                                    batch_loss_conv_sc[i] += torch.mean(criterion_l1(batch_conv_sc_, conv_sc_)) \
                                                                + torch.sqrt(torch.mean(criterion_l2(batch_conv_sc_, conv_sc_)))
                                    batch_loss_melsp[i] += torch.mean(criterion_l1(melsp_est, melsp)) \
                                                                + torch.sqrt(torch.mean(criterion_l2(melsp_est, melsp)))
                                    batch_loss_magsp[i] += torch.mean(criterion_l1(magsp_est, magsp)) \
                                                                 + torch.sqrt(torch.mean(criterion_l2(magsp_est, magsp)))
                                else:
                                    batch_loss_seg_conv[i] += torch.mean(criterion_l1(batch_seg_conv_, seg_conv_))
                                    batch_loss_conv_sc[i] += torch.mean(criterion_l1(batch_conv_sc_, conv_sc_))
                                    batch_loss_melsp[i] += torch.mean(criterion_l1(melsp_est, melsp))
                                    batch_loss_magsp[i] += torch.mean(criterion_l1(magsp_est, magsp))
                                batch_loss_melsp_dB[i] += torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_est_rest, min=1e-16))-melsp_rest_log))**2, -1)))
                                batch_loss_magsp_dB[i] += torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_est_rest, min=1e-16))-magsp_rest_log))**2, -1)))

                                batch_loss_ce_select_ = torch.mean(criterion_ce(batch_x_c_output_.reshape(-1, args.cf_dim), batch_x_c_.reshape(-1)).reshape(batch_x_c_output_.shape[0], -1), 0) # n_bands
                                batch_loss_ce_f_select_ = torch.mean(criterion_ce(batch_x_f_output_.reshape(-1, args.cf_dim), batch_x_f_.reshape(-1)).reshape(batch_x_f_output_.shape[0], -1), 0) # n_bands
                                batch_loss_err_select_ = torch.mean(torch.sum(criterion_l1(F.softmax(batch_x_c_output_, dim=-1), F.one_hot(batch_x_c_, num_classes=args.cf_dim).float()), -1), 0) # n_bands
                                batch_loss_err_f_select_ = torch.mean(torch.sum(criterion_l1(F.softmax(batch_x_f_output_, dim=-1), F.one_hot(batch_x_f_, num_classes=args.cf_dim).float()), -1), 0) # n_bands

                                batch_loss_ce_select[i] += batch_loss_ce_select_
                                batch_loss_ce_f_select[i] += batch_loss_ce_f_select_
                                batch_loss_err_select[i] += 100*batch_loss_err_select_
                                batch_loss_err_f_select[i] += 100*batch_loss_err_f_select_

                                batch_loss_fro_select_, batch_loss_l1_select_ = criterion_stft(batch_x_output[i][k,:slens_utt].transpose(1,0), batch_x_) # n_bands
                                batch_loss_fro_fb_select_, batch_loss_l1_fb_select_ = criterion_stft_fb(batch_x_output_fb[i][k,:slens_utt_fb], batch_x_fb_)

                                batch_loss_fro_select[i] += batch_loss_fro_select_
                                batch_loss_l1_select[i] += batch_loss_l1_select_
                                batch_loss_fro_fb[i] += batch_loss_fro_fb_select_
                                batch_loss_l1_fb[i] += batch_loss_l1_fb_select_

                                if i % 2 == 0:
                                    pdf_cv = batch_pdf_cv[i//2][k,:flens_utt]
                                    batch_loss_laplace_cv[i//2] += criterion_laplace(pdf_cv[:,:args.mel_dim], pdf_cv[:,args.mel_dim:], melsp)
                                    if flens_utt > 1:
                                        melsp_cv_est = batch_melsp_cv[i//2][k,:flens_utt]
                                        magsp_cv_est = batch_magsp_cv[i//2][k,:flens_utt]
                                        batch_loss_melsp_cv[i//2] += torch.mean(criterion_l1(melsp_cv_est, melsp)) \
                                                                    + torch.sqrt(torch.mean(criterion_l2(melsp_cv_est, melsp)))
                                        batch_loss_magsp_cv[i//2] += torch.mean(criterion_l1(magsp_cv_est, magsp)) \
                                                                     + torch.sqrt(torch.mean(criterion_l2(magsp_cv_est, magsp)))
                                    else:
                                        batch_loss_melsp_cv[i//2] += torch.mean(criterion_l1(batch_melsp_cv[i//2][k,:flens_utt], melsp))
                                        batch_loss_magsp_cv[i//2] += torch.mean(criterion_l1(batch_magsp_cv[i//2][k,:flens_utt], magsp))
                        for i in range(n_half_cyc_eval):
                            batch_loss_laplace[i] /= len_idx_select
                            batch_loss_melsp[i] /= len_idx_select
                            batch_loss_melsp_dB[i] /= len_idx_select
                            batch_loss_magsp[i] /= len_idx_select
                            batch_loss_magsp_dB[i] /= len_idx_select
                            batch_loss_seg_conv[i] /= len_idx_select
                            batch_loss_conv_sc[i] /= len_idx_select
                            batch_loss_h[i] /= len_idx_select
                            batch_loss_mid_smpl[i] /= len_idx_select
                            batch_loss_ce_select[i] /= len_idx_select #n_bands
                            batch_loss_ce_f_select[i] /= len_idx_select #n_bands
                            batch_loss_err_select[i] /= len_idx_select #n_bands
                            batch_loss_err_f_select[i] /= len_idx_select #n_bands
                            batch_loss_fro_select[i] /= len_idx_select #n_bands
                            batch_loss_l1_select[i] /= len_idx_select #n_bands
                            batch_loss_fro_fb[i] /= len_idx_select
                            batch_loss_l1_fb[i] /= len_idx_select
                            batch_loss_ce_c_avg[i] = batch_loss_ce_select[i].mean().item()
                            batch_loss_ce_f_avg[i] = batch_loss_ce_f_select[i].mean().item()
                            batch_loss_err_c_avg[i] = batch_loss_err_select[i].mean().item()
                            batch_loss_err_f_avg[i] = batch_loss_err_f_select[i].mean().item()
                            batch_loss_ce_avg[i] = (batch_loss_ce_c_avg[i] + batch_loss_ce_f_avg[i]) / 2
                            batch_loss_err_avg[i] = (batch_loss_err_c_avg[i] + batch_loss_err_f_avg[i]) / 2
                            batch_loss_fro_avg[i] = batch_loss_fro_select[i].mean().item()
                            batch_loss_l1_avg[i] = batch_loss_l1_select[i].mean().item()
                            total_eval_loss["eval/loss_ce-%d"%(i+1)].append(batch_loss_ce_avg[i])
                            total_eval_loss["eval/loss_err-%d"%(i+1)].append(batch_loss_err_avg[i])
                            total_eval_loss["eval/loss_ce_c-%d"%(i+1)].append(batch_loss_ce_c_avg[i])
                            total_eval_loss["eval/loss_err_c-%d"%(i+1)].append(batch_loss_err_c_avg[i])
                            total_eval_loss["eval/loss_ce_f-%d"%(i+1)].append(batch_loss_ce_f_avg[i])
                            total_eval_loss["eval/loss_err_f-%d"%(i+1)].append(batch_loss_err_f_avg[i])
                            loss_ce_avg[i].append(batch_loss_ce_avg[i])
                            loss_err_avg[i].append(batch_loss_err_avg[i])
                            loss_ce_c_avg[i].append(batch_loss_ce_c_avg[i])
                            loss_err_c_avg[i].append(batch_loss_err_c_avg[i])
                            loss_ce_f_avg[i].append(batch_loss_ce_f_avg[i])
                            loss_err_f_avg[i].append(batch_loss_err_f_avg[i])
                            for j in range(args.n_bands):
                                total_eval_loss["eval/loss_ce_c-%d-%d"%(i+1,j+1)].append(batch_loss_ce_select[i][j].item())
                                total_eval_loss["eval/loss_err_c-%d-%d"%(i+1,j+1)].append(batch_loss_err_select[i][j].item())
                                total_eval_loss["eval/loss_ce_f-%d-%d"%(i+1,j+1)].append(batch_loss_ce_f_select[i][j].item())
                                total_eval_loss["eval/loss_err_f-%d-%d"%(i+1,j+1)].append(batch_loss_err_f_select[i][j].item())
                                total_eval_loss["eval/loss_fro-%d-%d"%(i+1,j+1)].append(batch_loss_fro_select[i][j].item())
                                total_eval_loss["eval/loss_l1-%d-%d"%(i+1,j+1)].append(batch_loss_l1_select[i][j].item())
                                loss_ce[i][j].append(batch_loss_ce_select[i][j].item())
                                loss_err[i][j].append(batch_loss_err_select[i][j].item())
                                loss_ce_f[i][j].append(batch_loss_ce_f_select[i][j].item())
                                loss_err_f[i][j].append(batch_loss_err_f_select[i][j].item())
                                loss_fro[i][j].append(batch_loss_fro_select[i][j].item())
                                loss_l1[i][j].append(batch_loss_l1_select[i][j].item())
                            total_eval_loss["eval/loss_fro-%d"%(i+1)].append(batch_loss_fro_avg[i])
                            total_eval_loss["eval/loss_l1-%d"%(i+1)].append(batch_loss_l1_avg[i])
                            total_eval_loss["eval/loss_fro_fb-%d"%(i+1)].append(batch_loss_fro_fb[i].item())
                            total_eval_loss["eval/loss_l1_fb-%d"%(i+1)].append(batch_loss_l1_fb[i].item())
                            total_eval_loss["eval/loss_seg_conv-%d"%(i+1)].append(batch_loss_seg_conv[i].item())
                            total_eval_loss["eval/loss_conv_sc-%d"%(i+1)].append(batch_loss_conv_sc[i].item())
                            total_eval_loss["eval/loss_h-%d"%(i+1)].append(batch_loss_h[i].item())
                            total_eval_loss["eval/loss_mid_smpl-%d"%(i+1)].append(batch_loss_mid_smpl[i].item())
                            loss_fro_avg[i].append(batch_loss_fro_avg[i])
                            loss_l1_avg[i].append(batch_loss_l1_avg[i])
                            loss_fro_fb[i].append(batch_loss_fro_fb[i].item())
                            loss_l1_fb[i].append(batch_loss_l1_fb[i].item())
                            loss_seg_conv[i].append(batch_loss_seg_conv[i].item())
                            loss_conv_sc[i].append(batch_loss_conv_sc[i].item())
                            loss_h[i].append(batch_loss_h[i].item())
                            loss_mid_smpl[i].append(batch_loss_mid_smpl[i].item())
                            total_eval_loss["eval/loss_laplace-%d"%(i+1)].append(batch_loss_laplace[i].item())
                            total_eval_loss["eval/loss_melsp-%d"%(i+1)].append(batch_loss_melsp[i].item())
                            total_eval_loss["eval/loss_melsp_dB-%d"%(i+1)].append(batch_loss_melsp_dB[i].item())
                            total_eval_loss["eval/loss_magsp-%d"%(i+1)].append(batch_loss_magsp[i].item())
                            total_eval_loss["eval/loss_magsp_dB-%d"%(i+1)].append(batch_loss_magsp_dB[i].item())
                            loss_laplace[i].append(batch_loss_laplace[i].item())
                            loss_melsp[i].append(batch_loss_melsp[i].item())
                            loss_melsp_dB[i].append(batch_loss_melsp_dB[i].item())
                            loss_magsp[i].append(batch_loss_magsp[i].item())
                            loss_magsp_dB[i].append(batch_loss_magsp_dB[i].item())
                            if i % 2 == 0:
                                batch_loss_laplace_cv[i//2] /= len_idx_select
                                batch_loss_melsp_cv[i//2] /= len_idx_select
                                batch_loss_magsp_cv[i//2] /= len_idx_select
                                total_eval_loss["eval/loss_laplace_cv-%d"%(i+1)].append(batch_loss_laplace_cv[i//2].item())
                                total_eval_loss["eval/loss_melsp_cv-%d"%(i+1)].append(batch_loss_melsp_cv[i//2].item())
                                total_eval_loss["eval/loss_magsp_cv-%d"%(i+1)].append(batch_loss_magsp_cv[i//2].item())
                                loss_laplace_cv[i//2].append(batch_loss_laplace_cv[i//2].item())
                                loss_melsp_cv[i//2].append(batch_loss_melsp_cv[i//2].item())
                                loss_magsp_cv[i//2].append(batch_loss_magsp_cv[i//2].item())
                        if len(idx_select_full) > 0:
                            logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                            batch_melsp = torch.index_select(batch_melsp,0,idx_select_full)
                            batch_magsp = torch.index_select(batch_magsp,0,idx_select_full)
                            batch_sc = torch.index_select(batch_sc,0,idx_select_full)
                            batch_feat_in_sc = torch.index_select(batch_feat_in_sc,0,idx_select_full)
                            batch_feat_magsp_in_sc = torch.index_select(batch_feat_magsp_in_sc,0,idx_select_full)
                            batch_x_c = torch.index_select(batch_x_c,0,idx_select_full)
                            batch_x_f = torch.index_select(batch_x_f,0,idx_select_full)
                            batch_x = torch.index_select(batch_x,0,idx_select_full)
                            batch_x_fb = torch.index_select(batch_x_fb,0,idx_select_full)
                            x_c_output = torch.index_select(x_c_output,0,idx_select_full)
                            x_f_output = torch.index_select(x_f_output,0,idx_select_full)
                            seg_conv = torch.index_select(seg_conv,0,idx_select_full)
                            conv_sc = torch.index_select(conv_sc,0,idx_select_full)
                            out = torch.index_select(out,0,idx_select_full)
                            out_2 = torch.index_select(out_2,0,idx_select_full)
                            out_f = torch.index_select(out_f,0,idx_select_full)
                            signs_c = torch.index_select(signs_c,0,idx_select_full)
                            scales_c = torch.index_select(scales_c,0,idx_select_full)
                            logits_c = torch.index_select(logits_c,0,idx_select_full)
                            signs_f = torch.index_select(signs_f,0,idx_select_full)
                            scales_f = torch.index_select(scales_f,0,idx_select_full)
                            logits_f = torch.index_select(logits_f,0,idx_select_full)
                            qz_alpha_fix = torch.index_select(qz_alpha_fix,0,idx_select_full)
                            qz_alpha_e_fix = torch.index_select(qz_alpha_e_fix,0,idx_select_full)
                            n_batch_utt = batch_melsp.shape[0]
                            for i in range(n_half_cyc_eval):
                                batch_pdf_rec[i] = torch.index_select(batch_pdf_rec[i],0,idx_select_full)
                                batch_melsp_rec[i] = torch.index_select(batch_melsp_rec[i],0,idx_select_full)
                                batch_magsp_rec[i] = torch.index_select(batch_magsp_rec[i],0,idx_select_full)
                                batch_feat_rec_sc[i] = torch.index_select(batch_feat_rec_sc[i],0,idx_select_full)
                                batch_feat_magsp_rec_sc[i] = torch.index_select(batch_feat_magsp_rec_sc[i],0,idx_select_full)
                                batch_x_c_output[i] = torch.index_select(batch_x_c_output[i],0,idx_select_full)
                                batch_x_f_output[i] = torch.index_select(batch_x_f_output[i],0,idx_select_full)
                                batch_x_output[i] = torch.index_select(batch_x_output[i],0,idx_select_full)
                                batch_x_output_fb[i] = torch.index_select(batch_x_output_fb[i],0,idx_select_full)
                                batch_x_c_output_noclamp[i] = torch.index_select(batch_x_c_output_noclamp[i],0,idx_select_full)
                                batch_x_f_output_noclamp[i] = torch.index_select(batch_x_f_output_noclamp[i],0,idx_select_full)
                                batch_seg_conv[i] = torch.index_select(batch_seg_conv[i],0,idx_select_full)
                                batch_conv_sc[i] = torch.index_select(batch_conv_sc[i],0,idx_select_full)
                                batch_out[i] = torch.index_select(batch_out[i],0,idx_select_full)
                                batch_out_2[i] = torch.index_select(batch_out_2[i],0,idx_select_full)
                                batch_out_f[i] = torch.index_select(batch_out_f[i],0,idx_select_full)
                                batch_signs_c[i] = torch.index_select(batch_signs_c[i],0,idx_select_full)
                                batch_scales_c[i] = torch.index_select(batch_scales_c[i],0,idx_select_full)
                                batch_logits_c[i] = torch.index_select(batch_logits_c[i],0,idx_select_full)
                                batch_signs_f[i] = torch.index_select(batch_signs_f[i],0,idx_select_full)
                                batch_scales_f[i] = torch.index_select(batch_scales_f[i],0,idx_select_full)
                                batch_logits_f[i] = torch.index_select(batch_logits_f[i],0,idx_select_full)
                                qy_logits[i] = torch.index_select(qy_logits[i],0,idx_select_full)
                                qz_alpha[i] = torch.index_select(qz_alpha[i],0,idx_select_full)
                                qy_logits_e[i] = torch.index_select(qy_logits_e[i],0,idx_select_full)
                                qz_alpha_e[i] = torch.index_select(qz_alpha_e[i],0,idx_select_full)
                                if i % 2 == 0:
                                    batch_pdf_cv[i//2] = torch.index_select(batch_pdf_cv[i//2],0,idx_select_full)
                                    batch_melsp_cv[i//2] = torch.index_select(batch_melsp_cv[i//2],0,idx_select_full)
                                    batch_magsp_cv[i//2] = torch.index_select(batch_magsp_cv[i//2],0,idx_select_full)
                                    batch_feat_cv_sc[i//2] = torch.index_select(batch_feat_cv_sc[i//2],0,idx_select_full)
                                    batch_feat_magsp_cv_sc[i//2] = torch.index_select(batch_feat_magsp_cv_sc[i//2],0,idx_select_full)
                                    batch_sc_cv[i//2] = torch.index_select(batch_sc_cv[i//2],0,idx_select_full)
                        else:
                            logging.info("batch loss select (%.3f sec)" % (time.time() - start))
                            iter_count += 1
                            total += time.time() - start
                            continue

                    # loss_compute
                    melsp = batch_melsp
                    melsp_rest = (torch.exp(melsp)-1)/10000
                    melsp_rest_log = torch.log10(torch.clamp(melsp_rest, min=1e-16))
                    magsp = magsp_rest = batch_magsp
                    magsp_rest_log = torch.log10(torch.clamp(magsp_rest, min=1e-16))
                    sc_onehot = F.one_hot(batch_sc, num_classes=n_spk).float()
                    batch_sc_ = batch_sc.reshape(-1)
                    batch_loss_sc_feat_in_ = torch.mean(criterion_ce(batch_feat_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                    batch_loss_sc_feat_in = batch_loss_sc_feat_in_.mean()
                    batch_loss_sc_feat_magsp_in_ = torch.mean(criterion_ce(batch_feat_magsp_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                    batch_loss_sc_feat_magsp_in = batch_loss_sc_feat_magsp_in_.mean()
                    batch_x_c_onehot = F.one_hot(batch_x_c, num_classes=args.cf_dim).float()
                    batch_x_c = batch_x_c.reshape(-1)
                    batch_x_f_onehot = F.one_hot(batch_x_f, num_classes=args.cf_dim).float()
                    T = batch_x_f_onehot.shape[1]
                    batch_x_f = batch_x_f.reshape(-1)
                    batch_x_ = batch_x.transpose(1,2)
                    for i in range(n_half_cyc_eval):
                        ## reconst. [i % 2 == 0] / cyclic reconst. [i % 2 == 1]
                        pdf = batch_pdf_rec[i]
                        melsp_est = batch_melsp_rec[i]
                        melsp_est_rest = (torch.exp(melsp_est)-1)/10000
                        magsp_est = magsp_est_rest = batch_magsp_rec[i]
                        ## conversion
                        if i % 2 == 0:
                            pdf_cv = batch_pdf_cv[i//2]
                            melsp_cv = batch_melsp_cv[i//2]
                            magsp_cv = batch_magsp_cv[i//2]
                        else:
                            sc_cv_onehot = F.one_hot(batch_sc_cv[i//2], num_classes=n_spk).float()

                        batch_loss_laplace_ = criterion_laplace(pdf[:,:,:args.mel_dim], pdf[:,:,args.mel_dim:], melsp)
                        batch_loss_laplace[i] = batch_loss_laplace_.mean()

                        batch_loss_melsp_ = torch.mean(torch.mean(criterion_l1(melsp_est, melsp), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_est, melsp), -1), -1))
                        batch_loss_melsp[i] = batch_loss_melsp_.mean()
                        batch_loss_melsp_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_est_rest, min=1e-16))-melsp_rest_log))**2, -1)), -1)
                        batch_loss_melsp_dB[i] = batch_loss_melsp_dB_.mean()

                        batch_loss_magsp_ = torch.mean(torch.mean(criterion_l1(magsp_est, magsp), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_est, magsp), -1), -1))
                        batch_loss_magsp[i] = batch_loss_magsp_.mean()
                        batch_loss_magsp_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_est_rest, min=1e-16))-magsp_rest_log))**2, -1)), -1)
                        batch_loss_magsp_dB[i] = batch_loss_magsp_dB_.mean()

                        batch_loss_px[i] = batch_loss_laplace_.mean() \
                                            + batch_loss_melsp_.mean() + batch_loss_melsp_dB_.mean() \
                                                + batch_loss_magsp_.mean() + batch_loss_magsp_dB_.mean()

                        batch_loss_px_sum = batch_loss_laplace_.sum() \
                                            + batch_loss_melsp_.sum() + batch_loss_melsp_dB_.sum() \
                                                + batch_loss_magsp_.sum() + batch_loss_magsp_dB_.sum()

                        ## conversion
                        if i % 2 == 0:
                            batch_loss_laplace_cv[i//2] = torch.mean(criterion_laplace(pdf_cv[:,:,:args.mel_dim], pdf_cv[:,:,args.mel_dim:], melsp))
                            batch_loss_melsp_cv[i//2] = torch.mean(criterion_l1(melsp_cv, melsp)) \
                                                            + torch.sqrt(torch.mean(criterion_l2(melsp_cv, melsp)))
                            batch_loss_magsp_cv[i//2] = torch.mean(torch.mean(criterion_l1(magsp_cv, magsp), -1)) \
                                                            + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_cv, magsp), -1)))

                        # speaker-classifier on features and latent
                        batch_sc_cv_ = batch_sc_cv[i//2].reshape(-1)
                        batch_loss_sc_feat_ = torch.mean(criterion_ce(batch_feat_rec_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                        batch_loss_sc_feat[i] = batch_loss_sc_feat_.mean()
                        batch_loss_sc_feat_magsp_ = torch.mean(criterion_ce(batch_feat_magsp_rec_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                        batch_loss_sc_feat_magsp[i] = batch_loss_sc_feat_magsp_.mean()
                        batch_loss_sc_z_ = torch.mean(kl_categorical_categorical_logits(p_spk, logits_p_spk, batch_z_sc[i]), -1)
                        batch_loss_sc_z[i] = batch_loss_sc_z_.mean()
                        batch_loss_sc_feat_kl = batch_loss_sc_feat_.sum() + batch_loss_sc_feat_magsp_.sum() + (100*batch_loss_sc_z_).sum()
                        if i % 2 == 0:
                            batch_loss_sc_feat_cv_ = torch.mean(criterion_ce(batch_feat_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                            batch_loss_sc_feat_cv[i//2] = batch_loss_sc_feat_cv_.mean()
                            batch_loss_sc_feat_magsp_cv_ = torch.mean(criterion_ce(batch_feat_magsp_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                            batch_loss_sc_feat_magsp_cv[i//2] = batch_loss_sc_feat_magsp_cv_.mean()
                            batch_loss_sc_feat_kl += batch_loss_sc_feat_cv_.sum() + batch_loss_sc_feat_magsp_cv_.sum()

                        # KL-div lat., CE and error-percentage spk.
                        if i % 2 == 0:
                            batch_loss_qy_py_ = torch.mean(criterion_ce(qy_logits[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                            batch_loss_qy_py[i] = batch_loss_qy_py_.mean()
                            batch_loss_qy_py_err_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i], dim=-1), sc_onehot), -1), -1)
                            batch_loss_qy_py_err[i] = batch_loss_qy_py_err_.mean()
                        else:
                            batch_loss_qy_py_ = torch.mean(criterion_ce(qy_logits[i].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                            batch_loss_qy_py[i] = batch_loss_qy_py_.mean()
                            batch_loss_qy_py_err_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i], dim=-1), sc_cv_onehot), -1), -1)
                            batch_loss_qy_py_err[i] = batch_loss_qy_py_err_.mean()
                        batch_loss_qz_pz_ = kl_laplace_laplace(qz_alpha[i], qz_alpha_fix)
                        batch_loss_qz_pz[i] = batch_loss_qz_pz_.mean()
                        batch_loss_qz_pz_e_ = kl_laplace_laplace(qz_alpha_e[i], qz_alpha_e_fix)
                        batch_loss_qz_pz_e[i] = batch_loss_qz_pz_e_.mean()
                        batch_loss_qz_pz_kl = batch_loss_qz_pz_.sum() + batch_loss_qz_pz_e_.sum()
                        if i % 2 == 0:
                            batch_loss_qy_py_e_ = torch.mean(criterion_ce(qy_logits_e[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                            batch_loss_qy_py_e[i] = batch_loss_qy_py_e_.mean()
                            batch_loss_qy_py_err_e_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits_e[i], dim=-1), sc_onehot), -1), -1)
                            batch_loss_qy_py_err_e[i] = batch_loss_qy_py_err_e_.mean()
                        else:
                            batch_loss_qy_py_e_ = torch.mean(criterion_ce(qy_logits_e[i].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                            batch_loss_qy_py_e[i] = batch_loss_qy_py_e_.mean()
                            batch_loss_qy_py_err_e_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits_e[i], dim=-1), sc_cv_onehot), -1), -1)
                            batch_loss_qy_py_err_e[i] = batch_loss_qy_py_err_e_.mean()
                        batch_loss_qy_py_kl = batch_loss_qy_py_.sum() + batch_loss_qy_py_e_.sum() \
                                                + batch_loss_qy_py_err_.sum() + batch_loss_qy_py_err_e_.sum()

                        # cosine, rmse latent
                        if i > 0:
                            z_obs = torch.cat((z_e[i], z[i]), 2)
                            batch_loss_lat_cossim_ = torch.clamp(torch.sum(z_obs*z_ref, -1), min=1e-13) / torch.clamp(torch.sqrt(torch.sum(z_obs**2, -1))*z_ref_denom, min=1e-13)
                            batch_loss_lat_cossim[i] = batch_loss_lat_cossim_.mean()
                            batch_loss_lat_rmse_ = torch.sqrt(torch.mean(torch.sum((z_obs-z_ref)**2, -1), -1))
                            batch_loss_lat_rmse[i] = batch_loss_lat_rmse_.mean()
                            batch_loss_qz_pz_kl += batch_loss_lat_rmse_.sum() - torch.log(batch_loss_lat_cossim_).sum()
                        else:
                            z_ref = torch.cat((z_e[i], z[i]), 2)
                            z_ref_denom = torch.sqrt(torch.sum(z_ref**2, -1))

                        # waveform layer loss
                        batch_loss_seg_conv_ = torch.mean(torch.mean(criterion_l1(batch_seg_conv[i], seg_conv), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_seg_conv[i], seg_conv), -1), -1))
                        batch_loss_seg_conv[i] = batch_loss_seg_conv_.mean()

                        batch_loss_conv_sc_ = torch.mean(torch.mean(criterion_l1(batch_conv_sc[i], conv_sc), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_conv_sc[i], conv_sc), -1), -1))
                        batch_loss_conv_sc[i] = batch_loss_conv_sc_.mean()

                        batch_loss_h_ = torch.mean(torch.mean(criterion_l1(batch_out[i], out), -1), -1) \
                                            + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_out[i], out), -1), -1)) \
                                        + torch.mean(torch.mean(criterion_l1(batch_out_2[i], out_2), -1), -1) \
                                            + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_out_2[i], out_2), -1), -1)) \
                                        + torch.mean(torch.mean(criterion_l1(batch_out_f[i], out_f), -1), -1) \
                                            + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_out_f[i], out_f), -1), -1))
                        batch_loss_h[i] = batch_loss_h_.mean()

                        batch_loss_mid_smpl_ = torch.mean(torch.mean(torch.mean(criterion_l1(batch_signs_c[i], signs_c), -1), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_signs_c[i], signs_c), -1), -1), -1)) \
                                            + torch.mean(torch.mean(torch.mean(criterion_l1(batch_scales_c[i], scales_c), -1), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_scales_c[i], scales_c), -1), -1), -1)) \
                                            + torch.mean(torch.mean(torch.mean(criterion_l1(batch_logits_c[i], logits_c), -1), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_logits_c[i], logits_c), -1), -1), -1)) \
                                            + torch.mean(torch.mean(torch.mean(criterion_l1(batch_signs_f[i], signs_f), -1), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_signs_f[i], signs_f), -1), -1), -1)) \
                                            + torch.mean(torch.mean(torch.mean(criterion_l1(batch_scales_f[i], scales_f), -1), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_scales_f[i], scales_f), -1), -1), -1)) \
                                            + torch.mean(torch.mean(torch.mean(criterion_l1(batch_logits_f[i], logits_f), -1), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_logits_f[i], logits_f), -1), -1), -1)) \
                                            + torch.mean(torch.mean(torch.mean(criterion_l1(batch_x_c_output_noclamp[i], x_c_output), -1), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_x_c_output_noclamp[i], x_c_output), -1), -1), -1)) \
                                            + torch.mean(torch.mean(torch.mean(criterion_l1(batch_x_f_output_noclamp[i], x_f_output), -1), -1), -1) \
                                                + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_x_f_output_noclamp[i], x_f_output), -1), -1), -1))
                        batch_loss_mid_smpl[i] = batch_loss_mid_smpl_.mean()

                        batch_loss_wave = batch_loss_seg_conv_.sum() + batch_loss_conv_sc_.sum() \
                                            + batch_loss_h_.sum() + batch_loss_mid_smpl_.sum()

                        # waveform loss
                        batch_loss_ce_ = torch.mean(criterion_ce(batch_x_c_output[i].reshape(-1, args.cf_dim), batch_x_c).reshape(n_batch_utt, T, -1), 1) # B x n_bands
                        batch_loss_err_ = torch.mean(torch.sum(criterion_l1(F.softmax(batch_x_c_output[i], dim=-1), batch_x_c_onehot), -1), 1) # B x n_bands
                        batch_loss_ce_f_ = torch.mean(criterion_ce(batch_x_f_output[i].reshape(-1, args.cf_dim), batch_x_f).reshape(n_batch_utt, T, -1), 1) # B x n_bands
                        batch_loss_err_f_ = torch.mean(torch.sum(criterion_l1(F.softmax(batch_x_f_output[i], dim=-1), batch_x_f_onehot), -1), 1) # B x n_bands
                        logging.info(f'{batch_loss_err_.mean()}')
                        logging.info(f'{batch_loss_err_f_.mean()}')
                        batch_loss_wave += ((batch_loss_err_.sum() + batch_loss_err_f_.sum())/factors) \
                                            + batch_loss_err_.mean(-1).sum() + batch_loss_err_f_.mean(-1).sum() \
                                                + batch_loss_ce_.sum() + batch_loss_ce_f_.sum() \
                                                + batch_loss_ce_.mean(-1).sum() + batch_loss_ce_f_.mean(-1).sum()
                        batch_loss_err_ = 100*batch_loss_err_.mean(0) # n_bands
                        batch_loss_err_f_ = 100*batch_loss_err_f_.mean(0) # n_bands

                        batch_loss_ce_c_avg[i] = batch_loss_ce_.mean().item()
                        batch_loss_err_c_avg[i] = batch_loss_err_.mean().item()
                        batch_loss_ce_f_avg[i] = batch_loss_ce_f_.mean().item()
                        batch_loss_err_f_avg[i] = batch_loss_err_f_.mean().item()
                        batch_loss_ce_avg[i] = (batch_loss_ce_c_avg[i] + batch_loss_ce_f_avg[i]) / 2
                        batch_loss_err_avg[i] = (batch_loss_err_c_avg[i] + batch_loss_err_f_avg[i]) / 2
                        total_eval_loss["eval/loss_ce-%d"%(i+1)].append(batch_loss_ce_avg[i])
                        total_eval_loss["eval/loss_err-%d"%(i+1)].append(batch_loss_err_avg[i])
                        total_eval_loss["eval/loss_ce_c-%d"%(i+1)].append(batch_loss_ce_c_avg[i])
                        total_eval_loss["eval/loss_err_c-%d"%(i+1)].append(batch_loss_err_c_avg[i])
                        total_eval_loss["eval/loss_ce_f-%d"%(i+1)].append(batch_loss_ce_f_avg[i])
                        total_eval_loss["eval/loss_err_f-%d"%(i+1)].append(batch_loss_err_f_avg[i])
                        loss_ce_avg[i].append(batch_loss_ce_avg[i])
                        loss_err_avg[i].append(batch_loss_err_avg[i])
                        loss_ce_c_avg[i].append(batch_loss_ce_c_avg[i])
                        loss_err_c_avg[i].append(batch_loss_err_c_avg[i])
                        loss_ce_f_avg[i].append(batch_loss_ce_f_avg[i])
                        loss_err_f_avg[i].append(batch_loss_err_f_avg[i])
                        batch_loss_fro_, batch_loss_l1_ = criterion_stft(batch_x_output[i].transpose(1,2), batch_x_)
                        for j in range(args.n_bands):
                            batch_loss_ce[i][j] = batch_loss_ce_[:,j].mean().item()
                            batch_loss_err[i][j] = batch_loss_err_[j].item()
                            batch_loss_ce_f[i][j] = batch_loss_ce_f_[:,j].mean().item()
                            batch_loss_err_f[i][j] = batch_loss_err_f_[j].item()
                            batch_loss_fro[i][j] = batch_loss_fro_[:,j].mean().item()
                            batch_loss_l1[i][j] = batch_loss_l1_[:,j].mean().item()
                            total_eval_loss["eval/loss_ce_c-%d-%d"%(i+1,j+1)].append(batch_loss_ce[i][j])
                            total_eval_loss["eval/loss_err_c-%d-%d"%(i+1,j+1)].append(batch_loss_err[i][j])
                            total_eval_loss["eval/loss_ce_f-%d-%d"%(i+1,j+1)].append(batch_loss_ce_f[i][j])
                            total_eval_loss["eval/loss_err_f-%d-%d"%(i+1,j+1)].append(batch_loss_err_f[i][j])
                            total_eval_loss["eval/loss_fro-%d-%d"%(i+1,j+1)].append(batch_loss_fro[i][j])
                            total_eval_loss["eval/loss_l1-%d-%d"%(i+1,j+1)].append(batch_loss_l1[i][j])
                            loss_ce[i][j].append(batch_loss_ce[i][j])
                            loss_err[i][j].append(batch_loss_err[i][j])
                            loss_ce_f[i][j].append(batch_loss_ce_f[i][j])
                            loss_err_f[i][j].append(batch_loss_err_f[i][j])
                            loss_fro[i][j].append(batch_loss_fro[i][j])
                            loss_l1[i][j].append(batch_loss_l1[i][j])
                        batch_loss_fro_avg[i] = batch_loss_fro_.mean().item()
                        batch_loss_l1_avg[i] = batch_loss_l1_.mean().item()
                        batch_loss_fro_fb_, batch_loss_l1_fb_ = criterion_stft_fb(batch_x_output_fb[i], batch_x_fb)
                        batch_loss_wave += batch_loss_fro_.sum() + batch_loss_l1_.sum() \
                                            + batch_loss_fro_fb_.sum() + batch_loss_l1_fb_.sum()
                        batch_loss_fro_fb[i] = batch_loss_fro_fb_.mean().item()
                        batch_loss_l1_fb[i] = batch_loss_l1_fb_.mean().item()
                        total_eval_loss["eval/loss_fro-%d"%(i+1)].append(batch_loss_fro_avg[i])
                        total_eval_loss["eval/loss_l1-%d"%(i+1)].append(batch_loss_l1_avg[i])
                        total_eval_loss["eval/loss_fro_fb-%d"%(i+1)].append(batch_loss_fro_fb[i])
                        total_eval_loss["eval/loss_l1_fb-%d"%(i+1)].append(batch_loss_l1_fb[i])
                        total_eval_loss["eval/loss_seg_conv-%d"%(i+1)].append(batch_loss_seg_conv[i].item())
                        total_eval_loss["eval/loss_conv_sc-%d"%(i+1)].append(batch_loss_conv_sc[i].item())
                        total_eval_loss["eval/loss_seg_conv-%d"%(i+1)].append(batch_loss_seg_conv[i].item())
                        total_eval_loss["eval/loss_conv_sc-%d"%(i+1)].append(batch_loss_conv_sc[i].item())
                        total_eval_loss["eval/loss_h-%d"%(i+1)].append(batch_loss_h[i].item())
                        total_eval_loss["eval/loss_mid_smpl-%d"%(i+1)].append(batch_loss_mid_smpl[i].item())
                        loss_fro_avg[i].append(batch_loss_fro_avg[i])
                        loss_l1_avg[i].append(batch_loss_l1_avg[i])
                        loss_fro_fb[i].append(batch_loss_fro_fb[i])
                        loss_l1_fb[i].append(batch_loss_l1_fb[i])
                        loss_seg_conv[i].append(batch_loss_seg_conv[i].item())
                        loss_conv_sc[i].append(batch_loss_conv_sc[i].item())
                        loss_h[i].append(batch_loss_h[i].item())
                        loss_mid_smpl[i].append(batch_loss_mid_smpl[i].item())

                        # elbo
                        batch_loss_elbo[i] = batch_loss_px_sum + batch_loss_wave + batch_loss_qz_pz_kl + batch_loss_qy_py_kl + batch_loss_sc_feat_kl

                        total_eval_loss["eval/loss_elbo-%d"%(i+1)].append(batch_loss_elbo[i].item())
                        total_eval_loss["eval/loss_px-%d"%(i+1)].append(batch_loss_px[i].item())
                        total_eval_loss["eval/loss_qy_py-%d"%(i+1)].append(batch_loss_qy_py[i].item())
                        total_eval_loss["eval/loss_qy_py_err-%d"%(i+1)].append(batch_loss_qy_py_err[i].item())
                        total_eval_loss["eval/loss_qz_pz-%d"%(i+1)].append(batch_loss_qz_pz[i].item())
                        total_eval_loss["eval/loss_qy_py_e-%d"%(i+1)].append(batch_loss_qy_py_e[i].item())
                        total_eval_loss["eval/loss_qy_py_err_e-%d"%(i+1)].append(batch_loss_qy_py_err_e[i].item())
                        total_eval_loss["eval/loss_qz_pz_e-%d"%(i+1)].append(batch_loss_qz_pz_e[i].item())
                        total_eval_loss["eval/loss_sc_z-%d"%(i+1)].append(batch_loss_sc_z[i].item())
                        if i > 0:
                            total_eval_loss["eval/loss_cossim-%d"%(i+1)].append(batch_loss_lat_cossim[i].item())
                            total_eval_loss["eval/loss_rmse-%d"%(i+1)].append(batch_loss_lat_rmse[i].item())
                            loss_lat_cossim[i].append(batch_loss_lat_cossim[i].item())
                            loss_lat_rmse[i].append(batch_loss_lat_rmse[i].item())
                        else:
                            total_eval_loss["eval/loss_sc_feat_in"].append(batch_loss_sc_feat_in.item())
                            total_eval_loss["eval/loss_sc_feat_magsp_in"].append(batch_loss_sc_feat_magsp_in.item())
                            loss_sc_feat_in.append(batch_loss_sc_feat_in.item())
                            loss_sc_feat_magsp_in.append(batch_loss_sc_feat_magsp_in.item())
                        total_eval_loss["eval/loss_sc_feat-%d"%(i+1)].append(batch_loss_sc_feat[i].item())
                        total_eval_loss["eval/loss_sc_feat_magsp-%d"%(i+1)].append(batch_loss_sc_feat_magsp[i].item())
                        loss_elbo[i].append(batch_loss_elbo[i].item())
                        loss_px[i].append(batch_loss_px[i].item())
                        loss_qy_py[i].append(batch_loss_qy_py[i].item())
                        loss_qy_py_err[i].append(batch_loss_qy_py_err[i].item())
                        loss_qz_pz[i].append(batch_loss_qz_pz[i].item())
                        loss_qy_py_e[i].append(batch_loss_qy_py_e[i].item())
                        loss_qy_py_err_e[i].append(batch_loss_qy_py_err_e[i].item())
                        loss_qz_pz_e[i].append(batch_loss_qz_pz_e[i].item())
                        loss_sc_z[i].append(batch_loss_sc_z[i].item())
                        loss_sc_feat[i].append(batch_loss_sc_feat[i].item())
                        loss_sc_feat_magsp[i].append(batch_loss_sc_feat_magsp[i].item())
                        ## in-domain reconst.
                        total_eval_loss["eval/loss_laplace-%d"%(i+1)].append(batch_loss_laplace[i].item())
                        total_eval_loss["eval/loss_melsp-%d"%(i+1)].append(batch_loss_melsp[i].item())
                        total_eval_loss["eval/loss_melsp_dB-%d"%(i+1)].append(batch_loss_melsp_dB[i].item())
                        total_eval_loss["eval/loss_magsp-%d"%(i+1)].append(batch_loss_magsp[i].item())
                        total_eval_loss["eval/loss_magsp_dB-%d"%(i+1)].append(batch_loss_magsp_dB[i].item())
                        loss_laplace[i].append(batch_loss_laplace[i].item())
                        loss_melsp[i].append(batch_loss_melsp[i].item())
                        loss_melsp_dB[i].append(batch_loss_melsp_dB[i].item())
                        loss_magsp[i].append(batch_loss_magsp[i].item())
                        loss_magsp_dB[i].append(batch_loss_magsp_dB[i].item())
                        ## conversion
                        if i % 2 == 0:
                            total_eval_loss["eval/loss_sc_feat_cv-%d"%(i+1)].append(batch_loss_sc_feat_cv[i//2].item())
                            total_eval_loss["eval/loss_sc_feat_magsp_cv-%d"%(i+1)].append(batch_loss_sc_feat_magsp_cv[i//2].item())
                            total_eval_loss["eval/loss_laplace_cv-%d"%(i+1)].append(batch_loss_laplace_cv[i//2].item())
                            total_eval_loss["eval/loss_melsp_cv-%d"%(i+1)].append(batch_loss_melsp_cv[i//2].item())
                            total_eval_loss["eval/loss_magsp_cv-%d"%(i+1)].append(batch_loss_magsp_cv[i//2].item())
                            loss_sc_feat_cv[i//2].append(batch_loss_sc_feat_cv[i//2].item())
                            loss_sc_feat_magsp_cv[i//2].append(batch_loss_sc_feat_magsp_cv[i//2].item())
                            loss_laplace_cv[i//2].append(batch_loss_laplace_cv[i//2].item())
                            loss_melsp_cv[i//2].append(batch_loss_melsp_cv[i//2].item())
                            loss_magsp_cv[i//2].append(batch_loss_magsp_cv[i//2].item())

                    text_log = "batch eval loss [%d] %d %d %d %d %.3f %.3f " % (c_idx+1, x_ss, x_bs, f_ss, f_bs, batch_loss_sc_feat_in.item(), batch_loss_sc_feat_magsp_in.item())
                    for i in range(n_half_cyc_eval):
                        if i == 0:
                            text_log += "[%ld] %.3f , %.3f ; %.3f %.3f %% %.3f , %.3f %.3f %% %.3f ; %.3f , %.3f %.3f , %.3f %.3f ; " \
                                "%.3f %.3f , %.3f %.3f %.3f dB , %.3f %.3f %.3f dB ; %.3f %.3f , %.3f %.3f ; %.3f %.3f %% %.3f %.3f %% %.3f %.3f %% , %.3f %.3f , %.3f %.3f ; " % (i+1,
                                batch_loss_elbo[i].item(), batch_loss_px[i].item(),
                                    batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(), batch_loss_qz_pz[i].item(),
                                    batch_loss_qy_py_e[i].item(), batch_loss_qy_py_err_e[i].item(), batch_loss_qz_pz_e[i].item(),
                                    batch_loss_sc_z[i].item(), batch_loss_sc_feat[i].item(), batch_loss_sc_feat_cv[i//2].item(), batch_loss_sc_feat_magsp[i].item(), batch_loss_sc_feat_magsp_cv[i//2].item(),
                                        batch_loss_laplace[i].item(), batch_loss_laplace_cv[i//2].item(), batch_loss_melsp[i].item(), batch_loss_melsp_cv[i//2].item(), batch_loss_melsp_dB[i].item(),
                                        batch_loss_magsp[i].item(), batch_loss_magsp_cv[i//2].item(), batch_loss_magsp_dB[i].item(),
                                        batch_loss_seg_conv[i], batch_loss_conv_sc[i], batch_loss_h[i], batch_loss_mid_smpl[i],
                                        batch_loss_ce_avg[i], batch_loss_err_avg[i], batch_loss_ce_c_avg[i], batch_loss_err_c_avg[i], batch_loss_ce_f_avg[i], batch_loss_err_f_avg[i],
                                        batch_loss_fro_avg[i], batch_loss_l1_avg[i], batch_loss_fro_fb[i], batch_loss_l1_fb[i])
                        else:
                            text_log += "[%ld] %.3f , %.3f ; %.3f %.3f , %.3f %.3f %% %.3f , %.3f %.3f %% %.3f ; "\
                                "%.3f , %.3f , %.3f ; %.3f , %.3f %.3f dB , %.3f %.3f dB ; %.3f %.3f , %.3f %.3f ; %.3f %.3f %% %.3f %.3f %% %.3f %.3f %% , %.3f %.3f , %.3f %.3f ; " % (i+1,
                                batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_lat_cossim[i].item(), batch_loss_lat_rmse[i].item(),
                                    batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(), batch_loss_qz_pz[i].item(),
                                    batch_loss_qy_py_e[i].item(), batch_loss_qy_py_err_e[i].item(), batch_loss_qz_pz_e[i].item(),
                                        batch_loss_sc_z[i].item(), batch_loss_sc_feat[i].item(), batch_loss_sc_feat_magsp[i].item(),
                                            batch_loss_laplace[i].item(), batch_loss_melsp[i].item(), batch_loss_melsp_dB[i].item(), batch_loss_magsp[i].item(), batch_loss_magsp_dB[i].item(),
                                            batch_loss_seg_conv[i], batch_loss_conv_sc[i], batch_loss_h[i], batch_loss_mid_smpl[i],
                                            batch_loss_ce_avg[i], batch_loss_err_avg[i], batch_loss_ce_c_avg[i], batch_loss_err_c_avg[i], batch_loss_ce_f_avg[i], batch_loss_err_f_avg[i],
                                            batch_loss_fro_avg[i], batch_loss_l1_avg[i], batch_loss_fro_fb[i], batch_loss_l1_fb[i])
                        for j in range(args.n_bands):
                            text_log += "[%d-%d] %.3f %.3f %% %.3f %.3f %% , %.3f %.3f " % (i+1, j+1,
                                batch_loss_ce[i][j], batch_loss_err[i][j], batch_loss_ce_f[i][j], batch_loss_err_f[i][j],
                                    batch_loss_fro[i][j], batch_loss_l1[i][j])
                        text_log += ";; "
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            tmp_gv_1 = []
            tmp_gv_2 = []
            for j in range(n_spk):
                if len(gv_src_src[j]) > 0:
                    tmp_gv_1.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_src[j],
                                        axis=0))-np.log(gv_mean[j])))))
                if len(gv_src_trg[j]) > 0:
                    tmp_gv_2.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg[j],
                                        axis=0))-np.log(gv_mean[j])))))
            eval_loss_gv_src_src = np.mean(tmp_gv_1)
            eval_loss_gv_src_trg = np.mean(tmp_gv_2)
            total_eval_loss["eval/loss_gv_src_src"].append(eval_loss_gv_src_src)
            total_eval_loss["eval/loss_gv_src_trg"].append(eval_loss_gv_src_trg)
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
                eval_loss_lat_dist_rmse = np.mean(loss_lat_dist_rmse)
                eval_loss_lat_dist_rmse_std = np.std(loss_lat_dist_rmse)
                eval_loss_lat_dist_cossim = np.mean(loss_lat_dist_cossim)
                eval_loss_lat_dist_cossim_std = np.std(loss_lat_dist_cossim)
            for i in range(n_half_cyc_eval):
                eval_loss_elbo[i] = np.mean(loss_elbo[i])
                eval_loss_elbo_std[i] = np.std(loss_elbo[i])
                eval_loss_px[i] = np.mean(loss_px[i])
                eval_loss_px_std[i] = np.std(loss_px[i])
                eval_loss_qy_py[i] = np.mean(loss_qy_py[i])
                eval_loss_qy_py_std[i] = np.std(loss_qy_py[i])
                eval_loss_qy_py_err[i] = np.mean(loss_qy_py_err[i])
                eval_loss_qy_py_err_std[i] = np.std(loss_qy_py_err[i])
                eval_loss_qz_pz[i] = np.mean(loss_qz_pz[i])
                eval_loss_qz_pz_std[i] = np.std(loss_qz_pz[i])
                eval_loss_qy_py_e[i] = np.mean(loss_qy_py_e[i])
                eval_loss_qy_py_e_std[i] = np.std(loss_qy_py_e[i])
                eval_loss_qy_py_err_e[i] = np.mean(loss_qy_py_err_e[i])
                eval_loss_qy_py_err_e_std[i] = np.std(loss_qy_py_err_e[i])
                eval_loss_qz_pz_e[i] = np.mean(loss_qz_pz_e[i])
                eval_loss_qz_pz_e_std[i] = np.std(loss_qz_pz_e[i])
                if i > 0:
                    eval_loss_lat_cossim[i] = np.mean(loss_lat_cossim[i])
                    eval_loss_lat_cossim_std[i] = np.std(loss_lat_cossim[i])
                    eval_loss_lat_rmse[i] = np.mean(loss_lat_rmse[i])
                    eval_loss_lat_rmse_std[i] = np.std(loss_lat_rmse[i])
                eval_loss_sc_z[i] = np.mean(loss_sc_z[i])
                eval_loss_sc_z_std[i] = np.std(loss_sc_z[i])
                eval_loss_sc_feat[i] = np.mean(loss_sc_feat[i])
                eval_loss_sc_feat_std[i] = np.std(loss_sc_feat[i])
                eval_loss_sc_feat_magsp[i] = np.mean(loss_sc_feat_magsp[i])
                eval_loss_sc_feat_magsp_std[i] = np.std(loss_sc_feat_magsp[i])
                eval_loss_laplace[i] = np.mean(loss_laplace[i])
                eval_loss_laplace_std[i] = np.std(loss_laplace[i])
                eval_loss_melsp[i] = np.mean(loss_melsp[i])
                eval_loss_melsp_std[i] = np.std(loss_melsp[i])
                eval_loss_melsp_dB[i] = np.mean(loss_melsp_dB[i])
                eval_loss_melsp_dB_std[i] = np.std(loss_melsp_dB[i])
                eval_loss_magsp[i] = np.mean(loss_magsp[i])
                eval_loss_magsp_std[i] = np.std(loss_magsp[i])
                eval_loss_magsp_dB[i] = np.mean(loss_magsp_dB[i])
                eval_loss_magsp_dB_std[i] = np.std(loss_magsp_dB[i])
                eval_loss_seg_conv[i] = np.mean(loss_seg_conv[i])
                eval_loss_seg_conv_std[i] = np.std(loss_seg_conv[i])
                eval_loss_conv_sc[i] = np.mean(loss_conv_sc[i])
                eval_loss_conv_sc_std[i] = np.std(loss_conv_sc[i])
                eval_loss_h[i] = np.mean(loss_h[i])
                eval_loss_h_std[i] = np.std(loss_h[i])
                eval_loss_mid_smpl[i] = np.mean(loss_mid_smpl[i])
                eval_loss_mid_smpl_std[i] = np.std(loss_mid_smpl[i])
                eval_loss_ce_avg[i] = np.mean(loss_ce_avg[i])
                eval_loss_ce_avg_std[i] = np.std(loss_ce_avg[i])
                eval_loss_err_avg[i] = np.mean(loss_err_avg[i])
                eval_loss_err_avg_std[i] = np.std(loss_err_avg[i])
                eval_loss_ce_c_avg[i] = np.mean(loss_ce_c_avg[i])
                eval_loss_ce_c_avg_std[i] = np.std(loss_ce_c_avg[i])
                eval_loss_err_c_avg[i] = np.mean(loss_err_c_avg[i])
                eval_loss_err_c_avg_std[i] = np.std(loss_err_c_avg[i])
                eval_loss_ce_f_avg[i] = np.mean(loss_ce_f_avg[i])
                eval_loss_ce_f_avg_std[i] = np.std(loss_ce_f_avg[i])
                eval_loss_err_f_avg[i] = np.mean(loss_err_f_avg[i])
                eval_loss_err_f_avg_std[i] = np.std(loss_err_f_avg[i])
                eval_loss_fro_avg[i] = np.mean(loss_fro_avg[i])
                eval_loss_fro_avg_std[i] = np.std(loss_fro_avg[i])
                eval_loss_l1_avg[i] = np.mean(loss_l1_avg[i])
                eval_loss_l1_avg_std[i] = np.std(loss_l1_avg[i])
                eval_loss_fro_fb[i] = np.mean(loss_fro_fb[i])
                eval_loss_fro_fb_std[i] = np.std(loss_fro_fb[i])
                eval_loss_l1_fb[i] = np.mean(loss_l1_fb[i])
                eval_loss_l1_fb_std[i] = np.std(loss_l1_fb[i])
                for j in range(args.n_bands):
                    eval_loss_ce[i][j] = np.mean(loss_ce[i][j])
                    eval_loss_ce_std[i][j] = np.std(loss_ce[i][j])
                    eval_loss_err[i][j] = np.mean(loss_err[i][j])
                    eval_loss_err_std[i][j] = np.std(loss_err[i][j])
                    eval_loss_ce_f[i][j] = np.mean(loss_ce_f[i][j])
                    eval_loss_ce_f_std[i][j] = np.std(loss_ce_f[i][j])
                    eval_loss_err_f[i][j] = np.mean(loss_err_f[i][j])
                    eval_loss_err_f_std[i][j] = np.std(loss_err_f[i][j])
                    eval_loss_fro[i][j] = np.mean(loss_fro[i][j])
                    eval_loss_fro_std[i][j] = np.std(loss_fro[i][j])
                    eval_loss_l1[i][j] = np.mean(loss_l1[i][j])
                    eval_loss_l1_std[i][j] = np.std(loss_l1[i][j])
                if i % 2 == 0:
                    eval_loss_sc_feat_cv[i//2] = np.mean(loss_sc_feat_cv[i//2])
                    eval_loss_sc_feat_cv_std[i//2] = np.std(loss_sc_feat_cv[i//2])
                    eval_loss_sc_feat_magsp_cv[i//2] = np.mean(loss_sc_feat_magsp_cv[i//2])
                    eval_loss_sc_feat_magsp_cv_std[i//2] = np.std(loss_sc_feat_magsp_cv[i//2])
                    eval_loss_laplace_cv[i//2] = np.mean(loss_laplace_cv[i//2])
                    eval_loss_laplace_cv_std[i//2] = np.std(loss_laplace_cv[i//2])
                    eval_loss_melsp_cv[i//2] = np.mean(loss_melsp_cv[i//2])
                    eval_loss_melsp_cv_std[i//2] = np.std(loss_melsp_cv[i//2])
                    eval_loss_magsp_cv[i//2] = np.mean(loss_magsp_cv[i//2])
                    eval_loss_magsp_cv_std[i//2] = np.std(loss_magsp_cv[i//2])
            text_log = "(EPOCH:%d) average evaluation loss = %.6f (+- %.6f) ; " % (
                epoch_idx + 1, eval_loss_sc_feat_in, eval_loss_sc_feat_in_std)
            for i in range(n_half_cyc_eval):
                if i % 2 == 0:
                    text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB ; %.6f %.6f " % (i+1,
                        eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_px[i], eval_loss_px_std[i],
                        eval_loss_qy_py[i], eval_loss_qy_py_std[i], eval_loss_qy_py_err[i], eval_loss_qy_py_err_std[i], eval_loss_qz_pz[i], eval_loss_qz_pz_std[i],
                        eval_loss_qy_py_e[i], eval_loss_qy_py_e_std[i], eval_loss_qy_py_err_e[i], eval_loss_qy_py_err_e_std[i], eval_loss_qz_pz_e[i], eval_loss_qz_pz_e_std[i],
                        eval_loss_sc_z[i], eval_loss_sc_z_std[i], eval_loss_sc_feat[i], eval_loss_sc_feat_std[i], eval_loss_sc_feat_cv[i//2], eval_loss_sc_feat_cv_std[i//2],
                        eval_loss_sc_feat_magsp[i], eval_loss_sc_feat_magsp_std[i], eval_loss_sc_feat_magsp_cv[i//2], eval_loss_sc_feat_magsp_cv_std[i//2],
                        eval_loss_laplace[i], eval_loss_laplace_std[i], eval_loss_laplace_cv[i//2], eval_loss_laplace_cv_std[i//2],
                        eval_loss_melsp[i], eval_loss_melsp_std[i], eval_loss_melsp_cv[i//2], eval_loss_melsp_cv_std[i//2], eval_loss_melsp_dB[i], eval_loss_melsp_dB_std[i],
                        eval_loss_magsp[i], eval_loss_magsp_std[i], eval_loss_magsp_cv[i//2], eval_loss_magsp_cv_std[i//2], eval_loss_magsp_dB[i], eval_loss_magsp_dB_std[i],
                        eval_loss_gv_src_src, eval_loss_gv_src_trg)
                    if pair_exist:
                        text_log += "%.6f (+- %.6f) dB %.6f (+- %.6f) %.6f (+- %.6f) ; " % (
                            eval_loss_melsp_dB_src_trg, eval_loss_melsp_dB_src_trg_std,
                            eval_loss_lat_dist_rmse, eval_loss_lat_dist_rmse_std, eval_loss_lat_dist_cossim, eval_loss_lat_dist_cossim_std)
                    else:
                        text_log += "n/a (+- n/a) dB n/a (+- n/a) n/a (+- n/a) ; "
                else:
                    text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                            "%.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; " % (i+1,
                        eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_px[i], eval_loss_px_std[i],
                        eval_loss_lat_cossim[i], eval_loss_lat_cossim_std[i], eval_loss_lat_rmse[i], eval_loss_lat_rmse_std[i],
                        eval_loss_qy_py[i], eval_loss_qy_py_std[i], eval_loss_qy_py_err[i], eval_loss_qy_py_err_std[i], eval_loss_qz_pz[i], eval_loss_qz_pz_std[i],
                        eval_loss_qy_py_e[i], eval_loss_qy_py_e_std[i], eval_loss_qy_py_err_e[i], eval_loss_qy_py_err_e_std[i], eval_loss_qz_pz_e[i], eval_loss_qz_pz_e_std[i],
                        eval_loss_sc_z[i], eval_loss_sc_z_std[i], eval_loss_sc_feat[i], eval_loss_sc_feat_std[i], eval_loss_sc_feat_magsp[i], eval_loss_sc_feat_magsp_std[i],
                        eval_loss_laplace[i], eval_loss_laplace_std[i], eval_loss_melsp[i], eval_loss_melsp_std[i], eval_loss_melsp_dB[i], eval_loss_melsp_dB_std[i],
                        eval_loss_magsp[i], eval_loss_magsp_std[i], eval_loss_magsp_dB[i], eval_loss_magsp_dB_std[i])
                text_log += "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                        "%.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " % (
                    eval_loss_seg_conv[i], eval_loss_seg_conv_std[i], eval_loss_conv_sc[i], eval_loss_conv_sc_std[i],
                    eval_loss_h[i], eval_loss_h_std[i], eval_loss_mid_smpl[i], eval_loss_mid_smpl_std[i],
                    eval_loss_ce_avg[i], eval_loss_ce_avg_std[i], eval_loss_err_avg[i], eval_loss_err_avg_std[i],
                        eval_loss_ce_c_avg[i], eval_loss_ce_c_avg_std[i], eval_loss_err_c_avg[i], eval_loss_err_c_avg_std[i],
                            eval_loss_ce_f_avg[i], eval_loss_ce_f_avg_std[i], eval_loss_err_f_avg[i], eval_loss_err_f_avg_std[i],
                                eval_loss_fro_avg[i], eval_loss_fro_avg_std[i], eval_loss_l1_avg[i], eval_loss_l1_avg_std[i],
                                    eval_loss_fro_fb[i], eval_loss_fro_fb_std[i], eval_loss_l1_fb[i], eval_loss_l1_fb_std[i])
                for j in range(args.n_bands):
                    text_log += "[%d-%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% , %.6f (+- %.6f) %.6f (+- %.6f) " % (i+1, j+1,
                        eval_loss_ce[i][j], eval_loss_ce_std[i][j], eval_loss_err[i][j], eval_loss_err_std[i][j],
                            eval_loss_ce_f[i][j], eval_loss_ce_f_std[i][j], eval_loss_err_f[i][j], eval_loss_err_f_std[i][j],
                                eval_loss_fro[i][j], eval_loss_fro_std[i][j], eval_loss_l1[i][j], eval_loss_l1_std[i][j])
                text_log += ";; "
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            if (not sparse_min_flag) and (iter_idx + 1 >= t_ends[idx_stage]):
                sparse_check_flag = True
            if (not sparse_min_flag and sparse_check_flag) \
                or ((round(float(round(Decimal(str(eval_loss_err_avg[0])),2))-0.66,2) <= float(round(Decimal(str(min_eval_loss_err_avg[0])),2))) and \
                    (round(float(round(Decimal(str(eval_loss_l1_avg[0])),2))-0.09,2) <= float(round(Decimal(str(min_eval_loss_l1_avg[0])),2))) and \
                    (round(float(round(Decimal(str(eval_loss_l1_fb[0])),2))-0.09,2) <= float(round(Decimal(str(min_eval_loss_l1_fb[0])),2))) and \
                    (float(round(Decimal(str(eval_loss_laplace_cv[0]-eval_loss_laplace[0])),2)) >= round(float(round(Decimal(str(min_eval_loss_laplace_cv[0]-min_eval_loss_laplace[0])),2))-0.03,2)) and \
                    (round(float(round(Decimal(str(eval_loss_laplace[0])),2))-0.05,2) <= float(round(Decimal(str(min_eval_loss_laplace[0])),2))) and \
                    (round(float(round(Decimal(str(eval_loss_ce_avg[0]+eval_loss_ce_avg_std[0])),2))-0.02,2) <= float(round(Decimal(str(min_eval_loss_ce_avg[0]+min_eval_loss_ce_avg_std[0])),2)) \
                        or round(float(round(Decimal(str(eval_loss_ce_avg[0])),2))-0.02,2) <= float(round(Decimal(str(min_eval_loss_ce_avg[0])),2)))):
                round_eval_loss_err_avg = float(round(Decimal(str(eval_loss_err_avg[0])),2))
                round_min_eval_loss_err_avg = float(round(Decimal(str(min_eval_loss_err_avg[0])),2))
                if (round_eval_loss_err_avg <= round_min_eval_loss_err_avg) or (not err_flag and round_eval_loss_err_avg > round_min_eval_loss_err_avg) or (not sparse_min_flag and sparse_check_flag):
                    if sparse_min_flag:
                        if round_eval_loss_err_avg > round_min_eval_loss_err_avg:
                            err_flag = True
                        elif round_eval_loss_err_avg <= round_min_eval_loss_err_avg:
                            err_flag = False
                    elif sparse_check_flag:
                        sparse_min_flag = True
                        err_flag = False
                    min_eval_loss_gv_src_src = eval_loss_gv_src_src
                    min_eval_loss_gv_src_trg = eval_loss_gv_src_trg
                    min_eval_loss_sc_feat_in = eval_loss_sc_feat_in
                    min_eval_loss_sc_feat_in_std = eval_loss_sc_feat_in_std
                    min_eval_loss_sc_feat_magsp_in = eval_loss_sc_feat_magsp_in
                    min_eval_loss_sc_feat_magsp_in_std = eval_loss_sc_feat_magsp_in_std
                    if pair_exist:
                        min_eval_loss_melsp_dB_src_trg = eval_loss_melsp_dB_src_trg
                        min_eval_loss_melsp_dB_src_trg_std = eval_loss_melsp_dB_src_trg_std
                        min_eval_loss_lat_dist_rmse = eval_loss_lat_dist_rmse
                        min_eval_loss_lat_dist_rmse_std = eval_loss_lat_dist_rmse_std
                        min_eval_loss_lat_dist_cossim = eval_loss_lat_dist_cossim
                        min_eval_loss_lat_dist_cossim_std = eval_loss_lat_dist_cossim_std
                    for i in range(n_half_cyc_eval):
                        min_eval_loss_elbo[i] = eval_loss_elbo[i]
                        min_eval_loss_elbo_std[i] = eval_loss_elbo_std[i]
                        min_eval_loss_px[i] = eval_loss_px[i]
                        min_eval_loss_px_std[i] = eval_loss_px_std[i]
                        min_eval_loss_qy_py[i] = eval_loss_qy_py[i]
                        min_eval_loss_qy_py_std[i] = eval_loss_qy_py_std[i]
                        min_eval_loss_qy_py_err[i] = eval_loss_qy_py_err[i]
                        min_eval_loss_qy_py_err_std[i] = eval_loss_qy_py_err_std[i]
                        min_eval_loss_qz_pz[i] = eval_loss_qz_pz[i]
                        min_eval_loss_qz_pz_std[i] = eval_loss_qz_pz_std[i]
                        min_eval_loss_qy_py_e[i] = eval_loss_qy_py_e[i]
                        min_eval_loss_qy_py_e_std[i] = eval_loss_qy_py_e_std[i]
                        min_eval_loss_qy_py_err_e[i] = eval_loss_qy_py_err_e[i]
                        min_eval_loss_qy_py_err_e_std[i] = eval_loss_qy_py_err_e_std[i]
                        min_eval_loss_qz_pz_e[i] = eval_loss_qz_pz_e[i]
                        min_eval_loss_qz_pz_e_std[i] = eval_loss_qz_pz_e_std[i]
                        if i > 0:
                            min_eval_loss_lat_cossim[i] = eval_loss_lat_cossim[i]
                            min_eval_loss_lat_cossim_std[i] = eval_loss_lat_cossim_std[i]
                            min_eval_loss_lat_rmse[i] = eval_loss_lat_rmse[i]
                            min_eval_loss_lat_rmse_std[i] = eval_loss_lat_rmse_std[i]
                        min_eval_loss_sc_z[i] = eval_loss_sc_z[i]
                        min_eval_loss_sc_z_std[i] = eval_loss_sc_z_std[i]
                        min_eval_loss_sc_feat[i] = eval_loss_sc_feat[i]
                        min_eval_loss_sc_feat_std[i] = eval_loss_sc_feat_std[i]
                        min_eval_loss_sc_feat_magsp[i] = eval_loss_sc_feat_magsp[i]
                        min_eval_loss_sc_feat_magsp_std[i] = eval_loss_sc_feat_magsp_std[i]
                        min_eval_loss_laplace[i] = eval_loss_laplace[i]
                        min_eval_loss_laplace_std[i] = eval_loss_laplace_std[i]
                        min_eval_loss_melsp[i] = eval_loss_melsp[i]
                        min_eval_loss_melsp_std[i] = eval_loss_melsp_std[i]
                        min_eval_loss_melsp_dB[i] = eval_loss_melsp_dB[i]
                        min_eval_loss_melsp_dB_std[i] = eval_loss_melsp_dB_std[i]
                        min_eval_loss_magsp[i] = eval_loss_magsp[i]
                        min_eval_loss_magsp_std[i] = eval_loss_magsp_std[i]
                        min_eval_loss_magsp_dB[i] = eval_loss_magsp_dB[i]
                        min_eval_loss_magsp_dB_std[i] = eval_loss_magsp_dB_std[i]
                        min_eval_loss_seg_conv[i] = eval_loss_seg_conv[i]
                        min_eval_loss_seg_conv_std[i] = eval_loss_seg_conv_std[i]
                        min_eval_loss_conv_sc[i] = eval_loss_conv_sc[i]
                        min_eval_loss_conv_sc_std[i] = eval_loss_conv_sc_std[i]
                        min_eval_loss_h[i] = eval_loss_h[i]
                        min_eval_loss_h_std[i] = eval_loss_h_std[i]
                        min_eval_loss_mid_smpl[i] = eval_loss_mid_smpl[i]
                        min_eval_loss_mid_smpl_std[i] = eval_loss_mid_smpl_std[i]
                        min_eval_loss_ce_avg[i] = eval_loss_ce_avg[i]
                        min_eval_loss_ce_avg_std[i] = eval_loss_ce_avg_std[i]
                        min_eval_loss_err_avg[i] = eval_loss_err_avg[i]
                        min_eval_loss_err_avg_std[i] = eval_loss_err_avg_std[i]
                        min_eval_loss_ce_c_avg[i] = eval_loss_ce_c_avg[i]
                        min_eval_loss_ce_c_avg_std[i] = eval_loss_ce_c_avg_std[i]
                        min_eval_loss_err_c_avg[i] = eval_loss_err_c_avg[i]
                        min_eval_loss_err_c_avg_std[i] = eval_loss_err_c_avg_std[i]
                        min_eval_loss_ce_f_avg[i] = eval_loss_ce_f_avg[i]
                        min_eval_loss_ce_f_avg_std[i] = eval_loss_ce_f_avg_std[i]
                        min_eval_loss_err_f_avg[i] = eval_loss_err_f_avg[i]
                        min_eval_loss_err_f_avg_std[i] = eval_loss_err_f_avg_std[i]
                        min_eval_loss_fro_avg[i] = eval_loss_fro_avg[i]
                        min_eval_loss_fro_avg_std[i] = eval_loss_fro_avg_std[i]
                        min_eval_loss_l1_avg[i] = eval_loss_l1_avg[i]
                        min_eval_loss_l1_avg_std[i] = eval_loss_l1_avg_std[i]
                        min_eval_loss_fro_fb[i] = eval_loss_fro_fb[i]
                        min_eval_loss_fro_fb_std[i] = eval_loss_fro_fb_std[i]
                        min_eval_loss_l1_fb[i] = eval_loss_l1_fb[i]
                        min_eval_loss_l1_fb_std[i] = eval_loss_l1_fb_std[i]
                        for j in range(args.n_bands):
                            min_eval_loss_ce[i][j] = eval_loss_ce[i][j]
                            min_eval_loss_ce_std[i][j] = eval_loss_ce_std[i][j]
                            min_eval_loss_err[i][j] = eval_loss_err[i][j]
                            min_eval_loss_err_std[i][j] = eval_loss_err_std[i][j]
                            min_eval_loss_ce_f[i][j] = eval_loss_ce_f[i][j]
                            min_eval_loss_ce_f_std[i][j] = eval_loss_ce_f_std[i][j]
                            min_eval_loss_err_f[i][j] = eval_loss_err_f[i][j]
                            min_eval_loss_err_f_std[i][j] = eval_loss_err_f_std[i][j]
                            min_eval_loss_fro[i][j] = eval_loss_fro[i][j]
                            min_eval_loss_fro_std[i][j] = eval_loss_fro_std[i][j]
                            min_eval_loss_l1[i][j] = eval_loss_l1[i][j]
                            min_eval_loss_l1_std[i][j] = eval_loss_l1_std[i][j]
                        if i % 2 == 0:
                            min_eval_loss_sc_feat_cv[i//2] = eval_loss_sc_feat_cv[i//2]
                            min_eval_loss_sc_feat_cv_std[i//2] = eval_loss_sc_feat_cv_std[i//2]
                            min_eval_loss_sc_feat_magsp_cv[i//2] = eval_loss_sc_feat_magsp_cv[i//2]
                            min_eval_loss_sc_feat_magsp_cv_std[i//2] = eval_loss_sc_feat_magsp_cv_std[i//2]
                            min_eval_loss_laplace_cv[i//2] = eval_loss_laplace_cv[i//2]
                            min_eval_loss_laplace_cv_std[i//2] = eval_loss_laplace_cv_std[i//2]
                            min_eval_loss_melsp_cv[i//2] = eval_loss_melsp_cv[i//2]
                            min_eval_loss_melsp_cv_std[i//2] = eval_loss_melsp_cv_std[i//2]
                            min_eval_loss_magsp_cv[i//2] = eval_loss_magsp_cv[i//2]
                            min_eval_loss_magsp_cv_std[i//2] = eval_loss_magsp_cv_std[i//2]
                    min_idx = epoch_idx
                    #epoch_min_flag = True
                    change_min_flag = True
            if change_min_flag:
                text_log = "min_eval_loss = %.6f (+- %.6f) ; " % (
                    min_eval_loss_sc_feat_in, min_eval_loss_sc_feat_in_std)
                for i in range(n_half_cyc_eval):
                    if i % 2 == 0:
                        text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                "%.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " \
                                "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) dB ; %.6f %.6f " % (i+1,
                            min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_px[i], min_eval_loss_px_std[i],
                            min_eval_loss_qy_py[i], min_eval_loss_qy_py_std[i], min_eval_loss_qy_py_err[i], min_eval_loss_qy_py_err_std[i], min_eval_loss_qz_pz[i], min_eval_loss_qz_pz_std[i],
                            min_eval_loss_qy_py_e[i], min_eval_loss_qy_py_e_std[i], min_eval_loss_qy_py_err_e[i], min_eval_loss_qy_py_err_e_std[i], min_eval_loss_qz_pz_e[i], min_eval_loss_qz_pz_e_std[i],
                            min_eval_loss_sc_z[i], min_eval_loss_sc_z_std[i], min_eval_loss_sc_feat[i], min_eval_loss_sc_feat_std[i], min_eval_loss_sc_feat_cv[i//2], min_eval_loss_sc_feat_cv_std[i//2],
                            min_eval_loss_sc_feat_magsp[i], min_eval_loss_sc_feat_magsp_std[i], min_eval_loss_sc_feat_magsp_cv[i//2], min_eval_loss_sc_feat_magsp_cv_std[i//2],
                            min_eval_loss_laplace[i], min_eval_loss_laplace_std[i], min_eval_loss_laplace_cv[i//2], min_eval_loss_laplace_cv_std[i//2],
                            min_eval_loss_melsp[i], min_eval_loss_melsp_std[i], min_eval_loss_melsp_cv[i//2], min_eval_loss_melsp_cv_std[i//2], min_eval_loss_melsp_dB[i], min_eval_loss_melsp_dB_std[i],
                            min_eval_loss_magsp[i], min_eval_loss_magsp_std[i], min_eval_loss_magsp_cv[i//2], min_eval_loss_magsp_cv_std[i//2], min_eval_loss_magsp_dB[i], min_eval_loss_magsp_dB_std[i],
                            min_eval_loss_gv_src_src, min_eval_loss_gv_src_trg)
                        if pair_exist:
                            text_log += "%.6f (+- %.6f) dB %.6f (+- %.6f) %.6f (+- %.6f) ; " % (
                                min_eval_loss_melsp_dB_src_trg, min_eval_loss_melsp_dB_src_trg_std,
                                min_eval_loss_lat_dist_rmse, min_eval_loss_lat_dist_rmse_std, min_eval_loss_lat_dist_cossim, min_eval_loss_lat_dist_cossim_std)
                        else:
                            text_log += "n/a (+- n/a) dB n/a (+- n/a) n/a (+- n/a) ; "
                    else:
                        text_log += "[%ld] %.6f (+- %.6f) , %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                "%.6f (+- %.6f) , %.6f (+- %.6f) , %.6f (+- %.6f) ; " \
                                "%.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; " % (i+1,
                            min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_px[i], min_eval_loss_px_std[i],
                            min_eval_loss_lat_cossim[i], min_eval_loss_lat_cossim_std[i], min_eval_loss_lat_rmse[i], min_eval_loss_lat_rmse_std[i],
                            min_eval_loss_qy_py[i], min_eval_loss_qy_py_std[i], min_eval_loss_qy_py_err[i], min_eval_loss_qy_py_err_std[i], min_eval_loss_qz_pz[i], min_eval_loss_qz_pz_std[i],
                            min_eval_loss_qy_py_e[i], min_eval_loss_qy_py_e_std[i], min_eval_loss_qy_py_err_e[i], min_eval_loss_qy_py_err_e_std[i], min_eval_loss_qz_pz_e[i], min_eval_loss_qz_pz_e_std[i],
                            min_eval_loss_sc_z[i], min_eval_loss_sc_z_std[i], min_eval_loss_sc_feat[i], min_eval_loss_sc_feat_std[i], min_eval_loss_sc_feat_magsp[i], min_eval_loss_sc_feat_magsp_std[i],
                            min_eval_loss_laplace[i], min_eval_loss_laplace_std[i], min_eval_loss_melsp[i], min_eval_loss_melsp_std[i], min_eval_loss_melsp_dB[i], min_eval_loss_melsp_dB_std[i],
                            min_eval_loss_magsp[i], min_eval_loss_magsp_std[i], min_eval_loss_magsp_dB[i], min_eval_loss_magsp_dB_std[i])
                    text_log += "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; " % (
                        min_eval_loss_seg_conv[i], min_eval_loss_seg_conv_std[i], min_eval_loss_conv_sc[i], min_eval_loss_conv_sc_std[i],
                        min_eval_loss_h[i], min_eval_loss_h_std[i], min_eval_loss_mid_smpl[i], min_eval_loss_mid_smpl_std[i],
                        min_eval_loss_ce_avg[i], min_eval_loss_ce_avg_std[i], min_eval_loss_err_avg[i], min_eval_loss_err_avg_std[i],
                            min_eval_loss_ce_c_avg[i], min_eval_loss_ce_c_avg_std[i], min_eval_loss_err_c_avg[i], min_eval_loss_err_c_avg_std[i],
                                min_eval_loss_ce_f_avg[i], min_eval_loss_ce_f_avg_std[i], min_eval_loss_err_f_avg[i], min_eval_loss_err_f_avg_std[i],
                                    min_eval_loss_fro_avg[i], min_eval_loss_fro_avg_std[i], min_eval_loss_l1_avg[i], min_eval_loss_l1_avg_std[i],
                                        min_eval_loss_fro_fb[i], min_eval_loss_fro_fb_std[i], min_eval_loss_l1_fb[i], min_eval_loss_l1_fb_std[i])
                    for j in range(args.n_bands):
                        text_log += "[%d-%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% , %.6f (+- %.6f) %.6f (+- %.6f) " % (i+1, j+1,
                            min_eval_loss_ce[i][j], min_eval_loss_ce_std[i][j], min_eval_loss_err[i][j], min_eval_loss_err_std[i][j],
                                min_eval_loss_ce_f[i][j], min_eval_loss_ce_f_std[i][j], min_eval_loss_err_f[i][j], min_eval_loss_err_f_std[i][j],
                                    min_eval_loss_fro[i][j], min_eval_loss_fro_std[i][j], min_eval_loss_l1[i][j], min_eval_loss_l1_std[i][j])
                    text_log += ";; "
                logging.info("%s min_idx=%d" % (text_log, min_idx+1))
            #if ((epoch_idx + 1) % args.save_interval_epoch == 0) or (epoch_min_flag):
            if True:
                logging.info('save epoch:%d' % (epoch_idx+1))
                if model_waveform.use_weight_norm:
                    torch.nn.utils.remove_weight_norm(model_waveform.scale_in)
                save_checkpoint(args.expdir, model_encoder_melsp_fix, model_encoder_melsp, model_decoder_melsp,
                    model_encoder_excit_fix, model_encoder_excit, model_spk, model_classifier,
                    model_waveform, min_eval_loss_melsp_dB[0], min_eval_loss_melsp_dB_std[0], min_eval_loss_melsp_cv[0],
                    min_eval_loss_melsp[0], min_eval_loss_laplace_cv[0], min_eval_loss_laplace[0],
                    min_eval_loss_melsp_dB_src_trg, min_eval_loss_melsp_dB_src_trg_std, min_eval_loss_gv_src_trg,
                    min_eval_loss_ce_avg[0], min_eval_loss_ce_avg_std[0], min_eval_loss_err_avg[0], min_eval_loss_err_avg_std[0],
                    min_eval_loss_l1_avg[0], min_eval_loss_l1_fb[0], err_flag,
                    iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1,
                    model_spkidtr=model_spkidtr)
                if model_waveform.use_weight_norm:
                    torch.nn.utils.weight_norm(model_waveform.scale_in)
                for param in model_waveform.scale_in.parameters():
                    param.requires_grad = False
            total = 0
            iter_count = 0
            loss_sc_feat_in = []
            loss_sc_feat_magsp_in = []
            for i in range(args.n_half_cyc):
                loss_elbo[i] = []
                loss_px[i] = []
                loss_qy_py[i] = []
                loss_qy_py_err[i] = []
                loss_qz_pz[i] = []
                loss_qy_py_e[i] = []
                loss_qy_py_err_e[i] = []
                loss_qz_pz_e[i] = []
                loss_lat_cossim[i] = []
                loss_lat_rmse[i] = []
                loss_sc_z[i] = []
                loss_sc_feat[i] = []
                loss_sc_feat_cv[i] = []
                loss_sc_feat_magsp[i] = []
                loss_sc_feat_magsp_cv[i] = []
                loss_laplace[i] = []
                loss_melsp[i] = []
                loss_laplace_cv[i] = []
                loss_melsp_cv[i] = []
                loss_melsp_dB[i] = []
                loss_magsp[i] = []
                loss_magsp_cv[i] = []
                loss_magsp_dB[i] = []
                loss_ce_avg[i] = []
                loss_err_avg[i] = []
                loss_ce_c_avg[i] = []
                loss_err_c_avg[i] = []
                loss_ce_f_avg[i] = []
                loss_err_f_avg[i] = []
                loss_fro_avg[i] = []
                loss_l1_avg[i] = []
                loss_fro_fb[i] = []
                loss_l1_fb[i] = []
                loss_seg_conv[i] = []
                loss_conv_sc[i] = []
                loss_h[i] = []
                loss_mid_smpl[i] = []
                for j in range(args.n_bands):
                    loss_ce[i][j] = []
                    loss_err[i][j] = []
                    loss_ce_f[i][j] = []
                    loss_err_f[i][j] = []
                    loss_fro[i][j] = []
                    loss_l1[i][j] = []
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model_encoder_melsp_fix.train()
            model_encoder_melsp.train()
            model_decoder_melsp.train()
            model_encoder_excit_fix.train()
            model_encoder_excit.train()
            model_classifier.train()
            model_spk.train()
            if args.spkidtr_dim > 0:
                model_spkidtr.train()
            model_waveform.train()
            for param in model_encoder_melsp.parameters():
                param.requires_grad = True
            for param in model_encoder_melsp.scale_in.parameters():
                param.requires_grad = False
            for param in model_decoder_melsp.parameters():
                param.requires_grad = True
            for param in model_decoder_melsp.scale_out.parameters():
                param.requires_grad = False
            for param in model_encoder_excit.parameters():
                param.requires_grad = True
            for param in model_encoder_excit.scale_in.parameters():
                param.requires_grad = False
            for param in model_classifier.parameters():
                param.requires_grad = True
            for param in model_spk.parameters():
                param.requires_grad = True
            if args.spkidtr_dim > 0:
                for param in model_spkidtr.parameters():
                    param.requires_grad = True
            # start next epoch
            if iter_idx < args.step_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_x_fb, batch_x, batch_x_c, batch_x_f, batch_feat, batch_feat_magsp, batch_sc, batch_sc_cv_data, c_idx, utt_idx, featfile, \
                    x_bs, x_ss, f_bs, f_ss, slens, flens, n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, idx_select, idx_select_full, slens_acc, flens_acc = next(generator)
            else:
                break
        # feedforward and backpropagate current batch
        logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

        x_es = x_ss+x_bs
        f_es = f_ss+f_bs
        logging.info(f'{x_ss*args.n_bands} {x_bs*args.n_bands} {x_es*args.n_bands} {x_ss} {x_bs} {x_es} {f_ss} {f_bs} {f_es} {max_slen*args.n_bands} {max_slen} {max_flen}')

        # handle waveformb batch padding
        f_ss_pad_left = f_ss-wav_pad_left
        if f_es <= max_flen:
            f_es_pad_right = f_es+wav_pad_right
        else:
            f_es_pad_right = max_flen+wav_pad_right
        if f_ss_pad_left >= 0 and f_es_pad_right <= max_flen: # pad left and right available
            batch_feat_org_in = batch_feat[:,f_ss_pad_left:f_es_pad_right]
        elif f_es_pad_right <= max_flen: # pad right available, left need additional replicate
            batch_feat_org_in = F.pad(batch_feat[:,:f_es_pad_right].transpose(1,2), (-f_ss_pad_left,0), "replicate").transpose(1,2)
        elif f_ss_pad_left >= 0: # pad left available, right need additional replicate
            batch_feat_org_in = F.pad(batch_feat[:,f_ss_pad_left:max_flen].transpose(1,2), (0,f_es_pad_right-max_flen), "replicate").transpose(1,2)
        else: # pad left and right need additional replicate
            batch_feat_org_in = F.pad(batch_feat[:,:max_flen].transpose(1,2), (-f_ss_pad_left,f_es_pad_right-max_flen), "replicate").transpose(1,2)
        if x_ss > 0:
            if x_es <= max_slen:
                batch_x_c_prev = batch_x_c[:,x_ss-1:x_es-1]
                batch_x_f_prev = batch_x_f[:,x_ss-1:x_es-1]
                if args.lpc > 0:
                    if x_ss-args.lpc >= 0:
                        batch_x_c_lpc = batch_x_c[:,x_ss-args.lpc:x_es-1]
                        batch_x_f_lpc = batch_x_f[:,x_ss-args.lpc:x_es-1]
                    else:
                        batch_x_c_lpc = F.pad(batch_x_c[:,:x_es-1], (0, 0, -(x_ss-args.lpc), 0), "constant", args.c_pad)
                        batch_x_f_lpc = F.pad(batch_x_f[:,:x_es-1], (0, 0, -(x_ss-args.lpc), 0), "constant", args.f_pad)
                batch_x_c = batch_x_c[:,x_ss:x_es]
                batch_x_f = batch_x_f[:,x_ss:x_es]
                batch_x = batch_x[:,x_ss:x_es]
                batch_x_fb = batch_x_fb[:,x_ss*args.n_bands:x_es*args.n_bands]
            else:
                batch_x_c_prev = batch_x_c[:,x_ss-1:-1]
                batch_x_f_prev = batch_x_f[:,x_ss-1:-1]
                if args.lpc > 0:
                    if x_ss-args.lpc >= 0:
                        batch_x_c_lpc = batch_x_c[:,x_ss-args.lpc:-1]
                        batch_x_f_lpc = batch_x_f[:,x_ss-args.lpc:-1]
                    else:
                        batch_x_c_lpc = F.pad(batch_x_c[:,:-1], (0, 0, -(x_ss-args.lpc), 0), "constant", args.c_pad)
                        batch_x_f_lpc = F.pad(batch_x_f[:,:-1], (0, 0, -(x_ss-args.lpc), 0), "constant", args.f_pad)
                batch_x_c = batch_x_c[:,x_ss:]
                batch_x_f = batch_x_f[:,x_ss:]
                batch_x = batch_x[:,x_ss:]
                batch_x_fb = batch_x_fb[:,x_ss*args.n_bands:]
        else:
            batch_x_c_prev = F.pad(batch_x_c[:,:x_es-1], (0, 0, 1, 0), "constant", args.c_pad)
            batch_x_f_prev = F.pad(batch_x_f[:,:x_es-1], (0, 0, 1, 0), "constant", args.f_pad)
            if args.lpc > 0:
                batch_x_c_lpc = F.pad(batch_x_c[:,:x_es-1], (0, 0, args.lpc, 0), "constant", args.c_pad)
                batch_x_f_lpc = F.pad(batch_x_f[:,:x_es-1], (0, 0, args.lpc, 0), "constant", args.f_pad)
            batch_x_c = batch_x_c[:,:x_es]
            batch_x_f = batch_x_f[:,:x_es]
            batch_x = batch_x[:,:x_es]
            batch_x_fb = batch_x_fb[:,:x_es*args.n_bands]

        # handle first pad for input on melsp flow
        flag_cv = True
        i_cv = 0
        i_cv_in = 0
        f_ss_first_pad_left = f_ss-first_pad_left
        f_es_first_pad_right = f_es+first_pad_right
        i_end = args.n_half_cyc*4
        for i in range(i_end):
            if i % 4 == 0: #enc
                if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                    batch_feat_in[i] = batch_feat[:,f_ss_first_pad_left:f_es_first_pad_right]
                elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                    batch_feat_in[i] = F.pad(batch_feat[:,:f_es_first_pad_right].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
                elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                    batch_feat_in[i] = F.pad(batch_feat[:,f_ss_first_pad_left:max_flen].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                else: # pad left and right need additional replicate
                    batch_feat_in[i] = F.pad(batch_feat[:,:max_flen].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                f_ss_first_pad_left += enc_pad_left
                f_es_first_pad_right -= enc_pad_right
            else: #spk/dec/wav
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
                    f_ss_first_pad_left += spk_pad_left
                    f_es_first_pad_right -= spk_pad_right
                elif i % 4 == 2:
                    f_ss_first_pad_left += dec_pad_left
                    f_es_first_pad_right -= dec_pad_right
                elif i % 4 == 3:
                    f_ss_first_pad_left += wav_pad_left
                    f_es_first_pad_right -= wav_pad_right
        batch_melsp = batch_feat[:,f_ss:f_es]
        batch_magsp = batch_feat_magsp[:,f_ss:f_es]
        batch_sc = batch_sc[:,f_ss:f_es]
        for i in range(n_cv):
            batch_sc_cv[i] = batch_sc_cv_data[i][:,f_ss:f_es]

        if f_ss > 0:
            idx_in = 0
            i_cv_in = 0
            for i in range(0,args.n_half_cyc,2):
                i_cv = i//2
                j = i+1
                if len(del_index_utt) > 0:
                    h_feat_in_sc = torch.FloatTensor(np.delete(h_feat_in_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_feat_magsp_in_sc = torch.FloatTensor(np.delete(h_feat_magsp_in_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_x_org = torch.FloatTensor(np.delete(h_x_org.cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                    h_x_2_org = torch.FloatTensor(np.delete(h_x_2_org.cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                    h_f_org = torch.FloatTensor(np.delete(h_f_org.cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                    h_z[i] = torch.FloatTensor(np.delete(h_z[i].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_z_e[i] = torch.FloatTensor(np.delete(h_z_e[i].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_z_fix = torch.FloatTensor(np.delete(h_z_fix.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_z_e_fix = torch.FloatTensor(np.delete(h_z_e_fix.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_spk[i] = torch.FloatTensor(np.delete(h_spk[i].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_spk_cv[i_cv] = torch.FloatTensor(np.delete(h_spk_cv[i_cv].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_melsp[i] = torch.FloatTensor(np.delete(h_melsp[i].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_melsp_cv[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv[i_cv].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_z_sc[i] = torch.FloatTensor(np.delete(h_z_sc[i].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_feat_sc[i] = torch.FloatTensor(np.delete(h_feat_sc[i].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_feat_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_feat_cv_sc[i_cv].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_feat_magsp_sc[i] = torch.FloatTensor(np.delete(h_feat_magsp_sc[i].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_feat_magsp_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_feat_magsp_cv_sc[i_cv].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_x[i] = torch.FloatTensor(np.delete(h_x[i].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                    h_x_2[i] = torch.FloatTensor(np.delete(h_x_2[i].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                    h_f[i] = torch.FloatTensor(np.delete(h_f[i].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                    h_z[j] = torch.FloatTensor(np.delete(h_z[j].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_z_e[j] = torch.FloatTensor(np.delete(h_z_e[j].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_spk[j] = torch.FloatTensor(np.delete(h_spk[j].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_melsp[j] = torch.FloatTensor(np.delete(h_melsp[j].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_z_sc[j] = torch.FloatTensor(np.delete(h_z_sc[j].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_feat_sc[j] = torch.FloatTensor(np.delete(h_feat_sc[j].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_feat_magsp_sc[j] = torch.FloatTensor(np.delete(h_feat_magsp_sc[j].cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_x[j] = torch.FloatTensor(np.delete(h_x[j].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                    h_x_2[j] = torch.FloatTensor(np.delete(h_x_2[j].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                    h_f[j] = torch.FloatTensor(np.delete(h_f[j].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z[i])
                qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_e[i])
                _, qz_alpha_fix, z_fix, h_z_fix = model_encoder_melsp_fix(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_fix)
                _, qz_alpha_e_fix, z_e_fix, h_z_e_fix = model_encoder_excit_fix(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_e_fix)
                batch_feat_in_sc, h_feat_in_sc = model_classifier(feat=batch_melsp, h=h_feat_in_sc)
                batch_feat_magsp_in_sc, h_feat_magsp_in_sc = model_classifier(feat_aux=batch_magsp, h=h_feat_magsp_in_sc)
                seg_conv, conv_sc, out, out_2, out_f, signs_c, scales_c, logits_c, signs_f, scales_f, logits_f, x_c_output, x_f_output, h_x_org, h_x_2_org, h_f_org \
                    = model_waveform.gen_mid_feat_smpl(batch_feat_org_in, batch_x_c_prev, batch_x_f_prev, batch_x_c, h=h_x_org, h_2=h_x_2_org, h_f=h_f_org, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc)
                ## time-varying speaker conditionings
                z_cat = torch.cat((z_e[i], z[i]), 2)
                feat_len = qy_logits[i].shape[1]
                z[i] = z[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                z_e[i] = z_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                batch_z_sc[i], h_z_sc[i] = model_classifier(lat=torch.cat((z[i], z_e[i]), 2), h=h_z_sc[i])
                qy_logits[i] = qy_logits[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha[i] = qz_alpha[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qy_logits_e[i] = qy_logits_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha_e[i] = qz_alpha_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha_fix = qz_alpha_fix[:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha_e_fix = qz_alpha_e_fix[:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                idx_in += 1
                if args.spkidtr_dim > 0:
                    spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                    spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                    batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i])
                    batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv])
                else:
                    batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i])
                    batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv])
                ## melsp reconstruction & conversion
                idx_in += 1
                i_cv_in += 1
                if spk_pad_right > 0:
                    z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                    if args.spkidtr_dim > 0:
                        spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                        spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:-spk_pad_right]
                else:
                    z_cat = z_cat[:,spk_pad_left:]
                    if args.spkidtr_dim > 0:
                        spk_code_in = spk_code_in[:,spk_pad_left:]
                        spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:]
                if args.spkidtr_dim > 0:
                    batch_pdf_rec[i], batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                        outpad_right=outpad_rights[idx_in], h=h_melsp[i])
                    batch_pdf_cv[i_cv], batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                        outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv])
                else:
                    batch_pdf_rec[i], batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                        outpad_right=outpad_rights[idx_in], h=h_melsp[i])
                    batch_pdf_cv[i_cv], batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                        outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv])
                ## waveform reconstruction
                idx_in += 1
                batch_x_c_output_noclamp[i], batch_x_f_output_noclamp[i], batch_seg_conv[i], batch_conv_sc[i], \
                    batch_out[i], batch_out_2[i], batch_out_f[i], batch_signs_c[i], batch_scales_c[i], batch_logits_c[i], batch_signs_f[i], batch_scales_f[i], batch_logits_f[i], h_x[i], h_x_2[i], h_f[i] \
                        = model_waveform(batch_melsp_rec[i], batch_x_c_prev, batch_x_f_prev, batch_x_c, h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in],
                                x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc, ret_mid_feat=True, ret_mid_smpl=True)
                batch_x_c_output[i] = torch.clamp(batch_x_c_output_noclamp[i], min=MIN_CLAMP, max=MAX_CLAMP)
                batch_x_f_output[i] = torch.clamp(batch_x_f_output_noclamp[i], min=MIN_CLAMP, max=MAX_CLAMP)
                u = torch.empty_like(batch_x_c_output[i])
                logits_gumbel = F.softmax(batch_x_c_output[i] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                sample_indices_c = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                logits_gumbel = F.softmax(batch_x_f_output[i] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                sample_indices_f = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                batch_x_output[i] = decode_mu_law_torch(sample_indices_c*args.cf_dim+sample_indices_f)
                batch_x_output_fb[i] = pqmf.synthesis(batch_x_output[i].transpose(1,2))[:,0]
                if wav_pad_right > 0:
                    cv_feat = batch_melsp_cv[i_cv][:,wav_pad_left:-wav_pad_right].detach()
                else:
                    cv_feat = batch_melsp_cv[i_cv][:,wav_pad_left:].detach()
                idx_in_1 = idx_in-1
                feat_len = batch_melsp_rec[i].shape[1]
                batch_pdf_rec[i] = batch_pdf_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_melsp_rec[i] = batch_melsp_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_pdf_cv[i_cv] = batch_pdf_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_melsp_cv[i_cv] = batch_melsp_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_magsp_rec[i] = torch.matmul((torch.exp(batch_melsp_rec[i])-1)/10000, melfb_t)
                batch_magsp_cv[i_cv] = torch.matmul((torch.exp(batch_melsp_cv[i_cv])-1)/10000, melfb_t)
                batch_feat_rec_sc[i], h_feat_sc[i] = model_classifier(feat=batch_melsp_rec[i], h=h_feat_sc[i])
                batch_feat_cv_sc[i_cv], h_feat_cv_sc[i_cv] = model_classifier(feat=batch_melsp_cv[i_cv], h=h_feat_cv_sc[i_cv])
                batch_feat_magsp_rec_sc[i], h_feat_magsp_sc[i] = model_classifier(feat_aux=batch_magsp_rec[i], h=h_feat_magsp_sc[i])
                batch_feat_magsp_cv_sc[i_cv], h_feat_magsp_cv_sc[i_cv] = model_classifier(feat_aux=batch_magsp_cv[i_cv], h=h_feat_magsp_cv_sc[i_cv])
                ## cyclic reconstruction
                idx_in += 1
                qy_logits[j], qz_alpha[j], z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z[j])
                qy_logits_e[j], qz_alpha_e[j], z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[j])
                ## time-varying speaker conditionings
                z_cat = torch.cat((z_e[j], z[j]), 2)
                feat_len = qy_logits[j].shape[1]
                z[j] = z[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                z_e[j] = z_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                batch_z_sc[j], h_z_sc[j] = model_classifier(lat=torch.cat((z[j], z_e[j]), 2), h=h_z_sc[j])
                qy_logits[j] = qy_logits[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha[j] = qz_alpha[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qy_logits_e[j] = qy_logits_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha_e[j] = qz_alpha_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                idx_in += 1
                if args.spkidtr_dim > 0:
                    if dec_enc_pad_right > 0:
                        spk_code_in = spk_code_in[:,dec_enc_pad_left:-dec_enc_pad_right]
                    else:
                        spk_code_in = spk_code_in[:,dec_enc_pad_left:]
                    batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j])
                else:
                    batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j])
                ## melsp reconstruction
                idx_in += 1
                if spk_pad_right > 0:
                    z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                    if args.spkidtr_dim > 0:
                        spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                else:
                    z_cat = z_cat[:,spk_pad_left:]
                    if args.spkidtr_dim > 0:
                        spk_code_in = spk_code_in[:,spk_pad_left:]
                if args.spkidtr_dim > 0:
                    batch_pdf_rec[j], batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                            outpad_right=outpad_rights[idx_in], h=h_melsp[j])
                else:
                    batch_pdf_rec[j], batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                            outpad_right=outpad_rights[idx_in], h=h_melsp[j])
                ## waveform reconstruction
                idx_in += 1
                batch_x_c_output_noclamp[j], batch_x_f_output_noclamp[j], batch_seg_conv[j], batch_conv_sc[j], \
                    batch_out[j], batch_out_2[j], batch_out_f[j], batch_signs_c[j], batch_scales_c[j], batch_logits_c[j], batch_signs_f[j], batch_scales_f[j], batch_logits_f[j], h_x[j], h_x_2[j], h_f[j] \
                        = model_waveform(batch_melsp_rec[j], batch_x_c_prev, batch_x_f_prev, batch_x_c, h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in],
                                x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc, ret_mid_feat=True, ret_mid_smpl=True)
                batch_x_c_output[j] = torch.clamp(batch_x_c_output_noclamp[j], min=MIN_CLAMP, max=MAX_CLAMP)
                batch_x_f_output[j] = torch.clamp(batch_x_f_output_noclamp[j], min=MIN_CLAMP, max=MAX_CLAMP)
                logits_gumbel = F.softmax(batch_x_c_output[j] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                sample_indices_c = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                logits_gumbel = F.softmax(batch_x_f_output[j] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                sample_indices_f = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                batch_x_output[j] = decode_mu_law_torch(sample_indices_c*args.cf_dim+sample_indices_f)
                batch_x_output_fb[j] = pqmf.synthesis(batch_x_output[j].transpose(1,2))[:,0]
                idx_in_1 = idx_in-1
                feat_len = batch_melsp_rec[j].shape[1]
                batch_pdf_rec[j] = batch_pdf_rec[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_melsp_rec[j] = batch_melsp_rec[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_magsp_rec[j] = torch.matmul((torch.exp(batch_melsp_rec[j])-1)/10000, melfb_t)
                batch_feat_rec_sc[j], h_feat_sc[j] = model_classifier(feat=batch_melsp_rec[j], h=h_feat_sc[j])
                batch_feat_magsp_rec_sc[j], h_feat_magsp_sc[j] = model_classifier(feat_aux=batch_magsp_rec[j], h=h_feat_magsp_sc[j])
        else:
            idx_in = 0
            i_cv_in = 0
            for i in range(0,args.n_half_cyc,2):
                i_cv = i//2
                j = i+1
                qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in])
                qy_logits_e[i], qz_alpha_e[i], z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in])
                _, qz_alpha_fix, z_fix, h_z_fix = model_encoder_melsp_fix(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in])
                _, qz_alpha_e_fix, z_e_fix, h_z_e_fix = model_encoder_excit_fix(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in])
                batch_feat_in_sc, h_feat_in_sc = model_classifier(feat=batch_melsp)
                batch_feat_magsp_in_sc, h_feat_magsp_in_sc = model_classifier(feat_aux=batch_magsp)
                seg_conv, conv_sc, out, out_2, out_f, signs_c, scales_c, logits_c, signs_f, scales_f, logits_f, x_c_output, x_f_output, h_x_org, h_x_2_org, h_f_org \
                    = model_waveform.gen_mid_feat_smpl(batch_feat_org_in, batch_x_c_prev, batch_x_f_prev, batch_x_c, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc)
                ## time-varying speaker conditionings
                z_cat = torch.cat((z_e[i], z[i]), 2)
                feat_len = qy_logits[i].shape[1]
                z[i] = z[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                z_e[i] = z_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                batch_z_sc[i], h_z_sc[i] = model_classifier(lat=torch.cat((z[i], z_e[i]), 2))
                qy_logits[i] = qy_logits[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha[i] = qz_alpha[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qy_logits_e[i] = qy_logits_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha_e[i] = qz_alpha_e[i][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha_fix = qz_alpha_fix[:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha_e_fix = qz_alpha_e_fix[:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                idx_in += 1
                if args.spkidtr_dim > 0:
                    spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                    spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                    batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                    batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                else:
                    batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                    batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                ## melsp reconstruction & conversion
                idx_in += 1
                i_cv_in += 1
                if spk_pad_right > 0:
                    z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                    if args.spkidtr_dim > 0:
                        spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                        spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:-spk_pad_right]
                else:
                    z_cat = z_cat[:,spk_pad_left:]
                    if args.spkidtr_dim > 0:
                        spk_code_in = spk_code_in[:,spk_pad_left:]
                        spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:]
                if args.spkidtr_dim > 0:
                    batch_pdf_rec[i], batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                        outpad_right=outpad_rights[idx_in])
                    batch_pdf_cv[i_cv], batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                        outpad_right=outpad_rights[idx_in])
                else:
                    batch_pdf_rec[i], batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                        outpad_right=outpad_rights[idx_in])
                    batch_pdf_cv[i_cv], batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                        outpad_right=outpad_rights[idx_in])
                ## waveform reconstruction
                idx_in += 1
                batch_x_c_output_noclamp[i], batch_x_f_output_noclamp[i], batch_seg_conv[i], batch_conv_sc[i], \
                    batch_out[i], batch_out_2[i], batch_out_f[i], batch_signs_c[i], batch_scales_c[i], batch_logits_c[i], batch_signs_f[i], batch_scales_f[i], batch_logits_f[i], h_x[i], h_x_2[i], h_f[i] \
                        = model_waveform(batch_melsp_rec[i], batch_x_c_prev, batch_x_f_prev, batch_x_c, outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in],
                                x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc, ret_mid_feat=True, ret_mid_smpl=True)
                batch_x_c_output[i] = torch.clamp(batch_x_c_output_noclamp[i], min=MIN_CLAMP, max=MAX_CLAMP)
                batch_x_f_output[i] = torch.clamp(batch_x_f_output_noclamp[i], min=MIN_CLAMP, max=MAX_CLAMP)
                u = torch.empty_like(batch_x_c_output[i])
                logits_gumbel = F.softmax(batch_x_c_output[i] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                sample_indices_c = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                logits_gumbel = F.softmax(batch_x_f_output[i] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                sample_indices_f = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                batch_x_output[i] = decode_mu_law_torch(sample_indices_c*args.cf_dim+sample_indices_f)
                batch_x_output_fb[i] = pqmf.synthesis(batch_x_output[i].transpose(1,2))[:,0]
                if wav_pad_right > 0:
                    cv_feat = batch_melsp_cv[i_cv][:,wav_pad_left:-wav_pad_right].detach()
                else:
                    cv_feat = batch_melsp_cv[i_cv][:,wav_pad_left:].detach()
                idx_in_1 = idx_in-1
                feat_len = batch_melsp_rec[i].shape[1]
                batch_pdf_rec[i] = batch_pdf_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_melsp_rec[i] = batch_melsp_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_pdf_cv[i_cv] = batch_pdf_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_melsp_cv[i_cv] = batch_melsp_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_magsp_rec[i] = torch.matmul((torch.exp(batch_melsp_rec[i])-1)/10000, melfb_t)
                batch_magsp_cv[i_cv] = torch.matmul((torch.exp(batch_melsp_cv[i_cv])-1)/10000, melfb_t)
                batch_feat_rec_sc[i], h_feat_sc[i] = model_classifier(feat=batch_melsp_rec[i])
                batch_feat_cv_sc[i_cv], h_feat_cv_sc[i_cv] = model_classifier(feat=batch_melsp_cv[i_cv])
                batch_feat_magsp_rec_sc[i], h_feat_magsp_sc[i] = model_classifier(feat_aux=batch_magsp_rec[i])
                batch_feat_magsp_cv_sc[i_cv], h_feat_magsp_cv_sc[i_cv] = model_classifier(feat_aux=batch_magsp_cv[i_cv])
                ## cyclic reconstruction
                idx_in += 1
                qy_logits[j], qz_alpha[j], z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in])
                qy_logits_e[j], qz_alpha_e[j], z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in])
                ## time-varying speaker conditionings
                z_cat = torch.cat((z_e[j], z[j]), 2)
                feat_len = qy_logits[j].shape[1]
                z[j] = z[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                z_e[j] = z_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                batch_z_sc[j], h_z_sc[j] = model_classifier(lat=torch.cat((z[j], z_e[j]), 2))
                qy_logits[j] = qy_logits[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha[j] = qz_alpha[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qy_logits_e[j] = qy_logits_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                qz_alpha_e[j] = qz_alpha_e[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                idx_in += 1
                if args.spkidtr_dim > 0:
                    if dec_enc_pad_right > 0:
                        spk_code_in = spk_code_in[:,dec_enc_pad_left:-dec_enc_pad_right]
                    else:
                        spk_code_in = spk_code_in[:,dec_enc_pad_left:]
                    batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                else:
                    batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                ## melsp reconstruction
                idx_in += 1
                if spk_pad_right > 0:
                    z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                    if args.spkidtr_dim > 0:
                        spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                else:
                    z_cat = z_cat[:,spk_pad_left:]
                    if args.spkidtr_dim > 0:
                        spk_code_in = spk_code_in[:,spk_pad_left:]
                if args.spkidtr_dim > 0:
                    batch_pdf_rec[j], batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                            outpad_right=outpad_rights[idx_in])
                else:
                    batch_pdf_rec[j], batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                            outpad_right=outpad_rights[idx_in])
                ## waveform reconstruction
                idx_in += 1
                batch_x_c_output_noclamp[j], batch_x_f_output_noclamp[j], batch_seg_conv[j], batch_conv_sc[j], \
                    batch_out[j], batch_out_2[j], batch_out_f[j], batch_signs_c[j], batch_scales_c[j], batch_logits_c[j], batch_signs_f[j], batch_scales_f[j], batch_logits_f[j], h_x[j], h_x_2[j], h_f[j] \
                        = model_waveform(batch_melsp_rec[j], batch_x_c_prev, batch_x_f_prev, batch_x_c, outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in],
                                x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc, ret_mid_feat=True, ret_mid_smpl=True)
                batch_x_c_output[j] = torch.clamp(batch_x_c_output_noclamp[j], min=MIN_CLAMP, max=MAX_CLAMP)
                batch_x_f_output[j] = torch.clamp(batch_x_f_output_noclamp[j], min=MIN_CLAMP, max=MAX_CLAMP)
                logits_gumbel = F.softmax(batch_x_c_output[j] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                sample_indices_c = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                logits_gumbel = F.softmax(batch_x_f_output[j] - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1))), dim=-1)
                logits_gumbel_norm_1hot = F.threshold(logits_gumbel / torch.max(logits_gumbel,-1,keepdim=True)[0], eps_1, 0)
                sample_indices_f = torch.sum(logits_gumbel_norm_1hot*indices_1hot,-1)
                batch_x_output[j] = decode_mu_law_torch(sample_indices_c*args.cf_dim+sample_indices_f)
                batch_x_output_fb[j] = pqmf.synthesis(batch_x_output[j].transpose(1,2))[:,0]
                idx_in_1 = idx_in-1
                feat_len = batch_melsp_rec[j].shape[1]
                batch_pdf_rec[j] = batch_pdf_rec[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_melsp_rec[j] = batch_melsp_rec[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                batch_magsp_rec[j] = torch.matmul((torch.exp(batch_melsp_rec[j])-1)/10000, melfb_t)
                batch_feat_rec_sc[j], h_feat_sc[j] = model_classifier(feat=batch_melsp_rec[j])
                batch_feat_magsp_rec_sc[j], h_feat_magsp_sc[j] = model_classifier(feat_aux=batch_magsp_rec[j])

        # Losses computation
        batch_loss = 0

        # handle short ending
        if len(idx_select) > 0:
            len_idx_select = len(idx_select)
            logging.info('len_idx_select: '+str(len_idx_select))
            batch_loss_sc_feat_kl_select = 0
            batch_loss_sc_z_kl_select = 0
            batch_loss_qz_pz_kl_select = 0
            batch_loss_qy_py_ce_select = 0
            for i in range(args.n_half_cyc):
                batch_loss_laplace[i] = 0
                batch_loss_melsp[i] = 0
                batch_loss_magsp[i] = 0
                batch_loss_melsp_dB[i] = 0
                batch_loss_magsp_dB[i] = 0
                batch_loss_seg_conv[i] = 0
                batch_loss_conv_sc[i] = 0
                batch_loss_h[i] = 0
                batch_loss_mid_smpl[i] = 0
                batch_loss_ce_select[i] = 0
                batch_loss_ce_f_select[i] = 0
                batch_loss_err_select[i] = 0
                batch_loss_err_f_select[i] = 0
                batch_loss_fro_select[i] = 0
                batch_loss_l1_select[i] = 0
                batch_loss_fro_fb[i] = 0
                batch_loss_l1_fb[i] = 0
                if i % 2 == 0:
                    batch_loss_laplace_cv[i//2] = 0
                    batch_loss_melsp_cv[i//2] = 0
                    batch_loss_magsp_cv[i//2] = 0
            for j in range(len(idx_select)):
                k = idx_select[j]
                slens_utt = slens_acc[k]
                slens_utt_fb = slens_utt*args.n_bands
                flens_utt = flens_acc[k]
                logging.info('%s %d %d %d' % (featfile[k], slens_utt, slens_utt_fb, flens_utt))
                batch_x_c_ = batch_x_c[k,:slens_utt]
                batch_x_f_ = batch_x_f[k,:slens_utt]
                batch_x_ = batch_x[k,:slens_utt].transpose(1,0)
                batch_x_fb_ = batch_x_fb[k,:slens_utt_fb]
                seg_conv_ = seg_conv[k,:flens_utt]
                conv_sc_ = conv_sc[k,:flens_utt]
                scales_c_ = scales_c[k,:slens_utt]
                signs_c_ = signs_c[k,:slens_utt]
                logits_c_ = logits_c[k,:slens_utt]
                scales_f_ = scales_f[k,:slens_utt]
                signs_f_ = signs_f[k,:slens_utt]
                logits_f_ = logits_f[k,:slens_utt]
                x_c_output_ = x_c_output[k,:slens_utt]
                x_f_output_ = x_f_output[k,:slens_utt]
                out_ = out[k,:slens_utt]
                out_2_ = out_2[k,:slens_utt]
                out_f_ = out_f[k,:slens_utt]
                melsp = batch_melsp[k,:flens_utt]
                melsp_rest = (torch.exp(melsp)-1)/10000
                magsp = magsp_rest = batch_magsp[k,:flens_utt]
                qz_alpha_fix_ = qz_alpha_fix[k,:flens_utt]
                qz_alpha_e_fix_ = qz_alpha_e_fix[k,:flens_utt]
                melsp_rest_log = torch.log10(torch.clamp(melsp_rest, min=1e-16))
                magsp_rest_log = torch.log10(torch.clamp(magsp_rest, min=1e-16))

                batch_sc_ = batch_sc[k,:flens_utt]
                sc_onehot_ = F.one_hot(batch_sc_, num_classes=n_spk).float()
                batch_loss_sc_feat_kl_select += torch.mean(criterion_ce(batch_feat_in_sc[k,:flens_utt], batch_sc_)) \
                                                + torch.mean(criterion_ce(batch_feat_magsp_in_sc[k,:flens_utt], batch_sc_))
                for i in range(args.n_half_cyc):
                    batch_x_c_output_ = batch_x_c_output[i][k,:slens_utt]
                    batch_x_f_output_ = batch_x_f_output[i][k,:slens_utt]
                    batch_seg_conv_ = batch_seg_conv[i][k,:flens_utt]
                    batch_conv_sc_ = batch_conv_sc[i][k,:flens_utt]
                    batch_scales_c_ = batch_scales_c[i][k,:slens_utt]
                    batch_signs_c_ = batch_signs_c[i][k,:slens_utt]
                    batch_logits_c_ = batch_logits_c[i][k,:slens_utt]
                    batch_scales_f_ = batch_scales_f[i][k,:slens_utt]
                    batch_signs_f_ = batch_signs_f[i][k,:slens_utt]
                    batch_logits_f_ = batch_logits_f[i][k,:slens_utt]
                    batch_x_c_output_noclamp_ = batch_x_c_output_noclamp[i][k,:slens_utt]
                    batch_x_f_output_noclamp_ = batch_x_f_output_noclamp[i][k,:slens_utt]
                    batch_out_ = batch_out[i][k,:slens_utt]
                    batch_out_2_ = batch_out_2[i][k,:slens_utt]
                    batch_out_f_ = batch_out_f[i][k,:slens_utt]

                    qy_logits_select_ = qy_logits[i][k,:flens_utt]
                    qy_logits_e_select_ = qy_logits_e[i][k,:flens_utt]

                    pdf = batch_pdf_rec[i][k,:flens_utt]
                    melsp_est = batch_melsp_rec[i][k,:flens_utt]
                    melsp_est_rest = (torch.exp(melsp_est)-1)/10000
                    magsp_est = magsp_est_rest = batch_magsp_rec[i][k,:flens_utt]

                    batch_loss_laplace[i] += criterion_laplace(pdf[:,:args.mel_dim], pdf[:,args.mel_dim:], melsp)
                    batch_loss_h[i] += torch.mean(criterion_l1(batch_out_, out_)) \
                                        + torch.sqrt(torch.mean(criterion_l2(batch_out_, out_))) \
                                    + torch.mean(criterion_l1(batch_out_2_, out_2_)) \
                                        + torch.sqrt(torch.mean(criterion_l2(batch_out_2_, out_2_))) \
                                    + torch.mean(criterion_l1(batch_out_f_, out_f_)) \
                                        + torch.sqrt(torch.mean(criterion_l2(batch_out_f_, out_f_)))
                    batch_loss_mid_smpl[i] += torch.mean(criterion_l1(batch_signs_c_, signs_c_)) \
                                            + torch.sqrt(torch.mean(criterion_l2(batch_signs_c_, signs_c_))) \
                                        + torch.mean(criterion_l1(batch_scales_c_, scales_c_)) \
                                            + torch.sqrt(torch.mean(criterion_l2(batch_scales_c_, scales_c_))) \
                                        + torch.mean(criterion_l1(batch_logits_c_, logits_c_)) \
                                            + torch.sqrt(torch.mean(criterion_l2(batch_logits_c_, logits_c_))) \
                                        + torch.mean(criterion_l1(batch_signs_f_, signs_f_)) \
                                            + torch.sqrt(torch.mean(criterion_l2(batch_signs_f_, signs_f_))) \
                                        + torch.mean(criterion_l1(batch_scales_f_, scales_f_)) \
                                            + torch.sqrt(torch.mean(criterion_l2(batch_scales_f_, scales_f_))) \
                                        + torch.mean(criterion_l1(batch_logits_f_, logits_f_)) \
                                            + torch.sqrt(torch.mean(criterion_l2(batch_logits_f_, logits_f_))) \
                                        + torch.mean(criterion_l1(batch_x_c_output_noclamp_, x_c_output_)) \
                                            + torch.sqrt(torch.mean(criterion_l2(batch_x_c_output_noclamp_, x_c_output_))) \
                                        + torch.mean(criterion_l1(batch_x_f_output_noclamp_, x_f_output_)) \
                                            + torch.sqrt(torch.mean(criterion_l2(batch_x_f_output_noclamp_, x_f_output_)))

                    if flens_utt > 1:
                        batch_loss_seg_conv[i] += torch.mean(criterion_l1(batch_seg_conv_, seg_conv_)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(batch_seg_conv_, seg_conv_)))
                        batch_loss_conv_sc[i] += torch.mean(criterion_l1(batch_conv_sc_, conv_sc_)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(batch_conv_sc_, conv_sc_)))
                        batch_loss_melsp[i] += torch.mean(criterion_l1(melsp_est, melsp)) \
                                                    + torch.sqrt(torch.mean(criterion_l2(melsp_est, melsp)))
                        batch_loss_magsp[i] += torch.mean(criterion_l1(magsp_est, magsp)) \
                                                     + torch.sqrt(torch.mean(criterion_l2(magsp_est, magsp)))
                    else:
                        batch_loss_seg_conv[i] += torch.mean(criterion_l1(batch_seg_conv_, seg_conv_))
                        batch_loss_conv_sc[i] += torch.mean(criterion_l1(batch_conv_sc_, conv_sc_))
                        batch_loss_melsp[i] += torch.mean(criterion_l1(melsp_est, melsp))
                        batch_loss_magsp[i] += torch.mean(criterion_l1(magsp_est, magsp))
                    batch_loss_melsp_dB[i] += torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_est_rest, min=1e-16))-melsp_rest_log))**2, -1)))
                    batch_loss_magsp_dB[i] += torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_est_rest, min=1e-16))-magsp_rest_log))**2, -1)))

                    batch_loss_ce_select_ = torch.mean(criterion_ce(batch_x_c_output_.reshape(-1, args.cf_dim), batch_x_c_.reshape(-1)).reshape(batch_x_c_output_.shape[0], -1), 0) # n_bands
                    batch_loss_ce_f_select_ = torch.mean(criterion_ce(batch_x_f_output_.reshape(-1, args.cf_dim), batch_x_f_.reshape(-1)).reshape(batch_x_f_output_.shape[0], -1), 0) # n_bands
                    batch_loss_err_select_ = torch.mean(torch.sum(criterion_l1(F.softmax(batch_x_c_output_, dim=-1), F.one_hot(batch_x_c_, num_classes=args.cf_dim).float()), -1), 0) # n_bands
                    batch_loss_err_f_select_ = torch.mean(torch.sum(criterion_l1(F.softmax(batch_x_f_output_, dim=-1), F.one_hot(batch_x_f_, num_classes=args.cf_dim).float()), -1), 0) # n_bands

                    batch_loss_ce_select[i] += batch_loss_ce_select_
                    batch_loss_ce_f_select[i] += batch_loss_ce_f_select_
                    batch_loss_err_select[i] += 100*batch_loss_err_select_
                    batch_loss_err_f_select[i] += 100*batch_loss_err_f_select_

                    batch_loss_fro_select_, batch_loss_l1_select_ = criterion_stft(batch_x_output[i][k,:slens_utt].transpose(1,0), batch_x_) # n_bands
                    batch_loss_fro_fb_select_, batch_loss_l1_fb_select_ = criterion_stft_fb(batch_x_output_fb[i][k,:slens_utt_fb], batch_x_fb_)

                    batch_loss_fro_select[i] += batch_loss_fro_select_
                    batch_loss_l1_select[i] += batch_loss_l1_select_
                    batch_loss_fro_fb[i] += batch_loss_fro_fb_select_
                    batch_loss_l1_fb[i] += batch_loss_l1_fb_select_

                    batch_loss += batch_loss_ce_select_.sum() + batch_loss_ce_f_select_.sum() \
                                  + batch_loss_ce_select_.mean() + batch_loss_ce_f_select_.mean() \
                                  + ((batch_loss_err_select_.sum() + batch_loss_err_f_select_.sum())/factors) \
                                  + batch_loss_err_select_.mean() + batch_loss_err_f_select_.mean() \
                                      + batch_loss_fro_select_.sum() + batch_loss_l1_select_.sum() \
                                      + batch_loss_fro_fb_select_ + batch_loss_l1_fb_select_

                    batch_sc_cv_ = batch_sc_cv[i//2][k,:flens_utt]
                    if i % 2 == 0:
                        batch_loss_sc_feat_kl_select += torch.mean(criterion_ce(batch_feat_rec_sc[i][k,:flens_utt], batch_sc_)) \
                                                        + torch.mean(criterion_ce(batch_feat_magsp_rec_sc[i][k,:flens_utt], batch_sc_)) \
                                                        + torch.mean(criterion_ce(batch_feat_cv_sc[i//2][k,:flens_utt], batch_sc_cv_)) \
                                                        + torch.mean(criterion_ce(batch_feat_magsp_cv_sc[i//2][k,:flens_utt], batch_sc_cv_))
                    else:
                        batch_loss_sc_feat_kl_select += torch.mean(criterion_ce(batch_feat_rec_sc[i][k,:flens_utt], batch_sc_)) \
                                                        + torch.mean(criterion_ce(batch_feat_magsp_rec_sc[i][k,:flens_utt], batch_sc_))
                    batch_loss_sc_z_kl_select += torch.mean(kl_categorical_categorical_logits(p_spk, logits_p_spk, batch_z_sc[i][k,:flens_utt]))

                    if i % 2 == 0:
                        batch_loss_qy_py_ce_select += torch.mean(criterion_ce(qy_logits_select_, batch_sc_)) \
                                                            + torch.mean(criterion_ce(qy_logits_e_select_, batch_sc_)) \
                                                    + torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits_select_, dim=-1), sc_onehot_), -1)) \
                                                        + torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits_e_select_, dim=-1), sc_onehot_), -1))
                        pdf_cv = batch_pdf_cv[i//2][k,:flens_utt]
                        batch_loss_laplace_cv[i//2] += criterion_laplace(pdf_cv[:,:args.mel_dim], pdf_cv[:,args.mel_dim:], melsp)
                        if flens_utt > 1:
                            melsp_cv_est = batch_melsp_cv[i//2][k,:flens_utt]
                            magsp_cv_est = batch_magsp_cv[i//2][k,:flens_utt]
                            batch_loss_melsp_cv[i//2] += torch.mean(criterion_l1(melsp_cv_est, melsp)) \
                                                        + torch.sqrt(torch.mean(criterion_l2(melsp_cv_est, melsp)))
                            batch_loss_magsp_cv[i//2] += torch.mean(criterion_l1(magsp_cv_est, magsp)) \
                                                         + torch.sqrt(torch.mean(criterion_l2(magsp_cv_est, magsp)))
                        else:
                            batch_loss_melsp_cv[i//2] += torch.mean(criterion_l1(batch_melsp_cv[i//2][k,:flens_utt], melsp))
                            batch_loss_magsp_cv[i//2] += torch.mean(criterion_l1(batch_magsp_cv[i//2][k,:flens_utt], magsp))
                    else:
                        sc_cv_onehot_ = F.one_hot(batch_sc_cv_, num_classes=n_spk).float()
                        batch_loss_qy_py_ce_select += torch.mean(criterion_ce(qy_logits_select_, batch_sc_cv_)) \
                                                            + torch.mean(criterion_ce(qy_logits_e_select_, batch_sc_cv_)) \
                                                    + torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits_select_, dim=-1), sc_cv_onehot_), -1)) \
                                                        + torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits_e_select_, dim=-1), sc_cv_onehot_), -1))

                    batch_loss_qz_pz_kl_select += kl_laplace_laplace(qz_alpha[i][k,:flens_utt], qz_alpha_fix_) \
                                                    + kl_laplace_laplace(qz_alpha_e[i][k,:flens_utt], qz_alpha_e_fix_)

                    if i > 0:
                        z_obs = torch.cat((z_e[i][k,:flens_utt], z[i][k,:flens_utt]), 1)
                        batch_loss_qz_pz_kl_select += torch.mean(torch.log(torch.clamp(torch.sum(z_obs*z_ref, -1), min=1e-13) / torch.clamp(torch.sqrt(torch.sum(z_obs**2, -1))*z_ref_denom, min=1e-13))) \
                                                    + torch.sqrt(torch.mean(torch.sum((z_obs-z_ref)**2, -1)))
                    else:
                        z_ref = torch.cat((z_e[0][k,:flens_utt], z[0][k,:flens_utt]), 1)
                        z_ref_denom = torch.sqrt(torch.sum(z_ref**2, -1))
            batch_loss += batch_loss_sc_feat_kl_select + batch_loss_sc_z_kl_select \
                            + batch_loss_qy_py_ce_select + batch_loss_qz_pz_kl_select
            for i in range(args.n_half_cyc):
                batch_loss += batch_loss_laplace[i] \
                                + batch_loss_melsp[i] + batch_loss_melsp_dB[i] \
                                + batch_loss_magsp[i] + batch_loss_magsp_dB[i] \
                                + batch_loss_seg_conv[i] + batch_loss_conv_sc[i] \
                                + batch_loss_h[i] + batch_loss_mid_smpl[i]
                batch_loss_laplace[i] /= len_idx_select
                batch_loss_melsp[i] /= len_idx_select
                batch_loss_melsp_dB[i] /= len_idx_select
                batch_loss_magsp[i] /= len_idx_select
                batch_loss_magsp_dB[i] /= len_idx_select
                batch_loss_seg_conv[i] /= len_idx_select
                batch_loss_conv_sc[i] /= len_idx_select
                batch_loss_h[i] /= len_idx_select
                batch_loss_mid_smpl[i] /= len_idx_select
                batch_loss_ce_select[i] /= len_idx_select #n_bands
                batch_loss_ce_f_select[i] /= len_idx_select #n_bands
                batch_loss_err_select[i] /= len_idx_select #n_bands
                batch_loss_err_f_select[i] /= len_idx_select #n_bands
                batch_loss_fro_select[i] /= len_idx_select #n_bands
                batch_loss_l1_select[i] /= len_idx_select #n_bands
                batch_loss_fro_fb[i] /= len_idx_select
                batch_loss_l1_fb[i] /= len_idx_select
                batch_loss_ce_c_avg[i] = batch_loss_ce_select[i].mean().item()
                batch_loss_ce_f_avg[i] = batch_loss_ce_f_select[i].mean().item()
                batch_loss_err_c_avg[i] = batch_loss_err_select[i].mean().item()
                batch_loss_err_f_avg[i] = batch_loss_err_f_select[i].mean().item()
                batch_loss_ce_avg[i] = (batch_loss_ce_c_avg[i] + batch_loss_ce_f_avg[i]) / 2
                batch_loss_err_avg[i] = (batch_loss_err_c_avg[i] + batch_loss_err_f_avg[i]) / 2
                batch_loss_fro_avg[i] = batch_loss_fro_select[i].mean().item()
                batch_loss_l1_avg[i] = batch_loss_l1_select[i].mean().item()
                total_train_loss["train/loss_ce-%d"%(i+1)].append(batch_loss_ce_avg[i])
                total_train_loss["train/loss_err-%d"%(i+1)].append(batch_loss_err_avg[i])
                total_train_loss["train/loss_ce_c-%d"%(i+1)].append(batch_loss_ce_c_avg[i])
                total_train_loss["train/loss_err_c-%d"%(i+1)].append(batch_loss_err_c_avg[i])
                total_train_loss["train/loss_ce_f-%d"%(i+1)].append(batch_loss_ce_f_avg[i])
                total_train_loss["train/loss_err_f-%d"%(i+1)].append(batch_loss_err_f_avg[i])
                loss_ce_avg[i].append(batch_loss_ce_avg[i])
                loss_err_avg[i].append(batch_loss_err_avg[i])
                loss_ce_c_avg[i].append(batch_loss_ce_c_avg[i])
                loss_err_c_avg[i].append(batch_loss_err_c_avg[i])
                loss_ce_f_avg[i].append(batch_loss_ce_f_avg[i])
                loss_err_f_avg[i].append(batch_loss_err_f_avg[i])
                for j in range(args.n_bands):
                    total_train_loss["train/loss_ce_c-%d-%d"%(i+1,j+1)].append(batch_loss_ce_select[i][j].item())
                    total_train_loss["train/loss_err_c-%d-%d"%(i+1,j+1)].append(batch_loss_err_select[i][j].item())
                    total_train_loss["train/loss_ce_f-%d-%d"%(i+1,j+1)].append(batch_loss_ce_f_select[i][j].item())
                    total_train_loss["train/loss_err_f-%d-%d"%(i+1,j+1)].append(batch_loss_err_f_select[i][j].item())
                    total_train_loss["train/loss_fro-%d-%d"%(i+1,j+1)].append(batch_loss_fro_select[i][j].item())
                    total_train_loss["train/loss_l1-%d-%d"%(i+1,j+1)].append(batch_loss_l1_select[i][j].item())
                    loss_ce[i][j].append(batch_loss_ce_select[i][j].item())
                    loss_err[i][j].append(batch_loss_err_select[i][j].item())
                    loss_ce_f[i][j].append(batch_loss_ce_f_select[i][j].item())
                    loss_err_f[i][j].append(batch_loss_err_f_select[i][j].item())
                    loss_fro[i][j].append(batch_loss_fro_select[i][j].item())
                    loss_l1[i][j].append(batch_loss_l1_select[i][j].item())
                total_train_loss["train/loss_fro-%d"%(i+1)].append(batch_loss_fro_avg[i])
                total_train_loss["train/loss_l1-%d"%(i+1)].append(batch_loss_l1_avg[i])
                total_train_loss["train/loss_fro_fb-%d"%(i+1)].append(batch_loss_fro_fb[i].item())
                total_train_loss["train/loss_l1_fb-%d"%(i+1)].append(batch_loss_l1_fb[i].item())
                total_train_loss["train/loss_seg_conv-%d"%(i+1)].append(batch_loss_seg_conv[i].item())
                total_train_loss["train/loss_conv_sc-%d"%(i+1)].append(batch_loss_conv_sc[i].item())
                total_train_loss["train/loss_h-%d"%(i+1)].append(batch_loss_h[i].item())
                total_train_loss["train/loss_mid_smpl-%d"%(i+1)].append(batch_loss_mid_smpl[i].item())
                loss_fro_avg[i].append(batch_loss_fro_avg[i])
                loss_l1_avg[i].append(batch_loss_l1_avg[i])
                loss_fro_fb[i].append(batch_loss_fro_fb[i].item())
                loss_l1_fb[i].append(batch_loss_l1_fb[i].item())
                loss_seg_conv[i].append(batch_loss_seg_conv[i].item())
                loss_conv_sc[i].append(batch_loss_conv_sc[i].item())
                loss_h[i].append(batch_loss_h[i].item())
                loss_mid_smpl[i].append(batch_loss_mid_smpl[i].item())
                total_train_loss["train/loss_laplace-%d"%(i+1)].append(batch_loss_laplace[i].item())
                total_train_loss["train/loss_melsp-%d"%(i+1)].append(batch_loss_melsp[i].item())
                total_train_loss["train/loss_melsp_dB-%d"%(i+1)].append(batch_loss_melsp_dB[i].item())
                total_train_loss["train/loss_magsp-%d"%(i+1)].append(batch_loss_magsp[i].item())
                total_train_loss["train/loss_magsp_dB-%d"%(i+1)].append(batch_loss_magsp_dB[i].item())
                loss_laplace[i].append(batch_loss_laplace[i].item())
                loss_melsp[i].append(batch_loss_melsp[i].item())
                loss_melsp_dB[i].append(batch_loss_melsp_dB[i].item())
                loss_magsp[i].append(batch_loss_magsp[i].item())
                loss_magsp_dB[i].append(batch_loss_magsp_dB[i].item())
                if i % 2 == 0:
                    batch_loss_laplace_cv[i//2] /= len_idx_select
                    batch_loss_melsp_cv[i//2] /= len_idx_select
                    batch_loss_magsp_cv[i//2] /= len_idx_select
                    total_train_loss["train/loss_laplace_cv-%d"%(i+1)].append(batch_loss_laplace_cv[i//2].item())
                    total_train_loss["train/loss_melsp_cv-%d"%(i+1)].append(batch_loss_melsp_cv[i//2].item())
                    total_train_loss["train/loss_magsp_cv-%d"%(i+1)].append(batch_loss_magsp_cv[i//2].item())
                    loss_laplace_cv[i//2].append(batch_loss_laplace_cv[i//2].item())
                    loss_melsp_cv[i//2].append(batch_loss_melsp_cv[i//2].item())
                    loss_magsp_cv[i//2].append(batch_loss_magsp_cv[i//2].item())
            if len(idx_select_full) > 0:
                logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                batch_melsp = torch.index_select(batch_melsp,0,idx_select_full)
                batch_magsp = torch.index_select(batch_magsp,0,idx_select_full)
                batch_sc = torch.index_select(batch_sc,0,idx_select_full)
                batch_feat_in_sc = torch.index_select(batch_feat_in_sc,0,idx_select_full)
                batch_feat_magsp_in_sc = torch.index_select(batch_feat_magsp_in_sc,0,idx_select_full)
                batch_x_c = torch.index_select(batch_x_c,0,idx_select_full)
                batch_x_f = torch.index_select(batch_x_f,0,idx_select_full)
                batch_x = torch.index_select(batch_x,0,idx_select_full)
                batch_x_fb = torch.index_select(batch_x_fb,0,idx_select_full)
                x_c_output = torch.index_select(x_c_output,0,idx_select_full)
                x_f_output = torch.index_select(x_f_output,0,idx_select_full)
                seg_conv = torch.index_select(seg_conv,0,idx_select_full)
                conv_sc = torch.index_select(conv_sc,0,idx_select_full)
                out = torch.index_select(out,0,idx_select_full)
                out_2 = torch.index_select(out_2,0,idx_select_full)
                out_f = torch.index_select(out_f,0,idx_select_full)
                signs_c = torch.index_select(signs_c,0,idx_select_full)
                scales_c = torch.index_select(scales_c,0,idx_select_full)
                logits_c = torch.index_select(logits_c,0,idx_select_full)
                signs_f = torch.index_select(signs_f,0,idx_select_full)
                scales_f = torch.index_select(scales_f,0,idx_select_full)
                logits_f = torch.index_select(logits_f,0,idx_select_full)
                qz_alpha_fix = torch.index_select(qz_alpha_fix,0,idx_select_full)
                qz_alpha_e_fix = torch.index_select(qz_alpha_e_fix,0,idx_select_full)
                n_batch_utt = batch_melsp.shape[0]
                for i in range(args.n_half_cyc):
                    batch_pdf_rec[i] = torch.index_select(batch_pdf_rec[i],0,idx_select_full)
                    batch_melsp_rec[i] = torch.index_select(batch_melsp_rec[i],0,idx_select_full)
                    batch_magsp_rec[i] = torch.index_select(batch_magsp_rec[i],0,idx_select_full)
                    batch_feat_rec_sc[i] = torch.index_select(batch_feat_rec_sc[i],0,idx_select_full)
                    batch_feat_magsp_rec_sc[i] = torch.index_select(batch_feat_magsp_rec_sc[i],0,idx_select_full)
                    batch_x_c_output[i] = torch.index_select(batch_x_c_output[i],0,idx_select_full)
                    batch_x_f_output[i] = torch.index_select(batch_x_f_output[i],0,idx_select_full)
                    batch_x_output[i] = torch.index_select(batch_x_output[i],0,idx_select_full)
                    batch_x_output_fb[i] = torch.index_select(batch_x_output_fb[i],0,idx_select_full)
                    batch_x_c_output_noclamp[i] = torch.index_select(batch_x_c_output_noclamp[i],0,idx_select_full)
                    batch_x_f_output_noclamp[i] = torch.index_select(batch_x_f_output_noclamp[i],0,idx_select_full)
                    batch_seg_conv[i] = torch.index_select(batch_seg_conv[i],0,idx_select_full)
                    batch_conv_sc[i] = torch.index_select(batch_conv_sc[i],0,idx_select_full)
                    batch_out[i] = torch.index_select(batch_out[i],0,idx_select_full)
                    batch_out_2[i] = torch.index_select(batch_out_2[i],0,idx_select_full)
                    batch_out_f[i] = torch.index_select(batch_out_f[i],0,idx_select_full)
                    batch_signs_c[i] = torch.index_select(batch_signs_c[i],0,idx_select_full)
                    batch_scales_c[i] = torch.index_select(batch_scales_c[i],0,idx_select_full)
                    batch_logits_c[i] = torch.index_select(batch_logits_c[i],0,idx_select_full)
                    batch_signs_f[i] = torch.index_select(batch_signs_f[i],0,idx_select_full)
                    batch_scales_f[i] = torch.index_select(batch_scales_f[i],0,idx_select_full)
                    batch_logits_f[i] = torch.index_select(batch_logits_f[i],0,idx_select_full)
                    qy_logits[i] = torch.index_select(qy_logits[i],0,idx_select_full)
                    qz_alpha[i] = torch.index_select(qz_alpha[i],0,idx_select_full)
                    qy_logits_e[i] = torch.index_select(qy_logits_e[i],0,idx_select_full)
                    qz_alpha_e[i] = torch.index_select(qz_alpha_e[i],0,idx_select_full)
                    if i % 2 == 0:
                        batch_pdf_cv[i//2] = torch.index_select(batch_pdf_cv[i//2],0,idx_select_full)
                        batch_melsp_cv[i//2] = torch.index_select(batch_melsp_cv[i//2],0,idx_select_full)
                        batch_magsp_cv[i//2] = torch.index_select(batch_magsp_cv[i//2],0,idx_select_full)
                        batch_feat_cv_sc[i//2] = torch.index_select(batch_feat_cv_sc[i//2],0,idx_select_full)
                        batch_feat_magsp_cv_sc[i//2] = torch.index_select(batch_feat_magsp_cv_sc[i//2],0,idx_select_full)
                        batch_sc_cv[i//2] = torch.index_select(batch_sc_cv[i//2],0,idx_select_full)
            else:
                optimizer.zero_grad()
                batch_loss.backward()
                flag = False
                explode_model = ""
                for name, param in model_encoder_melsp.named_parameters():
                    if param.requires_grad:
                        grad_norm = param.grad.norm()
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            flag = True
                            explode_model = "enc_melsp"
                            break
                if not flag:
                    for name, param in model_encoder_excit.named_parameters():
                        if param.requires_grad:
                            grad_norm = param.grad.norm()
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                flag = True
                                explode_model = "enc_excit"
                                break
                if not flag:
                    for name, param in model_decoder_melsp.named_parameters():
                        if param.requires_grad:
                            grad_norm = param.grad.norm()
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                flag = True
                                explode_model = "dec_melsp"
                                break
                if not flag:
                    for name, param in model_spk.named_parameters():
                        if param.requires_grad:
                            grad_norm = param.grad.norm()
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                flag = True
                                explode_model = "spk"
                                break
                if not flag and args.spkidtr_dim > 0:
                    for name, param in model_spkidtr.named_parameters():
                        if param.requires_grad:
                            grad_norm = param.grad.norm()
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                flag = True
                                explode_model = "spkidtr"
                                break
                if flag:
                    logging.info("explode grad %s" % (explode_model))
                    optimizer.zero_grad()
                    text_log = "batch loss_select %lf " % (batch_loss.item())
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    continue
                torch.nn.utils.clip_grad_norm_(model_encoder_melsp.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(model_encoder_excit.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(model_decoder_melsp.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(model_spk.parameters(), 10)
                if args.spkidtr_dim > 0:
                    torch.nn.utils.clip_grad_norm_(model_spkidtr.parameters(), 10)
                optimizer.step()

                with torch.no_grad():
                    if idx_stage < args.n_stage-1 and iter_idx + 1 == t_starts[idx_stage+1]:
                        idx_stage += 1
                    if idx_stage > 0:
                        sparsify(model_encoder_melsp, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                        sparsify(model_encoder_excit, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                        sparsify(model_decoder_melsp, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                    else:
                        sparsify(model_encoder_melsp, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])
                        sparsify(model_encoder_excit, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])
                        sparsify(model_decoder_melsp, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])

                text_log = "batch loss_select %lf " % (batch_loss.item())
                logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                iter_idx += 1
                #if iter_idx % args.save_interval_iter == 0:
                #    logging.info('save iter:%d' % (iter_idx))
                #    save_checkpoint(args.expdir, model_encoder, model_decoder, \
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
        magsp_rest_log = torch.log10(torch.clamp(magsp_rest, min=1e-16))
        sc_onehot = F.one_hot(batch_sc, num_classes=n_spk).float()
        batch_sc_ = batch_sc.reshape(-1)
        batch_loss_sc_feat_in_ = torch.mean(criterion_ce(batch_feat_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
        batch_loss_sc_feat_in = batch_loss_sc_feat_in_.mean()
        batch_loss_sc_feat_magsp_in_ = torch.mean(criterion_ce(batch_feat_magsp_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
        batch_loss_sc_feat_magsp_in = batch_loss_sc_feat_magsp_in_.mean()
        batch_loss += batch_loss_sc_feat_in_.sum() + batch_loss_sc_feat_magsp_in_.sum()
        batch_x_c_onehot = F.one_hot(batch_x_c, num_classes=args.cf_dim).float()
        batch_x_c = batch_x_c.reshape(-1)
        batch_x_f_onehot = F.one_hot(batch_x_f, num_classes=args.cf_dim).float()
        T = batch_x_f_onehot.shape[1]
        batch_x_f = batch_x_f.reshape(-1)
        batch_x_ = batch_x.transpose(1,2)
        for i in range(args.n_half_cyc):
            ## reconst. [i % 2 == 0] / cyclic reconst. [i % 2 == 1]
            pdf = batch_pdf_rec[i]
            melsp_est = batch_melsp_rec[i]
            melsp_est_rest = (torch.exp(melsp_est)-1)/10000
            magsp_est = magsp_est_rest = batch_magsp_rec[i]
            ## conversion
            if i % 2 == 0:
                pdf_cv = batch_pdf_cv[i//2]
                melsp_cv = batch_melsp_cv[i//2]
                magsp_cv = batch_magsp_cv[i//2]
            else:
                sc_cv_onehot = F.one_hot(batch_sc_cv[i//2], num_classes=n_spk).float()

            batch_loss_laplace_ = criterion_laplace(pdf[:,:,:args.mel_dim], pdf[:,:,args.mel_dim:], melsp)
            batch_loss_laplace[i] = batch_loss_laplace_.mean()

            batch_loss_melsp_ = torch.mean(torch.mean(criterion_l1(melsp_est, melsp), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_est, melsp), -1), -1))
            batch_loss_melsp[i] = batch_loss_melsp_.mean()
            batch_loss_melsp_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_est_rest, min=1e-16))-melsp_rest_log))**2, -1)), -1)
            batch_loss_melsp_dB[i] = batch_loss_melsp_dB_.mean()

            batch_loss_magsp_ = torch.mean(torch.mean(criterion_l1(magsp_est, magsp), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_est, magsp), -1), -1))
            batch_loss_magsp[i] = batch_loss_magsp_.mean()
            batch_loss_magsp_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_est_rest, min=1e-16))-magsp_rest_log))**2, -1)), -1)
            batch_loss_magsp_dB[i] = batch_loss_magsp_dB_.mean()

            batch_loss_px[i] = batch_loss_laplace_.mean() \
                                + batch_loss_melsp_.mean() + batch_loss_melsp_dB_.mean() \
                                    + batch_loss_magsp_.mean() + batch_loss_magsp_dB_.mean()

            batch_loss_px_sum = batch_loss_laplace_.sum() \
                                + batch_loss_melsp_.sum() + batch_loss_melsp_dB_.sum() \
                                    + batch_loss_magsp_.sum() + batch_loss_magsp_dB_.sum()

            ## conversion
            if i % 2 == 0:
                batch_loss_laplace_cv[i//2] = torch.mean(criterion_laplace(pdf_cv[:,:,:args.mel_dim], pdf_cv[:,:,args.mel_dim:], melsp))
                batch_loss_melsp_cv[i//2] = torch.mean(criterion_l1(melsp_cv, melsp)) \
                                                + torch.sqrt(torch.mean(criterion_l2(melsp_cv, melsp)))
                batch_loss_magsp_cv[i//2] = torch.mean(torch.mean(criterion_l1(magsp_cv, magsp), -1)) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_cv, magsp), -1)))

            # speaker-classifier on features and latent
            batch_sc_cv_ = batch_sc_cv[i//2].reshape(-1)
            batch_loss_sc_feat_ = torch.mean(criterion_ce(batch_feat_rec_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_feat[i] = batch_loss_sc_feat_.mean()
            batch_loss_sc_feat_magsp_ = torch.mean(criterion_ce(batch_feat_magsp_rec_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_feat_magsp[i] = batch_loss_sc_feat_magsp_.mean()
            batch_loss_sc_z_ = torch.mean(kl_categorical_categorical_logits(p_spk, logits_p_spk, batch_z_sc[i]), -1)
            batch_loss_sc_z[i] = batch_loss_sc_z_.mean()
            batch_loss_sc_feat_kl = batch_loss_sc_feat_.sum() + batch_loss_sc_feat_magsp_.sum() + (100*batch_loss_sc_z_).sum()
            if i % 2 == 0:
                batch_loss_sc_feat_cv_ = torch.mean(criterion_ce(batch_feat_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                batch_loss_sc_feat_cv[i//2] = batch_loss_sc_feat_cv_.mean()
                batch_loss_sc_feat_magsp_cv_ = torch.mean(criterion_ce(batch_feat_magsp_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                batch_loss_sc_feat_magsp_cv[i//2] = batch_loss_sc_feat_magsp_cv_.mean()
                batch_loss_sc_feat_kl += batch_loss_sc_feat_cv_.sum() + batch_loss_sc_feat_magsp_cv_.sum()

            # KL-div lat., CE and error-percentage spk.
            if i % 2 == 0:
                batch_loss_qy_py_ = torch.mean(criterion_ce(qy_logits[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                batch_loss_qy_py[i] = batch_loss_qy_py_.mean()
                batch_loss_qy_py_err_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i], dim=-1), sc_onehot), -1), -1)
                batch_loss_qy_py_err[i] = batch_loss_qy_py_err_.mean()
            else:
                batch_loss_qy_py_ = torch.mean(criterion_ce(qy_logits[i].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                batch_loss_qy_py[i] = batch_loss_qy_py_.mean()
                batch_loss_qy_py_err_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i], dim=-1), sc_cv_onehot), -1), -1)
                batch_loss_qy_py_err[i] = batch_loss_qy_py_err_.mean()
            batch_loss_qz_pz_ = kl_laplace_laplace(qz_alpha[i], qz_alpha_fix)
            batch_loss_qz_pz[i] = batch_loss_qz_pz_.mean()
            batch_loss_qz_pz_e_ = kl_laplace_laplace(qz_alpha_e[i], qz_alpha_e_fix)
            batch_loss_qz_pz_e[i] = batch_loss_qz_pz_e_.mean()
            batch_loss_qz_pz_kl = batch_loss_qz_pz_.sum() + batch_loss_qz_pz_e_.sum()
            if i % 2 == 0:
                batch_loss_qy_py_e_ = torch.mean(criterion_ce(qy_logits_e[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
                batch_loss_qy_py_e[i] = batch_loss_qy_py_e_.mean()
                batch_loss_qy_py_err_e_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits_e[i], dim=-1), sc_onehot), -1), -1)
                batch_loss_qy_py_err_e[i] = batch_loss_qy_py_err_e_.mean()
            else:
                batch_loss_qy_py_e_ = torch.mean(criterion_ce(qy_logits_e[i].reshape(-1, n_spk), batch_sc_cv_).reshape(n_batch_utt, -1), -1)
                batch_loss_qy_py_e[i] = batch_loss_qy_py_e_.mean()
                batch_loss_qy_py_err_e_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits_e[i], dim=-1), sc_cv_onehot), -1), -1)
                batch_loss_qy_py_err_e[i] = batch_loss_qy_py_err_e_.mean()
            batch_loss_qy_py_kl = batch_loss_qy_py_.sum() + batch_loss_qy_py_e_.sum() \
                                    + batch_loss_qy_py_err_.sum() + batch_loss_qy_py_err_e_.sum()

            # cosine, rmse latent
            if i > 0:
                z_obs = torch.cat((z_e[i], z[i]), 2)
                batch_loss_lat_cossim_ = torch.clamp(torch.sum(z_obs*z_ref, -1), min=1e-13) / torch.clamp(torch.sqrt(torch.sum(z_obs**2, -1))*z_ref_denom, min=1e-13)
                batch_loss_lat_cossim[i] = batch_loss_lat_cossim_.mean()
                batch_loss_lat_rmse_ = torch.sqrt(torch.mean(torch.sum((z_obs-z_ref)**2, -1), -1))
                batch_loss_lat_rmse[i] = batch_loss_lat_rmse_.mean()
                batch_loss_qz_pz_kl += batch_loss_lat_rmse_.sum() - torch.log(batch_loss_lat_cossim_).sum()
            else:
                z_ref = torch.cat((z_e[i], z[i]), 2)
                z_ref_denom = torch.sqrt(torch.sum(z_ref**2, -1))

            # waveform layer loss
            batch_loss_seg_conv_ = torch.mean(torch.mean(criterion_l1(batch_seg_conv[i], seg_conv), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_seg_conv[i], seg_conv), -1), -1))
            batch_loss_seg_conv[i] = batch_loss_seg_conv_.mean()

            batch_loss_conv_sc_ = torch.mean(torch.mean(criterion_l1(batch_conv_sc[i], conv_sc), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_conv_sc[i], conv_sc), -1), -1))
            batch_loss_conv_sc[i] = batch_loss_conv_sc_.mean()

            batch_loss_h_ = torch.mean(torch.mean(criterion_l1(batch_out[i], out), -1), -1) \
                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_out[i], out), -1), -1)) \
                            + torch.mean(torch.mean(criterion_l1(batch_out_2[i], out_2), -1), -1) \
                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_out_2[i], out_2), -1), -1)) \
                            + torch.mean(torch.mean(criterion_l1(batch_out_f[i], out_f), -1), -1) \
                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_out_f[i], out_f), -1), -1))
            batch_loss_h[i] = batch_loss_h_.mean()

            batch_loss_mid_smpl_ = torch.mean(torch.mean(torch.mean(criterion_l1(batch_signs_c[i], signs_c), -1), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_signs_c[i], signs_c), -1), -1), -1)) \
                                + torch.mean(torch.mean(torch.mean(criterion_l1(batch_scales_c[i], scales_c), -1), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_scales_c[i], scales_c), -1), -1), -1)) \
                                + torch.mean(torch.mean(torch.mean(criterion_l1(batch_logits_c[i], logits_c), -1), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_logits_c[i], logits_c), -1), -1), -1)) \
                                + torch.mean(torch.mean(torch.mean(criterion_l1(batch_signs_f[i], signs_f), -1), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_signs_f[i], signs_f), -1), -1), -1)) \
                                + torch.mean(torch.mean(torch.mean(criterion_l1(batch_scales_f[i], scales_f), -1), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_scales_f[i], scales_f), -1), -1), -1)) \
                                + torch.mean(torch.mean(torch.mean(criterion_l1(batch_logits_f[i], logits_f), -1), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_logits_f[i], logits_f), -1), -1), -1)) \
                                + torch.mean(torch.mean(torch.mean(criterion_l1(batch_x_c_output_noclamp[i], x_c_output), -1), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_x_c_output_noclamp[i], x_c_output), -1), -1), -1)) \
                                + torch.mean(torch.mean(torch.mean(criterion_l1(batch_x_f_output_noclamp[i], x_f_output), -1), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(torch.mean(criterion_l2(batch_x_f_output_noclamp[i], x_f_output), -1), -1), -1))
            batch_loss_mid_smpl[i] = batch_loss_mid_smpl_.mean()

            batch_loss_wave = batch_loss_seg_conv_.sum() + batch_loss_conv_sc_.sum() \
                                + batch_loss_h_.sum() + batch_loss_mid_smpl_.sum()

            # waveform loss
            batch_loss_ce_ = torch.mean(criterion_ce(batch_x_c_output[i].reshape(-1, args.cf_dim), batch_x_c).reshape(n_batch_utt, T, -1), 1) # B x n_bands
            batch_loss_err_ = torch.mean(torch.sum(criterion_l1(F.softmax(batch_x_c_output[i], dim=-1), batch_x_c_onehot), -1), 1) # B x n_bands
            batch_loss_ce_f_ = torch.mean(criterion_ce(batch_x_f_output[i].reshape(-1, args.cf_dim), batch_x_f).reshape(n_batch_utt, T, -1), 1) # B x n_bands
            batch_loss_err_f_ = torch.mean(torch.sum(criterion_l1(F.softmax(batch_x_f_output[i], dim=-1), batch_x_f_onehot), -1), 1) # B x n_bands
            logging.info(f'{batch_loss_err_.mean()}')
            logging.info(f'{batch_loss_err_f_.mean()}')
            batch_loss_wave += ((batch_loss_err_.sum() + batch_loss_err_f_.sum())/factors) \
                                + batch_loss_err_.mean(-1).sum() + batch_loss_err_f_.mean(-1).sum() \
                                    + batch_loss_ce_.sum() + batch_loss_ce_f_.sum() \
                                    + batch_loss_ce_.mean(-1).sum() + batch_loss_ce_f_.mean(-1).sum()
            batch_loss_err_ = 100*batch_loss_err_.mean(0) # n_bands
            batch_loss_err_f_ = 100*batch_loss_err_f_.mean(0) # n_bands

            batch_loss_ce_c_avg[i] = batch_loss_ce_.mean().item()
            batch_loss_err_c_avg[i] = batch_loss_err_.mean().item()
            batch_loss_ce_f_avg[i] = batch_loss_ce_f_.mean().item()
            batch_loss_err_f_avg[i] = batch_loss_err_f_.mean().item()
            batch_loss_ce_avg[i] = (batch_loss_ce_c_avg[i] + batch_loss_ce_f_avg[i]) / 2
            batch_loss_err_avg[i] = (batch_loss_err_c_avg[i] + batch_loss_err_f_avg[i]) / 2
            total_train_loss["train/loss_ce-%d"%(i+1)].append(batch_loss_ce_avg[i])
            total_train_loss["train/loss_err-%d"%(i+1)].append(batch_loss_err_avg[i])
            total_train_loss["train/loss_ce_c-%d"%(i+1)].append(batch_loss_ce_c_avg[i])
            total_train_loss["train/loss_err_c-%d"%(i+1)].append(batch_loss_err_c_avg[i])
            total_train_loss["train/loss_ce_f-%d"%(i+1)].append(batch_loss_ce_f_avg[i])
            total_train_loss["train/loss_err_f-%d"%(i+1)].append(batch_loss_err_f_avg[i])
            loss_ce_avg[i].append(batch_loss_ce_avg[i])
            loss_err_avg[i].append(batch_loss_err_avg[i])
            loss_ce_c_avg[i].append(batch_loss_ce_c_avg[i])
            loss_err_c_avg[i].append(batch_loss_err_c_avg[i])
            loss_ce_f_avg[i].append(batch_loss_ce_f_avg[i])
            loss_err_f_avg[i].append(batch_loss_err_f_avg[i])
            batch_loss_fro_, batch_loss_l1_ = criterion_stft(batch_x_output[i].transpose(1,2), batch_x_)
            for j in range(args.n_bands):
                batch_loss_ce[i][j] = batch_loss_ce_[:,j].mean().item()
                batch_loss_err[i][j] = batch_loss_err_[j].item()
                batch_loss_ce_f[i][j] = batch_loss_ce_f_[:,j].mean().item()
                batch_loss_err_f[i][j] = batch_loss_err_f_[j].item()
                batch_loss_fro[i][j] = batch_loss_fro_[:,j].mean().item()
                batch_loss_l1[i][j] = batch_loss_l1_[:,j].mean().item()
                total_train_loss["train/loss_ce_c-%d-%d"%(i+1,j+1)].append(batch_loss_ce[i][j])
                total_train_loss["train/loss_err_c-%d-%d"%(i+1,j+1)].append(batch_loss_err[i][j])
                total_train_loss["train/loss_ce_f-%d-%d"%(i+1,j+1)].append(batch_loss_ce_f[i][j])
                total_train_loss["train/loss_err_f-%d-%d"%(i+1,j+1)].append(batch_loss_err_f[i][j])
                total_train_loss["train/loss_fro-%d-%d"%(i+1,j+1)].append(batch_loss_fro[i][j])
                total_train_loss["train/loss_l1-%d-%d"%(i+1,j+1)].append(batch_loss_l1[i][j])
                loss_ce[i][j].append(batch_loss_ce[i][j])
                loss_err[i][j].append(batch_loss_err[i][j])
                loss_ce_f[i][j].append(batch_loss_ce_f[i][j])
                loss_err_f[i][j].append(batch_loss_err_f[i][j])
                loss_fro[i][j].append(batch_loss_fro[i][j])
                loss_l1[i][j].append(batch_loss_l1[i][j])
            batch_loss_fro_avg[i] = batch_loss_fro_.mean().item()
            batch_loss_l1_avg[i] = batch_loss_l1_.mean().item()
            batch_loss_fro_fb_, batch_loss_l1_fb_ = criterion_stft_fb(batch_x_output_fb[i], batch_x_fb)
            batch_loss_wave += batch_loss_fro_.sum() + batch_loss_l1_.sum() \
                                + batch_loss_fro_fb_.sum() + batch_loss_l1_fb_.sum()
            batch_loss_fro_fb[i] = batch_loss_fro_fb_.mean().item()
            batch_loss_l1_fb[i] = batch_loss_l1_fb_.mean().item()
            total_train_loss["train/loss_fro-%d"%(i+1)].append(batch_loss_fro_avg[i])
            total_train_loss["train/loss_l1-%d"%(i+1)].append(batch_loss_l1_avg[i])
            total_train_loss["train/loss_fro_fb-%d"%(i+1)].append(batch_loss_fro_fb[i])
            total_train_loss["train/loss_l1_fb-%d"%(i+1)].append(batch_loss_l1_fb[i])
            total_train_loss["train/loss_seg_conv-%d"%(i+1)].append(batch_loss_seg_conv[i].item())
            total_train_loss["train/loss_conv_sc-%d"%(i+1)].append(batch_loss_conv_sc[i].item())
            total_train_loss["train/loss_seg_conv-%d"%(i+1)].append(batch_loss_seg_conv[i].item())
            total_train_loss["train/loss_conv_sc-%d"%(i+1)].append(batch_loss_conv_sc[i].item())
            total_train_loss["train/loss_h-%d"%(i+1)].append(batch_loss_h[i].item())
            total_train_loss["train/loss_mid_smpl-%d"%(i+1)].append(batch_loss_mid_smpl[i].item())
            loss_fro_avg[i].append(batch_loss_fro_avg[i])
            loss_l1_avg[i].append(batch_loss_l1_avg[i])
            loss_fro_fb[i].append(batch_loss_fro_fb[i])
            loss_l1_fb[i].append(batch_loss_l1_fb[i])
            loss_seg_conv[i].append(batch_loss_seg_conv[i].item())
            loss_conv_sc[i].append(batch_loss_conv_sc[i].item())
            loss_h[i].append(batch_loss_h[i].item())
            loss_mid_smpl[i].append(batch_loss_mid_smpl[i].item())

            # elbo
            batch_loss_elbo[i] = batch_loss_px_sum + batch_loss_wave + batch_loss_qz_pz_kl + batch_loss_qy_py_kl + batch_loss_sc_feat_kl
            batch_loss += batch_loss_elbo[i]

            total_train_loss["train/loss_elbo-%d"%(i+1)].append(batch_loss_elbo[i].item())
            total_train_loss["train/loss_px-%d"%(i+1)].append(batch_loss_px[i].item())
            total_train_loss["train/loss_qy_py-%d"%(i+1)].append(batch_loss_qy_py[i].item())
            total_train_loss["train/loss_qy_py_err-%d"%(i+1)].append(batch_loss_qy_py_err[i].item())
            total_train_loss["train/loss_qz_pz-%d"%(i+1)].append(batch_loss_qz_pz[i].item())
            total_train_loss["train/loss_qy_py_e-%d"%(i+1)].append(batch_loss_qy_py_e[i].item())
            total_train_loss["train/loss_qy_py_err_e-%d"%(i+1)].append(batch_loss_qy_py_err_e[i].item())
            total_train_loss["train/loss_qz_pz_e-%d"%(i+1)].append(batch_loss_qz_pz_e[i].item())
            total_train_loss["train/loss_sc_z-%d"%(i+1)].append(batch_loss_sc_z[i].item())
            if i > 0:
                total_train_loss["train/loss_cossim-%d"%(i+1)].append(batch_loss_lat_cossim[i].item())
                total_train_loss["train/loss_rmse-%d"%(i+1)].append(batch_loss_lat_rmse[i].item())
                loss_lat_cossim[i].append(batch_loss_lat_cossim[i].item())
                loss_lat_rmse[i].append(batch_loss_lat_rmse[i].item())
            else:
                total_train_loss["train/loss_sc_feat_in"].append(batch_loss_sc_feat_in.item())
                total_train_loss["train/loss_sc_feat_magsp_in"].append(batch_loss_sc_feat_magsp_in.item())
                loss_sc_feat_in.append(batch_loss_sc_feat_in.item())
                loss_sc_feat_magsp_in.append(batch_loss_sc_feat_magsp_in.item())
            total_train_loss["train/loss_sc_feat-%d"%(i+1)].append(batch_loss_sc_feat[i].item())
            total_train_loss["train/loss_sc_feat_magsp-%d"%(i+1)].append(batch_loss_sc_feat_magsp[i].item())
            loss_elbo[i].append(batch_loss_elbo[i].item())
            loss_px[i].append(batch_loss_px[i].item())
            loss_qy_py[i].append(batch_loss_qy_py[i].item())
            loss_qy_py_err[i].append(batch_loss_qy_py_err[i].item())
            loss_qz_pz[i].append(batch_loss_qz_pz[i].item())
            loss_qy_py_e[i].append(batch_loss_qy_py_e[i].item())
            loss_qy_py_err_e[i].append(batch_loss_qy_py_err_e[i].item())
            loss_qz_pz_e[i].append(batch_loss_qz_pz_e[i].item())
            loss_sc_z[i].append(batch_loss_sc_z[i].item())
            loss_sc_feat[i].append(batch_loss_sc_feat[i].item())
            loss_sc_feat_magsp[i].append(batch_loss_sc_feat_magsp[i].item())
            ## in-domain reconst.
            total_train_loss["train/loss_laplace-%d"%(i+1)].append(batch_loss_laplace[i].item())
            total_train_loss["train/loss_melsp-%d"%(i+1)].append(batch_loss_melsp[i].item())
            total_train_loss["train/loss_melsp_dB-%d"%(i+1)].append(batch_loss_melsp_dB[i].item())
            total_train_loss["train/loss_magsp-%d"%(i+1)].append(batch_loss_magsp[i].item())
            total_train_loss["train/loss_magsp_dB-%d"%(i+1)].append(batch_loss_magsp_dB[i].item())
            loss_laplace[i].append(batch_loss_laplace[i].item())
            loss_melsp[i].append(batch_loss_melsp[i].item())
            loss_melsp_dB[i].append(batch_loss_melsp_dB[i].item())
            loss_magsp[i].append(batch_loss_magsp[i].item())
            loss_magsp_dB[i].append(batch_loss_magsp_dB[i].item())
            ## conversion
            if i % 2 == 0:
                total_train_loss["train/loss_sc_feat_cv-%d"%(i+1)].append(batch_loss_sc_feat_cv[i//2].item())
                total_train_loss["train/loss_sc_feat_magsp_cv-%d"%(i+1)].append(batch_loss_sc_feat_magsp_cv[i//2].item())
                total_train_loss["train/loss_laplace_cv-%d"%(i+1)].append(batch_loss_laplace_cv[i//2].item())
                total_train_loss["train/loss_melsp_cv-%d"%(i+1)].append(batch_loss_melsp_cv[i//2].item())
                total_train_loss["train/loss_magsp_cv-%d"%(i+1)].append(batch_loss_magsp_cv[i//2].item())
                loss_sc_feat_cv[i//2].append(batch_loss_sc_feat_cv[i//2].item())
                loss_sc_feat_magsp_cv[i//2].append(batch_loss_sc_feat_magsp_cv[i//2].item())
                loss_laplace_cv[i//2].append(batch_loss_laplace_cv[i//2].item())
                loss_melsp_cv[i//2].append(batch_loss_melsp_cv[i//2].item())
                loss_magsp_cv[i//2].append(batch_loss_magsp_cv[i//2].item())

        optimizer.zero_grad()
        batch_loss.backward()
        flag = False
        explode_model = ""
        for name, param in model_encoder_melsp.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    flag = True
                    explode_model = "enc_melsp"
                    break
        if not flag:
            for name, param in model_encoder_excit.named_parameters():
                if param.requires_grad:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        flag = True
                        explode_model = "enc_excit"
                        break
        if not flag:
            for name, param in model_decoder_melsp.named_parameters():
                if param.requires_grad:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        flag = True
                        explode_model = "dec_melsp"
                        break
        if not flag:
            for name, param in model_spk.named_parameters():
                if param.requires_grad:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        flag = True
                        explode_model = "spk"
                        break
        if not flag and args.spkidtr_dim > 0:
            for name, param in model_spkidtr.named_parameters():
                if param.requires_grad:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        flag = True
                        explode_model = "spkidtr"
                        break
        if flag:
            logging.info("explode grad %s" % (explode_model))
            optimizer.zero_grad()
            text_log = "batch loss [%d] %d %d %d %d %.3f %.3f " % (c_idx+1, x_ss, x_bs, f_ss, f_bs, batch_loss_sc_feat_in.item(), batch_loss_sc_feat_magsp_in.item())
            for i in range(args.n_half_cyc):
                if i == 0:
                    text_log += "[%ld] %.3f , %.3f ; %.3f %.3f %% %.3f , %.3f %.3f %% %.3f ; %.3f , %.3f %.3f , %.3f %.3f ; " \
                        "%.3f %.3f , %.3f %.3f %.3f dB , %.3f %.3f %.3f dB ; %.3f %.3f , %.3f %.3f ; %.3f %.3f %% %.3f %.3f %% %.3f %.3f %% , %.3f %.3f , %.3f %.3f ; " % (i+1,
                        batch_loss_elbo[i].item(), batch_loss_px[i].item(),
                            batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(), batch_loss_qz_pz[i].item(),
                            batch_loss_qy_py_e[i].item(), batch_loss_qy_py_err_e[i].item(), batch_loss_qz_pz_e[i].item(),
                            batch_loss_sc_z[i].item(), batch_loss_sc_feat[i].item(), batch_loss_sc_feat_cv[i//2].item(), batch_loss_sc_feat_magsp[i].item(), batch_loss_sc_feat_magsp_cv[i//2].item(),
                                batch_loss_laplace[i].item(), batch_loss_laplace_cv[i//2].item(), batch_loss_melsp[i].item(), batch_loss_melsp_cv[i//2].item(), batch_loss_melsp_dB[i].item(),
                                batch_loss_magsp[i].item(), batch_loss_magsp_cv[i//2].item(), batch_loss_magsp_dB[i].item(),
                                batch_loss_seg_conv[i], batch_loss_conv_sc[i], batch_loss_h[i], batch_loss_mid_smpl[i],
                                batch_loss_ce_avg[i], batch_loss_err_avg[i], batch_loss_ce_c_avg[i], batch_loss_err_c_avg[i], batch_loss_ce_f_avg[i], batch_loss_err_f_avg[i],
                                batch_loss_fro_avg[i], batch_loss_l1_avg[i], batch_loss_fro_fb[i], batch_loss_l1_fb[i])
                else:
                    text_log += "[%ld] %.3f , %.3f ; %.3f %.3f , %.3f %.3f %% %.3f , %.3f %.3f %% %.3f ; "\
                        "%.3f , %.3f , %.3f ; %.3f , %.3f %.3f dB , %.3f %.3f dB ; %.3f %.3f , %.3f %.3f ; %.3f %.3f %% %.3f %.3f %% %.3f %.3f %% , %.3f %.3f , %.3f %.3f ; " % (i+1,
                        batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_lat_cossim[i].item(), batch_loss_lat_rmse[i].item(),
                            batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(), batch_loss_qz_pz[i].item(),
                            batch_loss_qy_py_e[i].item(), batch_loss_qy_py_err_e[i].item(), batch_loss_qz_pz_e[i].item(),
                                batch_loss_sc_z[i].item(), batch_loss_sc_feat[i].item(), batch_loss_sc_feat_magsp[i].item(),
                                    batch_loss_laplace[i].item(), batch_loss_melsp[i].item(), batch_loss_melsp_dB[i].item(), batch_loss_magsp[i].item(), batch_loss_magsp_dB[i].item(),
                                    batch_loss_seg_conv[i], batch_loss_conv_sc[i], batch_loss_h[i], batch_loss_mid_smpl[i],
                                    batch_loss_ce_avg[i], batch_loss_err_avg[i], batch_loss_ce_c_avg[i], batch_loss_err_c_avg[i], batch_loss_ce_f_avg[i], batch_loss_err_f_avg[i],
                                    batch_loss_fro_avg[i], batch_loss_l1_avg[i], batch_loss_fro_fb[i], batch_loss_l1_fb[i])
                for j in range(args.n_bands):
                    text_log += "[%d-%d] %.3f %.3f %% %.3f %.3f %% , %.3f %.3f " % (i+1, j+1,
                        batch_loss_ce[i][j], batch_loss_err[i][j], batch_loss_ce_f[i][j], batch_loss_err_f[i][j],
                            batch_loss_fro[i][j], batch_loss_l1[i][j])
                text_log += ";; "
            logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
            continue
        torch.nn.utils.clip_grad_norm_(model_encoder_melsp.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(model_encoder_excit.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(model_decoder_melsp.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(model_spk.parameters(), 10)
        if args.spkidtr_dim > 0:
            torch.nn.utils.clip_grad_norm_(model_spkidtr.parameters(), 10)
        optimizer.step()

        with torch.no_grad():
            if idx_stage < args.n_stage-1 and iter_idx + 1 == t_starts[idx_stage+1]:
                idx_stage += 1
            if idx_stage > 0:
                sparsify(model_encoder_melsp, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                sparsify(model_encoder_excit, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                sparsify(model_decoder_melsp, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
            else:
                sparsify(model_encoder_melsp, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])
                sparsify(model_encoder_excit, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])
                sparsify(model_decoder_melsp, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])

        text_log = "batch loss [%d] %d %d %d %d %.3f %.3f " % (c_idx+1, x_ss, x_bs, f_ss, f_bs, batch_loss_sc_feat_in.item(), batch_loss_sc_feat_magsp_in.item())
        for i in range(args.n_half_cyc):
            if i == 0:
                text_log += "[%ld] %.3f , %.3f ; %.3f %.3f %% %.3f , %.3f %.3f %% %.3f ; %.3f , %.3f %.3f , %.3f %.3f ; " \
                    "%.3f %.3f , %.3f %.3f %.3f dB , %.3f %.3f %.3f dB ; %.3f %.3f , %.3f %.3f ; %.3f %.3f %% %.3f %.3f %% %.3f %.3f %% , %.3f %.3f , %.3f %.3f ; " % (i+1,
                    batch_loss_elbo[i].item(), batch_loss_px[i].item(),
                        batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(), batch_loss_qz_pz[i].item(),
                        batch_loss_qy_py_e[i].item(), batch_loss_qy_py_err_e[i].item(), batch_loss_qz_pz_e[i].item(),
                        batch_loss_sc_z[i].item(), batch_loss_sc_feat[i].item(), batch_loss_sc_feat_cv[i//2].item(), batch_loss_sc_feat_magsp[i].item(), batch_loss_sc_feat_magsp_cv[i//2].item(),
                            batch_loss_laplace[i].item(), batch_loss_laplace_cv[i//2].item(), batch_loss_melsp[i].item(), batch_loss_melsp_cv[i//2].item(), batch_loss_melsp_dB[i].item(),
                            batch_loss_magsp[i].item(), batch_loss_magsp_cv[i//2].item(), batch_loss_magsp_dB[i].item(),
                            batch_loss_seg_conv[i], batch_loss_conv_sc[i], batch_loss_h[i], batch_loss_mid_smpl[i],
                            batch_loss_ce_avg[i], batch_loss_err_avg[i], batch_loss_ce_c_avg[i], batch_loss_err_c_avg[i], batch_loss_ce_f_avg[i], batch_loss_err_f_avg[i],
                            batch_loss_fro_avg[i], batch_loss_l1_avg[i], batch_loss_fro_fb[i], batch_loss_l1_fb[i])
            else:
                text_log += "[%ld] %.3f , %.3f ; %.3f %.3f , %.3f %.3f %% %.3f , %.3f %.3f %% %.3f ; "\
                    "%.3f , %.3f , %.3f ; %.3f , %.3f %.3f dB , %.3f %.3f dB ; %.3f %.3f , %.3f %.3f ; %.3f %.3f %% %.3f %.3f %% %.3f %.3f %% , %.3f %.3f , %.3f %.3f ; " % (i+1,
                    batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_lat_cossim[i].item(), batch_loss_lat_rmse[i].item(),
                        batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(), batch_loss_qz_pz[i].item(),
                        batch_loss_qy_py_e[i].item(), batch_loss_qy_py_err_e[i].item(), batch_loss_qz_pz_e[i].item(),
                            batch_loss_sc_z[i].item(), batch_loss_sc_feat[i].item(), batch_loss_sc_feat_magsp[i].item(),
                                batch_loss_laplace[i].item(), batch_loss_melsp[i].item(), batch_loss_melsp_dB[i].item(), batch_loss_magsp[i].item(), batch_loss_magsp_dB[i].item(),
                                batch_loss_seg_conv[i], batch_loss_conv_sc[i], batch_loss_h[i], batch_loss_mid_smpl[i],
                                batch_loss_ce_avg[i], batch_loss_err_avg[i], batch_loss_ce_c_avg[i], batch_loss_err_c_avg[i], batch_loss_ce_f_avg[i], batch_loss_err_f_avg[i],
                                batch_loss_fro_avg[i], batch_loss_l1_avg[i], batch_loss_fro_fb[i], batch_loss_l1_fb[i])
            for j in range(args.n_bands):
                text_log += "[%d-%d] %.3f %.3f %% %.3f %.3f %% , %.3f %.3f " % (i+1, j+1,
                    batch_loss_ce[i][j], batch_loss_err[i][j], batch_loss_ce_f[i][j], batch_loss_err_f[i][j],
                        batch_loss_fro[i][j], batch_loss_l1[i][j])
            text_log += ";; "
        logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
        iter_idx += 1
        #if iter_idx % args.save_interval_iter == 0:
        #    logging.info('save iter:%d' % (iter_idx))
        #    save_checkpoint(args.expdir, model_encoder, model_decoder, \
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


    logging.info("Maximum step is reached, please check the development optimum index, or continue training by increasing maximum step.")


if __name__ == "__main__":
    main()
