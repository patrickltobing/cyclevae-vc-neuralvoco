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
from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER
from vcneuvoco import GRU_EXCIT_DECODER, SPKID_TRANSFORM_LAYER, GRU_SPK
from vcneuvoco import GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF, encode_mu_law

import torch_optimizer as optim

from dataset import FeatureDatasetCycMceplf0WavVAE, FeatureDatasetEvalCycMceplf0WavVAE, padding

from dtw_c import dtw_c as dtw

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)


def train_generator(dataloader, device, batch_size, upsampling_factor, n_cv, limit_count=None, n_bands=10):
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
            max_slen = np.max(slens) ## get max samples length
            xs_c = batch['x_c'][:,:max_slen].to(device)
            xs_f = batch['x_f'][:,:max_slen].to(device)
            flens = batch['flen'].data.numpy()
            max_flen = np.max(flens) ## get max frames length
            feat = batch['feat'][:,:max_flen].to(device)
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
                    xs_c = torch.LongTensor(np.delete(xs_c.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    xs_f = torch.LongTensor(np.delete(xs_f.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    slens = np.delete(slens, del_index_utt, axis=0)
                    flens = np.delete(flens, del_index_utt, axis=0)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc = torch.LongTensor(np.delete(sc.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    for j in range(n_cv):
                        sc_cv[j] = torch.LongTensor(np.delete(sc_cv[j].cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                        feat_cv[j] = torch.FloatTensor(np.delete(feat_cv[j].cpu().data.numpy(), del_index_utt, axis=0)).to(device)
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
                yield xs_c, xs_f, feat, sc, sc_cv, feat_cv, c_idx, idx, featfiles, x_bs, x_ss, f_bs, f_ss, slens, flens, \
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

        yield [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


def eval_generator(dataloader, device, batch_size, upsampling_factor, limit_count=None, n_bands=10):
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
            max_flen = np.max(flens) ## get max frames length
            max_flen_trg = np.max(flens_trg)
            max_flen_spc_src = np.max(flens_spc_src)
            max_flen_spc_src_trg = np.max(flens_spc_src_trg)
            xs_c = batch['x_c'][:,:max_slen].to(device)
            xs_f = batch['x_f'][:,:max_slen].to(device)
            feat = batch['h_src'][:,:max_flen].to(device)
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
                    #if flens_acc[i] >= f_bs:
                    #    slens_acc[i] -= delta
                    #    flens_acc[i] -= delta_frm
                    #else:
                    if flens_acc[i] <= 0:
                        del_index_utt.append(i)
                if len(del_index_utt) > 0:
                    slens = np.delete(slens, del_index_utt, axis=0)
                    flens = np.delete(flens, del_index_utt, axis=0)
                    flens_trg = np.delete(flens_trg, del_index_utt, axis=0)
                    flens_spc_src = np.delete(flens_spc_src, del_index_utt, axis=0)
                    flens_spc_src_trg = np.delete(flens_spc_src_trg, del_index_utt, axis=0)
                    xs_c = torch.LongTensor(np.delete(xs_c.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    xs_f = torch.LongTensor(np.delete(xs_f.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_trg = torch.FloatTensor(np.delete(feat_trg.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc = torch.LongTensor(np.delete(sc.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    sc_cv = torch.LongTensor(np.delete(sc_cv.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat_cv = torch.FloatTensor(np.delete(feat_cv.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
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
                yield xs_c, xs_f, feat, feat_trg, sc, sc_cv, feat_cv, c_idx, idx, featfiles, x_bs, x_ss, f_bs, f_ss, slens, flens, \
                    n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, file_src_trg_flag, spcidx_src, \
                        spcidx_src_trg, flens_spc_src, flens_spc_src_trg, feat_full, sc_full, sc_cv_full, \
                            idx_select, idx_select_full, slens_acc, flens_acc
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

        yield [], [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


def save_checkpoint(checkpoint_dir, model_encoder_melsp, model_decoder_melsp, model_encoder_excit, model_decoder_excit,
        model_spk, model_waveform, min_eval_loss_ce_avg, min_eval_loss_ce_avg_std, min_eval_loss_err_avg,
        min_eval_loss_err_avg_std, iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, iterations, model_spkidtr=None):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    model_encoder_melsp.cpu()
    model_decoder_melsp.cpu()
    model_encoder_excit.cpu()
    model_decoder_excit.cpu()
    model_spk.cpu()
    if model_spkidtr is not None:
        model_spkidtr.cpu()
    model_waveform.cpu()
    checkpoint = {
        "model_encoder_melsp": model_encoder_melsp.state_dict(),
        "model_decoder_melsp": model_decoder_melsp.state_dict(),
        "model_encoder_excit": model_encoder_excit.state_dict(),
        "model_decoder_excit": model_decoder_excit.state_dict(),
        "model_spk": model_spk.state_dict(),
        "model_waveform": model_waveform.state_dict(),
        "min_eval_loss_ce_avg": min_eval_loss_ce_avg,
        "min_eval_loss_ce_avg_std": min_eval_loss_ce_avg_std,
        "min_eval_loss_err_avg": min_eval_loss_err_avg,
        "min_eval_loss_err_avg_std": min_eval_loss_err_avg_std,
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
    model_decoder_melsp.cuda()
    model_encoder_excit.cuda()
    model_decoder_excit.cuda()
    model_spk.cuda()
    if model_spkidtr is not None:
        model_spkidtr.cuda()
    model_waveform.cuda()
    logging.info("%d-iter checkpoint created." % iterations)


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
    parser.add_argument("--waveforms",
                        type=str, help="directory or list of wav files")
    parser.add_argument("--waveforms_eval_list",
                        type=str, help="directory or list of evaluation wav files")
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--feats_eval_list", required=True,
                        type=str, help="directory or list of evaluation feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="directory or list of evaluation wav files")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    # network structure setting
    parser.add_argument("--upsampling_factor", default=120,
                        type=int, help="number of dimension of aux feats")
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
    parser.add_argument("--hidden_units_wave", default=384,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_units_wave_2", default=24,
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
    parser.add_argument("--kernel_size_lf0", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_lf0", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--kernel_size_wave", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_wave", default=1,
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
    parser.add_argument("--spkidtr_dim", default=2,
                        type=int, help="number of dimension of reduced one-hot spk-dim (if 0 not apply reduction)")
    parser.add_argument("--lpc", default=6,
                        type=int, help="kernel size of dilated causal convolution")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=8,
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
    parser.add_argument("--causal_conv_spk", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_dec", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_lf0", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_wave", default=False,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_enc", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_spk", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_dec", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_lf0", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_wave", default=0,
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
    parser.add_argument("--n_bands", default=10,
                        type=int, help="number of bands")
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
                        type=str, help="model path for cyclevae")
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
    wav_eval_src_list = args.waveforms_eval_list.split('@')
    feat_eval_src_list = args.feats_eval_list.split('@')
    assert(n_spk == len(feat_eval_src_list))
    assert(n_spk == len(wav_eval_src_list))

    args.cap_dim = read_hdf5(args.stats, "/mean_feat_mceplf0cap")[3:args.full_excit_dim].shape[0]

    # save args as conf
    args.string_path = "/log_1pmelmagsp"
    args.n_quantize = 1024
    args.cf_dim = int(np.sqrt(args.n_quantize))
    args.half_n_quantize = args.n_quantize // 2
    args.c_pad = args.half_n_quantize // args.cf_dim
    args.f_pad = args.half_n_quantize % args.cf_dim
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
        red_dim=args.mel_dim,
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
        red_dim=args.mel_dim,
        do_prob=args.do_prob)
    logging.info(model_decoder_excit)
    if (args.spkidtr_dim > 0):
        model_spkidtr = SPKID_TRANSFORM_LAYER(
            n_spk=n_spk,
            spkidtr_dim=args.spkidtr_dim)
        logging.info(model_spkidtr)
    else:
        model_spkidtr = None
    model_spk = GRU_SPK(
        n_spk=n_spk,
        feat_dim=args.lat_dim+args.lat_dim_e,
        hidden_units=32,
        kernel_size=args.kernel_size_spk,
        dilation_size=args.dilation_size_spk,
        causal_conv=args.causal_conv_spk,
        pad_first=True,
        right_size=args.right_size_spk,
        red_dim=args.mel_dim,
        do_prob=args.do_prob)
    logging.info(model_spk)
    model_waveform = GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF(
        feat_dim=n_spk*2+args.lat_dim+args.lat_dim_e,
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
        n_spk=n_spk,
        scale_in_flag=False,
        red_dim=args.mel_dim,
        do_prob=args.do_prob)
    logging.info(model_waveform)
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_l1 = torch.nn.L1Loss(reduction='none')
    criterion_l2 = torch.nn.MSELoss(reduction='none')

    # send to gpu
    if torch.cuda.is_available():
        model_encoder_melsp.cuda()
        model_decoder_melsp.cuda()
        model_encoder_excit.cuda()
        model_decoder_excit.cuda()
        model_spk.cuda()
        if (args.spkidtr_dim > 0):
            model_spkidtr.cuda()
        model_waveform.cuda()
        criterion_ce.cuda()
        criterion_l1.cuda()
        criterion_l2.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)

    model_encoder_melsp.train()
    model_decoder_melsp.train()
    model_encoder_excit.train()
    model_decoder_excit.train()
    model_spk.train()
    if (args.spkidtr_dim > 0):
        model_spkidtr.train()
    model_waveform.train()

    parameters = filter(lambda p: p.requires_grad, model_encoder_melsp.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder_melsp): %.3f million' % parameters)
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
    if (args.spkidtr_dim > 0):
        parameters = filter(lambda p: p.requires_grad, model_spkidtr.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        logging.info('Trainable Parameters (spkidtr): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_waveform.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (waveform): %.3f million' % parameters)

    for param in model_encoder_melsp.parameters():
        param.requires_grad = False
    for param in model_decoder_melsp.parameters():
        param.requires_grad = False
    for param in model_encoder_excit.parameters():
        param.requires_grad = False
    for param in model_decoder_excit.parameters():
        param.requires_grad = False
    for param in model_spk.parameters():
        param.requires_grad = False
    for param in model_waveform.parameters():
        param.requires_grad = True

    module_list = list(model_waveform.in_red.parameters())
    module_list += list(model_waveform.conv.parameters()) + list(model_waveform.conv_s_c.parameters())
    module_list += list(model_waveform.embed_c_wav.parameters()) + list(model_waveform.embed_f_wav.parameters())
    module_list += list(model_waveform.gru.parameters())
    module_list += list(model_waveform.gru_2.parameters()) + list(model_waveform.out.parameters())
    module_list += list(model_waveform.gru_f.parameters()) + list(model_waveform.out_f.parameters())

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
        model_decoder_melsp.load_state_dict(checkpoint["model_decoder_melsp"])
        model_encoder_excit.load_state_dict(checkpoint["model_encoder_excit"])
        model_decoder_excit.load_state_dict(checkpoint["model_decoder_excit"])
        model_spk.load_state_dict(checkpoint["model_spk"])
        if (args.spkidtr_dim > 0):
            model_spkidtr.load_state_dict(checkpoint["model_spkidtr"])
        model_waveform.load_state_dict(checkpoint["model_waveform"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)

    def zero_wav_pad(x): return padding(x, args.pad_len*(args.upsampling_factor // args.n_bands), value=args.half_n_quantize)
    def zero_feat_pad(x): return padding(x, args.pad_len, value=None)
    pad_wav_transform = transforms.Compose([zero_wav_pad])
    pad_feat_transform = transforms.Compose([zero_feat_pad])

    wav_transform = transforms.Compose([lambda x: encode_mu_law(x, args.n_quantize)])

    n_rec = args.n_half_cyc + args.n_half_cyc%2
    n_cv = int(args.n_half_cyc/2+args.n_half_cyc%2)

    stats_list = args.stats_list.split('@')
    assert(n_spk == len(stats_list))

    if os.path.isdir(args.waveforms):
        filenames = sorted(find_files(args.waveforms, "*.wav", use_dir_name=False))
        wav_list = [args.waveforms + "/" + filename for filename in filenames]
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)
    if os.path.isdir(args.feats):
        feat_list = [args.feats + "/" + filename for filename in filenames]
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)
    logging.info("number of training data = %d." % len(feat_list))
    dataset = FeatureDatasetCycMceplf0WavVAE(feat_list, pad_feat_transform, spk_list, stats_list,
                    args.n_half_cyc, args.string_path, excit_dim=args.full_excit_dim, wav_list=wav_list,
                        upsampling_factor=args.upsampling_factor, pad_wav_transform=pad_wav_transform, wav_transform=wav_transform,
                            n_bands=args.n_bands, cf_dim=args.cf_dim, pad_left=model_waveform.pad_left, pad_right=model_waveform.pad_right)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_utt, shuffle=True, num_workers=args.n_workers)
    #generator = train_generator(dataloader, device, args.batch_size, args.upsampling_factor, n_cv, limit_count=20, n_bands=args.n_bands)
    #generator = train_generator(dataloader, device, args.batch_size, args.upsampling_factor, n_cv, limit_count=1, n_bands=args.n_bands)
    generator = train_generator(dataloader, device, args.batch_size, args.upsampling_factor, n_cv, limit_count=None, n_bands=args.n_bands)

    # define generator evaluation
    wav_list_eval_src_list = [None]*n_spk
    for i in range(n_spk):
        if os.path.isdir(wav_eval_src_list[i]):
            wav_list_eval_src_list[i] = sorted(find_files(wav_eval_src_list[i], "*.wav", use_dir_name=False))
        elif os.path.isfile(wav_eval_src_list[i]):
            wav_list_eval_src_list[i] = read_txt(wav_eval_src_list[i])
        else:
            logging.error("%s should be directory or list." % (wav_eval_src_list[i]))
            sys.exit(1)
    feat_list_eval_src_list = [None]*n_spk
    for i in range(n_spk):
        if os.path.isdir(feat_eval_src_list[i]):
            feat_list_eval_src_list[i] = sorted(find_files(feat_eval_src_list[i], "*.h5", use_dir_name=False))
        elif os.path.isfile(feat_eval_src_list[i]):
            feat_list_eval_src_list[i] = read_txt(feat_eval_src_list[i])
        else:
            logging.error("%s should be directory or list." % (feat_eval_src_list[i]))
            sys.exit(1)
    assert len(wav_list_eval_src_list[0]) == len(feat_list_eval_src_list[0])
    dataset_eval = FeatureDatasetEvalCycMceplf0WavVAE(feat_list_eval_src_list, pad_feat_transform, spk_list,
                    stats_list, args.string_path, excit_dim=args.full_excit_dim, upsampling_factor=args.upsampling_factor,
                        wav_list=wav_list_eval_src_list, pad_wav_transform=pad_wav_transform, wav_transform=wav_transform,
                            n_bands=args.n_bands, cf_dim=args.cf_dim, pad_left=model_waveform.pad_left, pad_right=model_waveform.pad_right)
    n_eval_data = len(dataset_eval.file_list_src)
    logging.info("number of evaluation data = %d." % n_eval_data)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size_utt_eval, shuffle=False, num_workers=args.n_workers)
    #generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=1, n_bands=args.n_bands)
    generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=None, n_bands=args.n_bands)

    writer = SummaryWriter(args.expdir)
    total_train_loss = defaultdict(list)
    total_eval_loss = defaultdict(list)

    gv_mean = [None]*n_spk
    for i in range(n_spk):
        gv_mean[i] = read_hdf5(stats_list[i], "/gv_melsp_mean")

    density_deltas_ = args.densities.split('-')
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
    lf0_pad_left = model_decoder_excit.pad_left
    lf0_pad_right = model_decoder_excit.pad_right
    logging.info(f'lf0_pad_left: {lf0_pad_left}')
    logging.info(f'lf0_pad_right: {lf0_pad_right}')
    wav_pad_left = model_waveform.pad_left
    wav_pad_right = model_waveform.pad_right
    logging.info(f'wav_pad_left: {wav_pad_left}')
    logging.info(f'wav_pad_right: {wav_pad_right}')
    dec_pad_left = model_decoder_melsp.pad_left
    dec_pad_right = model_decoder_melsp.pad_right
    logging.info(f'dec_pad_left: {dec_pad_left}')
    logging.info(f'dec_pad_right: {dec_pad_right}')
    dec_enc_pad_left = dec_pad_left + enc_pad_left
    dec_enc_pad_right = dec_pad_right + enc_pad_right
    first_pad_left = (enc_pad_left + spk_pad_left + lf0_pad_left + wav_pad_left + dec_pad_left)*args.n_half_cyc
    first_pad_right = (enc_pad_right + spk_pad_right + lf0_pad_right + wav_pad_right + dec_pad_right)*args.n_half_cyc
    logging.info(f'first_pad_left: {first_pad_left}')
    logging.info(f'first_pad_right: {first_pad_right}')
    outpad_lefts = [None]*args.n_half_cyc*5
    outpad_rights = [None]*args.n_half_cyc*5
    outpad_lefts[0] = first_pad_left-enc_pad_left
    outpad_rights[0] = first_pad_right-enc_pad_right
    for i in range(1,args.n_half_cyc*5):
        if i % 5 == 4:
            outpad_lefts[i] = outpad_lefts[i-1]-dec_pad_left
            outpad_rights[i] = outpad_rights[i-1]-dec_pad_right
        elif i % 5 == 3:
            outpad_lefts[i] = outpad_lefts[i-1]-wav_pad_left
            outpad_rights[i] = outpad_rights[i-1]-wav_pad_right
        elif i % 5 == 2:
            outpad_lefts[i] = outpad_lefts[i-1]-lf0_pad_left
            outpad_rights[i] = outpad_rights[i-1]-lf0_pad_right
        elif i % 5 == 1:
            outpad_lefts[i] = outpad_lefts[i-1]-spk_pad_left
            outpad_rights[i] = outpad_rights[i-1]-spk_pad_right
        else:
            outpad_lefts[i] = outpad_lefts[i-1]-enc_pad_left
            outpad_rights[i] = outpad_rights[i-1]-enc_pad_right
    logging.info(outpad_lefts)
    logging.info(outpad_rights)
    batch_feat_in = [None]*args.n_half_cyc*5
    batch_sc_in = [None]*args.n_half_cyc*5
    batch_sc_cv_in = [None]*n_cv*4
    total = 0
    iter_count = 0
    batch_excit_cv = [None]*n_cv
    z = [None]*n_rec
    z_e = [None]*n_rec
    batch_melsp_rec = [None]*n_rec
    batch_melsp_cv = [None]*n_cv
    batch_lf0_rec = [None]*n_rec
    batch_lf0_cv = [None]*n_cv
    batch_x_c_output = [None]*n_rec
    batch_x_f_output = [None]*n_rec
    h_z = [None]*n_rec
    h_z_e = [None]*n_rec
    h_spk = [None]*n_rec
    h_spk_cv = [None]*n_cv
    h_melsp = [None]*n_rec
    h_lf0 = [None]*n_rec
    h_melsp_cv = [None]*n_cv
    h_lf0_cv = [None]*n_cv
    h_x = [None]*n_rec
    h_x_2 = [None]*n_rec
    h_f = [None]*n_rec
    loss_ce_avg = [None]*n_rec
    loss_err_avg = [None]*n_rec
    loss_ce_c_avg = [None]*n_rec
    loss_err_c_avg = [None]*n_rec
    loss_ce_f_avg = [None]*n_rec
    loss_err_f_avg = [None]*n_rec
    loss_ce = [None]*n_rec
    loss_err = [None]*n_rec
    loss_ce_f = [None]*n_rec
    loss_err_f = [None]*n_rec
    loss_uv_src_trg = []
    loss_f0_src_trg = []
    loss_uvcap_src_trg = []
    loss_cap_src_trg = []
    loss_lat_dist_rmse = []
    loss_lat_dist_cossim = []
    for i in range(n_rec):
        loss_ce[i] = [None]*args.n_bands
        loss_err[i] = [None]*args.n_bands
        loss_ce_f[i] = [None]*args.n_bands
        loss_err_f[i] = [None]*args.n_bands
        loss_ce_avg[i] = []
        loss_err_avg[i] = []
        loss_ce_c_avg[i] = []
        loss_err_c_avg[i] = []
        loss_ce_f_avg[i] = []
        loss_err_f_avg[i] = []
        for j in range(args.n_bands):
            loss_ce[i][j] = []
            loss_err[i][j] = []
            loss_ce_f[i][j] = []
            loss_err_f[i][j] = []
    batch_loss_ce_avg = [None]*n_rec
    batch_loss_err_avg = [None]*n_rec
    batch_loss_ce_c_avg = [None]*n_rec
    batch_loss_err_c_avg = [None]*n_rec
    batch_loss_ce_f_avg = [None]*n_rec
    batch_loss_err_f_avg = [None]*n_rec
    batch_loss_ce = [None]*n_rec
    batch_loss_err = [None]*n_rec
    batch_loss_ce_f = [None]*n_rec
    batch_loss_err_f = [None]*n_rec
    batch_loss_ce_select = [None]*n_rec
    batch_loss_err_select = [None]*n_rec
    batch_loss_ce_f_select = [None]*n_rec
    batch_loss_err_f_select = [None]*n_rec
    batch_loss_ce_select_avg = [None]*n_rec
    batch_loss_err_select_avg = [None]*n_rec
    batch_loss_ce_c_select_avg = [None]*n_rec
    batch_loss_err_c_select_avg = [None]*n_rec
    batch_loss_ce_f_select_avg = [None]*n_rec
    batch_loss_err_f_select_avg = [None]*n_rec
    for i in range(n_rec):
        batch_loss_ce[i] = [None]*args.n_bands
        batch_loss_err[i] = [None]*args.n_bands
        batch_loss_ce_f[i] = [None]*args.n_bands
        batch_loss_err_f[i] = [None]*args.n_bands
    n_half_cyc_eval = min(2,args.n_half_cyc)
    n_rec_eval = n_half_cyc_eval + n_half_cyc_eval%2
    n_cv_eval = int(n_half_cyc_eval/2+n_half_cyc_eval%2)
    first_pad_left_eval = (enc_pad_left + spk_pad_left + lf0_pad_left + wav_pad_left + dec_pad_left)*n_half_cyc_eval
    first_pad_right_eval = (enc_pad_right + spk_pad_right + lf0_pad_right + wav_pad_right + dec_pad_right)*n_half_cyc_eval
    logging.info(f'first_pad_left_eval: {first_pad_left_eval}')
    logging.info(f'first_pad_right_eval: {first_pad_right_eval}')
    first_pad_left_eval_utt_dec = spk_pad_left + lf0_pad_left
    first_pad_right_eval_utt_dec = spk_pad_right + lf0_pad_right
    logging.info(f'first_pad_left_eval_utt_dec: {first_pad_left_eval_utt_dec}')
    logging.info(f'first_pad_right_eval_utt_dec: {first_pad_right_eval_utt_dec}')
    first_pad_left_eval_utt = enc_pad_left + first_pad_left_eval_utt_dec
    first_pad_right_eval_utt = enc_pad_right + first_pad_right_eval_utt_dec
    logging.info(f'first_pad_left_eval_utt: {first_pad_left_eval_utt}')
    logging.info(f'first_pad_right_eval_utt: {first_pad_right_eval_utt}')
    eval_loss_ce_avg = [None]*n_half_cyc_eval
    eval_loss_ce_avg_std = [None]*n_half_cyc_eval
    eval_loss_err_avg = [None]*n_half_cyc_eval
    eval_loss_err_avg_std = [None]*n_half_cyc_eval
    eval_loss_ce_c_avg = [None]*n_half_cyc_eval
    eval_loss_ce_c_avg_std = [None]*n_half_cyc_eval
    eval_loss_err_c_avg = [None]*n_half_cyc_eval
    eval_loss_err_c_avg_std = [None]*n_half_cyc_eval
    eval_loss_ce_f_avg = [None]*n_half_cyc_eval
    eval_loss_ce_f_avg_std = [None]*n_half_cyc_eval
    eval_loss_err_f_avg = [None]*n_half_cyc_eval
    eval_loss_err_f_avg_std = [None]*n_half_cyc_eval
    eval_loss_ce = [None]*n_half_cyc_eval
    eval_loss_ce_std = [None]*n_half_cyc_eval
    eval_loss_err = [None]*n_half_cyc_eval
    eval_loss_err_std = [None]*n_half_cyc_eval
    eval_loss_ce_f = [None]*n_half_cyc_eval
    eval_loss_ce_f_std = [None]*n_half_cyc_eval
    eval_loss_err_f = [None]*n_half_cyc_eval
    eval_loss_err_f_std = [None]*n_half_cyc_eval
    for i in range(n_half_cyc_eval):
        eval_loss_ce[i] = [None]*args.n_bands
        eval_loss_ce_std[i] = [None]*args.n_bands
        eval_loss_err[i] = [None]*args.n_bands
        eval_loss_err_std[i] = [None]*args.n_bands
        eval_loss_ce_f[i] = [None]*args.n_bands
        eval_loss_ce_f_std[i] = [None]*args.n_bands
        eval_loss_err_f[i] = [None]*args.n_bands
        eval_loss_err_f_std[i] = [None]*args.n_bands
    min_eval_loss_ce_avg = [None]*n_half_cyc_eval
    min_eval_loss_ce_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_err_avg = [None]*n_half_cyc_eval
    min_eval_loss_err_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_ce_c_avg = [None]*n_half_cyc_eval
    min_eval_loss_ce_c_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_err_c_avg = [None]*n_half_cyc_eval
    min_eval_loss_err_c_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_ce_f_avg = [None]*n_half_cyc_eval
    min_eval_loss_ce_f_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_err_f_avg = [None]*n_half_cyc_eval
    min_eval_loss_err_f_avg_std = [None]*n_half_cyc_eval
    min_eval_loss_ce = [None]*n_half_cyc_eval
    min_eval_loss_ce_std = [None]*n_half_cyc_eval
    min_eval_loss_err = [None]*n_half_cyc_eval
    min_eval_loss_err_std = [None]*n_half_cyc_eval
    min_eval_loss_ce_f = [None]*n_half_cyc_eval
    min_eval_loss_ce_f_std = [None]*n_half_cyc_eval
    min_eval_loss_err_f = [None]*n_half_cyc_eval
    min_eval_loss_err_f_std = [None]*n_half_cyc_eval
    for i in range(n_half_cyc_eval):
        min_eval_loss_ce[i] = [None]*args.n_bands
        min_eval_loss_ce_std[i] = [None]*args.n_bands
        min_eval_loss_err[i] = [None]*args.n_bands
        min_eval_loss_err_std[i] = [None]*args.n_bands
        min_eval_loss_ce_f[i] = [None]*args.n_bands
        min_eval_loss_ce_f_std[i] = [None]*args.n_bands
        min_eval_loss_err_f[i] = [None]*args.n_bands
        min_eval_loss_err_f_std[i] = [None]*args.n_bands
    min_eval_loss_ce_avg[0] = 99999999.99
    min_eval_loss_ce_avg_std[0] = 99999999.99
    min_eval_loss_err_avg[0] = 99999999.99
    min_eval_loss_err_avg_std[0] = 99999999.99
    iter_idx = 0
    min_idx = -1
    change_min_flag = False
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        min_eval_loss_ce_avg[0] = checkpoint["min_eval_loss_ce_avg"]
        min_eval_loss_ce_avg_std[0] = checkpoint["min_eval_loss_ce_avg_std"]
        min_eval_loss_err_avg[0] = checkpoint["min_eval_loss_err_avg"]
        min_eval_loss_err_avg_std[0] = checkpoint["min_eval_loss_err_avg_std"]
        iter_idx = checkpoint["iter_idx"]
        min_idx = checkpoint["min_idx"]
    while idx_stage < args.n_stage-1 and iter_idx + 1 >= t_starts[idx_stage+1]:
        idx_stage += 1
        logging.info(idx_stage)
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        batch_x_c, batch_x_f, batch_feat, batch_sc, batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
            x_bs, x_ss, f_bs, f_ss, slens, flens, n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, idx_select, idx_select_full, slens_acc, flens_acc = next(generator)
        if c_idx < 0: # summarize epoch
            # save current epoch model
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # report current epoch
            text_log = "(EPOCH:%d) average optimization loss =" % (epoch_idx + 1)
            for i in range(n_half_cyc_eval):
                text_log += " [%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %%" % (i+1,
                        np.mean(loss_ce_avg[i]), np.std(loss_ce_avg[i]), np.mean(loss_err_avg[i]), np.std(loss_err_avg[i]), 
                            np.mean(loss_ce_c_avg[i]), np.std(loss_ce_c_avg[i]), np.mean(loss_err_c_avg[i]), np.std(loss_err_c_avg[i]), 
                                np.mean(loss_ce_f_avg[i]), np.std(loss_ce_f_avg[i]), np.mean(loss_err_f_avg[i]), np.std(loss_err_f_avg[i]))
                text_log += " ;"
                for j in range(args.n_bands):
                    text_log += " [%d-%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %%" % (i+1, j+1,
                            np.mean(loss_ce[i][j]), np.std(loss_ce[i][j]), np.mean(loss_err[i][j]), np.std(loss_err[i][j]), 
                                np.mean(loss_ce_f[i][j]), np.std(loss_ce_f[i][j]), np.mean(loss_err_f[i][j]), np.std(loss_err_f[i][j]))
                text_log += " ;;"
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            logging.info("estimated time until max. epoch = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * total))))
            # compute loss in evaluation data
            total = 0
            iter_count = 0
            loss_uv_src_trg = []
            loss_f0_src_trg = []
            loss_uvcap_src_trg = []
            loss_cap_src_trg = []
            loss_lat_dist_rmse = []
            loss_lat_dist_cossim = []
            for i in range(n_rec):
                loss_ce_avg[i] = []
                loss_err_avg[i] = []
                loss_ce_c_avg[i] = []
                loss_err_c_avg[i] = []
                loss_ce_f_avg[i] = []
                loss_err_f_avg[i] = []
                for j in range(args.n_bands):
                    loss_ce[i][j] = []
                    loss_err[i][j] = []
                    loss_ce_f[i][j] = []
                    loss_err_f[i][j] = []
            model_encoder_melsp.eval()
            model_decoder_melsp.eval()
            model_encoder_excit.eval()
            model_decoder_excit.eval()
            model_spk.eval()
            if args.spkidtr_dim > 0:
                model_spkidtr.eval()
            model_waveform.eval()
            for param in model_waveform.parameters():
                param.requires_grad = False
            pair_exist = False
            logging.info("Evaluation data")
            while True:
                with torch.no_grad():
                    start = time.time()
                    batch_x_c, batch_x_f, batch_feat_data, batch_feat_trg_data, batch_sc_data, \
                        batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
                        x_bs, x_ss, f_bs, f_ss, slens, flens, n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, src_trg_flag, \
                            spcidx_src, spcidx_src_trg, flens_spc_src, flens_spc_src_trg, \
                                batch_feat_data_full, batch_sc_data_full, batch_sc_cv_data_full, \
                                    idx_select, idx_select_full, slens_acc, flens_acc = next(generator_eval)
                    if c_idx < 0:
                        break

                    x_es = x_ss+x_bs
                    f_es = f_ss+f_bs
                    logging.info(f'{x_ss} {x_bs} {x_es} {max_slen} {f_ss} {f_bs} {f_es} {max_flen}')
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
                    else:
                        batch_x_c_prev = F.pad(batch_x_c[:,:x_es-1], (0, 0, 1, 0), "constant", args.c_pad)
                        batch_x_f_prev = F.pad(batch_x_f[:,:x_es-1], (0, 0, 1, 0), "constant", args.f_pad)
                        if args.lpc > 0:
                            batch_x_c_lpc = F.pad(batch_x_c[:,:x_es-1], (0, 0, args.lpc, 0), "constant", args.c_pad)
                            batch_x_f_lpc = F.pad(batch_x_f[:,:x_es-1], (0, 0, args.lpc, 0), "constant", args.f_pad)
                        batch_x_c = batch_x_c[:,:x_es]
                        batch_x_f = batch_x_f[:,:x_es]
                    # handle first pad for input features
                    flag_cv = True
                    i_cv = 0
                    i_cv_in = 0
                    f_ss_first_pad_left = f_ss-first_pad_left_eval
                    f_es_first_pad_right = f_es+first_pad_right_eval
                    i_end = n_half_cyc_eval*5
                    for i in range(i_end):
                        if i % 5 == 0: #enc
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
                        else: #spk/lf0/wav/dec
                            if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                                batch_sc_in[i] = batch_sc_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                                if flag_cv:
                                    batch_sc_cv_in[i_cv_in] = batch_sc_cv_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                                    i_cv_in += 1
                                    if i % 5 == 4:
                                        i_cv += 1
                                        flag_cv = False
                                else:
                                    if (i + 1) % 10 == 0:
                                        flag_cv = True
                            elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                                batch_sc_in[i] = F.pad(batch_sc_data[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                                    i_cv_in += 1
                                    if i % 5 == 4:
                                        i_cv += 1
                                        flag_cv = False
                                else:
                                    if (i + 1) % 10 == 0:
                                        flag_cv = True
                            elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                                diff_pad = f_es_first_pad_right - max_flen
                                batch_sc_in[i] = F.pad(batch_sc_data[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                                    i_cv_in += 1
                                    if i % 5 == 4:
                                        i_cv += 1
                                        flag_cv = False
                                else:
                                    if (i + 1) % 10 == 0:
                                        flag_cv = True
                            else: # pad left and right need additional replicate
                                diff_pad = f_es_first_pad_right - max_flen
                                batch_sc_in[i] = F.pad(batch_sc_data[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                                    i_cv_in += 1
                                    if i % 5 == 4:
                                        i_cv += 1
                                        flag_cv = False
                                else:
                                    if (i + 1) % 10 == 0:
                                        flag_cv = True
                            if i % 5 == 1:
                                f_ss_first_pad_left += spk_pad_left
                                f_es_first_pad_right -= spk_pad_right
                            elif i % 5 == 2:
                                f_ss_first_pad_left += lf0_pad_left
                                f_es_first_pad_right -= lf0_pad_right
                            elif i % 5 == 3:
                                f_ss_first_pad_left += wav_pad_left
                                f_es_first_pad_right -= wav_pad_right
                            elif i % 5 == 4:
                                f_ss_first_pad_left += dec_pad_left
                                f_es_first_pad_right -= dec_pad_right
                    batch_melsp = batch_feat_data[:,f_ss:f_es,args.full_excit_dim:]
                    batch_excit = batch_feat_data[:,f_ss:f_es,:args.full_excit_dim]
                    batch_melsp_data_full = batch_feat_data_full[:,:,args.full_excit_dim:]
                    batch_melsp_trg_data = batch_feat_trg_data[:,:,args.full_excit_dim:]
                    batch_excit_trg_data = batch_feat_trg_data[:,:,:args.full_excit_dim]
                    batch_excit_cv[0] = batch_feat_cv_data[:,f_ss:f_es]

                    if f_ss > 0:
                        idx_in = 0
                        i_cv_in = 0
                        for i in range(0,n_half_cyc_eval,2):
                            i_cv = i//2
                            j = i+1
                            if len(del_index_utt) > 0:
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
                                h_x[i] = torch.FloatTensor(np.delete(h_x[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_x_2[i] = torch.FloatTensor(np.delete(h_x_2[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_f[i] = torch.FloatTensor(np.delete(h_f[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp[i] = torch.FloatTensor(np.delete(h_melsp[i].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                h_melsp_cv[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv[i_cv].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                                if n_half_cyc_eval > 1:
                                    h_z[j] = torch.FloatTensor(np.delete(h_z[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_z_e[j] = torch.FloatTensor(np.delete(h_z_e[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_spk[j] = torch.FloatTensor(np.delete(h_spk[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_lf0[j] = torch.FloatTensor(np.delete(h_lf0[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_x[j] = torch.FloatTensor(np.delete(h_x[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_x_2[j] = torch.FloatTensor(np.delete(h_x_2[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    h_f[j] = torch.FloatTensor(np.delete(h_f[j].cpu().data.numpy(),
                                                                    del_index_utt, axis=1)).to(device)
                                    if n_half_cyc_eval > 2:
                                        h_melsp[j] = torch.FloatTensor(np.delete(h_melsp[j].cpu().data.numpy(),
                                                                        del_index_utt, axis=1)).to(device)
                            ## latent infer.
                            if i > 0:
                                idx_in += 1
                                i_cv_in += 1
                                cyc_rec_feat = batch_melsp_rec[i-1]
                                _, _, z[i], h_z[i] = model_encoder_melsp(cyc_rec_feat, outpad_right=outpad_rights[idx_in], h=h_z[i], sampling=False)
                                _, _, z_e[i], h_z_e[i] = model_encoder_excit(cyc_rec_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[i], sampling=False)
                            else:
                                _, _, z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z[i], sampling=False)
                                _, _, z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_e[i], sampling=False)
                            ## time-varying speaker conditionings
                            idx_in += 1
                            z_cat = torch.cat((z_e[i], z[i]), 2)
                            if args.spkidtr_dim > 0:
                                spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                                spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                                batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv])
                            else:
                                batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv])
                            ## excit reconstruction & conversion
                            idx_in += 1
                            i_cv_in += 1
                            if spk_pad_right > 0:
                                z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                                z_e[i] = z_e[i][:,spk_pad_left:-spk_pad_right]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:-spk_pad_right]
                            else:
                                z_cat = z_cat[:,spk_pad_left:]
                                z_e[i] = z_e[i][:,spk_pad_left:]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_lf0_rec[i], h_lf0[i] \
                                        = model_decoder_excit(z_e[i], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[i])
                                batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                        = model_decoder_excit(z_e[i], y=spk_cv_code_in, aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], h=h_lf0_cv[i_cv])
                            else:
                                batch_lf0_rec[i], h_lf0[i] \
                                        = model_decoder_excit(z_e[i], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[i])
                                batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                        = model_decoder_excit(z_e[i], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], h=h_lf0_cv[i_cv])
                            ## waveform reconstruction
                            idx_in += 1
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
                            if args.lpc > 0:
                                if args.spkidtr_dim > 0:
                                    batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                        = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                            spk_code=spk_code_in, spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                                h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                        = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                            spk_code=batch_sc_in[idx_in], spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                                h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                            else:
                                if args.spkidtr_dim > 0:
                                    batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                        = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                            spk_code=spk_code_in, spk_aux=batch_spk,
                                                h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                        = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                            spk_code=batch_sc_in[idx_in], spk_aux=batch_spk,
                                                h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                            ## melsp reconstruction & conversion
                            idx_in += 1
                            i_cv_in += 1
                            if wav_pad_right > 0:
                                z_cat = z_cat[:,wav_pad_left:-wav_pad_right]
                                e_in = batch_lf0_rec[i][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                                e_cv_in = batch_lf0_cv[i_cv][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,wav_pad_left:-wav_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,wav_pad_left:-wav_pad_right]
                                batch_spk = batch_spk[:,wav_pad_left:-wav_pad_right]
                                batch_spk_cv = batch_spk_cv[:,wav_pad_left:-wav_pad_right]
                            else:
                                z_cat = z_cat[:,wav_pad_left:]
                                e_in = batch_lf0_rec[i][:,wav_pad_left:,:args.excit_dim]
                                e_cv_in = batch_lf0_cv[i_cv][:,wav_pad_left:,:args.excit_dim]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,wav_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,wav_pad_left:]
                                batch_spk = batch_spk[:,wav_pad_left:]
                                batch_spk_cv = batch_spk_cv[:,wav_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                    e=e_in, outpad_right=outpad_rights[idx_in], h=h_melsp[i])
                                batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                                    e=e_cv_in, outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv])
                            else:
                                batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=e_in, outpad_right=outpad_rights[idx_in], h=h_melsp[i])
                                batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    e=e_cv_in, outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv])
                            ## cyclic reconstruction, latent infer.
                            if n_half_cyc_eval > 1:
                                idx_in += 1
                                cv_feat = batch_melsp_cv[i_cv]
                                _, _, z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z[j], sampling=False)
                                _, _, z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[j], sampling=False)
                                ## time-varying speaker conditionings
                                idx_in += 1
                                z_cat = torch.cat((z_e[j], z[j]), 2)
                                if args.spkidtr_dim > 0:
                                    if dec_enc_pad_right > 0:
                                        spk_code_in = spk_code_in[:,dec_enc_pad_left:-dec_enc_pad_right]
                                    else:
                                        spk_code_in = spk_code_in[:,dec_enc_pad_left:]
                                    batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j])
                                else:
                                    batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j])
                                ## excit reconstruction
                                idx_in += 1
                                if spk_pad_right > 0:
                                    z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                                    z_e[j] = z_e[j][:,spk_pad_left:-spk_pad_right]
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                                else:
                                    z_cat = z_cat[:,spk_pad_left:]
                                    z_e[j] = z_e[j][:,spk_pad_left:]
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,spk_pad_left:]
                                if args.spkidtr_dim > 0:
                                    batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[j])
                                else:
                                    batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[j])
                                ## waveform cyclic reconstruction
                                idx_in += 1
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
                                if args.lpc > 0:
                                    if args.spkidtr_dim > 0:
                                        batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                            = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                                 spk_code=spk_code_in, spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                                    h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                    else:
                                        batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                            = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                                 spk_code=batch_sc_in[idx_in], spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                                    h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                else:
                                    if args.spkidtr_dim > 0:
                                        batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                            = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                                 spk_code=spk_code_in, spk_aux=batch_spk,
                                                    h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                    else:
                                        batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                            = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                                 spk_code=batch_sc_in[idx_in], spk_aux=batch_spk,
                                                    h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                ## melsp cyclic reconstruction
                                if n_half_cyc_eval > 2:
                                    idx_in += 1
                                    if wav_pad_right > 0:
                                        z_cat = z_cat[:,wav_pad_left:-wav_pad_right]
                                        e_in = batch_lf0_rec[j][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                                        if args.spkidtr_dim > 0:
                                            spk_code_in = spk_code_in[:,wav_pad_left:-wav_pad_right]
                                        batch_spk = batch_spk[:,wav_pad_left:-wav_pad_right]
                                    else:
                                        z_cat = z_cat[:,wav_pad_left:]
                                        e_in = batch_lf0_rec[j][:,wav_pad_left:,:args.excit_dim]
                                        if args.spkidtr_dim > 0:
                                            spk_code_in = spk_code_in[:,wav_pad_left:]
                                        batch_spk = batch_spk[:,wav_pad_left:]
                                    if args.spkidtr_dim > 0:
                                        batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                                e=e_in, outpad_right=outpad_rights[idx_in], h=h_melsp[j])
                                    else:
                                        batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                                e=e_in, outpad_right=outpad_rights[idx_in], h=h_melsp[j])
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
                            trj_lat_src_e = trj_lat_src_e[:,spk_pad_left:-spk_pad_right]
                            trj_spk_code = trj_spk_code[:,spk_pad_left:-spk_pad_right]
                            trj_spk_cv_code = trj_spk_cv_code[:,spk_pad_left:-spk_pad_right]
                        else:
                            z_cat = z_cat[:,spk_pad_left:]
                            trj_lat_src_e = trj_lat_src_e[:,spk_pad_left:]
                            trj_spk_code = trj_spk_code[:,spk_pad_left:]
                            trj_spk_cv_code = trj_spk_cv_code[:,spk_pad_left:]
                        trj_src_src_uvlf0, _ = model_decoder_excit(trj_lat_src_e, y=trj_spk_code, aux=trj_spk)
                        trj_src_trg_uvlf0, _ = model_decoder_excit(trj_lat_src_e, y=trj_spk_cv_code, aux=trj_spk_cv)

                        if pair_flag:
                            if lf0_pad_right > 0:
                                trj_lat_src = z_cat[:,lf0_pad_left:-lf0_pad_right]
                                trj_src_trg_uvlf0 = trj_src_trg_uvlf0[:,lf0_pad_left:-lf0_pad_right]
                            else:
                                trj_lat_src = z_cat[:,lf0_pad_left:]
                                trj_src_trg_uvlf0 = trj_src_trg_uvlf0[:,lf0_pad_left:]
                            batch_melsp_trg_data_in = F.pad(batch_melsp_trg_data.transpose(1,2), (enc_pad_left,enc_pad_right), "replicate").transpose(1,2)
                            _, _, trj_lat_trg, _ = model_encoder_melsp(batch_melsp_trg_data_in,sampling=False)
                            _, _, trj_lat_trg_e, _ = model_encoder_excit(batch_melsp_trg_data_in, sampling=False)
                            trj_lat_trg = torch.cat((trj_lat_trg_e, trj_lat_trg), 2)

                            for k in range(n_batch_utt):
                                if src_trg_flag[k]:
                                    # spcidx lat
                                    trj_lat_src_ = np.array(torch.index_select(trj_lat_src[k],0,spcidx_src[k,:flens_spc_src[k]]).cpu().data.numpy(), dtype=np.float64)
                                    trj_lat_trg_ = np.array(torch.index_select(trj_lat_trg[k],0,spcidx_src_trg[k,:flens_spc_src_trg[k]]).cpu().data.numpy(), dtype=np.float64)
                                    # spcidx excit, trg
                                    trj_src_trg_uvlf0_ = torch.index_select(trj_src_trg_uvlf0[k],0,spcidx_src[k,:flens_spc_src[k]])
                                    trj_trg_uvlf0_ = torch.index_select(batch_excit_trg_data[k],0,spcidx_src_trg[k,:flens_spc_src_trg[k]])
                                    # excit feats
                                    trj_src_trg_uv = trj_src_trg_uvlf0_[:,0]
                                    trj_src_trg_f0 = torch.exp(trj_src_trg_uvlf0_[:,1])
                                    trj_src_trg_uvcap = trj_src_trg_uvlf0_[:,2]
                                    trj_src_trg_cap = -torch.exp(trj_src_trg_uvlf0_[:,3:])
                                    trj_trg_uv = trj_trg_uvlf0_[:,0]
                                    trj_trg_f0 = torch.exp(trj_trg_uvlf0_[:,1])
                                    trj_trg_uvcap = trj_trg_uvlf0_[:,2]
                                    trj_trg_cap = -torch.exp(trj_trg_uvlf0_[:,3:])
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
                                    _, twf_cossim_lat, batch_lat_cdist_srctrg2, _ = dtw.dtw_org_to_trg(\
                                        trj_lat_src_, trj_lat_trg_, mcd=0)
                                    # RMSE
                                    batch_lat_dist_rmse = (batch_lat_dist_srctrg1+batch_lat_dist_srctrg2)/2
                                    # Cos-sim
                                    batch_lat_dist_cossim = (batch_lat_cdist_srctrg1+batch_lat_cdist_srctrg2)/2
                                    # excit dtw
                                    twf_cossim_lat = torch.LongTensor(twf_cossim_lat[:,0]).cuda()
                                    trj_src_trg_uv = torch.index_select(trj_src_trg_uv,0,twf_cossim_lat)
                                    trj_src_trg_f0 = torch.index_select(trj_src_trg_f0,0,twf_cossim_lat)
                                    batch_uv_src_trg = torch.mean(100*torch.abs(trj_src_trg_uv-trj_trg_uv)).item()
                                    batch_f0_src_trg = torch.sqrt(torch.mean((trj_src_trg_f0-trj_trg_f0)**2)).item()
                                    trj_src_trg_uvcap = torch.index_select(trj_src_trg_uvcap,0,twf_cossim_lat)
                                    trj_src_trg_cap = torch.index_select(trj_src_trg_cap,0,twf_cossim_lat)
                                    batch_uvcap_src_trg = torch.mean(100*torch.abs(trj_src_trg_uvcap-trj_trg_uvcap)).item()
                                    batch_cap_src_trg = torch.mean(torch.sum(torch.abs(trj_src_trg_cap-trj_trg_cap), -1)).item()
                                    loss_uv_src_trg.append(batch_uv_src_trg)
                                    loss_f0_src_trg.append(batch_f0_src_trg)
                                    loss_uvcap_src_trg.append(batch_uvcap_src_trg)
                                    loss_cap_src_trg.append(batch_cap_src_trg)
                                    loss_lat_dist_rmse.append(batch_lat_dist_rmse)
                                    loss_lat_dist_cossim.append(batch_lat_dist_cossim)
                                    total_eval_loss["eval/loss_uv_src_trg"].append(batch_uv_src_trg)
                                    total_eval_loss["eval/loss_f0_src_trg"].append(batch_f0_src_trg)
                                    total_eval_loss["eval/loss_uvcap_src_trg"].append(batch_uvcap_src_trg)
                                    total_eval_loss["eval/loss_cap_src_trg"].append(batch_cap_src_trg)
                                    total_eval_loss["eval/loss_lat_dist_rmse"].append(batch_lat_dist_rmse)
                                    total_eval_loss["eval/loss_lat_dist_cossim"].append(batch_lat_dist_cossim)
                                    logging.info('acc cv %s %s %.3f %% %.3f Hz ' \
                                    '%.3f %% %.3f dB %.3f %.3f' % (featfile[k], spk_cv[k], 
                                            batch_uv_src_trg, batch_f0_src_trg, batch_uvcap_src_trg,
                                            batch_cap_src_trg, batch_lat_dist_rmse, batch_lat_dist_cossim))
                        idx_in = 0
                        i_cv_in = 0
                        for i in range(0,n_half_cyc_eval,2):
                            i_cv = i//2
                            j = i+1
                            ## latent infer.
                            if i > 0:
                                idx_in += 1
                                i_cv_in += 1
                                cyc_rec_feat = batch_melsp_rec[i-1]
                                _, _, z[i], h_z[i] = model_encoder_melsp(cyc_rec_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                                _, _, z_e[i], h_z_e[i] = model_encoder_excit(cyc_rec_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                            else:
                                _, _, z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], sampling=False)
                                _, _, z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], sampling=False)
                            ## time-varying speaker conditionings
                            idx_in += 1
                            z_cat = torch.cat((z_e[i], z[i]), 2)
                            if args.spkidtr_dim > 0:
                                spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                                spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                                batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                            else:
                                batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                                batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                            ## excit reconstruction & conversion
                            idx_in += 1
                            i_cv_in += 1
                            if spk_pad_right > 0:
                                z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                                z_e[i] = z_e[i][:,spk_pad_left:-spk_pad_right]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:-spk_pad_right]
                            else:
                                z_cat = z_cat[:,spk_pad_left:]
                                z_e[i] = z_e[i][:,spk_pad_left:]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,spk_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_lf0_rec[i], h_lf0[i] \
                                        = model_decoder_excit(z_e[i], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in])
                                batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                        = model_decoder_excit(z_e[i], y=spk_cv_code_in, aux=batch_spk_cv, outpad_right=outpad_rights[idx_in])
                            else:
                                batch_lf0_rec[i], h_lf0[i] \
                                        = model_decoder_excit(z_e[i], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in])
                                batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                        = model_decoder_excit(z_e[i], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv, outpad_right=outpad_rights[idx_in])
                            ## waveform reconstruction
                            idx_in += 1
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
                            if args.lpc > 0:
                                if args.spkidtr_dim > 0:
                                    batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                        = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                            spk_code=spk_code_in, spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                                outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                        = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                            spk_code=batch_sc_in[idx_in], spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                                outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                            else:
                                if args.spkidtr_dim > 0:
                                    batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                        = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                            spk_code=spk_code_in, spk_aux=batch_spk,
                                                outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                        = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                            spk_code=batch_sc_in[idx_in], spk_aux=batch_spk,
                                                outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                            ## melsp reconstruction & conversion
                            idx_in += 1
                            i_cv_in += 1
                            if wav_pad_right > 0:
                                z_cat = z_cat[:,wav_pad_left:-wav_pad_right]
                                e_in = batch_lf0_rec[i][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                                e_cv_in = batch_lf0_cv[i_cv][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,wav_pad_left:-wav_pad_right]
                                    spk_cv_code_in = spk_cv_code_in[:,wav_pad_left:-wav_pad_right]
                                batch_spk = batch_spk[:,wav_pad_left:-wav_pad_right]
                                batch_spk_cv = batch_spk_cv[:,wav_pad_left:-wav_pad_right]
                            else:
                                z_cat = z_cat[:,wav_pad_left:]
                                e_in = batch_lf0_rec[i][:,wav_pad_left:,:args.excit_dim]
                                e_cv_in = batch_lf0_cv[i_cv][:,wav_pad_left:,:args.excit_dim]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,wav_pad_left:]
                                    spk_cv_code_in = spk_cv_code_in[:,wav_pad_left:]
                                batch_spk = batch_spk[:,wav_pad_left:]
                                batch_spk_cv = batch_spk_cv[:,wav_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                    e=e_in, outpad_right=outpad_rights[idx_in])
                                batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                                    e=e_cv_in, outpad_right=outpad_rights[idx_in])
                            else:
                                batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                    e=e_in, outpad_right=outpad_rights[idx_in])
                                batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                                    e=e_cv_in, outpad_right=outpad_rights[idx_in])
                            ## cyclic reconstruction, latent infer.
                            if n_half_cyc_eval > 1:
                                idx_in += 1
                                cv_feat = batch_melsp_cv[i_cv]
                                _, _, z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                                _, _, z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], sampling=False)
                                ## time-varying speaker conditionings
                                idx_in += 1
                                z_cat = torch.cat((z_e[j], z[j]), 2)
                                if args.spkidtr_dim > 0:
                                    if dec_enc_pad_right > 0:
                                        spk_code_in = spk_code_in[:,dec_enc_pad_left:-dec_enc_pad_right]
                                    else:
                                        spk_code_in = spk_code_in[:,dec_enc_pad_left:]
                                    batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in])
                                ## excit reconstruction
                                idx_in += 1
                                if spk_pad_right > 0:
                                    z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                                    z_e[j] = z_e[j][:,spk_pad_left:-spk_pad_right]
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                                else:
                                    z_cat = z_cat[:,spk_pad_left:]
                                    z_e[j] = z_e[j][:,spk_pad_left:]
                                    if args.spkidtr_dim > 0:
                                        spk_code_in = spk_code_in[:,spk_pad_left:]
                                if args.spkidtr_dim > 0:
                                    batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in])
                                else:
                                    batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in])
                                ## waveform cyclic reconstruction
                                idx_in += 1
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
                                if args.lpc > 0:
                                    if args.spkidtr_dim > 0:
                                        batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                            = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                                 spk_code=spk_code_in, spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                                    outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                    else:
                                        batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                            = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                                 spk_code=batch_sc_in[idx_in], spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                                    outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                else:
                                    if args.spkidtr_dim > 0:
                                        batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                            = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                                 spk_code=spk_code_in, spk_aux=batch_spk,
                                                    outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                    else:
                                        batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                            = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                                 spk_code=batch_sc_in[idx_in], spk_aux=batch_spk,
                                                    outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in])
                                ## melsp cyclic reconstruction
                                if n_half_cyc_eval > 2:
                                    idx_in += 1
                                    if wav_pad_right > 0:
                                        z_cat = z_cat[:,wav_pad_left:-wav_pad_right]
                                        e_in = batch_lf0_rec[j][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                                        if args.spkidtr_dim > 0:
                                            spk_code_in = spk_code_in[:,wav_pad_left:-wav_pad_right]
                                        batch_spk = batch_spk[:,wav_pad_left:-wav_pad_right]
                                    else:
                                        z_cat = z_cat[:,wav_pad_left:]
                                        e_in = batch_lf0_rec[j][:,wav_pad_left:,:args.excit_dim]
                                        if args.spkidtr_dim > 0:
                                            spk_code_in = spk_code_in[:,wav_pad_left:]
                                        batch_spk = batch_spk[:,wav_pad_left:]
                                    if args.spkidtr_dim > 0:
                                        batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                                e=e_in, outpad_right=outpad_rights[idx_in])
                                    else:
                                        batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                                e=e_in, outpad_right=outpad_rights[idx_in])

                    # samples check
                    i = np.random.randint(0, batch_melsp_rec[0].shape[0])
                    logging.info("%d %s %d %d %d %d %s" % (i, \
                        os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                            f_ss, f_es, flens[i], max_flen, spk_cv[i]))
                    #logging.info(batch_melsp_rec[0][i,:2,:4])
                    #if n_half_cyc_eval > 1: 
                    #    logging.info(batch_melsp_rec[1][i,:2,:4])
                    #logging.info(batch_melsp[i,:2,:4])
                    #logging.info(batch_melsp_cv[0][i,:2,:4])
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
                    #logging.info(torch.max(z[0][i,5:10], -1))
                    #unique, counts = np.unique(torch.max(z[0][i], -1)[1].cpu().data.numpy(), return_counts=True)
                    #logging.info(dict(zip(unique, counts)))

                    # handle short ending
                    if len(idx_select) > 0:
                        logging.info('len_idx_select: '+str(len(idx_select)))
                        for i in range(n_half_cyc_eval):
                            batch_loss_ce_select[i] = 0
                            batch_loss_err_select[i] = 0
                            batch_loss_ce_f_select[i] = 0
                            batch_loss_err_f_select[i] = 0
                        for j in range(len(idx_select)):
                            k = idx_select[j]
                            slens_utt = slens_acc[k]
                            flens_utt = flens_acc[k]
                            logging.info('%s %d %d' % (featfile[k], slens_utt, flens_utt))
                            batch_x_c_ = batch_x_c[k,:slens_utt]
                            batch_x_f_ = batch_x_f[k,:slens_utt]
                            one_hot_x_c = F.one_hot(batch_x_c_, num_classes=args.cf_dim).float()
                            one_hot_x_f = F.one_hot(batch_x_f_, num_classes=args.cf_dim).float()
                            batch_x_c_ = batch_x_c_.reshape(-1)
                            batch_x_f_ = batch_x_f_.reshape(-1)
                            # T x n_bands x 256 --> (T x n_bands) x 256 --> T x n_bands
                            for i in range(n_half_cyc_eval):
                                batch_x_c_output_ = batch_x_c_output[i][k,:slens_utt]
                                batch_x_f_output_ = batch_x_f_output[i][k,:slens_utt]
                                batch_loss_ce_select_ = torch.mean(criterion_ce(batch_x_c_output_.reshape(-1, args.cf_dim), batch_x_c_).reshape(slens_utt, -1), 0) # n_bands
                                batch_loss_ce_f_select_ = torch.mean(criterion_ce(batch_x_f_output_.reshape(-1, args.cf_dim), batch_x_f_).reshape(slens_utt, -1), 0) # n_bands
                                batch_loss_ce_select[i] += batch_loss_ce_select_
                                batch_loss_ce_f_select[i] += batch_loss_ce_f_select_
                                batch_loss_err_select[i] += torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_c_output_, dim=-1), one_hot_x_c), -1), 0) # n_bands
                                batch_loss_err_f_select[i] += torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_f_output_, dim=-1), one_hot_x_f), -1), 0) # n_bands
                        for i in range(n_half_cyc_eval):
                            batch_loss_ce_select[i] /= len(idx_select)
                            batch_loss_err_select[i] /= len(idx_select)
                            batch_loss_ce_f_select[i] /= len(idx_select)
                            batch_loss_err_f_select[i] /= len(idx_select)
                            batch_loss_ce_c_select_avg[i] = batch_loss_ce_select[i].mean().item()
                            batch_loss_err_c_select_avg[i] = batch_loss_err_select[i].mean().item()
                            batch_loss_ce_f_select_avg[i] = batch_loss_ce_f_select[i].mean().item()
                            batch_loss_err_f_select_avg[i] = batch_loss_err_f_select[i].mean().item()
                            batch_loss_ce_select_avg[i] = (batch_loss_ce_c_select_avg[i] + batch_loss_ce_f_select_avg[i])/2
                            batch_loss_err_select_avg[i] = (batch_loss_err_c_select_avg[i] + batch_loss_err_f_select_avg[i])/2
                            total_eval_loss["eval/loss_ce-%d"%(i+1)].append(batch_loss_ce_select_avg[i])
                            total_eval_loss["eval/loss_err-%d"%(i+1)].append(batch_loss_err_select_avg[i])
                            total_eval_loss["eval/loss_ce_c-%d"%(i+1)].append(batch_loss_ce_c_select_avg[i])
                            total_eval_loss["eval/loss_err_c-%d"%(i+1)].append(batch_loss_err_c_select_avg[i])
                            total_eval_loss["eval/loss_ce_f-%d"%(i+1)].append(batch_loss_ce_f_select_avg[i])
                            total_eval_loss["eval/loss_err_f-%d"%(i+1)].append(batch_loss_err_f_select_avg[i])
                            loss_ce_avg[i].append(batch_loss_ce_select_avg[i])
                            loss_err_avg[i].append(batch_loss_err_select_avg[i])
                            loss_ce_c_avg[i].append(batch_loss_ce_c_select_avg[i])
                            loss_err_c_avg[i].append(batch_loss_err_c_select_avg[i])
                            loss_ce_f_avg[i].append(batch_loss_ce_f_select_avg[i])
                            loss_err_f_avg[i].append(batch_loss_err_f_select_avg[i])
                            for j in range(args.n_bands):
                                total_eval_loss["eval/loss_ce_c-%d-%d"%(i+1,j+1)].append(batch_loss_ce_select[i][j].item())
                                total_eval_loss["eval/loss_err_c-%d-%d"%(i+1,j+1)].append(batch_loss_err_select[i][j].item())
                                total_eval_loss["eval/loss_ce_f-%d-%d"%(i+1,j+1)].append(batch_loss_ce_f_select[i][j].item())
                                total_eval_loss["eval/loss_err_f-%d-%d"%(i+1,j+1)].append(batch_loss_err_f_select[i][j].item())
                                loss_ce[i][j].append(batch_loss_ce_select[i][j].item())
                                loss_err[i][j].append(batch_loss_err_select[i][j].item())
                                loss_ce_f[i][j].append(batch_loss_ce_f_select[i][j].item())
                                loss_err_f[i][j].append(batch_loss_err_f_select[i][j].item())
                        if len(idx_select_full) > 0:
                            logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                            batch_x_c = torch.index_select(batch_x_c,0,idx_select_full)
                            batch_x_f = torch.index_select(batch_x_f,0,idx_select_full)
                            for i in range(n_half_cyc_eval):
                                batch_x_c_output[i] = torch.index_select(batch_x_c_output[i],0,idx_select_full)
                                batch_x_f_output[i] = torch.index_select(batch_x_f_output[i],0,idx_select_full)
                            n_batch_utt = batch_x_c.shape[0]
                        elif batch_loss > 0:
                            logging.info("batch eval loss select %.3f (%.3f sec)" % (batch_loss.item(), time.time() - start))
                            iter_count += 1
                            continue
                        else:
                            continue

                    # loss_compute
                    one_hot_x_c = F.one_hot(batch_x_c, num_classes=args.cf_dim).float()
                    one_hot_x_f = F.one_hot(batch_x_f, num_classes=args.cf_dim).float()
                    T = batch_x_c.shape[1]
                    for i in range(n_half_cyc_eval):
                        batch_loss_ce_ = torch.mean(criterion_ce(batch_x_c_output[i].reshape(-1, args.cf_dim), batch_x_c.reshape(-1)).reshape(n_batch_utt, T, -1), 1) # B x n_bands
                        batch_loss_err_ = torch.mean(torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_c_output[i], dim=-1), one_hot_x_c), -1), 1), 0) # n_bands
                        batch_loss_ce_f_ = torch.mean(criterion_ce(batch_x_f_output[i].reshape(-1, args.cf_dim), batch_x_f.reshape(-1)).reshape(n_batch_utt, T, -1), 1) # B x n_bands
                        batch_loss_err_f_ = torch.mean(torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_f_output[i], dim=-1), one_hot_x_f), -1), 1), 0) # n_bands

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
                        for j in range(args.n_bands):
                            batch_loss_ce[i][j] = batch_loss_ce_[:,j].mean().item()
                            batch_loss_err[i][j] = batch_loss_err_[j].item()
                            batch_loss_ce_f[i][j] = batch_loss_ce_f_[:,j].mean().item()
                            batch_loss_err_f[i][j] = batch_loss_err_f_[j].item()
                            total_eval_loss["eval/loss_ce_c-%d-%d"%(i+1,j+1)].append(batch_loss_ce[i][j])
                            total_eval_loss["eval/loss_err_c-%d-%d"%(i+1,j+1)].append(batch_loss_err[i][j])
                            total_eval_loss["eval/loss_ce_f-%d-%d"%(i+1,j+1)].append(batch_loss_ce_f[i][j])
                            total_eval_loss["eval/loss_err_f-%d-%d"%(i+1,j+1)].append(batch_loss_err_f[i][j])
                            loss_ce[i][j].append(batch_loss_ce[i][j])
                            loss_err[i][j].append(batch_loss_err[i][j])
                            loss_ce_f[i][j].append(batch_loss_ce_f[i][j])
                            loss_err_f[i][j].append(batch_loss_err_f[i][j])

                    text_log = "batch eval loss [%d] %d %d %d %d %d :" % (c_idx+1, max_slen, x_ss, x_bs, f_ss, f_bs)
                    for i in range(n_half_cyc_eval):
                        text_log += " [%d] %.3f %.3f %% %.3f %.3f %% %.3f %.3f %%" % (i+1, batch_loss_ce_avg[i], batch_loss_err_avg[i],
                                batch_loss_ce_c_avg[i], batch_loss_err_c_avg[i], batch_loss_ce_f_avg[i], batch_loss_err_f_avg[i])
                        text_log += " ;"
                        for j in range(args.n_bands):
                            text_log += " [%d-%d] %.3f %.3f %% %.3f %.3f %%" % (i+1, j+1,
                                batch_loss_ce[i][j], batch_loss_err[i][j], batch_loss_ce_f[i][j], batch_loss_err_f[i][j])
                        text_log += " ;;"
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            logging.info('sme %d' % (epoch_idx + 1))
            for key in total_eval_loss.keys():
                total_eval_loss[key] = np.mean(total_eval_loss[key])
                logging.info(f"(Steps: {iter_idx}) {key} = {total_eval_loss[key]:.4f}.")
            write_to_tensorboard(writer, iter_idx, total_eval_loss)
            total_eval_loss = defaultdict(list)
            if pair_exist:
                eval_loss_uv_src_trg = np.mean(loss_uv_src_trg)
                eval_loss_uv_src_trg_std = np.std(loss_uv_src_trg)
                eval_loss_f0_src_trg = np.mean(loss_f0_src_trg)
                eval_loss_f0_src_trg_std = np.std(loss_f0_src_trg)
                eval_loss_uvcap_src_trg = np.mean(loss_uvcap_src_trg)
                eval_loss_uvcap_src_trg_std = np.std(loss_uvcap_src_trg)
                eval_loss_cap_src_trg = np.mean(loss_cap_src_trg)
                eval_loss_cap_src_trg_std = np.std(loss_cap_src_trg)
                eval_loss_lat_dist_rmse = np.mean(loss_lat_dist_rmse)
                eval_loss_lat_dist_rmse_std = np.std(loss_lat_dist_rmse)
                eval_loss_lat_dist_cossim = np.mean(loss_lat_dist_cossim)
                eval_loss_lat_dist_cossim_std = np.std(loss_lat_dist_cossim)
            for i in range(n_half_cyc_eval):
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
                for j in range(args.n_bands):
                    eval_loss_ce[i][j] = np.mean(loss_ce[i][j])
                    eval_loss_ce_std[i][j] = np.std(loss_ce[i][j])
                    eval_loss_err[i][j] = np.mean(loss_err[i][j])
                    eval_loss_err_std[i][j] = np.std(loss_err[i][j])
                    eval_loss_ce_f[i][j] = np.mean(loss_ce_f[i][j])
                    eval_loss_ce_f_std[i][j] = np.std(loss_ce_f[i][j])
                    eval_loss_err_f[i][j] = np.mean(loss_err_f[i][j])
                    eval_loss_err_f_std[i][j] = np.std(loss_err_f[i][j])
            text_log = "(EPOCH:%d) average evaluation loss =" % (epoch_idx+1)
            for i in range(n_half_cyc_eval):
                text_log += " [%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %%" % (i+1,
                        eval_loss_ce_avg[i], eval_loss_ce_avg_std[i], eval_loss_err_avg[i], eval_loss_err_avg_std[i],
                            eval_loss_ce_c_avg[i], eval_loss_ce_c_avg_std[i], eval_loss_err_c_avg[i], eval_loss_err_c_avg_std[i],
                                eval_loss_ce_f_avg[i], eval_loss_ce_f_avg_std[i], eval_loss_err_f_avg[i], eval_loss_err_f_avg_std[i])
                text_log += " ;"
                for j in range(args.n_bands):
                    text_log += " [%d-%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %%" % (i+1, j+1,
                            eval_loss_ce[i][j], eval_loss_ce_std[i][j], eval_loss_err[i][j], eval_loss_err_std[i][j],
                                eval_loss_ce_f[i][j], eval_loss_ce_f_std[i][j], eval_loss_err_f[i][j], eval_loss_err_f_std[i][j])
                text_log += " ;;"
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            if (eval_loss_ce_avg[0]+eval_loss_ce_avg_std[0]) <= (min_eval_loss_ce_avg[0]+min_eval_loss_ce_avg_std[0]) \
                or eval_loss_ce_avg[0] <= min_eval_loss_ce_avg[0] \
                    or round(eval_loss_ce_avg[0]+eval_loss_ce_avg_std[0],2) <= round(min_eval_loss_ce_avg[0]+min_eval_loss_ce_avg_std[0],2) \
                        or round(eval_loss_ce_avg[0],2) <= round(min_eval_loss_ce_avg[0],2) \
                            or (eval_loss_err_avg[0] <= min_eval_loss_err_avg[0]) and ((round(eval_loss_ce_avg[0],2)-0.01) <= round(min_eval_loss_ce_avg[0],2)) \
                                or ((eval_loss_err_avg[0]+eval_loss_err_avg_std[0]) <= (min_eval_loss_err_avg[0]+min_eval_loss_err_avg_std[0])) and ((round(eval_loss_ce_avg[0],2)-0.01) <= round(min_eval_loss_ce_avg[0],2)):
                if pair_exist:
                    min_eval_loss_uv_src_trg = eval_loss_uv_src_trg
                    min_eval_loss_uv_src_trg_std = eval_loss_uv_src_trg_std
                    min_eval_loss_f0_src_trg = eval_loss_f0_src_trg
                    min_eval_loss_f0_src_trg_std = eval_loss_f0_src_trg_std
                    min_eval_loss_uvcap_src_trg = eval_loss_uvcap_src_trg
                    min_eval_loss_uvcap_src_trg_std = eval_loss_uvcap_src_trg_std
                    min_eval_loss_cap_src_trg = eval_loss_cap_src_trg
                    min_eval_loss_cap_src_trg_std = eval_loss_cap_src_trg_std
                    min_eval_loss_lat_dist_rmse = eval_loss_lat_dist_rmse
                    min_eval_loss_lat_dist_rmse_std = eval_loss_lat_dist_rmse_std
                    min_eval_loss_lat_dist_cossim = eval_loss_lat_dist_cossim
                    min_eval_loss_lat_dist_cossim_std = eval_loss_lat_dist_cossim_std
                for i in range(n_half_cyc_eval):
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
                    for j in range(args.n_bands):
                        min_eval_loss_ce[i][j] = eval_loss_ce[i][j]
                        min_eval_loss_ce_std[i][j] = eval_loss_ce_std[i][j]
                        min_eval_loss_err[i][j] = eval_loss_err[i][j]
                        min_eval_loss_err_std[i][j] = eval_loss_err_std[i][j]
                        min_eval_loss_ce_f[i][j] = eval_loss_ce_f[i][j]
                        min_eval_loss_ce_f_std[i][j] = eval_loss_ce_f_std[i][j]
                        min_eval_loss_err_f[i][j] = eval_loss_err_f[i][j]
                        min_eval_loss_err_f_std[i][j] = eval_loss_err_f_std[i][j]
                min_idx = epoch_idx
                #epoch_min_flag = True
                change_min_flag = True
            if change_min_flag:
                text_log = "min_eval_loss ="
                for i in range(n_half_cyc_eval):
                    text_log += " [%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %%" % (i+1,
                            min_eval_loss_ce_avg[i], min_eval_loss_ce_avg_std[i], min_eval_loss_err_avg[i], min_eval_loss_err_avg_std[i],
                                min_eval_loss_ce_c_avg[i], min_eval_loss_ce_c_avg_std[i], min_eval_loss_err_c_avg[i], min_eval_loss_err_c_avg_std[i],
                                    min_eval_loss_ce_f_avg[i], min_eval_loss_ce_f_avg_std[i], min_eval_loss_err_f_avg[i], min_eval_loss_err_f_avg_std[i])
                    text_log += " ;"
                    for j in range(args.n_bands):
                        text_log += " [%d-%d] %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) %.6f (+- %.6f) %%" % (i+1, j+1,
                                min_eval_loss_ce[i][j], min_eval_loss_ce_std[i][j], min_eval_loss_err[i][j], min_eval_loss_err_std[i][j],
                                    min_eval_loss_ce_f[i][j], min_eval_loss_ce_f_std[i][j], min_eval_loss_err_f[i][j], min_eval_loss_err_f_std[i][j])
                    text_log += " ;;"
                logging.info("%s min_idx=%d" % (text_log, min_idx+1))
            #if ((epoch_idx + 1) % args.save_interval_epoch == 0) or (epoch_min_flag):
            if True:
                logging.info('save epoch:%d' % (epoch_idx+1))
                save_checkpoint(args.expdir, model_encoder_melsp, model_decoder_melsp, model_encoder_excit, model_decoder_excit,
                    model_spk, model_waveform, min_eval_loss_ce_avg[0], min_eval_loss_ce_avg_std[0], min_eval_loss_err_avg[0], min_eval_loss_err_avg_std[0],
                    iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1, model_spkidtr=model_spkidtr)
            total = 0
            iter_count = 0
            for i in range(n_rec):
                loss_ce_avg[i] = []
                loss_err_avg[i] = []
                loss_ce_c_avg[i] = []
                loss_err_c_avg[i] = []
                loss_ce_f_avg[i] = []
                loss_err_f_avg[i] = []
                for j in range(args.n_bands):
                    loss_ce[i][j] = []
                    loss_err[i][j] = []
                    loss_ce_f[i][j] = []
                    loss_err_f[i][j] = []
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model_encoder_melsp.train()
            model_decoder_melsp.train()
            model_encoder_excit.train()
            model_decoder_excit.train()
            model_spk.train()
            if args.spkidtr_dim > 0:
                model_spkidtr.train()
            model_waveform.train()
            for param in model_waveform.parameters():
                param.requires_grad = True
            # start next epoch
            if epoch_idx < args.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_x_c, batch_x_f, batch_feat, batch_sc, batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
                    x_bs, x_ss, f_bs, f_ss, slens, flens, n_batch_utt, del_index_utt, max_slen, max_flen, spk_cv, idx_select, idx_select_full, slens_acc, flens_acc = next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

            x_es = x_ss+x_bs
            f_es = f_ss+f_bs
            logging.info(f'{x_ss} {x_bs} {x_es} {max_slen} {f_ss} {f_bs} {f_es} {max_flen}')
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
            else:
                batch_x_c_prev = F.pad(batch_x_c[:,:x_es-1], (0, 0, 1, 0), "constant", args.c_pad)
                batch_x_f_prev = F.pad(batch_x_f[:,:x_es-1], (0, 0, 1, 0), "constant", args.f_pad)
                if args.lpc > 0:
                    batch_x_c_lpc = F.pad(batch_x_c[:,:x_es-1], (0, 0, args.lpc, 0), "constant", args.c_pad)
                    batch_x_f_lpc = F.pad(batch_x_f[:,:x_es-1], (0, 0, args.lpc, 0), "constant", args.f_pad)
                batch_x_c = batch_x_c[:,:x_es]
                batch_x_f = batch_x_f[:,:x_es]
            # handle first pad for input features
            flag_cv = True
            i_cv = 0
            i_cv_in = 0
            f_ss_first_pad_left = f_ss-first_pad_left
            f_es_first_pad_right = f_es+first_pad_right
            i_end = args.n_half_cyc*5
            for i in range(i_end):
                if i % 5 == 0: #enc
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
                else: #spk/lf0/wav//spec
                    if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                        batch_sc_in[i] = batch_sc[:,f_ss_first_pad_left:f_es_first_pad_right]
                        if flag_cv:
                            batch_sc_cv_in[i_cv_in] = batch_sc_cv_data[i_cv][:,f_ss_first_pad_left:f_es_first_pad_right]
                            i_cv_in += 1
                            if i % 5 == 4:
                                i_cv += 1
                                flag_cv = False
                        else:
                            if (i + 1) % 10 == 0:
                                flag_cv = True
                    elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                        batch_sc_in[i] = F.pad(batch_sc[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[i_cv][:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                            i_cv_in += 1
                            if i % 5 == 4:
                                i_cv += 1
                                flag_cv = False
                        else:
                            if (i + 1) % 10 == 0:
                                flag_cv = True
                    elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                        diff_pad = f_es_first_pad_right - max_flen
                        batch_sc_in[i] = F.pad(batch_sc[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[i_cv][:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                            i_cv_in += 1
                            if i % 5 == 4:
                                i_cv += 1
                                flag_cv = False
                        else:
                            if (i + 1) % 10 == 0:
                                flag_cv = True
                    else: # pad left and right need additional replicate
                        diff_pad = f_es_first_pad_right - max_flen
                        batch_sc_in[i] = F.pad(batch_sc[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv_in] = F.pad(batch_sc_cv_data[i_cv][:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                            i_cv_in += 1
                            if i % 5 == 4:
                                i_cv += 1
                                flag_cv = False
                        else:
                            if (i + 1) % 10 == 0:
                                flag_cv = True
                    if i % 5 == 1:
                        f_ss_first_pad_left += spk_pad_left
                        f_es_first_pad_right -= spk_pad_right
                    elif i % 5 == 2:
                        f_ss_first_pad_left += lf0_pad_left
                        f_es_first_pad_right -= lf0_pad_right
                    elif i % 5 == 3:
                        f_ss_first_pad_left += wav_pad_left
                        f_es_first_pad_right -= wav_pad_right
                    elif i % 5 == 4:
                        f_ss_first_pad_left += dec_pad_left
                        f_es_first_pad_right -= dec_pad_right
            batch_excit = batch_feat[:,f_ss:f_es,:args.full_excit_dim]
            for i in range(n_cv):
                batch_excit_cv[i] = batch_feat_cv_data[i][:,f_ss:f_es]

            if f_ss > 0:
                idx_in = 0
                i_cv_in = 0
                for i in range(0,args.n_half_cyc,2):
                    i_cv = i//2
                    j = i+1
                    if len(del_index_utt) > 0:
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
                        h_x[i] = torch.FloatTensor(np.delete(h_x[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_x_2[i] = torch.FloatTensor(np.delete(h_x_2[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_f[i] = torch.FloatTensor(np.delete(h_f[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_melsp[i] = torch.FloatTensor(np.delete(h_melsp[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_melsp_cv[i_cv] = torch.FloatTensor(np.delete(h_melsp_cv[i_cv].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        if args.n_half_cyc > 1:
                            h_z[j] = torch.FloatTensor(np.delete(h_z[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_z_e[j] = torch.FloatTensor(np.delete(h_z_e[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_spk[j] = torch.FloatTensor(np.delete(h_spk[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_lf0[j] = torch.FloatTensor(np.delete(h_lf0[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_x[j] = torch.FloatTensor(np.delete(h_x[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_x_2[j] = torch.FloatTensor(np.delete(h_x_2[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            h_f[j] = torch.FloatTensor(np.delete(h_f[j].cpu().data.numpy(),
                                                            del_index_utt, axis=1)).to(device)
                            if args.n_half_cyc > 2:
                                h_melsp[j] = torch.FloatTensor(np.delete(h_melsp[j].cpu().data.numpy(),
                                                                del_index_utt, axis=1)).to(device)
                    ## latent infer.
                    if i > 0:
                        idx_in += 1
                        i_cv_in += 1
                        cyc_rec_feat = batch_melsp_rec[i-1].detach()
                        _, _, z[i], h_z[i] = model_encoder_melsp(cyc_rec_feat, outpad_right=outpad_rights[idx_in], h=h_z[i], do=True)
                        _, _, z_e[i], h_z_e[i] = model_encoder_excit(cyc_rec_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[i], do=True)
                    else:
                        _, _, z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z[i], do=True)
                        _, _, z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], h=h_z_e[i], do=True)
                    ## time-varying speaker conditionings
                    idx_in += 1
                    z_cat = torch.cat((z_e[i], z[i]), 2)
                    if args.spkidtr_dim > 0:
                        spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                        spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                        batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i], do=True)
                        batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv], do=True)
                    else:
                        batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[i], do=True)
                        batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk_cv[i_cv], do=True)
                    ## excit reconstruction & conversion
                    idx_in += 1
                    i_cv_in += 1
                    if spk_pad_right > 0:
                        z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                        z_e[i] = z_e[i][:,spk_pad_left:-spk_pad_right]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                            spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:-spk_pad_right]
                    else:
                        z_cat = z_cat[:,spk_pad_left:]
                        z_e[i] = z_e[i][:,spk_pad_left:]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,spk_pad_left:]
                            spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:]
                    if args.spkidtr_dim > 0:
                        batch_lf0_rec[i], h_lf0[i] \
                                = model_decoder_excit(z_e[i], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[i], do=True)
                        batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                = model_decoder_excit(z_e[i], y=spk_cv_code_in, aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], h=h_lf0_cv[i_cv], do=True)
                    else:
                        batch_lf0_rec[i], h_lf0[i] \
                                = model_decoder_excit(z_e[i], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[i], do=True)
                        batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                = model_decoder_excit(z_e[i], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], h=h_lf0_cv[i_cv], do=True)
                    ## waveform reconstruction
                    idx_in += 1
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
                    if args.lpc > 0:
                        if args.spkidtr_dim > 0:
                            batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                    spk_code=spk_code_in, spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                        h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                    spk_code=batch_sc_in[idx_in], spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                        h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        if args.spkidtr_dim > 0:
                            batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                    spk_code=spk_code_in, spk_aux=batch_spk,
                                        h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                    spk_code=batch_sc_in[idx_in], spk_aux=batch_spk,
                                        h=h_x[i], h_2=h_x_2[i], h_f=h_f[i], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                    ## melsp reconstruction & conversion
                    idx_in += 1
                    i_cv_in += 1
                    if wav_pad_right > 0:
                        z_cat = z_cat[:,wav_pad_left:-wav_pad_right]
                        e_in = batch_lf0_rec[i][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                        e_cv_in = batch_lf0_cv[i_cv][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,wav_pad_left:-wav_pad_right]
                            spk_cv_code_in = spk_cv_code_in[:,wav_pad_left:-wav_pad_right]
                        batch_spk = batch_spk[:,wav_pad_left:-wav_pad_right]
                        batch_spk_cv = batch_spk_cv[:,wav_pad_left:-wav_pad_right]
                    else:
                        z_cat = z_cat[:,wav_pad_left:]
                        e_in = batch_lf0_rec[i][:,wav_pad_left:,:args.excit_dim]
                        e_cv_in = batch_lf0_cv[i_cv][:,wav_pad_left:,:args.excit_dim]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,wav_pad_left:]
                            spk_cv_code_in = spk_cv_code_in[:,wav_pad_left:]
                        batch_spk = batch_spk[:,wav_pad_left:]
                        batch_spk_cv = batch_spk_cv[:,wav_pad_left:]
                    if args.spkidtr_dim > 0:
                        batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                            e=e_in, outpad_right=outpad_rights[idx_in], h=h_melsp[i], do=True)
                        batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                            e=e_cv_in, outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv], do=True)
                    else:
                        batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                            e=e_in, outpad_right=outpad_rights[idx_in], h=h_melsp[i], do=True)
                        batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                            e=e_cv_in, outpad_right=outpad_rights[idx_in], h=h_melsp_cv[i_cv], do=True)
                    ## cyclic reconstruction, latent infer.
                    if args.n_half_cyc > 1:
                        idx_in += 1
                        cv_feat = batch_melsp_cv[i_cv].detach()
                        _, _, z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z[j], do=True)
                        _, _, z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], h=h_z_e[j], do=True)
                        ## time-varying speaker conditionings
                        idx_in += 1
                        z_cat = torch.cat((z_e[j], z[j]), 2)
                        if args.spkidtr_dim > 0:
                            if dec_enc_pad_right > 0:
                                spk_code_in = spk_code_in[:,dec_enc_pad_left:-dec_enc_pad_right]
                            else:
                                spk_code_in = spk_code_in[:,dec_enc_pad_left:]
                            batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j], do=True)
                        else:
                            batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], h=h_spk[j], do=True)
                        ## excit reconstruction
                        idx_in += 1
                        if spk_pad_right > 0:
                            z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                            z_e[j] = z_e[j][:,spk_pad_left:-spk_pad_right]
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                        else:
                            z_cat = z_cat[:,spk_pad_left:]
                            z_e[j] = z_e[j][:,spk_pad_left:]
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,spk_pad_left:]
                        if args.spkidtr_dim > 0:
                            batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[j], do=True)
                        else:
                            batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], h=h_lf0[j], do=True)
                        ## waveform cyclic reconstruction
                        idx_in += 1
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
                        if args.lpc > 0:
                            if args.spkidtr_dim > 0:
                                batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                         spk_code=spk_code_in, spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                            h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                            else:
                                batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                         spk_code=batch_sc_in[idx_in], spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                            h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            if args.spkidtr_dim > 0:
                                batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                         spk_code=spk_code_in, spk_aux=batch_spk,
                                            h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                            else:
                                batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                         spk_code=batch_sc_in[idx_in], spk_aux=batch_spk,
                                            h=h_x[j], h_2=h_x_2[j], h_f=h_f[j], outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        ## melsp cyclic reconstruction
                        if args.n_half_cyc > 2:
                            idx_in += 1
                            if wav_pad_right > 0:
                                z_cat = z_cat[:,wav_pad_left:-wav_pad_right]
                                e_in = batch_lf0_rec[j][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,wav_pad_left:-wav_pad_right]
                                batch_spk = batch_spk[:,wav_pad_left:-wav_pad_right]
                            else:
                                z_cat = z_cat[:,wav_pad_left:]
                                e_in = batch_lf0_rec[j][:,wav_pad_left:,:args.excit_dim]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,wav_pad_left:]
                                batch_spk = batch_spk[:,wav_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                        e=e_in, outpad_right=outpad_rights[idx_in], h=h_melsp[j], do=True)
                            else:
                                batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                        e=e_in, outpad_right=outpad_rights[idx_in], h=h_melsp[j], do=True)
            else:
                idx_in = 0
                i_cv_in = 0
                for i in range(0,args.n_half_cyc,2):
                    i_cv = i//2
                    j = i+1
                    ## latent infer.
                    if i > 0:
                        idx_in += 1
                        i_cv_in += 1
                        cyc_rec_feat = batch_melsp_rec[i-1].detach()
                        _, _, z[i], h_z[i] = model_encoder_melsp(cyc_rec_feat, outpad_right=outpad_rights[idx_in], do=True)
                        _, _, z_e[i], h_z_e[i] = model_encoder_excit(cyc_rec_feat, outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        _, _, z[i], h_z[i] = model_encoder_melsp(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        _, _, z_e[i], h_z_e[i] = model_encoder_excit(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                    ## time-varying speaker conditionings
                    idx_in += 1
                    z_cat = torch.cat((z_e[i], z[i]), 2)
                    if args.spkidtr_dim > 0:
                        spk_code_in = model_spkidtr(batch_sc_in[idx_in])
                        spk_cv_code_in = model_spkidtr(batch_sc_cv_in[i_cv_in])
                        batch_spk, h_spk[i] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                        batch_spk_cv, h_spk_cv[i_cv] = model_spk(spk_cv_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        batch_spk, h_spk[i] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                        batch_spk_cv, h_spk_cv[i_cv] = model_spk(batch_sc_cv_in[i_cv_in], z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                    ## excit reconstruction & conversion
                    idx_in += 1
                    i_cv_in += 1
                    if spk_pad_right > 0:
                        z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                        z_e[i] = z_e[i][:,spk_pad_left:-spk_pad_right]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                            spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:-spk_pad_right]
                    else:
                        z_cat = z_cat[:,spk_pad_left:]
                        z_e[i] = z_e[i][:,spk_pad_left:]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,spk_pad_left:]
                            spk_cv_code_in = spk_cv_code_in[:,spk_pad_left:]
                    if args.spkidtr_dim > 0:
                        batch_lf0_rec[i], h_lf0[i] \
                                = model_decoder_excit(z_e[i], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], do=True)
                        batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                = model_decoder_excit(z_e[i], y=spk_cv_code_in, aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        batch_lf0_rec[i], h_lf0[i] \
                                = model_decoder_excit(z_e[i], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], do=True)
                        batch_lf0_cv[i_cv], h_lf0_cv[i_cv] \
                                = model_decoder_excit(z_e[i], y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv, outpad_right=outpad_rights[idx_in], do=True)
                    ## waveform reconstruction
                    idx_in += 1
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
                    if args.lpc > 0:
                        if args.spkidtr_dim > 0:
                            batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                    spk_code=spk_code_in, spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                        outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                    spk_code=batch_sc_in[idx_in], spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                        outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        if args.spkidtr_dim > 0:
                            batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                    spk_code=spk_code_in, spk_aux=batch_spk,
                                        outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_x_c_output[i], batch_x_f_output[i], h_x[i], h_x_2[i], h_f[i] \
                                = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                    spk_code=batch_sc_in[idx_in], spk_aux=batch_spk,
                                        outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                    ## melsp reconstruction & conversion
                    idx_in += 1
                    i_cv_in += 1
                    if wav_pad_right > 0:
                        z_cat = z_cat[:,wav_pad_left:-wav_pad_right]
                        e_in = batch_lf0_rec[i][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                        e_cv_in = batch_lf0_cv[i_cv][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,wav_pad_left:-wav_pad_right]
                            spk_cv_code_in = spk_cv_code_in[:,wav_pad_left:-wav_pad_right]
                        batch_spk = batch_spk[:,wav_pad_left:-wav_pad_right]
                        batch_spk_cv = batch_spk_cv[:,wav_pad_left:-wav_pad_right]
                    else:
                        z_cat = z_cat[:,wav_pad_left:]
                        e_in = batch_lf0_rec[i][:,wav_pad_left:,:args.excit_dim]
                        e_cv_in = batch_lf0_cv[i_cv][:,wav_pad_left:,:args.excit_dim]
                        if args.spkidtr_dim > 0:
                            spk_code_in = spk_code_in[:,wav_pad_left:]
                            spk_cv_code_in = spk_cv_code_in[:,wav_pad_left:]
                        batch_spk = batch_spk[:,wav_pad_left:]
                        batch_spk_cv = batch_spk_cv[:,wav_pad_left:]
                    if args.spkidtr_dim > 0:
                        batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                            e=e_in, outpad_right=outpad_rights[idx_in], do=True)
                        batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=spk_cv_code_in, aux=batch_spk_cv,
                                            e=e_cv_in, outpad_right=outpad_rights[idx_in], do=True)
                    else:
                        batch_melsp_rec[i], h_melsp[i] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                            e=e_in, outpad_right=outpad_rights[idx_in], do=True)
                        batch_melsp_cv[i_cv], h_melsp_cv[i_cv] = model_decoder_melsp(z_cat, y=batch_sc_cv_in[i_cv_in], aux=batch_spk_cv,
                                            e=e_cv_in, outpad_right=outpad_rights[idx_in], do=True)
                    ## cyclic reconstruction, latent infer.
                    if args.n_half_cyc > 1:
                        idx_in += 1
                        cv_feat = batch_melsp_cv[i_cv].detach()
                        _, _, z[j], h_z[j] = model_encoder_melsp(cv_feat, outpad_right=outpad_rights[idx_in], do=True)
                        _, _, z_e[j], h_z_e[j] = model_encoder_excit(cv_feat, outpad_right=outpad_rights[idx_in], do=True)
                        ## time-varying speaker conditionings
                        idx_in += 1
                        z_cat = torch.cat((z_e[j], z[j]), 2)
                        if args.spkidtr_dim > 0:
                            if dec_enc_pad_right > 0:
                                spk_code_in = spk_code_in[:,dec_enc_pad_left:-dec_enc_pad_right]
                            else:
                                spk_code_in = spk_code_in[:,dec_enc_pad_left:]
                            batch_spk, h_spk[j] = model_spk(spk_code_in, z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_spk, h_spk[j] = model_spk(batch_sc_in[idx_in], z=z_cat, outpad_right=outpad_rights[idx_in], do=True)
                        ## excit reconstruction
                        idx_in += 1
                        if spk_pad_right > 0:
                            z_cat = z_cat[:,spk_pad_left:-spk_pad_right]
                            z_e[j] = z_e[j][:,spk_pad_left:-spk_pad_right]
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,spk_pad_left:-spk_pad_right]
                        else:
                            z_cat = z_cat[:,spk_pad_left:]
                            z_e[j] = z_e[j][:,spk_pad_left:]
                            if args.spkidtr_dim > 0:
                                spk_code_in = spk_code_in[:,spk_pad_left:]
                        if args.spkidtr_dim > 0:
                            batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=spk_code_in, aux=batch_spk, outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            batch_lf0_rec[j], h_lf0[j] = model_decoder_excit(z_e[j], y=batch_sc_in[idx_in], aux=batch_spk, outpad_right=outpad_rights[idx_in], do=True)
                        ## waveform cyclic reconstruction
                        idx_in += 1
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
                        if args.lpc > 0:
                            if args.spkidtr_dim > 0:
                                batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                         spk_code=spk_code_in, spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                            outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                            else:
                                batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                         spk_code=batch_sc_in[idx_in], spk_aux=batch_spk, x_c_lpc=batch_x_c_lpc, x_f_lpc=batch_x_f_lpc,
                                            outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        else:
                            if args.spkidtr_dim > 0:
                                batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                         spk_code=spk_code_in, spk_aux=batch_spk,
                                            outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                            else:
                                batch_x_c_output[j], batch_x_f_output[j], h_x[j], h_x_2[j], h_f[j] \
                                    = model_waveform(z_cat, batch_x_c_prev, batch_x_f_prev, batch_x_c,
                                         spk_code=batch_sc_in[idx_in], spk_aux=batch_spk,
                                            outpad_left=outpad_lefts[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        ## melsp cyclic reconstruction
                        if args.n_half_cyc > 2:
                            idx_in += 1
                            if wav_pad_right > 0:
                                z_cat = z_cat[:,wav_pad_left:-wav_pad_right]
                                e_in = batch_lf0_rec[j][:,wav_pad_left:-wav_pad_right,:args.excit_dim]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,wav_pad_left:-wav_pad_right]
                                batch_spk = batch_spk[:,wav_pad_left:-wav_pad_right]
                            else:
                                z_cat = z_cat[:,wav_pad_left:]
                                e_in = batch_lf0_rec[j][:,wav_pad_left:,:args.excit_dim]
                                if args.spkidtr_dim > 0:
                                    spk_code_in = spk_code_in[:,wav_pad_left:]
                                batch_spk = batch_spk[:,wav_pad_left:]
                            if args.spkidtr_dim > 0:
                                batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=spk_code_in, aux=batch_spk,
                                                        e=e_in, outpad_right=outpad_rights[idx_in], do=True)
                            else:
                                batch_melsp_rec[j], h_melsp[j] = model_decoder_melsp(z_cat, y=batch_sc_in[idx_in], aux=batch_spk,
                                                        e=e_in, outpad_right=outpad_rights[idx_in], do=True)

            # samples check
            with torch.no_grad():
                i = np.random.randint(0, batch_melsp_rec[0].shape[0])
                logging.info("%d %s %d %d %d %d %s" % (i, \
                    os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                        f_ss, f_es, flens[i], max_flen, spk_cv[0][i]))
                #logging.info(batch_melsp_rec[0][i,:2,:4])
                #if args.n_half_cyc > 1:
                #    logging.info(batch_melsp_rec[1][i,:2,:4])
                #logging.info(batch_melsp[i,:2,:4])
                #logging.info(batch_melsp_cv[0][i,:2,:4])
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
                #logging.info(torch.max(z[0][i,5:10], -1))
                #unique, counts = np.unique(torch.max(z[0][i], -1)[1].cpu().data.numpy(), return_counts=True)
                #logging.info(dict(zip(unique, counts)))

            # Losses computation
            batch_loss = 0

            # handle short ending
            if len(idx_select) > 0:
                logging.info('len_idx_select: '+str(len(idx_select)))
                for i in range(args.n_half_cyc):
                    batch_loss_ce_select[i] = 0
                    batch_loss_err_select[i] = 0
                    batch_loss_ce_f_select[i] = 0
                    batch_loss_err_f_select[i] = 0
                for j in range(len(idx_select)):
                    k = idx_select[j]
                    slens_utt = slens_acc[k]
                    flens_utt = flens_acc[k]
                    logging.info('%s %d %d' % (featfile[k], slens_utt, flens_utt))
                    batch_x_c_ = batch_x_c[k,:slens_utt]
                    batch_x_f_ = batch_x_f[k,:slens_utt]
                    one_hot_x_c = F.one_hot(batch_x_c_, num_classes=args.cf_dim).float()
                    one_hot_x_f = F.one_hot(batch_x_f_, num_classes=args.cf_dim).float()
                    batch_x_c_ = batch_x_c_.reshape(-1)
                    batch_x_f_ = batch_x_f_.reshape(-1)
                    # T x n_bands x 256 --> (T x n_bands) x 256 --> T x n_bands
                    for i in range(args.n_half_cyc):
                        batch_x_c_output_ = batch_x_c_output[i][k,:slens_utt]
                        batch_x_f_output_ = batch_x_f_output[i][k,:slens_utt]
                        batch_loss_ce_select_ = torch.mean(criterion_ce(batch_x_c_output_.reshape(-1, args.cf_dim), batch_x_c_).reshape(slens_utt, -1), 0) # n_bands
                        batch_loss_ce_f_select_ = torch.mean(criterion_ce(batch_x_f_output_.reshape(-1, args.cf_dim), batch_x_f_).reshape(slens_utt, -1), 0) # n_bands
                        batch_loss_ce_select[i] += batch_loss_ce_select_
                        batch_loss_ce_f_select[i] += batch_loss_ce_f_select_
                        batch_loss += batch_loss_ce_select_.sum() + batch_loss_ce_f_select_.sum() \
                                        + batch_loss_ce_select_.mean() + batch_loss_ce_f_select_.mean() #360/405
                        batch_loss_err_select[i] += torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_c_output_, dim=-1), one_hot_x_c), -1), 0) # n_bands
                        batch_loss_err_f_select[i] += torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_f_output_, dim=-1), one_hot_x_f), -1), 0) # n_bands
                for i in range(args.n_half_cyc):
                    batch_loss_ce_select[i] /= len(idx_select)
                    batch_loss_err_select[i] /= len(idx_select)
                    batch_loss_ce_f_select[i] /= len(idx_select)
                    batch_loss_err_f_select[i] /= len(idx_select)
                    batch_loss_ce_c_select_avg[i] = batch_loss_ce_select[i].mean().item()
                    batch_loss_err_c_select_avg[i] = batch_loss_err_select[i].mean().item()
                    batch_loss_ce_f_select_avg[i] = batch_loss_ce_f_select[i].mean().item()
                    batch_loss_err_f_select_avg[i] = batch_loss_err_f_select[i].mean().item()
                    batch_loss_ce_select_avg[i] = (batch_loss_ce_c_select_avg[i] + batch_loss_ce_f_select_avg[i])/2
                    batch_loss_err_select_avg[i] = (batch_loss_err_c_select_avg[i] + batch_loss_err_f_select_avg[i])/2
                    total_train_loss["train/loss_ce-%d"%(i+1)].append(batch_loss_ce_select_avg[i])
                    total_train_loss["train/loss_err-%d"%(i+1)].append(batch_loss_err_select_avg[i])
                    total_train_loss["train/loss_ce_c-%d"%(i+1)].append(batch_loss_ce_c_select_avg[i])
                    total_train_loss["train/loss_err_c-%d"%(i+1)].append(batch_loss_err_c_select_avg[i])
                    total_train_loss["train/loss_ce_f-%d"%(i+1)].append(batch_loss_ce_f_select_avg[i])
                    total_train_loss["train/loss_err_f-%d"%(i+1)].append(batch_loss_err_f_select_avg[i])
                    loss_ce_avg[i].append(batch_loss_ce_select_avg[i])
                    loss_err_avg[i].append(batch_loss_err_select_avg[i])
                    loss_ce_c_avg[i].append(batch_loss_ce_c_select_avg[i])
                    loss_err_c_avg[i].append(batch_loss_err_c_select_avg[i])
                    loss_ce_f_avg[i].append(batch_loss_ce_f_select_avg[i])
                    loss_err_f_avg[i].append(batch_loss_err_f_select_avg[i])
                    for j in range(args.n_bands):
                        total_train_loss["train/loss_ce_c-%d-%d"%(i+1,j+1)].append(batch_loss_ce_select[i][j].item())
                        total_train_loss["train/loss_err_c-%d-%d"%(i+1,j+1)].append(batch_loss_err_select[i][j].item())
                        total_train_loss["train/loss_ce_f-%d-%d"%(i+1,j+1)].append(batch_loss_ce_f_select[i][j].item())
                        total_train_loss["train/loss_err_f-%d-%d"%(i+1,j+1)].append(batch_loss_err_f_select[i][j].item())
                        loss_ce[i][j].append(batch_loss_ce_select[i][j].item())
                        loss_err[i][j].append(batch_loss_err_select[i][j].item())
                        loss_ce_f[i][j].append(batch_loss_ce_f_select[i][j].item())
                        loss_err_f[i][j].append(batch_loss_err_f_select[i][j].item())
                if len(idx_select_full) > 0:
                    logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                    batch_x_c = torch.index_select(batch_x_c,0,idx_select_full)
                    batch_x_f = torch.index_select(batch_x_f,0,idx_select_full)
                    for i in range(args.n_half_cyc):
                        batch_x_c_output[i] = torch.index_select(batch_x_c_output[i],0,idx_select_full)
                        batch_x_f_output[i] = torch.index_select(batch_x_f_output[i],0,idx_select_full)
                    n_batch_utt = batch_x_c.shape[0]
                elif batch_loss > 0:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    flag = False
                    for name, param in model_waveform.named_parameters():
                        if param.requires_grad:
                            grad_norm = param.grad.norm()
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                flag = True
                    if flag:
                        logging.info("explode grad")
                        optimizer.zero_grad()
                        continue
                    torch.nn.utils.clip_grad_norm_(model_waveform.parameters(), 10)
                    optimizer.step()

                    with torch.no_grad():
                        if idx_stage < args.n_stage-1 and iter_idx + 1 == t_starts[idx_stage+1]:
                            idx_stage += 1
                        if idx_stage > 0:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                        else:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])

                    logging.info("batch loss select %.3f (%.3f sec)" % (batch_loss.item(), time.time() - start))
                    iter_idx += 1
                    #if iter_idx % args.save_interval_iter == 0:
                    #    logging.info('save iter:%d' % (iter_idx))
                    #    save_checkpoint(args.expdir, model_waveform, optimizer, np.random.get_state(), torch.get_rng_state(), iter_idx)
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
                else:
                    continue

            # loss_compute
            one_hot_x_c = F.one_hot(batch_x_c, num_classes=args.cf_dim).float()
            one_hot_x_f = F.one_hot(batch_x_f, num_classes=args.cf_dim).float()
            T = batch_x_c.shape[1]
            for i in range(args.n_half_cyc):
                batch_loss_ce_ = torch.mean(criterion_ce(batch_x_c_output[i].reshape(-1, args.cf_dim), batch_x_c.reshape(-1)).reshape(n_batch_utt, T, -1), 1) # B x n_bands
                batch_loss_err_ = torch.mean(torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_c_output[i], dim=-1), one_hot_x_c), -1), 1), 0) # n_bands
                batch_loss_ce_f_ = torch.mean(criterion_ce(batch_x_f_output[i].reshape(-1, args.cf_dim), batch_x_f.reshape(-1)).reshape(n_batch_utt, T, -1), 1) # B x n_bands
                batch_loss_err_f_ = torch.mean(torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_f_output[i], dim=-1), one_hot_x_f), -1), 1), 0) # n_bands

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
                for j in range(args.n_bands):
                    batch_loss_ce[i][j] = batch_loss_ce_[:,j].mean().item()
                    batch_loss_err[i][j] = batch_loss_err_[j].item()
                    batch_loss_ce_f[i][j] = batch_loss_ce_f_[:,j].mean().item()
                    batch_loss_err_f[i][j] = batch_loss_err_f_[j].item()
                    total_train_loss["train/loss_ce_c-%d-%d"%(i+1,j+1)].append(batch_loss_ce[i][j])
                    total_train_loss["train/loss_err_c-%d-%d"%(i+1,j+1)].append(batch_loss_err[i][j])
                    total_train_loss["train/loss_ce_f-%d-%d"%(i+1,j+1)].append(batch_loss_ce_f[i][j])
                    total_train_loss["train/loss_err_f-%d-%d"%(i+1,j+1)].append(batch_loss_err_f[i][j])
                    loss_ce[i][j].append(batch_loss_ce[i][j])
                    loss_err[i][j].append(batch_loss_err[i][j])
                    loss_ce_f[i][j].append(batch_loss_ce_f[i][j])
                    loss_err_f[i][j].append(batch_loss_err_f[i][j])

                batch_loss += batch_loss_ce_.sum() + batch_loss_ce_f_.sum() \
                                + batch_loss_ce_.mean(-1).sum() + batch_loss_ce_f_.mean(-1).sum() #360/405[clamp]

            optimizer.zero_grad()
            batch_loss.backward()
            flag = False
            for name, param in model_waveform.named_parameters():
                if param.requires_grad:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        flag = True
            if flag:
                logging.info("explode grad")
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model_waveform.parameters(), 10)
            optimizer.step()

            with torch.no_grad():
                if idx_stage < args.n_stage-1 and iter_idx + 1 == t_starts[idx_stage+1]:
                    idx_stage += 1
                if idx_stage > 0:
                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                else:
                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])

            text_log = "batch loss [%d] %d %d %d %d %d :" % (c_idx+1, max_slen, x_ss, x_bs, f_ss, f_bs)
            for i in range(args.n_half_cyc):
                text_log += " [%d] %.3f %.3f %% %.3f %.3f %% %.3f %.3f %%" % (i+1, batch_loss_ce_avg[i], batch_loss_err_avg[i],
                        batch_loss_ce_c_avg[i], batch_loss_err_c_avg[i], batch_loss_ce_f_avg[i], batch_loss_err_f_avg[i])
                text_log += " ;"
                for j in range(args.n_bands):
                    text_log += " [%d-%d] %.3f %.3f %% %.3f %.3f %%" % (i+1, j+1,
                        batch_loss_ce[i][j], batch_loss_err[i][j], batch_loss_ce_f[i][j], batch_loss_err_f[i][j])
                text_log += " ;;"
            logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
            iter_idx += 1
            #if iter_idx % args.save_interval_iter == 0:
            #    logging.info('save iter:%d' % (iter_idx))
            #    save_checkpoint(args.expdir, model_waveform, optimizer, np.random.get_state(), torch.get_rng_state(), iter_idx)
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
