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
from vcneuvoco import GRU_WAVE_DECODER_DUALGRU_COMPACT_LPCSEG
from vcneuvoco import encode_mu_law
from radam import RAdam

from dataset import FeatureDatasetNeuVoco, padding

#import warnings
#warnings.filterwarnings('ignore')

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)


def data_generator(dataloader, device, batch_size, upsampling_factor, limit_count=None, seg=5, batch_sizes=None):
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
            slens = batch['slen'].data.numpy()
            flens = batch['flen'].data.numpy()
            max_slen = np.max(slens) ## get max samples length
            max_flen = np.max(flens) ## get max samples length
            xs = batch['x'][:,:max_slen].to(device)
            feat = batch['feat'][:,:max_flen].to(device)
            featfiles = batch['featfile']
            n_batch_utt = feat.size(0)

            if batch_sizes is not None:
                batch_size = batch_sizes[np.random.randint(3)]
            len_frm = max_flen
            len_smpl = max_slen
            x_ss = 0
            f_ss = 0
            x_bs = batch_size*upsampling_factor
            f_bs = batch_size
            delta_smpl = batch_size*upsampling_factor
            delta_frm = batch_size
            flens_acc = np.array(flens)
            slens_acc = np.array(slens)
            while True:
                del_index_utt = []
                idx_select = []
                idx_select_full = []
                for i in range(n_batch_utt):
                    if flens_acc[i] <= 0:
                        del_index_utt.append(i)
                if len(del_index_utt) > 0:
                    xs = torch.LongTensor(np.delete(xs.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    featfiles = np.delete(featfiles, del_index_utt, axis=0)
                    flens_acc = np.delete(flens_acc, del_index_utt, axis=0)
                    slens_acc = np.delete(slens_acc, del_index_utt, axis=0)
                    n_batch_utt -= len(del_index_utt)
                for i in range(n_batch_utt):
                    if flens_acc[i] < f_bs:
                        idx_select.append(i)
                if len(idx_select) > 0:
                    idx_select_full = torch.LongTensor(np.delete(np.arange(n_batch_utt), idx_select, axis=0)).to(device)
                    idx_select = torch.LongTensor(idx_select).to(device)
                yield xs, feat, c_idx, idx, featfiles, x_bs, f_bs, x_ss, f_ss, n_batch_utt, del_index_utt, max_slen, \
                    slens_acc, idx_select, idx_select_full
                for i in range(n_batch_utt):
                    flens_acc[i] -= delta_frm
                    slens_acc[i] -= delta_smpl

                count += 1
                if limit_count is not None and count > limit_count:
                    break
                #len_frm -= delta_frm
                len_smpl -= delta_smpl
                #if len_frm > 0:
                if len_smpl >= seg:
                    x_ss += delta_smpl
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

        yield [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], []


#def eval_generator(dataloader, device, batch_size, upsampling_factor, limit_count=None, seg=5):
def eval_generator(dataloader, device, batch_size, upsampling_factor, limit_count=None):
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
            slens = batch['slen'].data.numpy()
            flens = batch['flen'].data.numpy()
            max_slen = np.max(slens) ## get max samples length
            max_flen = np.max(flens) ## get max samples length
            xs = batch['x'][:,:max_slen].to(device)
            feat = batch['feat'][:,:max_flen].to(device)
            featfiles = batch['featfile']
            n_batch_utt = feat.size(0)

            len_frm = max_flen
            len_smpl = max_slen
            x_ss = 0
            f_ss = 0
            x_bs = batch_size*upsampling_factor
            f_bs = batch_size
            delta_smpl = batch_size*upsampling_factor
            delta_frm = batch_size
            flens_acc = np.array(flens)
            while True:
                del_index_utt = []
                for i in range(n_batch_utt):
                    if flens_acc[i] >= f_bs:
                        flens_acc[i] -= delta_frm
                    else:
                        del_index_utt.append(i)
                if len(del_index_utt) > 0:
                    xs = torch.LongTensor(np.delete(xs.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    featfiles = np.delete(featfiles, del_index_utt, axis=0)
                    flens_acc = np.delete(flens_acc, del_index_utt, axis=0)
                    n_batch_utt -= len(del_index_utt)
                yield xs, feat, c_idx, idx, featfiles, x_bs, f_bs, x_ss, f_ss, n_batch_utt, del_index_utt, max_slen

                count += 1
                if limit_count is not None and count > limit_count:
                    break
                len_frm -= delta_frm
                #len_smpl -= delta_smpl
                if len_frm >= f_bs:
                #if len_frm >= seg:
                    x_ss += delta_smpl
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

        yield [], [], -1, -1, [], [], [], [], [], [], [], []


def save_checkpoint(checkpoint_dir, model_waveform,
        optimizer, numpy_random_state, torch_random_state, iterations):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    model_waveform.cpu()
    checkpoint = {
        "model_waveform": model_waveform.state_dict(),
        "optimizer": optimizer.state_dict(),
        "numpy_random_state": numpy_random_state,
        "torch_random_state": torch_random_state,
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    model_waveform.cuda()
    logging.info("%d-iter checkpoint created." % iterations)


def write_to_tensorboard(writer, steps, loss):
    """Write to tensorboard."""
    for key, value in loss.items():
        writer.add_scalar(key, value, steps)


def sparsify(model_waveform, iter_idx, t_start, t_end, interval, densities, densities_p=None, density_conv_s_c=None, density_conv_s_c_p=None,
        density_out=None, density_out_p=None):
    if iter_idx < t_start or ((iter_idx-t_start) % interval != 0 and iter_idx < t_end):
        pass
    else:
        logging.info('sparsify: %ld %ld %ld %ld' % (iter_idx, t_start, t_end, interval))
        p = model_waveform.gru.weight_hh_l0 #recurrent
        nb = p.shape[0] // p.shape[1]
        N = p.shape[0] // nb
        N_16 = N // 16
        ones = torch.diag(torch.ones(N, device=p.device))
        #q = model_waveform.gru.weight_ih_l0 #input
        #M = q.shape[0] // nb
        #M_16 = M // 16
        #M_row = q.shape[1]
        #if M == M_row:
        #    if M == N:
        #        ones_ = ones
        #    else:
        #        ones_ = torch.diag(torch.ones(M, device=q.device))
        if densities_p is not None:
            for k in range(nb):
                density = densities[k]
                if iter_idx < t_end:
                    r = 1 - (iter_idx-t_start)/(t_end - t_start)
                    density = densities_p[k] - (densities_p[k]-densities[k])*(1 - r)**5
                logging.info('%ld: %lf %lf %lf' % (k+1, densities_p[k], densities[k], density))
                #recurrent weight
                A = p[k*N:(k+1)*N, :]
                L = (A - torch.diag(torch.diag(A))).transpose(1, 0).reshape(N, N_16, 16) #horizontal block structure (16) in input part, in real-time simultaneously computed for each 16 output using 2 registers (256x2 bits)
                S = torch.sum(L*L, -1)
                SS, _ = torch.sort(S.reshape(-1))
                thresh = SS[round(N*N_16*(1-density))]
                mask = torch.clamp(torch.repeat_interleave((S>=thresh).float(), 16, dim=1) + ones, max=1).transpose(1,0)
                p[k*N:(k+1)*N, :] = p[k*N:(k+1)*N, :]*mask
        #        #input weight
        #        A = q[k*M:(k+1)*M, :]
        #        if M == M_row:
        #            L = (A - torch.diag(torch.diag(A))).transpose(1, 0).reshape(M, M_16, 16) #horizontal block structure (16) in input part, in real-time simultaneously computed for each 16 output using 2 registers (256x2 bits)
        #        else:
        #            L = A.transpose(1, 0).reshape(M_row, M_16, 16) #horizontal block structure (16) in input part, in real-time simultaneously computed for each 16 output using 2 registers (256x2 bits)
        #        S = torch.sum(L*L, -1)
        #        SS, _ = torch.sort(S.reshape(-1))
        #        thresh = SS[round(M_row*M_16*(1-density))]
        #        if M == M_row:
        #            mask = torch.clamp(torch.repeat_interleave((S>=thresh).float(), 16, dim=1) + ones_, max=1).transpose(1,0)
        #        else:
        #            mask = torch.repeat_interleave((S>=thresh).float(), 16, dim=1).transpose(1,0)
        #        q[k*M:(k+1)*M, :] = q[k*M:(k+1)*M, :]*mask
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
        #        #input weight
        #        A = q[k*M:(k+1)*M, :]
        #        if M == M_row:
        #            L = (A - torch.diag(torch.diag(A))).transpose(1, 0).reshape(M, M_16, 16)
        #        else:
        #            L = A.transpose(1, 0).reshape(M_row, M_16, 16)
        #        S = torch.sum(L*L, -1)
        #        SS, _ = torch.sort(S.reshape(-1))
        #        thresh = SS[round(M_row*M_16*(1-density))]
        #        if M == M_row:
        #            mask = torch.clamp(torch.repeat_interleave((S>=thresh).float(), 16, dim=1) + ones_, max=1).transpose(1,0)
        #        else:
        #            mask = torch.repeat_interleave((S>=thresh).float(), 16, dim=1).transpose(1,0)
        #        q[k*M:(k+1)*M, :] = q[k*M:(k+1)*M, :]*mask
        #out.out
        if density_out is not None:
            s = model_waveform.out.out.weight #conv after upsampling before multiplication with waveform for GRU input
            O = s.shape[0]
            O_16 = O // 16
            P = s.shape[1]
            density = density_out
            if iter_idx < t_end:
                r = 1 - (iter_idx-t_start)/(t_end - t_start)
                if density_out_p is not None:
                    density = density_out_p - (density_out_p-density_out)*(1 - r)**5
                else:
                    density = 1 - (1-density_out)*(1 - r)**5
            if density_out_p is not None:
                logging.info('%lf %lf %lf' % (density_out_p, density_out, density))
            else:
                logging.info('1 %lf %lf' % (density_out, density))
            A = s[:, :, 0]
            L = A.transpose(1, 0).reshape(P, O_16, 16) #horizontal block structure (16) in input part, in real-time simultaneously computed for each 16 output using 2 registers (256x2 bits)
            S = torch.sum(L*L, -1)
            SS, _ = torch.sort(S.reshape(-1))
            thresh = SS[round(P*O_16*(1-density))]
            mask = torch.repeat_interleave((S>=thresh).float(), 16, dim=1).transpose(1,0)
            s[:, :, 0] = s[:, :, 0]*mask


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--waveforms",
                        type=str, help="directory or list of wav files")
    parser.add_argument("--waveforms_eval",
                        type=str, help="directory or list of evaluation wav files")
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--feats_eval", required=True,
                        type=str, help="directory or list of evaluation feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="directory or list of evaluation wav files")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    # network structure setting
    parser.add_argument("--upsampling_factor", default=120,
                        type=int, help="number of dimension of aux feats")
    parser.add_argument("--hidden_units_wave", default=384,
                        type=int, help="depth of dilation")
    parser.add_argument("--hidden_units_wave_2", default=32,
                        type=int, help="depth of dilation")
    parser.add_argument("--kernel_size_wave", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_wave", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--mcep_dim", default=50,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--seg", default=2,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--lpc", default=4,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--right_size", default=0,
                        type=int, help="kernel size of dilated causal convolution")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=10,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--epoch_count", default=100,
                        type=int, help="number of training epochs")
    parser.add_argument("--do_prob", default=0,
                        type=float, help="dropout probability")
    parser.add_argument("--batch_size_utt", default=5,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--batch_size_utt_eval", default=5,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--n_workers", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--n_quantize", default=256,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_wave", default=False,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--n_stage", default=4,
                        type=int, help="number of sparsification stages")
    parser.add_argument("--t_start", default=20001,
                        type=int, help="iter idx to start sparsify")
    parser.add_argument("--t_end", default=2520000,
                        type=int, help="iter idx to finish densitiy sparsify")
    parser.add_argument("--interval", default=100,
                        type=int, help="interval in finishing densitiy sparsify")
    parser.add_argument("--densities", default="0.27-0.27-0.39",
                        type=str, help="final densitiy of reset, update, new hidden gate matrices")
    # other setting
    parser.add_argument("--pad_len", default=3000,
                        type=int, help="seed number")
    parser.add_argument("--save_interval_iter", default=5000,
                        type=int, help="interval steps to logr")
    parser.add_argument("--save_interval_epoch", default=10,
                        type=int, help="interval steps to logr")
    parser.add_argument("--log_interval_steps", default=50,
                        type=int, help="interval steps to logr")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--pretrained", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--string_path", default=None,
                        type=str, help="model path to restart training")
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

    #if args.pretrained is None:
    if 'mel' in args.string_path:
        mean_stats = torch.FloatTensor(read_hdf5(args.stats, "/mean_melsp"))
        scale_stats = torch.FloatTensor(read_hdf5(args.stats, "/scale_melsp"))
        args.excit_dim = 0
        #mean_stats = torch.FloatTensor(np.r_[read_hdf5(args.stats, "/mean_feat_mceplf0cap")[:2], read_hdf5(args.stats, "/mean_melsp")])
        #scale_stats = torch.FloatTensor(np.r_[read_hdf5(args.stats, "/scale_feat_mceplf0cap")[:2], read_hdf5(args.stats, "/scale_melsp")])
        #args.excit_dim = 2
        #mean_stats = torch.FloatTensor(np.r_[read_hdf5(args.stats, "/mean_feat_mceplf0cap")[:6], read_hdf5(args.stats, "/mean_melsp")])
        #scale_stats = torch.FloatTensor(np.r_[read_hdf5(args.stats, "/scale_feat_mceplf0cap")[:6], read_hdf5(args.stats, "/scale_melsp")])
        #args.excit_dim = 6
    else:
        mean_stats = torch.FloatTensor(read_hdf5(args.stats, "/mean_"+args.string_path.replace("/","")))
        scale_stats = torch.FloatTensor(read_hdf5(args.stats, "/scale_"+args.string_path.replace("/","")))
        if mean_stats.shape[0] > args.mcep_dim+2:
            if 'feat_org_lf0' in args.string_path:
                args.cap_dim = mean_stats.shape[0]-(args.mcep_dim+2)
                args.excit_dim = 2+args.cap_dim
            else:
                args.cap_dim = mean_stats.shape[0]-(args.mcep_dim+3)
                args.excit_dim = 2+1+args.cap_dim
            #args.cap_dim = mean_stats.shape[0]-(args.mcep_dim+2)
            #args.excit_dim = 2+args.cap_dim
        else:
            args.cap_dim = None
            args.excit_dim = 2

    # save args as conf
    if args.n_stage < 3:
        args.n_stage = 3
    elif args.n_stage > 5:
        args.n_stage = 5
    #if args.batch_size < 10:
    #    args.batch_size = 10
    torch.save(args, args.expdir + "/model.conf")
    #batch_sizes = [None]*3
    #batch_sizes[0] = int(args.batch_size*0.5)
    #batch_sizes[1] = int(args.batch_size)
    #batch_sizes[2] = int(args.batch_size*1.5)
    #logging.info(batch_sizes)

    # define network
    model_waveform = GRU_WAVE_DECODER_DUALGRU_COMPACT_LPCSEG(
        feat_dim=args.mcep_dim+args.excit_dim,
        upsampling_factor=args.upsampling_factor,
        hidden_units=args.hidden_units_wave,
        hidden_units_2=args.hidden_units_wave_2,
        kernel_size=args.kernel_size_wave,
        dilation_size=args.dilation_size_wave,
        seg=args.seg,
        lpc=args.lpc,
        causal_conv=args.causal_conv_wave,
        right_size=args.right_size,
        do_prob=args.do_prob)
    logging.info(model_waveform)
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_l1 = torch.nn.L1Loss(reduction='none')

    # send to gpu
    if torch.cuda.is_available():
        model_waveform.cuda()
        criterion_ce.cuda()
        criterion_l1.cuda()
        mean_stats = mean_stats.cuda()
        scale_stats = scale_stats.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)

    model_waveform.train()

    model_waveform.scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/scale_stats.data),2))
    model_waveform.scale_in.bias = torch.nn.Parameter(-(mean_stats.data/scale_stats.data))

    for param in model_waveform.parameters():
        param.requires_grad = True
    for param in model_waveform.scale_in.parameters():
        param.requires_grad = False

    parameters = filter(lambda p: p.requires_grad, model_waveform.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (waveform): %.3f million' % parameters)

    module_list = list(model_waveform.conv.parameters())
    #module_list += list(model_waveform.conv_s_c.parameters()) + list(model_waveform.conv_s_x.parameters())
    module_list += list(model_waveform.conv_s_c.parameters()) + list(model_waveform.embed_wav.parameters())
    module_list += list(model_waveform.gru.parameters()) + list(model_waveform.gru_2.parameters())
    module_list += list(model_waveform.out.parameters())

    optimizer = RAdam(module_list, lr=args.lr)
    #optimizer = torch.optim.Adam(module_list, lr=args.lr)

    # resume
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained)
        model_waveform.load_state_dict(checkpoint["model_waveform"])
        epoch_idx = checkpoint["iterations"]
        logging.info("pretrained from %d-iter checkpoint." % epoch_idx)
        epoch_idx = 0
    elif args.resume is not None:
        checkpoint = torch.load(args.resume)
        model_waveform.load_state_dict(checkpoint["model_waveform"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)
    else:
        epoch_idx = 0

    def zero_wav_pad(x): return padding(x, args.pad_len*args.upsampling_factor, value=0.0)  # noqa: E704
    def zero_feat_pad(x): return padding(x, args.pad_len, value=0.0)  # noqa: E704
    pad_wav_transform = transforms.Compose([zero_wav_pad])
    pad_feat_transform = transforms.Compose([zero_feat_pad])

    wav_transform = transforms.Compose([lambda x: encode_mu_law(x, args.n_quantize)])

    # define generator training
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
    assert len(wav_list) == len(feat_list)
    logging.info("number of training data = %d." % len(feat_list))
    dataset = FeatureDatasetNeuVoco(wav_list, feat_list, pad_wav_transform, pad_feat_transform, args.upsampling_factor, 
                    args.string_path, wav_transform=wav_transform)
                    #args.string_path, wav_transform=wav_transform, with_excit=False)
                    #args.string_path, wav_transform=wav_transform, with_excit=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_utt, shuffle=True, num_workers=args.n_workers)
    #generator = data_generator(dataloader, device, args.batch_size, args.upsampling_factor, limit_count=1, seg=args.seg, batch_sizes=batch_sizes)
    #generator = data_generator(dataloader, device, args.batch_size, args.upsampling_factor, limit_count=None, seg=args.seg, batch_sizes=batch_sizes)
    #generator = data_generator(dataloader, device, args.batch_size, args.upsampling_factor, limit_count=1, seg=args.seg)
    generator = data_generator(dataloader, device, args.batch_size, args.upsampling_factor, limit_count=None, seg=args.seg)

    # define generator evaluation
    if os.path.isdir(args.waveforms_eval):
        filenames = sorted(find_files(args.waveforms_eval, "*.wav", use_dir_name=False))
        wav_list_eval = [args.waveforms + "/" + filename for filename in filenames]
    elif os.path.isfile(args.waveforms_eval):
        wav_list_eval = read_txt(args.waveforms_eval)
    else:
        logging.error("--waveforms_eval should be directory or list.")
        sys.exit(1)
    if os.path.isdir(args.feats_eval):
        feat_list_eval = [args.feats_eval + "/" + filename for filename in filenames]
    elif os.path.isfile(args.feats):
        feat_list_eval = read_txt(args.feats_eval)
    else:
        logging.error("--feats_eval should be directory or list.")
        sys.exit(1)
    assert len(wav_list_eval) == len(feat_list_eval)
    logging.info("number of evaluation data = %d." % len(feat_list_eval))
    dataset_eval = FeatureDatasetNeuVoco(wav_list_eval, feat_list_eval, pad_wav_transform, pad_feat_transform, args.upsampling_factor, 
                    args.string_path, wav_transform=wav_transform)
                    #args.string_path, wav_transform=wav_transform, with_excit=False)
                    #args.string_path, wav_transform=wav_transform, with_excit=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size_utt_eval, shuffle=False, num_workers=args.n_workers)
    ##generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=1, seg=args.seg)
    #generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=None, seg=args.seg)
    #generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=1)
    #generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=None)
    #generator_eval = data_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=1)
    generator_eval = data_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=None)

    writer = SummaryWriter(args.expdir)
    total_train_loss = defaultdict(list)
    total_eval_loss = defaultdict(list)

    flag_conv_s_c = False
    #conv_s_c_params = model_waveform.conv_s_c.weight.shape[0]*model_waveform.conv_s_c.weight.shape[1]
    #densities_conv_s_c = [None]*args.n_stage
    #density_conv_s_c = float(16384/conv_s_c_params)
    #if conv_s_c_params > 16384:
    #    density_delta_conv_s_c = (1-density_conv_s_c)/args.n_stage
    #    logging.info(density_conv_s_c)
    #    logging.info(density_delta_conv_s_c)
    #    densities_conv_s_c[0] = 1-density_delta_conv_s_c
    #    for i in range(1,args.n_stage):
    #        if i < args.n_stage-1:
    #            densities_conv_s_c[i] = densities_conv_s_c[i-1]-density_delta_conv_s_c
    #        else:
    #            densities_conv_s_c[i] = density_conv_s_c
    #    logging.info(densities_conv_s_c)
    #    flag_conv_s_c = True

    flag_out = False
    out_params = model_waveform.out.out.weight.shape[0]*model_waveform.out.out.weight.shape[1]
    densities_out = [None]*args.n_stage
    density_out = float(8192/out_params)
    if out_params > 8192:
        density_delta_out = (1-density_out)/args.n_stage
        logging.info(density_out)
        logging.info(density_delta_out)
        densities_out[0] = 1-density_delta_out
        for i in range(1,args.n_stage):
            if i < args.n_stage-1:
                densities_out[i] = densities_out[i-1]-density_delta_out
            else:
                densities_out[i] = density_out
        logging.info(densities_out)
        flag_out = True

    #sparsify = lpcnet.Sparsify(2000, 40000, 400, (0.05, 0.05, 0.2))
    density_deltas_ = args.densities.split('-')
    density_deltas = [None]*len(density_deltas_)
    for i in range(len(density_deltas_)):
        density_deltas[i] = (1-float(density_deltas_[i]))/args.n_stage
    t_deltas = [None]*args.n_stage
    t_starts = [None]*args.n_stage
    t_ends = [None]*args.n_stage
    densities = [None]*args.n_stage
    t_delta = args.t_end - args.t_start + 1
    #t_deltas[0] = round((1/(args.n_stage-1))*0.6*t_delta)
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
            #t_deltas[i] = round((1/(args.n_stage-1))*0.6*t_delta)
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
            #t_deltas[i] = round(0.4*t_delta)
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
    total = 0
    iter_count = 0
    batch_x_i = [None]*model_waveform.seg
    batch_x_output_i = [None]*model_waveform.seg
    batch_loss_ce = [None]*model_waveform.seg
    batch_loss_prc = [None]*model_waveform.seg
    batch_loss_ce_sum_select_ = [None]*model_waveform.seg
    batch_loss_prc_sum_select_ = [None]*model_waveform.seg
    loss_ce = [None]*model_waveform.seg
    loss_prc = [None]*model_waveform.seg
    eval_loss_ce = [None]*model_waveform.seg
    eval_loss_ce_std = [None]*model_waveform.seg
    eval_loss_prc = [None]*model_waveform.seg
    eval_loss_prc_std = [None]*model_waveform.seg
    min_eval_loss_ce = [None]*model_waveform.seg
    min_eval_loss_ce_std = [None]*model_waveform.seg
    min_eval_loss_prc = [None]*model_waveform.seg
    min_eval_loss_prc_std = [None]*model_waveform.seg
    for i in range(model_waveform.seg):
        loss_ce[i] = []
        loss_prc[i] = []
    prev_n_batch_utt = args.batch_size_utt
    min_eval_loss_ce[-1] = 99999999.99
    min_eval_loss_ce_std[-1] = 99999999.99
    iter_idx = 0
    min_idx = -1
    #min_eval_loss_ce[0] = 2.199576
    #min_eval_loss_ce_std[0] = 0.817393
    #iter_idx = 69908
    #min_idx = 6
    while idx_stage < args.n_stage-1 and iter_idx + 1 >= t_starts[idx_stage+1]:
        idx_stage += 1
        logging.info(idx_stage)
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        batch_x, batch_feat, c_idx, utt_idx, featfile, x_bs, f_bs, x_ss, f_ss, n_batch_utt, \
            del_index_utt, max_slen, slens_acc, idx_select, idx_select_full = next(generator)
        if c_idx < 0: # summarize epoch
            # save current epoch model
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # report current epoch
            text_log = "(EPOCH:%d) average optimization loss =" % (epoch_idx + 1)
            for i in range(model_waveform.seg_1,-1,-1):
                text_log += " [%d] %.6f (+- %.6f) %.6f (+- %.6f) %% ;" % (i+1, \
                    np.mean(loss_ce[i]), np.std(loss_ce[i]), np.mean(loss_prc[i]), np.std(loss_prc[i]))
            logging.info("%s; (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            logging.info("estimated time until max. epoch = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * total))))
            # compute loss in evaluation data
            total = 0
            iter_count = 0
            for i in range(model_waveform.seg):
                loss_ce[i] = []
                loss_prc[i] = []
            model_waveform.eval()
            for param in model_waveform.parameters():
                param.requires_grad = False
            logging.info("Evaluation data")
            while True:
                with torch.no_grad():
                    start = time.time()
                    batch_x, batch_feat, c_idx, utt_idx, featfile, x_bs, f_bs, x_ss, f_ss, n_batch_utt, \
                        del_index_utt, max_slen, slens_acc, idx_select, idx_select_full = next(generator_eval)
                    if c_idx < 0:
                        break

                    x_es = x_ss+x_bs
                    f_es = f_ss+f_bs

                    logging.info("%d %d %d %d %d" % (max_slen, x_ss, x_bs, f_ss, f_bs))
                    if max_slen >= x_es:
                        if model_waveform.lpc > 0:
                            # lpc from 1st group to segth group (index t-(seg-1) to t) --> index lpc: t-(seg-1)-n_lpc, seg-1+lpc = lpc_offset
                            if x_ss-model_waveform.lpc >= 0:
                                batch_x_lpc = batch_x[:,x_ss-model_waveform.lpc:x_es-1]
                            else:
                                batch_x_lpc = F.pad(batch_x[:,:x_es-1], (-(x_ss-model_waveform.lpc), 0), "constant", args.n_quantize // 2) 
                        # seg+seg-1 -> generate seg group samples [1st,2nd,segth_samples] in seg, seg-1+seg = seg_offset
                        if x_ss-model_waveform.seg >= 0:
                            batch_x_prev = batch_x[:,x_ss-model_waveform.seg:x_es-model_waveform.seg]
                        else:
                            batch_x_prev = F.pad(batch_x[:,:x_es-model_waveform.seg], (-(x_ss-model_waveform.seg), 0), "constant", args.n_quantize // 2)
                        batch_x = batch_x[:,x_ss:x_es]
                        batch_feat = batch_feat[:,f_ss:f_es]
                    else:
                        if model_waveform.lpc > 0:
                            # lpc from 1st group to segth group (index t-(seg-1) to t) --> index lpc: t-(seg-1)-n_lpc, seg-1+lpc = lpc_offset
                            if x_ss-model_waveform.lpc >= 0:
                                batch_x_lpc = batch_x[:,x_ss-model_waveform.lpc:-1]
                            else:
                                batch_x_lpc = F.pad(batch_x[:,:-1], (-(x_ss-model_waveform.lpc), 0), "constant", args.n_quantize // 2)
                        # seg+seg-1 -> generate seg group samples [1st,2nd,segth_samples] in seg, seg-1+seg = seg_offset
                        if x_ss-model_waveform.seg_offset >= 0:
                            batch_x_prev = batch_x[:,x_ss-model_waveform.seg:-model_waveform.seg]
                        else:
                            batch_x_prev = F.pad(batch_x[:,:-model_waveform.seg], (-(x_ss-model_waveform.seg), 0), "constant", args.n_quantize // 2)
                        batch_x = batch_x[:,x_ss:]
                        batch_feat = batch_feat[:,f_ss:]

                    # prev ground-truth wave. for prediction calc. for each 1 shift segment
                    if model_waveform.lpc > 0:
                        # B x T --> B x T x K --> B x T_seg x K x seg --> B x T_seg x seg x K
                        x_lpc = batch_x_lpc.unfold(1, model_waveform.lpc, 1).unfold(1, model_waveform.seg, model_waveform.seg).permute(0,1,3,2)

                    # feedforward
                    if f_ss > 0:
                        if len(del_index_utt) > 0:
                            h_x = torch.FloatTensor(np.delete(h_x.cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                            h_x_2 = torch.FloatTensor(np.delete(h_x_2.cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                        if model_waveform.lpc > 0:
                            batch_x_output, h_x, h_x_2 = model_waveform(batch_feat, batch_x_prev, x_lpc=x_lpc, h=h_x, h_2=h_x_2, shift1=False)
                        else:
                            batch_x_output, h_x, h_x_2 = model_waveform(batch_feat, batch_x_prev, h=h_x, h_2=h_x_2, shift1=False)
                    else:
                        if model_waveform.lpc > 0:
                            batch_x_output, h_x, h_x_2 = model_waveform(batch_feat, batch_x_prev, x_lpc=x_lpc, shift1=False)
                        else:
                            batch_x_output, h_x, h_x_2 = model_waveform(batch_feat, batch_x_prev, shift1=False)

                    j = np.random.randint(0, batch_x_output.shape[0])
                    logging.info("%s" % (os.path.join(os.path.basename(os.path.dirname(featfile[j])),os.path.basename(featfile[j]))))

                    # handle short ending
                    if len(idx_select) > 0:
                        logging.info('len_idx_select: '+str(len(idx_select)))
                        batch_loss_ce_sum_select = 0
                        batch_loss_prc_sum_select = 0
                        for j in range(len(idx_select)):
                            k = idx_select[j]
                            slens_utt = slens_acc[k]
                            logging.info('%s %d' % (featfile[k], slens_utt))
                            batch_x_i_ = batch_x[k,:slens_utt]
                            batch_x_output_i_ = batch_x_output[k,:slens_utt]
                            # loss
                            batch_loss_ce_sum_select += torch.mean(criterion_ce(batch_x_output_i_, batch_x_i_))
                            batch_loss_prc_sum_select += torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_output_i_, dim=-1), F.one_hot(batch_x_i_, num_classes=args.n_quantize).float()), -1))
                        batch_loss_ce_sum_select /= len(idx_select)
                        batch_loss_prc_sum_select /= len(idx_select)
                        total_eval_loss["eval/loss_ce-%d"%(model_waveform.seg)].append(batch_loss_ce_sum_select.item())
                        loss_ce[-1].append(batch_loss_ce_sum_select.item())
                        total_eval_loss["eval/loss_prc-%d"%(model_waveform.seg)].append(batch_loss_prc_sum_select.item())
                        loss_prc[-1].append(batch_loss_prc_sum_select.item())
                        if len(idx_select_full) > 0:
                            logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                            batch_x = torch.index_select(batch_x, 0, idx_select_full)
                            batch_x_output = torch.index_select(batch_x_output, 0, idx_select_full)
                        else:
                            text_log = "batch loss_select %.3f %.3f " % (batch_loss_ce_sum_select, batch_loss_prc_sum_select)
                            logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                            iter_count += 1
                            total += time.time() - start
                            continue

                    # loss
                    batch_loss_ce[-1] = torch.mean(criterion_ce(batch_x_output.reshape(-1, args.n_quantize), batch_x.reshape(-1))).item()
                    total_eval_loss["eval/loss_ce-%d"%(model_waveform.seg)].append(batch_loss_ce[-1])
                    loss_ce[-1].append(batch_loss_ce[-1])
                    batch_loss_prc[-1] = torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_output, dim=-1), F.one_hot(batch_x, num_classes=args.n_quantize).float()), -1)).item()
                    total_eval_loss["eval/loss_prc-%d"%(model_waveform.seg)].append(batch_loss_prc[-1])
                    loss_prc[-1].append(batch_loss_prc[-1])

                    text_log = "batch eval loss [%d] %d %d %d %d %d :" % (c_idx+1, max_slen, x_ss, x_bs, f_ss, f_bs)
                    text_log += " %.3f %.3f %% ;" % (batch_loss_ce[-1], batch_loss_prc[-1])
                    logging.info("%s; (%.3f sec)" % (text_log, time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            logging.info('sme')
            for key in total_eval_loss.keys():
                total_eval_loss[key] = np.mean(total_eval_loss[key])
                logging.info(f"(Steps: {iter_idx}) {key} = {total_eval_loss[key]:.4f}.")
            write_to_tensorboard(writer, iter_idx, total_eval_loss)
            total_eval_loss = defaultdict(list)
            eval_loss_ce[-1] = np.mean(loss_ce[-1])
            eval_loss_ce_std[-1] = np.std(loss_ce[-1])
            eval_loss_prc[-1] = np.mean(loss_prc[-1])
            eval_loss_prc_std[-1] = np.std(loss_prc[-1])
            text_log = "(EPOCH:%d) average evaluation loss =" % (epoch_idx + 1)
            text_log += " %.6f (+- %.6f) %.6f (+- %.6f) %% ;" % (eval_loss_ce[-1], eval_loss_ce_std[-1], \
                        eval_loss_prc[-1], eval_loss_prc_std[-1])
            logging.info("%s; (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            if (eval_loss_ce[-1]+eval_loss_ce_std[-1]) <= (min_eval_loss_ce[-1]+min_eval_loss_ce_std[-1]) \
                or (eval_loss_ce[-1] <= min_eval_loss_ce[-1]):
                min_eval_loss_ce[-1] = eval_loss_ce[-1]
                min_eval_loss_ce_std[-1] = eval_loss_ce_std[-1]
                min_eval_loss_prc[-1] = eval_loss_prc[-1]
                min_eval_loss_prc_std[-1] = eval_loss_prc_std[-1]
                min_idx = epoch_idx
                change_min_flag = True
            if change_min_flag:
                text_log = "min_eval_loss ="
                text_log += " %.6f (+- %.6f) %.6f (+- %.6f) %% ;" % (min_eval_loss_ce[-1], min_eval_loss_ce_std[-1], \
                            min_eval_loss_prc[-1], min_eval_loss_prc_std[-1])
                logging.info("%s; min_idx=%d" % (text_log, min_idx + 1))
            #if ((epoch_idx + 1) % args.save_interval_epoch == 0) or (epoch_min_flag):
            #    logging.info('save epoch:%d' % (epoch_idx+1))
            #    save_checkpoint(args.expdir, model_waveform, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1)
            logging.info('save epoch:%d' % (epoch_idx+1))
            save_checkpoint(args.expdir, model_waveform, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1)
            prev_n_batch_utt = args.batch_size_utt
            total = 0
            iter_count = 0
            for i in range(model_waveform.seg):
                loss_ce[i] = []
                loss_prc[i] = []
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model_waveform.train()
            for param in model_waveform.parameters():
                param.requires_grad = True
            for param in model_waveform.scale_in.parameters():
                param.requires_grad = False
            # start next epoch
            if epoch_idx < args.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_x, batch_feat, c_idx, utt_idx, featfile, x_bs, f_bs, x_ss, f_ss, n_batch_utt, \
                    del_index_utt, max_slen, slens_acc, idx_select, idx_select_full = next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

            x_es = x_ss+x_bs
            f_es = f_ss+f_bs

            logging.info("%d %d %d %d %d" % (max_slen, x_ss, x_bs, f_ss, f_bs))
            if max_slen >= x_es:
                if model_waveform.lpc > 0:
                    # lpc from 1st group to segth group (index t-(seg-1) to t) --> index lpc: t-(seg-1)-n_lpc, seg-1+lpc = lpc_offset
                    if x_ss-model_waveform.lpc_offset >= 0:
                        batch_x_lpc = batch_x[:,x_ss-model_waveform.lpc_offset:x_es-1]
                    else:
                        batch_x_lpc = F.pad(batch_x[:,:x_es-1], (-(x_ss-model_waveform.lpc_offset), 0), "constant", args.n_quantize // 2) 
                # seg+seg-1 -> generate seg group samples [1st,2nd,segth_samples] in seg, seg-1+seg = seg_offset
                if x_ss-model_waveform.seg_offset >= 0:
                    batch_x_prev = batch_x[:,x_ss-model_waveform.seg_offset:x_es-model_waveform.seg]
                else:
                    batch_x_prev = F.pad(batch_x[:,:x_es-model_waveform.seg], (-(x_ss-model_waveform.seg_offset), 0), "constant", args.n_quantize // 2)
                batch_x = batch_x[:,x_ss:x_es]
                if f_ss > 0:
                    batch_feat = batch_feat[:,f_ss-1:f_es]
                else:
                    batch_feat = batch_feat[:,:f_es]
            else:
                if model_waveform.lpc > 0:
                    # lpc from 1st group to segth group (index t-(seg-1) to t) --> index lpc: t-(seg-1)-n_lpc, seg-1+lpc = lpc_offset
                    if x_ss-model_waveform.lpc_offset >= 0:
                        batch_x_lpc = batch_x[:,x_ss-model_waveform.lpc_offset:-1]
                    else:
                        batch_x_lpc = F.pad(batch_x[:,:-1], (-(x_ss-model_waveform.lpc_offset), 0), "constant", args.n_quantize // 2)
                # seg+seg-1 -> generate seg group samples [1st,2nd,segth_samples] in seg, seg-1+seg = seg_offset
                if x_ss-model_waveform.seg_offset >= 0:
                    batch_x_prev = batch_x[:,x_ss-model_waveform.seg_offset:-model_waveform.seg]
                else:
                    batch_x_prev = F.pad(batch_x[:,:-model_waveform.seg], (-(x_ss-model_waveform.seg_offset), 0), "constant", args.n_quantize // 2)
                batch_x = batch_x[:,x_ss:]
                if f_ss > 0:
                    batch_feat = batch_feat[:,f_ss-1:]

            # prev ground-truth wave. for prediction calc. for each 1 shift segment
            if model_waveform.lpc > 0:
                # B x T --> B x T x K --> B x T_seg x K x seg --> B x T_seg x seg x K
                x_lpc = batch_x_lpc[:,:-model_waveform.seg_1].unfold(1, model_waveform.lpc, 1).unfold(1, model_waveform.seg, model_waveform.seg).permute(0,1,3,2)
                for i in range(1,model_waveform.seg):
                    if i < model_waveform.seg_1:
                        x_lpc = torch.cat((x_lpc, batch_x_lpc[:,i:-model_waveform.seg_1+i].unfold(1, model_waveform.lpc, 1).unfold(1, model_waveform.seg, model_waveform.seg).permute(0,1,3,2)), 0)
                    else:
                        x_lpc = torch.cat((x_lpc, batch_x_lpc[:,i:].unfold(1, model_waveform.lpc, 1).unfold(1, model_waveform.seg, model_waveform.seg).permute(0,1,3,2)), 0)

            # feedforward
            if f_ss > 0:
                if len(del_index_utt) > 0:
                    idx_batch_seg_s = 0
                    idx_batch_seg_e = prev_n_batch_utt
                    # handle hidden state per group of batch (because of 1 shift even though segment output)
                    for i in range(model_waveform.seg):
                        if i > 0:
                            h_x_ = torch.cat((h_x_, torch.FloatTensor(np.delete(h_x[:,idx_batch_seg_s:idx_batch_seg_e].cpu().data.numpy(), del_index_utt, axis=1)).to(device)), 1)
                            h_x_2_ = torch.cat((h_x_2_, torch.FloatTensor(np.delete(h_x_2[:,idx_batch_seg_s:idx_batch_seg_e].cpu().data.numpy(), del_index_utt, axis=1)).to(device)), 1)
                        else:
                            h_x_ = torch.FloatTensor(np.delete(h_x[:,idx_batch_seg_s:idx_batch_seg_e].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                            h_x_2_ = torch.FloatTensor(np.delete(h_x_2[:,idx_batch_seg_s:idx_batch_seg_e].cpu().data.numpy(), del_index_utt, axis=1)).to(device)
                        idx_batch_seg_s = idx_batch_seg_e
                        idx_batch_seg_e += prev_n_batch_utt
                    h_x = h_x_
                    h_x_2 = h_x_2_
                if model_waveform.lpc > 0:
                    batch_x_output, h_x, h_x_2 = model_waveform(batch_feat, batch_x_prev, h=h_x, h_2=h_x_2, x_lpc=x_lpc, do=True)
                else:
                    batch_x_output, h_x, h_x_2 = model_waveform(batch_feat, batch_x_prev, h=h_x, h_2=h_x_2, do=True)
            else:
                if model_waveform.lpc > 0:
                    batch_x_output, h_x, h_x_2 = model_waveform(batch_feat, batch_x_prev, x_lpc=x_lpc, do=True, first=True)
                else:
                    batch_x_output, h_x, h_x_2 = model_waveform(batch_feat, batch_x_prev, do=True, first=True)
            prev_n_batch_utt = n_batch_utt

            batch_loss = 0 
            # handle short ending
            if len(idx_select) > 0:
                logging.info('len_idx_select: '+str(len(idx_select)))
                batch_loss_ce_sum_select = 0
                for i in range(model_waveform.seg):
                    batch_loss_ce_sum_select_[i] = 0
                    batch_loss_prc_sum_select_[i] = 0
                for j in range(len(idx_select)):
                    k = idx_select[j]
                    slens_utt = slens_acc[k]
                    logging.info('%s %d' % (featfile[k], slens_utt))
                    for i in range(model_waveform.seg): #from t-(seg-1) to t for 1 shift segment grouping
                        batch_x_i_ = batch_x[k,:slens_utt-model_waveform.seg_1+i] # ground truth not include seg_offset at first
                        # discard the leading index for t-(seg-1) due to shift 1 segment output
                        batch_x_output_i_ = batch_x_output[k,model_waveform.seg_1-i:slens_utt]
                        # loss
                        batch_loss_ce_sum_select__ = torch.mean(criterion_ce(batch_x_output_i_, batch_x_i_))
                        batch_loss_ce_sum_select += batch_loss_ce_sum_select__
                        batch_loss_ce_sum_select_[i] += batch_loss_ce_sum_select__
                        batch_loss_prc_sum_select_[i] += torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_output_i_, dim=-1), F.one_hot(batch_x_i_, num_classes=args.n_quantize).float()), -1))
                batch_loss += batch_loss_ce_sum_select
                for i in range(model_waveform.seg):
                    batch_loss_ce_sum_select_[i] /= len(idx_select)
                    total_train_loss["train/loss_ce-%d"%(i+1)].append(batch_loss_ce_sum_select_[i].item())
                    loss_ce[i].append(batch_loss_ce_sum_select_[i].item())
                    batch_loss_prc_sum_select_[i] /= len(idx_select)
                    total_train_loss["train/loss_prc-%d"%(i+1)].append(batch_loss_prc_sum_select_[i].item())
                    loss_prc[i].append(batch_loss_prc_sum_select_[i].item())
                if len(idx_select_full) > 0:
                    logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                    batch_x = torch.index_select(batch_x, 0, idx_select_full)
                    idx_batch_seg_s = 0
                    idx_batch_seg_e = n_batch_utt
                    for i in range(model_waveform.seg):
                        if i > 0:
                            batch_x_output_ = torch.cat((batch_x_output_, torch.index_select(batch_x_output[idx_batch_seg_s:idx_batch_seg_e], 0, idx_select_full)), 0)
                        else:
                            batch_x_output_ = torch.index_select(batch_x_output[idx_batch_seg_s:idx_batch_seg_e], 0, idx_select_full)
                        idx_batch_seg_s = idx_batch_seg_e
                        idx_batch_seg_e += n_batch_utt
                    batch_x_output = batch_x_output_
                else:
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
                        if not flag_conv_s_c:
                            if not flag_out:
                                if idx_stage > 0:
                                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                                else:
                                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])
                            else:
                                if idx_stage > 0:
                                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1], \
                                        density_out=densities_out[idx_stage], density_out_p=densities_out[idx_stage-1])
                                else:
                                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], \
                                        density_out=densities_out[idx_stage])
                        else:
                            if not flag_out:
                                if idx_stage > 0:
                                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1], \
                                                density_conv_s_c=densities_conv_s_c[idx_stage], density_conv_s_c_p=densities_conv_s_c[idx_stage-1])
                                else:
                                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], \
                                                density_conv_s_c=densities_conv_s_c[idx_stage])
                            else:
                                if idx_stage > 0:
                                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1], \
                                                density_conv_s_c=densities_conv_s_c[idx_stage], density_conv_s_c_p=densities_conv_s_c[idx_stage-1], \
                                                    density_out=densities_out[idx_stage], density_out_p=densities_out[idx_stage-1])
                                else:
                                    sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], \
                                                density_conv_s_c=densities_conv_s_c[idx_stage], density_out=densities_out[idx_stage])

                    text_log = "batch loss_select %lf " % (batch_loss.item())
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    iter_idx += 1
                    if iter_idx % args.save_interval_iter == 0:
                        logging.info('save iter:%d' % (iter_idx))
                        save_checkpoint(args.expdir, model_waveform, optimizer, np.random.get_state(), \
                            torch.get_rng_state(), iter_idx)
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

            # batch segment handling
            if len(idx_select) > 0:
                n_batch_utt = len(idx_select_full)
            idx_batch_seg_s = 0
            idx_batch_seg_e = n_batch_utt
            for i in range(model_waveform.seg): #from t-(seg-1) to t for 1 shift segment grouping
                if i < model_waveform.seg_1:
                    batch_x_i[i] = batch_x[:,:-model_waveform.seg_1+i] # ground truth not include seg_offset at first
                else:
                    batch_x_i[i] = batch_x
                # discard the leading index for t-(seg-1) due to shift 1 segment output
                batch_x_output_i[i] = batch_x_output[idx_batch_seg_s:idx_batch_seg_e,model_waveform.seg_1-i:]
                idx_batch_seg_s = idx_batch_seg_e
                idx_batch_seg_e += n_batch_utt

            # loss
            batch_loss_ce_sum = 0
            for i in range(model_waveform.seg_1,-1,-1):
                batch_loss_ce_ = torch.mean(criterion_ce(batch_x_output_i[i].reshape(-1, args.n_quantize), batch_x_i[i].reshape(-1)).reshape(batch_x_output_i[i].shape[0], -1), -1)
                batch_loss_ce[i] = batch_loss_ce_.mean().item()
                total_train_loss["train/loss_ce-%d"%(i+1)].append(batch_loss_ce[i])
                loss_ce[i].append(batch_loss_ce[i])
                batch_loss_ce_sum += batch_loss_ce_.sum()
                batch_loss_prc[i] = torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_output_i[i], dim=-1), F.one_hot(batch_x_i[i], num_classes=args.n_quantize).float()), -1)).item()
                total_train_loss["train/loss_prc-%d"%(i+1)].append(batch_loss_prc[i])
                loss_prc[i].append(batch_loss_prc[i])
                if i == model_waveform.seg_1:
                    i = np.random.randint(0, batch_x_output_i[i].shape[0])
                    logging.info("%s" % (os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i]))))
            batch_loss += batch_loss_ce_sum

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
                if not flag_conv_s_c:
                    if not flag_out:
                        if idx_stage > 0:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1])
                        else:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage])
                    else:
                        if idx_stage > 0:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1], \
                                density_out=densities_out[idx_stage], density_out_p=densities_out[idx_stage-1])
                        else:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], \
                                density_out=densities_out[idx_stage])
                else:
                    if not flag_out:
                        if idx_stage > 0:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1], \
                                        density_conv_s_c=densities_conv_s_c[idx_stage], density_conv_s_c_p=densities_conv_s_c[idx_stage-1])
                        else:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], \
                                        density_conv_s_c=densities_conv_s_c[idx_stage])
                    else:
                        if idx_stage > 0:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], densities_p=densities[idx_stage-1], \
                                        density_conv_s_c=densities_conv_s_c[idx_stage], density_conv_s_c_p=densities_conv_s_c[idx_stage-1], \
                                            density_out=densities_out[idx_stage], density_out_p=densities_out[idx_stage-1])
                        else:
                            sparsify(model_waveform, iter_idx + 1, t_starts[idx_stage], t_ends[idx_stage], args.interval, densities[idx_stage], \
                                        density_conv_s_c=densities_conv_s_c[idx_stage], density_out=densities_out[idx_stage])

            text_log = "batch loss [%d] %d %d %d %d %d :" % (c_idx+1, max_slen, x_ss, x_bs, f_ss, f_bs)
            for i in range(model_waveform.seg_1,-1,-1):
                text_log += " [%d] %.3f %.3f %% ;" % (i+1, batch_loss_ce[i], batch_loss_prc[i])
            logging.info("%s; (%.3f sec)" % (text_log, time.time() - start))
            iter_idx += 1
            if iter_idx % args.save_interval_iter == 0:
                logging.info('save iter:%d' % (iter_idx))
                save_checkpoint(args.expdir, model_waveform, optimizer, np.random.get_state(), torch.get_rng_state(), iter_idx)
            iter_count += 1
            if iter_idx % args.log_interval_steps == 0:
                logging.info('smt')
                for key in total_train_loss.keys():
                    total_train_loss[key] = np.mean(total_train_loss[key])
                    logging.info(f"(Steps: {iter_idx}) {key} = {total_train_loss[key]:.4f}.")
                write_to_tensorboard(writer, iter_idx, total_train_loss)
                total_train_loss = defaultdict(list)
            total += time.time() - start


    # save final model
    model_waveform.cpu()
    torch.save({"model_waveform": model_waveform.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
