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
from vcneuvoco import GRU_POST_NET
from vcneuvoco import sampling_laplace, kl_laplace, ModulationSpectrumLoss, LaplaceLoss

import torch_optimizer as optim

from dataset import FeatureDatasetVAE, padding

import librosa
from dtw_c import dtw_c as dtw

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)


def train_generator(dataloader, device, batch_size, limit_count=None):
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
            sc = batch['sc'][:,:max_flen].to(device)
            featfiles = batch['featfile']
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
                    featfiles = np.delete(featfiles, del_index_utt, axis=0)
                    flens_acc = np.delete(flens_acc, del_index_utt, axis=0)
                    n_batch_utt -= len(del_index_utt)
                for i in range(n_batch_utt):
                    if flens_acc[i] < f_bs:
                        idx_select.append(i)
                if len(idx_select) > 0:
                    idx_select_full = torch.LongTensor(np.delete(np.arange(n_batch_utt), idx_select, axis=0)).to(device)
                    idx_select = torch.LongTensor(idx_select).to(device)
                yield feat, feat_magsp, sc, c_idx, idx, featfiles, f_bs, f_ss, flens, \
                    n_batch_utt, del_index_utt, max_flen, idx_select, idx_select_full, flens_acc
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

        yield [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], []


def eval_generator(dataloader, device, batch_size, limit_count=None):
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
            sc = batch['sc'][:,:max_flen].to(device)
            featfiles = batch['featfile']
            n_batch_utt = feat.size(0)

            len_frm = max_flen
            f_ss = 0
            f_bs = batch_size
            delta_frm = batch_size
            flens_acc = np.array(flens)
            yield feat, feat_magsp, sc, c_idx, idx, featfiles, flens_acc, n_batch_utt, max_flen

            if limit_count is not None and count > limit_count:
                break
            c_idx += 1
            count += 1
            #if c_idx > 0:
            #if c_idx > 1:
            #if c_idx > 2:
            #    break

        yield [], [], [], -1, -1, [], [], [], []


def save_checkpoint(checkpoint_dir, model_encoders, n_enc, model_decoder, model_post, model_classifier,
        min_eval_loss_melsp_x_sum_post_dB, min_eval_loss_melsp_x_sum_post_dB_std,
        min_eval_loss_magsp_x_sum_post_dB, min_eval_loss_magsp_x_sum_post_dB_std,
        iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, iterations, model_spkidtr=None):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    for i in range(n_enc):
        model_encoders[i].cpu()
    model_decoder.cpu()
    model_post.cpu()
    model_classifier.cpu()
    checkpoint = {
        "model_decoder": model_decoder.state_dict(),
        "model_post": model_post.state_dict(),
        "model_classifier": model_classifier.state_dict(),
        "min_eval_loss_melsp_x_sum_post_dB": min_eval_loss_melsp_x_sum_post_dB,
        "min_eval_loss_melsp_x_sum_post_dB_std": min_eval_loss_melsp_x_sum_post_dB_std,
        "min_eval_loss_magsp_x_sum_post_dB": min_eval_loss_melsp_x_sum_post_dB,
        "min_eval_loss_magsp_x_sum_post_dB_std": min_eval_loss_melsp_x_sum_post_dB_std,
        "iter_idx": iter_idx,
        "min_idx": min_idx,
        "optimizer": optimizer.state_dict(),
        "numpy_random_state": numpy_random_state,
        "torch_random_state": torch_random_state,
        "iterations": iterations}
    for i in range(n_enc):
        checkpoint["model_encoder-%d"%(i+1)] = model_encoders[i].state_dict()
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    for i in range(n_enc):
        model_encoders[i].cuda()
    model_decoder.cuda()
    model_post.cuda()
    model_classifier.cuda()
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
    parser.add_argument("--feats_eval", required=True,
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
    parser.add_argument("--kernel_size_post", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_post", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--spk_list", required=True,
                        type=str, help="kernel size of dilated causal convolution")
    parser.add_argument("--stats_list", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument("--lat_dim", default=64,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--mel_dim", default=80,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--fftl", default=2048,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--fs", default=24000,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--n_enc", default=4,
                        type=int, help="kernel size of dilated causal convolution")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=30,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--epoch_count", default=500,
                        type=int, help="number of training epochs")
    parser.add_argument("--do_prob", default=0.5,
                        type=float, help="dropout probability")
    parser.add_argument("--batch_size_utt", default=8,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--batch_size_utt_eval", default=14,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--n_workers", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_enc", default=False,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_dec", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_post", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_enc", default=2,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_dec", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--right_size_post", default=0,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
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

    mean_stats = torch.FloatTensor(read_hdf5(args.stats, "/mean_melsp"))
    scale_stats = torch.FloatTensor(read_hdf5(args.stats, "/scale_melsp"))

    # save args as conf
    #args.fftsize = 2 ** (len(bin(args.batch_size)) - 2 + 1)
    args.fftsize = 2 ** (len(bin(500)) - 2 + 1)
    args.string_path = "/log_1pmelmagsp"
    if n_spk <= 32:
        args.hidden_units_cls = 32
    elif n_spk <= 64:
        args.hidden_units_cls = 64
    elif n_spk <= 128:
        args.hidden_units_cls = 128
    elif n_spk <= 256:
        args.hidden_units_cls = 256
    else:
        args.hidden_units_cls = 512
    torch.save(args, args.expdir + "/model.conf")

    # define network
    model_encoders = [None]*args.n_enc
    for i in range(args.n_enc):
        model_encoders[i] = GRU_VAE_ENCODER(
            in_dim=args.mel_dim,
            lat_dim=args.lat_dim,
            hidden_layers=args.hidden_layers_enc,
            hidden_units=args.hidden_units_enc,
            kernel_size=args.kernel_size_enc,
            dilation_size=args.dilation_size_enc,
            causal_conv=args.causal_conv_enc,
            pad_first=True,
            right_size=args.right_size_enc,
            n_spk=None,
            do_prob=args.do_prob)
        logging.info(model_encoders[i])
    model_decoder = GRU_SPEC_DECODER(
        feat_dim=args.lat_dim,
        out_dim=args.mel_dim,
        hidden_layers=args.hidden_layers_dec,
        hidden_units=args.hidden_units_dec,
        kernel_size=args.kernel_size_dec,
        dilation_size=args.dilation_size_dec,
        causal_conv=args.causal_conv_dec,
        pad_first=True,
        right_size=args.right_size_dec,
        n_spk=None,
        do_prob=args.do_prob)
    logging.info(model_decoder)
    model_post = GRU_POST_NET(
        spec_dim=args.mel_dim,
        excit_dim=None,
        n_spk=None,
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
    model_classifier = GRU_LAT_FEAT_CLASSIFIER(
        lat_dim=args.lat_dim,
        feat_dim=args.mel_dim,
        feat_aux_dim=args.fftl//2+1,
        n_spk=n_spk,
        hidden_units=args.hidden_units_cls,
        hidden_layers=1)
    logging.info(model_classifier)
    criterion_ms = ModulationSpectrumLoss(args.fftsize, post=True)
    criterion_laplace = LaplaceLoss()
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_l1 = torch.nn.L1Loss(reduction='none')
    criterion_l2 = torch.nn.MSELoss(reduction='none')
    melfb_t = torch.FloatTensor(np.linalg.pinv(librosa.filters.mel(args.fs, args.fftl, n_mels=args.mel_dim)).T)

    # send to gpu
    if torch.cuda.is_available():
        for i in range(args.n_enc):
            model_encoders[i].cuda()
        model_decoder.cuda()
        model_post.cuda()
        model_classifier.cuda()
        criterion_ms.cuda()
        criterion_laplace.cuda()
        criterion_ce.cuda()
        criterion_l1.cuda()
        criterion_l2.cuda()
        mean_stats = mean_stats.cuda()
        scale_stats = scale_stats.cuda()
        melfb_t = melfb_t.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)

    for i in range(args.n_enc):
        model_encoders[i].train()
    model_decoder.train()
    model_post.train()
    model_classifier.train()

    for i in range(args.n_enc):
        if model_encoders[i].use_weight_norm:
            torch.nn.utils.remove_weight_norm(model_encoders[i].scale_in)
    if model_decoder.use_weight_norm:
        torch.nn.utils.remove_weight_norm(model_decoder.scale_out)
    if model_post.use_weight_norm:
        torch.nn.utils.remove_weight_norm(model_post.scale_in)
        torch.nn.utils.remove_weight_norm(model_post.scale_out)

    for i in range(args.n_enc):
        model_encoders[i].scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/scale_stats.data),2))
        model_encoders[i].scale_in.bias = torch.nn.Parameter(-(mean_stats.data/scale_stats.data))
    model_decoder.scale_out.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(scale_stats.data),2))
    model_decoder.scale_out.bias = torch.nn.Parameter(mean_stats.data)
    model_post.scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/scale_stats.data),2))
    model_post.scale_in.bias = torch.nn.Parameter(-(mean_stats.data/scale_stats.data))
    model_post.scale_out.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(scale_stats.data),2))
    model_post.scale_out.bias = torch.nn.Parameter(mean_stats.data)

    for i in range(args.n_enc):
        if model_encoders[i].use_weight_norm:
            torch.nn.utils.weight_norm(model_encoders[i].scale_in)
    if model_decoder.use_weight_norm:
        torch.nn.utils.weight_norm(model_decoder.scale_out)
    if model_post.use_weight_norm:
        torch.nn.utils.weight_norm(model_post.scale_in)
        torch.nn.utils.weight_norm(model_post.scale_out)

    for i in range(args.n_enc):
        parameters = filter(lambda p: p.requires_grad, model_encoders[i].parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        logging.info('Trainable Parameters (encoder-%d): %.3f million' % (i+1,parameters))
    parameters = filter(lambda p: p.requires_grad, model_decoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_post.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (post): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_classifier.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (classifier): %.3f million' % parameters)

    for i in range(args.n_enc):
        for param in model_encoders[i].parameters():
            param.requires_grad = True
        for param in model_encoders[i].scale_in.parameters():
            param.requires_grad = False
    for param in model_decoder.parameters():
        param.requires_grad = False
    for param in model_post.parameters():
        param.requires_grad = False
    for param in model_classifier.parameters():
        param.requires_grad = False

    module_list = list(model_encoders[0].conv.parameters())
    module_list += list(model_encoders[0].gru.parameters()) + list(model_encoders[0].out.parameters())

    for i in range(1, args.n_enc):
        module_list += list(model_encoders[i].conv.parameters())
        module_list += list(model_encoders[i].gru.parameters()) + list(model_encoders[i].out.parameters())

    # model = ...
    optimizer = optim.RAdam(
        module_list,
        lr= args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    checkpoint = torch.load(args.gen_model)
    model_decoder.load_state_dict(checkpoint["model_decoder"])
    model_post.load_state_dict(checkpoint["model_post"])
    model_classifier.load_state_dict(checkpoint["model_classifier"])
    epoch_idx = checkpoint["iterations"]
    logging.info("gen_model from %d-iter checkpoint." % epoch_idx)
    epoch_idx = 0

    # resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        for i in range(args.n_enc):
            model_encoders[i].load_state_dict(checkpoint["model_encoder-"%(i+1)])
        model_decoder.load_state_dict(checkpoint["model_decoder"])
        model_post.load_state_dict(checkpoint["model_post"])
        model_classifier.load_state_dict(checkpoint["model_classifier"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)

    def zero_feat_pad(x): return padding(x, args.pad_len, value=None)
    pad_feat_transform = transforms.Compose([zero_feat_pad])

    stats_list = args.stats_list.split('@')

    if os.path.isdir(args.feats):
        feat_list = [args.feats + "/" + filename for filename in filenames]
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)
    logging.info("number of training data = %d." % len(feat_list))
    dataset = FeatureDatasetVAE(feat_list, pad_feat_transform, args.string_path, magsp=True, spk_list=spk_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_utt, shuffle=True, num_workers=args.n_workers)
    #generator = train_generator(dataloader, device, args.batch_size, limit_count=1)
    #generator = train_generator(dataloader, device, args.batch_size, limit_count=20)
    generator = train_generator(dataloader, device, args.batch_size, limit_count=None)

    # define generator evaluation
    if os.path.isdir(args.feats_eval):
        feat_list_eval = [args.feats_eval + "/" + filename for filename in filenames]
    elif os.path.isfile(args.feats_eval):
        feat_list_eval = read_txt(args.feats_eval)
    else:
        logging.error("--feats_eval should be directory or list.")
        sys.exit(1)
    logging.info("number of evaluation data = %d." % len(feat_list_eval))
    dataset_eval = FeatureDatasetVAE(feat_list_eval, pad_feat_transform, args.string_path, magsp=True, spk_list=spk_list)
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
    enc_pad_left = model_encoders[0].pad_left
    enc_pad_right = model_encoders[0].pad_right
    logging.info(f'enc_pad_left: {enc_pad_left}')
    logging.info(f'enc_pad_right: {enc_pad_right}')
    dec_pad_left = model_decoder.pad_left
    dec_pad_right = model_decoder.pad_right
    logging.info(f'dec_pad_left: {dec_pad_left}')
    logging.info(f'dec_pad_right: {dec_pad_right}')
    post_pad_left = model_post.pad_left
    post_pad_right = model_post.pad_right
    logging.info(f'post_pad_left: {dec_pad_left}')
    logging.info(f'post_pad_right: {dec_pad_right}')
    first_pad_left = enc_pad_left + dec_pad_left + post_pad_left
    first_pad_right = enc_pad_right + dec_pad_right + post_pad_right
    logging.info(f'first_pad_left: {first_pad_left}')
    logging.info(f'first_pad_right: {first_pad_right}')
    outpad_lefts = [None]*3
    outpad_rights = [None]*3
    outpad_lefts[0] = first_pad_left-enc_pad_left
    outpad_rights[0] = first_pad_right-enc_pad_right
    outpad_lefts[1] = outpad_lefts[0]-dec_pad_left
    outpad_rights[1] = outpad_rights[0]-dec_pad_right
    outpad_lefts[2] = outpad_lefts[1]-post_pad_left
    outpad_rights[2] = outpad_rights[1]-post_pad_right
    logging.info(outpad_lefts)
    logging.info(outpad_rights)
    batch_qzx = [None]*args.n_enc
    batch_z = [None]*args.n_enc
    batch_x = [None]*args.n_enc
    batch_z_sc = [None]*args.n_enc
    batch_pxz_post = [None]*args.n_enc
    batch_x_post = [None]*args.n_enc
    batch_x_sc = [None]*args.n_enc
    batch_x_post_sc = [None]*args.n_enc
    h_z = [None]*args.n_enc
    h_x = [None]*args.n_enc
    h_z_sc = [None]*args.n_enc
    h_x_post = [None]*args.n_enc
    h_x_sc = [None]*args.n_enc
    h_x_post_sc = [None]*args.n_enc
    total = 0
    iter_count = 0
    loss_elbo = []
    loss_pxz_post = [None]*args.n_enc
    loss_qzx_pz = [None]*args.n_enc
    loss_sc_z = [None]*args.n_enc
    loss_sc_x = [None]*args.n_enc
    loss_sc_x_post = [None]*args.n_enc
    loss_melsp_x_dB = [None]*args.n_enc
    loss_melsp_x_post_dB = [None]*args.n_enc
    loss_ms_norm_x = [None]*args.n_enc
    loss_ms_err_x = [None]*args.n_enc
    loss_ms_norm_x_post = [None]*args.n_enc
    loss_ms_err_x_post = [None]*args.n_enc
    for i in range(args.n_enc):
        loss_pxz_post[i] = []
        loss_qzx_pz[i] = []
        loss_sc_z[i] = []
        loss_sc_x[i] = []
        loss_sc_x_post[i] = []
        loss_melsp_x_dB[i] = []
        loss_melsp_x_post_dB[i] = []
        loss_ms_norm_x[i] = []
        loss_ms_err_x[i] = []
        loss_ms_norm_x_post[i] = []
        loss_ms_err_x_post[i] = []
    loss_pxz_sum_post = []
    loss_sc_x_sum = []
    loss_sc_x_sum_post = []
    loss_sc_magsp_x_sum = []
    loss_sc_magsp_x_sum_post = []
    loss_melsp_x_sum = []
    loss_melsp_x_sum_dB = []
    loss_melsp_x_sum_post = []
    loss_melsp_x_sum_post_dB = []
    loss_magsp_x_sum = []
    loss_magsp_x_sum_dB = []
    loss_magsp_x_sum_post = []
    loss_magsp_x_sum_post_dB = []
    loss_ms_norm_x_sum = []
    loss_ms_err_x_sum = []
    loss_ms_norm_x_sum_post = []
    loss_ms_err_x_sum_post = []
    loss_ms_norm_magsp_x_sum = []
    loss_ms_err_magsp_x_sum = []
    loss_ms_norm_magsp_x_sum_post = []
    loss_ms_err_magsp_x_sum_post = []
    melsp_x = [None]*args.n_enc
    melsp_x_rest = [None]*args.n_enc
    melsp_x_post = [None]*args.n_enc
    melsp_x_post_rest = [None]*args.n_enc
    batch_loss_pxz_post_ = [None]*args.n_enc
    batch_loss_pxz_post = [None]*args.n_enc
    batch_loss_melsp_x_dB_ = [None]*args.n_enc
    batch_loss_melsp_x_post_dB_ = [None]*args.n_enc
    batch_loss_melsp_x_dB = [None]*args.n_enc
    batch_loss_melsp_x_post_dB = [None]*args.n_enc
    batch_loss_ms_norm_x_ = [None]*args.n_enc
    batch_loss_ms_err_x_ = [None]*args.n_enc
    batch_loss_ms_norm_x_post_ = [None]*args.n_enc
    batch_loss_ms_err_x_post_ = [None]*args.n_enc
    batch_loss_ms_norm_x = [None]*args.n_enc
    batch_loss_ms_err_x = [None]*args.n_enc
    batch_loss_ms_norm_x_post = [None]*args.n_enc
    batch_loss_ms_err_x_post = [None]*args.n_enc
    batch_loss_qzx_pz_ = [None]*args.n_enc
    batch_loss_qzx_pz = [None]*args.n_enc
    batch_loss_sc_z_ = [None]*args.n_enc
    batch_loss_sc_x_ = [None]*args.n_enc
    batch_loss_sc_x_post_ = [None]*args.n_enc
    batch_loss_sc_z = [None]*args.n_enc
    batch_loss_sc_x = [None]*args.n_enc
    batch_loss_sc_x_post = [None]*args.n_enc
    batch_loss_pxz_post = [None]*args.n_enc
    batch_loss_qzx_pz = [None]*args.n_enc
    batch_loss_sc_z = [None]*args.n_enc
    batch_loss_sc_x = [None]*args.n_enc
    batch_loss_sc_x_post = [None]*args.n_enc
    batch_loss_melsp_x_dB = [None]*args.n_enc
    batch_loss_melsp_x_post_dB = [None]*args.n_enc
    batch_loss_ms_norm_x = [None]*args.n_enc
    batch_loss_ms_err_x = [None]*args.n_enc
    batch_loss_ms_norm_x_post = [None]*args.n_enc
    batch_loss_ms_err_x_post = [None]*args.n_enc
    eval_loss_pxz_post = [None]*args.n_enc
    eval_loss_pxz_post_std = [None]*args.n_enc
    eval_loss_qzx_pz = [None]*args.n_enc
    eval_loss_qzx_pz_std = [None]*args.n_enc
    eval_loss_sc_z = [None]*args.n_enc
    eval_loss_sc_z_std = [None]*args.n_enc
    eval_loss_sc_x = [None]*args.n_enc
    eval_loss_sc_x_std = [None]*args.n_enc
    eval_loss_sc_x_post = [None]*args.n_enc
    eval_loss_sc_x_post_std = [None]*args.n_enc
    eval_loss_melsp_x_dB = [None]*args.n_enc
    eval_loss_melsp_x_dB_std = [None]*args.n_enc
    eval_loss_melsp_x_post_dB = [None]*args.n_enc
    eval_loss_melsp_x_post_dB_std = [None]*args.n_enc
    eval_loss_ms_norm_x = [None]*args.n_enc
    eval_loss_ms_norm_x_std = [None]*args.n_enc
    eval_loss_ms_err_x = [None]*args.n_enc
    eval_loss_ms_err_x_std = [None]*args.n_enc
    eval_loss_ms_norm_x_post = [None]*args.n_enc
    eval_loss_ms_norm_x_post_std = [None]*args.n_enc
    eval_loss_ms_err_x_post = [None]*args.n_enc
    eval_loss_ms_err_x_post_std = [None]*args.n_enc
    eval_loss_gv_x = [None]*args.n_enc
    eval_loss_gv_x_post = [None]*args.n_enc
    min_eval_loss_pxz_post = [None]*args.n_enc
    min_eval_loss_pxz_post_std = [None]*args.n_enc
    min_eval_loss_qzx_pz = [None]*args.n_enc
    min_eval_loss_qzx_pz_std = [None]*args.n_enc
    min_eval_loss_sc_z = [None]*args.n_enc
    min_eval_loss_sc_z_std = [None]*args.n_enc
    min_eval_loss_sc_x = [None]*args.n_enc
    min_eval_loss_sc_x_std = [None]*args.n_enc
    min_eval_loss_sc_x_post = [None]*args.n_enc
    min_eval_loss_sc_x_post_std = [None]*args.n_enc
    min_eval_loss_melsp_x_dB = [None]*args.n_enc
    min_eval_loss_melsp_x_dB_std = [None]*args.n_enc
    min_eval_loss_melsp_x_post_dB = [None]*args.n_enc
    min_eval_loss_melsp_x_post_dB_std = [None]*args.n_enc
    min_eval_loss_ms_norm_x = [None]*args.n_enc
    min_eval_loss_ms_norm_x_std = [None]*args.n_enc
    min_eval_loss_ms_err_x = [None]*args.n_enc
    min_eval_loss_ms_err_x_std = [None]*args.n_enc
    min_eval_loss_ms_norm_x_post = [None]*args.n_enc
    min_eval_loss_ms_norm_x_post_std = [None]*args.n_enc
    min_eval_loss_ms_err_x_post = [None]*args.n_enc
    min_eval_loss_ms_err_x_post_std = [None]*args.n_enc
    min_eval_loss_gv_x = [None]*args.n_enc
    min_eval_loss_gv_x_post = [None]*args.n_enc
    tmp_gv_1 = [None]*args.n_enc
    tmp_gv_2 = [None]*args.n_enc
    gv_x = [None]*args.n_enc
    gv_x_post = [None]*args.n_enc
    for i in range(args.n_enc):
        gv_x[i] = [None]*n_spk
        gv_x_post[i] = [None]*n_spk
    gv_x_sum = [None]*n_spk
    gv_x_sum_post = [None]*n_spk
    min_eval_loss_melsp_x_sum_post_dB = 99999999.99
    min_eval_loss_melsp_x_sum_post_dB_std = 99999999.99
    min_eval_loss_magsp_x_sum_post_dB = 99999999.99
    min_eval_loss_magsp_x_sum_post_dB_std = 99999999.99
    iter_idx = 0
    min_idx = -1
    change_min_flag = False
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        min_eval_loss_melsp_x_sum_post_dB = checkpoint["min_eval_loss_melsp_x_sum_post_dB"]
        min_eval_loss_melsp_x_sum_post_dB_std = checkpoint["min_eval_loss_melsp_x_sum_post_dB_std"]
        min_eval_loss_magsp_x_sum_post_dB = checkpoint["min_eval_loss_magsp_x_sum_post_dB"]
        min_eval_loss_magsp_x_sum_post_dB_std = checkpoint["min_eval_loss_magsp_x_sum_post_dB_std"]
        iter_idx = checkpoint["iter_idx"]
        min_idx = checkpoint["min_idx"]
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        batch_feat, batch_feat_magsp, batch_sc, c_idx, utt_idx, featfile, f_bs, f_ss, flens, n_batch_utt, \
            del_index_utt, max_flen, idx_select, idx_select_full, flens_acc = next(generator)
        if c_idx < 0: # summarize epoch
            # save current epoch model
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # report current epoch
            text_log = "(EPOCH:%d) average optimization loss " % (epoch_idx + 1)
            text_log += "%.6f (+- %.6f) ; " % (np.mean(loss_elbo), np.std(loss_elbo))
            for i in range(args.n_enc):
                text_log += "[%d] %.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                    "%.6f (+- %.6f) dB %.6f (+- %.6f) dB ; "\
                    "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (i+1,
                    np.mean(loss_pxz_post[i]), np.std(loss_pxz_post[i]), np.mean(loss_qzx_pz[i]), np.std(loss_qzx_pz[i]),
                    np.mean(loss_sc_z[i]), np.std(loss_sc_z[i]), np.mean(loss_sc_x[i]), np.std(loss_sc_x[i]), np.mean(loss_sc_x_post[i]), np.std(loss_sc_x_post[i]),
                    np.mean(loss_melsp_x_dB[i]), np.std(loss_melsp_x_dB[i]), np.mean(loss_melsp_x_post_dB[i]), np.std(loss_melsp_x_post_dB[i]), 
                    np.mean(loss_ms_norm_x[i]), np.std(loss_ms_norm_x[i]), np.mean(loss_ms_err_x[i]), np.std(loss_ms_err_x[i]),
                    np.mean(loss_ms_norm_x_post[i]), np.std(loss_ms_norm_x_post[i]), np.mean(loss_ms_err_x_post[i]), np.std(loss_ms_err_x_post[i]))
            text_log += "[+] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (
                np.mean(loss_pxz_sum_post), np.std(loss_pxz_sum_post),
                np.mean(loss_sc_x_sum), np.std(loss_sc_x_sum), np.mean(loss_sc_x_sum_post), np.std(loss_sc_x_sum_post),
                np.mean(loss_sc_magsp_x_sum), np.std(loss_sc_magsp_x_sum), np.mean(loss_sc_magsp_x_sum_post), np.std(loss_sc_magsp_x_sum_post),
                np.mean(loss_melsp_x_sum), np.std(loss_melsp_x_sum), np.mean(loss_melsp_x_sum_dB), np.std(loss_melsp_x_sum_dB),
                np.mean(loss_melsp_x_sum_post), np.std(loss_melsp_x_sum_post), np.mean(loss_melsp_x_sum_post_dB), np.std(loss_melsp_x_sum_post_dB),
                np.mean(loss_magsp_x_sum), np.std(loss_magsp_x_sum), np.mean(loss_magsp_x_sum_dB), np.std(loss_magsp_x_sum_dB),
                np.mean(loss_magsp_x_sum_post), np.std(loss_magsp_x_sum_post), np.mean(loss_magsp_x_sum_post_dB), np.std(loss_magsp_x_sum_post_dB),
                np.mean(loss_ms_norm_x_sum), np.std(loss_ms_norm_x_sum), np.mean(loss_ms_err_x_sum), np.std(loss_ms_err_x_sum),
                np.mean(loss_ms_norm_x_sum_post), np.std(loss_ms_norm_x_sum_post), np.mean(loss_ms_err_x_sum_post), np.std(loss_ms_err_x_sum_post),
                np.mean(loss_ms_norm_magsp_x_sum), np.std(loss_ms_norm_magsp_x_sum), np.mean(loss_ms_err_magsp_x_sum), np.std(loss_ms_err_magsp_x_sum),
                np.mean(loss_ms_norm_magsp_x_sum_post), np.std(loss_ms_norm_magsp_x_sum_post), np.mean(loss_ms_err_magsp_x_sum_post), np.std(loss_ms_err_magsp_x_sum_post))
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            logging.info("estimated time until max. epoch = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * total))))
            # compute loss in evaluation data
            total = 0
            iter_count = 0
            loss_elbo = []
            for i in range(args.n_enc):
                loss_pxz_post[i] = []
                loss_qzx_pz[i] = []
                loss_sc_z[i] = []
                loss_sc_x[i] = []
                loss_sc_x_post[i] = []
                loss_melsp_x_dB[i] = []
                loss_melsp_x_post_dB[i] = []
                loss_ms_norm_x[i] = []
                loss_ms_err_x[i] = []
                loss_ms_norm_x_post[i] = []
                loss_ms_err_x_post[i] = []
            loss_pxz_sum_post = []
            loss_sc_x_sum = []
            loss_sc_x_sum_post = []
            loss_sc_magsp_x_sum = []
            loss_sc_magsp_x_sum_post = []
            loss_melsp_x_sum = []
            loss_melsp_x_sum_dB = []
            loss_melsp_x_sum_post = []
            loss_melsp_x_sum_post_dB = []
            loss_magsp_x_sum = []
            loss_magsp_x_sum_dB = []
            loss_magsp_x_sum_post = []
            loss_magsp_x_sum_post_dB = []
            loss_ms_norm_x_sum = []
            loss_ms_err_x_sum = []
            loss_ms_norm_x_sum_post = []
            loss_ms_err_x_sum_post = []
            loss_ms_norm_magsp_x_sum = []
            loss_ms_err_magsp_x_sum = []
            loss_ms_norm_magsp_x_sum_post = []
            loss_ms_err_magsp_x_sum_post = []
            for i in range(n_spk):
                for j in range(args.n_enc):
                    gv_x[j][i] = []
                    gv_x_post[j][i] = []
                gv_x_sum[i] = []
                gv_x_sum_post[i] = []
            for i in range(args.n_enc):
                model_encoders[i].eval()
            model_decoder.eval()
            model_post.eval()
            model_classifier.eval()
            for i in range(args.n_enc):
                for param in model_encoders[i].parameters():
                    param.requires_grad = False
            logging.info("Evaluation data")
            while True:
                with torch.no_grad():
                    start = time.time()
                    batch_melsp, batch_magsp, batch_sc, c_idx, utt_idx, featfile, flens, n_batch_utt, max_flen = next(generator_eval)
                    if c_idx < 0:
                        break

                    # handle first pad for input on melsp
                    batch_x_in = F.pad(batch_melsp.transpose(1,2), (first_pad_left,first_pad_right), "replicate").transpose(1,2)

                    ## lat. infer.
                    idx_in = 0
                    for i in range(args.n_enc):
                        batch_qzx[i], batch_z[i], _ = model_encoders[i](batch_x_in, sampling=False)
                        batch_qzx[i] = batch_qzx[i][:,outpad_lefts[idx_in]:batch_qzx[i].shape[1]-outpad_rights[idx_in]]
                    ## reconst.
                    idx_in += 1
                    for i in range(args.n_enc):
                        batch_x[i], _ = model_decoder(batch_z[i])
                        if i > 0:
                            batch_x_sum += (batch_x[i].exp()-1)/10000
                        else:
                            batch_x_sum = (batch_x[i].exp()-1)/10000
                    batch_x_sum = torch.log(torch.clamp(batch_x_sum, min=1e-13)*10000+1)
                    idx_in_1 = idx_in-1
                    for i in range(args.n_enc):
                        batch_z[i] = batch_z[i][:,outpad_lefts[idx_in_1]:batch_z[i].shape[1]-outpad_rights[idx_in_1]]
                        batch_z_sc[i], _ = model_classifier(lat=batch_z[i])
                    ## post.
                    idx_in += 1
                    for i in range(args.n_enc):
                        batch_pxz_post[i], batch_x_post[i], _ = model_post(batch_x[i])
                    batch_pxz_sum_post, batch_x_sum_post, _ = model_post(batch_x_sum)
                    if model_post.pad_right > 0:
                        batch_x_sum = batch_x_sum[:,model_post.pad_left:-model_post.pad_right]
                    else:
                        batch_x_sum = batch_x_sum[:,model_post.pad_left:]
                    idx_in_1 = idx_in-1
                    for i in range(args.n_enc):
                        batch_x[i] = batch_x[i][:,outpad_lefts[idx_in_1]:batch_x[i].shape[1]-outpad_rights[idx_in_1]]
                        batch_x_sc[i], _ = model_classifier(feat=batch_x[i])
                        batch_x_post_sc[i], _ = model_classifier(feat=batch_x_post[i])
                    batch_magsp_x_sum = torch.matmul((torch.exp(batch_x_sum)-1)/10000, melfb_t)
                    batch_magsp_x_sum_post = torch.matmul((torch.exp(batch_x_sum_post)-1)/10000, melfb_t)
                    batch_x_sum_sc, _ = model_classifier(feat=batch_x_sum)
                    batch_x_sum_post_sc, _ = model_classifier(feat=batch_x_sum_post)
                    batch_magsp_x_sum_sc, _ = model_classifier(feat_aux=batch_magsp_x_sum)
                    batch_magsp_x_sum_post_sc, _ = model_classifier(feat_aux=batch_magsp_x_sum_post)
                   
                    # samples check
                    i = np.random.randint(0, batch_x[0].shape[0])
                    logging.info("%d %s %d %d" % (i, \
                        os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                            flens[i], max_flen))
                    for j in range(args.n_enc):
                        logging.info(batch_x[j][i,:2,:4])
                        logging.info(batch_x_post[j][i,:2,:4])
                    logging.info(batch_x_sum[i,:2,:4])
                    logging.info(batch_x_sum_post[i,:2,:4])
                    logging.info(batch_melsp[i,:2,:4])
                    logging.info(batch_magsp_x_sum[i,:2,:4])
                    logging.info(batch_magsp_x_sum_post[i,:2,:4])
                    logging.info(batch_magsp[i,:2,:4])

                    for k in range(n_batch_utt):
                        flens_utt = flens[k]
                        ## melsp
                        melsp = batch_melsp[k,:flens_utt]
                        melsp_rest = (torch.exp(melsp)-1)/10000

                        for i in range(args.n_enc):
                            melsp_x[i] = batch_x[i][k,:flens_utt]
                            melsp_x_rest[i] = (torch.exp(melsp_x[i])-1)/10000

                            melsp_x_post[i] = batch_x_post[i][k,:flens_utt]
                            melsp_x_post_rest[i] = (torch.exp(melsp_x_post[i])-1)/10000

                        melsp_x_sum = batch_x_sum[k,:flens_utt]
                        melsp_x_sum_rest = (torch.exp(melsp_x_sum)-1)/10000

                        melsp_x_sum_post = batch_x_sum_post[k,:flens_utt]
                        melsp_x_sum_post_rest = (torch.exp(melsp_x_sum_post)-1)/10000

                        ## GV stat
                        spk_src_idx = spk_list.index(os.path.basename(os.path.dirname(featfile[k])))
                        for i in range(args.n_enc):
                            gv_x[i][spk_src_idx].append(torch.var(melsp_x_rest[i], 0).cpu().data.numpy())
                            gv_x_post[i][spk_src_idx].append(torch.var(melsp_x_post_rest[i], 0).cpu().data.numpy())
                        gv_x_sum[spk_src_idx].append(torch.var(melsp_x_sum_rest, 0).cpu().data.numpy())
                        gv_x_sum_post[spk_src_idx].append(torch.var(melsp_x_sum_post_rest, 0).cpu().data.numpy())

                        ## melsp density
                        for i in range(args.n_enc):
                            batch_pxz_post_ = batch_pxz_post[i][k,:flens_utt]
                            batch_loss_pxz_post_[i] = criterion_laplace(batch_pxz_post_[:,:args.mel_dim], batch_pxz_post_[:,args.mel_dim:], melsp).mean()
                        batch_pxz_sum_post_ = batch_pxz_sum_post[k,:flens_utt]
                        batch_loss_pxz_sum_post_ = criterion_laplace(batch_pxz_sum_post_[:,:args.mel_dim], batch_pxz_sum_post_[:,args.mel_dim:], melsp).mean()

                        if k > 0:
                            for i in range(args.n_enc):
                                batch_loss_pxz_post[i] += batch_loss_pxz_post_[i]
                            batch_loss_pxz_sum_post += batch_loss_pxz_sum_post_
                            batch_loss_p += batch_loss_pxz_sum_post_
                        else:
                            for i in range(args.n_enc):
                                batch_loss_pxz_post[i] = batch_loss_pxz_post_[i]
                            batch_loss_pxz_sum_post = batch_loss_pxz_sum_post_
                            batch_loss_p = batch_loss_pxz_sum_post_

                        ## melsp acc
                        for i in range(args.n_enc):
                            batch_loss_melsp_x_dB_[i] = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_rest[i], min=1e-16))
                                                                    -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)))
                            batch_loss_melsp_x_post_dB_[i] = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_post_rest[i], min=1e-16))
                                                                    -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)))
                        batch_loss_melsp_x_sum_ = torch.mean(torch.sum(criterion_l1(melsp_x_sum, melsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_x_sum, melsp), -1))) \
                                                    + torch.mean(torch.mean(criterion_l1(melsp_x_sum, melsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_x_sum, melsp), -1)))
                        batch_loss_melsp_x_sum_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_sum_rest, min=1e-16))
                                                                -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)))
                        batch_loss_melsp_x_sum_post_ = torch.mean(torch.sum(criterion_l1(melsp_x_sum_post, melsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_x_sum_post, melsp), -1))) \
                                                    + torch.mean(torch.mean(criterion_l1(melsp_x_sum_post, melsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_x_sum_post, melsp), -1)))
                        batch_loss_melsp_x_sum_post_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_sum_post_rest, min=1e-16))
                                                                -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)))
                        if k > 0:
                            for i in range(args.n_enc):
                                batch_loss_melsp_x_dB[i] += batch_loss_melsp_x_dB_[i]
                                batch_loss_melsp_x_post_dB[i] += batch_loss_melsp_x_post_dB_[i]
                            batch_loss_melsp_x_sum += batch_loss_melsp_x_sum_
                            batch_loss_melsp_x_sum_dB += batch_loss_melsp_x_sum_dB_
                            batch_loss_melsp_x_sum_post += batch_loss_melsp_x_sum_post_
                            batch_loss_melsp_x_sum_post_dB += batch_loss_melsp_x_sum_post_dB_
                        else:
                            for i in range(args.n_enc):
                                batch_loss_melsp_x_dB[i] = batch_loss_melsp_x_dB_[i]
                                batch_loss_melsp_x_post_dB[i] = batch_loss_melsp_x_post_dB_[i]
                            batch_loss_melsp_x_sum = batch_loss_melsp_x_sum_
                            batch_loss_melsp_x_sum_dB = batch_loss_melsp_x_sum_dB_
                            batch_loss_melsp_x_sum_post = batch_loss_melsp_x_sum_post_
                            batch_loss_melsp_x_sum_post_dB = batch_loss_melsp_x_sum_post_dB_

                        ## magsp acc
                        magsp = batch_magsp[k,:flens_utt]
                        magsp_x_sum = batch_magsp_x_sum[k,:flens_utt]
                        magsp_x_sum_post = batch_magsp_x_sum_post[k,:flens_utt]
                        batch_loss_magsp_x_sum_ = torch.mean(torch.sum(criterion_l1(magsp_x_sum, magsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_x_sum, magsp), -1))) \
                                                    + torch.mean(torch.mean(criterion_l1(magsp_x_sum, magsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_x_sum, magsp), -1)))
                        batch_loss_magsp_x_sum_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_x_sum, min=1e-16))
                                                                -torch.log10(torch.clamp(magsp, min=1e-16))))**2, -1)))
                        batch_loss_magsp_x_sum_post_ = torch.mean(torch.sum(criterion_l1(magsp_x_sum_post, magsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_x_sum_post, magsp), -1))) \
                                                    + torch.mean(torch.mean(criterion_l1(magsp_x_sum_post, magsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_x_sum_post, magsp), -1)))
                        batch_loss_magsp_x_sum_post_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_x_sum_post, min=1e-16))
                                                                -torch.log10(torch.clamp(magsp, min=1e-16))))**2, -1)))
                        if k > 0:
                            batch_loss_magsp_x_sum += batch_loss_magsp_x_sum_
                            batch_loss_magsp_x_sum_dB += batch_loss_magsp_x_sum_dB_
                            batch_loss_magsp_x_sum_post += batch_loss_magsp_x_sum_post_
                            batch_loss_magsp_x_sum_post_dB += batch_loss_magsp_x_sum_post_dB_
                        else:
                            batch_loss_magsp_x_sum = batch_loss_magsp_x_sum_
                            batch_loss_magsp_x_sum_dB = batch_loss_magsp_x_sum_dB_
                            batch_loss_magsp_x_sum_post = batch_loss_magsp_x_sum_post_
                            batch_loss_magsp_x_sum_post_dB = batch_loss_magsp_x_sum_post_dB_

                        batch_loss_p += batch_loss_melsp_x_sum_ + batch_loss_melsp_x_sum_post_ \
                                        + batch_loss_magsp_x_sum_ + batch_loss_magsp_x_sum_post_

                        ## melsp ms
                        for i in range(args.n_enc):
                            batch_loss_ms_norm_x_[i], batch_loss_ms_err_x_[i] = criterion_ms(melsp_x_rest[i], melsp_rest)
                            batch_loss_ms_norm_x_post_[i], batch_loss_ms_err_x_post_[i] = criterion_ms(melsp_x_post_rest[i], melsp_rest)
                        batch_loss_ms_norm_x_sum_, batch_loss_ms_err_x_sum_ = criterion_ms(melsp_x_sum_rest, melsp_rest)
                        if not torch.isinf(batch_loss_ms_norm_x_sum_) and not torch.isnan(batch_loss_ms_norm_x_sum_):
                            batch_loss_p += batch_loss_ms_norm_x_sum_
                        if not torch.isinf(batch_loss_ms_err_x_sum_) and not torch.isnan(batch_loss_ms_err_x_sum_):
                            batch_loss_p += batch_loss_ms_err_x_sum_
                        batch_loss_ms_norm_x_sum_post_, batch_loss_ms_err_x_sum_post_ = criterion_ms(melsp_x_sum_post_rest, melsp_rest)
                        if not torch.isinf(batch_loss_ms_norm_x_sum_post_) and not torch.isnan(batch_loss_ms_norm_x_sum_post_):
                            batch_loss_p += batch_loss_ms_norm_x_sum_post_
                        if not torch.isinf(batch_loss_ms_err_x_sum_post_) and not torch.isnan(batch_loss_ms_err_x_sum_post_):
                            batch_loss_p += batch_loss_ms_err_x_sum_post_
                        batch_loss_ms_norm_magsp_x_sum_, batch_loss_ms_err_magsp_x_sum_ = criterion_ms(magsp_x_sum, magsp)
                        if not torch.isinf(batch_loss_ms_norm_magsp_x_sum_) and not torch.isnan(batch_loss_ms_norm_magsp_x_sum_):
                            batch_loss_p += batch_loss_ms_norm_magsp_x_sum_
                        if not torch.isinf(batch_loss_ms_err_magsp_x_sum_) and not torch.isnan(batch_loss_ms_err_magsp_x_sum_):
                            batch_loss_p += batch_loss_ms_err_magsp_x_sum_
                        batch_loss_ms_norm_magsp_x_sum_post_, batch_loss_ms_err_magsp_x_sum_post_ = criterion_ms(magsp_x_sum_post, magsp)
                        if not torch.isinf(batch_loss_ms_norm_magsp_x_sum_post_) and not torch.isnan(batch_loss_ms_norm_magsp_x_sum_post_):
                            batch_loss_p += batch_loss_ms_norm_magsp_x_sum_post_
                        if not torch.isinf(batch_loss_ms_err_magsp_x_sum_post_) and not torch.isnan(batch_loss_ms_err_magsp_x_sum_post_):
                            batch_loss_p += batch_loss_ms_err_magsp_x_sum_post_
                        if k > 0:
                            for i in range(args.n_enc):
                                batch_loss_ms_norm_x[i] += batch_loss_ms_norm_x_[i]
                                batch_loss_ms_err_x[i] += batch_loss_ms_err_x_[i]
                                batch_loss_ms_norm_x_post[i] += batch_loss_ms_norm_x_post_[i]
                                batch_loss_ms_err_x_post[i] += batch_loss_ms_err_x_post_[i]
                            batch_loss_ms_norm_x_sum += batch_loss_ms_norm_x_sum_
                            batch_loss_ms_err_x_sum += batch_loss_ms_err_x_sum_
                            batch_loss_ms_norm_x_sum_post += batch_loss_ms_norm_x_sum_post_
                            batch_loss_ms_err_x_sum_post += batch_loss_ms_err_x_sum_post_
                            batch_loss_ms_norm_magsp_x_sum += batch_loss_ms_norm_magsp_x_sum_
                            batch_loss_ms_err_magsp_x_sum += batch_loss_ms_err_magsp_x_sum_
                            batch_loss_ms_norm_magsp_x_sum_post += batch_loss_ms_norm_magsp_x_sum_post_
                            batch_loss_ms_err_magsp_x_sum_post += batch_loss_ms_err_magsp_x_sum_post_
                        else:
                            for i in range(args.n_enc):
                                batch_loss_ms_norm_x[i] = batch_loss_ms_norm_x_[i]
                                batch_loss_ms_err_x[i] = batch_loss_ms_err_x_[i]
                                batch_loss_ms_norm_x_post[i] = batch_loss_ms_norm_x_post_[i]
                                batch_loss_ms_err_x_post[i] = batch_loss_ms_err_x_post_[i]
                            batch_loss_ms_norm_x_sum = batch_loss_ms_norm_x_sum_
                            batch_loss_ms_err_x_sum = batch_loss_ms_err_x_sum_
                            batch_loss_ms_norm_x_sum_post = batch_loss_ms_norm_x_sum_post_
                            batch_loss_ms_err_x_sum_post = batch_loss_ms_err_x_sum_post_
                            batch_loss_ms_norm_magsp_x_sum = batch_loss_ms_norm_magsp_x_sum_
                            batch_loss_ms_err_magsp_x_sum = batch_loss_ms_err_magsp_x_sum_
                            batch_loss_ms_norm_magsp_x_sum_post = batch_loss_ms_norm_magsp_x_sum_post_
                            batch_loss_ms_err_magsp_x_sum_post = batch_loss_ms_err_magsp_x_sum_post_

                        # KL-div. lat.
                        for i in range(args.n_enc):
                            batch_loss_qzx_pz_[i] = torch.mean(torch.sum(kl_laplace(batch_qzx[i][k,:flens_utt]), -1))
                        if k > 0:
                            for i in range(args.n_enc):
                                batch_loss_qzx_pz[i] += batch_loss_qzx_pz_[i]
                                batch_loss_q += batch_loss_qzx_pz_[i]
                        else:
                            for i in range(args.n_enc):
                                batch_loss_qzx_pz[i] = batch_loss_qzx_pz_[i]
                                batch_loss_q = batch_loss_qzx_pz_[i]

                        # lat/melsp/magsp cls
                        batch_sc_ = batch_sc[k,:flens_utt]
                        for i in range(args.n_enc):
                            batch_loss_sc_z_[i] = torch.mean(criterion_ce(batch_z_sc[i][k,:flens_utt], batch_sc_))
                            batch_loss_sc_x_[i] = torch.mean(criterion_ce(batch_x_sc[i][k,:flens_utt], batch_sc_))
                            batch_loss_sc_x_post_[i] = torch.mean(criterion_ce(batch_x_post_sc[i][k,:flens_utt], batch_sc_))
                        batch_loss_sc_x_sum_ = torch.mean(criterion_ce(batch_x_sum_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_x_sum_post_ = torch.mean(criterion_ce(batch_x_sum_post_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_magsp_x_sum_ = torch.mean(criterion_ce(batch_magsp_x_sum_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_magsp_x_sum_post_ = torch.mean(criterion_ce(batch_magsp_x_sum_post_sc[k,:flens_utt], batch_sc_))
                        if k > 0:
                            for i in range(args.n_enc):
                                batch_loss_sc_z[i] += batch_loss_sc_z_[i]
                                batch_loss_sc_x[i] += batch_loss_sc_x_[i]
                                batch_loss_sc_x_post[i] += batch_loss_sc_x_post_[i]
                            batch_loss_sc_x_sum += batch_loss_sc_x_sum_
                            batch_loss_sc_x_sum_post += batch_loss_sc_x_sum_post_
                            batch_loss_sc_magsp_x_sum += batch_loss_sc_magsp_x_sum_
                            batch_loss_sc_magsp_x_sum_post += batch_loss_sc_magsp_x_sum_post_
                            batch_loss_sc += batch_loss_sc_x_sum_ + batch_loss_sc_x_sum_post_ \
                                            + batch_loss_sc_magsp_x_sum_ + batch_loss_sc_magsp_x_sum_post_
                        else:
                            for i in range(args.n_enc):
                                batch_loss_sc_z[i] = batch_loss_sc_z_[i]
                                batch_loss_sc_x[i] = batch_loss_sc_x_[i]
                                batch_loss_sc_x_post[i] = batch_loss_sc_x_post_[i]
                            batch_loss_sc_x_sum = batch_loss_sc_x_sum_
                            batch_loss_sc_x_sum_post = batch_loss_sc_x_sum_post_
                            batch_loss_sc_magsp_x_sum = batch_loss_sc_magsp_x_sum_
                            batch_loss_sc_magsp_x_sum_post = batch_loss_sc_magsp_x_sum_post_
                            batch_loss_sc = batch_loss_sc_x_sum_ + batch_loss_sc_x_sum_post_ \
                                            + batch_loss_sc_magsp_x_sum_ + batch_loss_sc_magsp_x_sum_post_

                        # elbo
                        if k > 0:
                            batch_loss_elbo += batch_loss_p + batch_loss_q + batch_loss_sc
                        else:
                            batch_loss_elbo = batch_loss_p + batch_loss_q + batch_loss_sc

                    for i in range(args.n_enc):
                        batch_loss_pxz_post[i] /= n_batch_utt
                        batch_loss_qzx_pz[i] /= n_batch_utt
                        batch_loss_sc_z[i] /= n_batch_utt
                        batch_loss_sc_x[i] /= n_batch_utt
                        batch_loss_sc_x_post[i] /= n_batch_utt
                        batch_loss_melsp_x_dB[i] /= n_batch_utt
                        batch_loss_melsp_x_post_dB[i] /= n_batch_utt
                        batch_loss_ms_norm_x[i] /= n_batch_utt
                        batch_loss_ms_err_x[i] /= n_batch_utt
                        batch_loss_ms_norm_x_post[i] /= n_batch_utt
                        batch_loss_ms_err_x_post[i] /= n_batch_utt
                    batch_loss_sc_x_sum /= n_batch_utt
                    batch_loss_pxz_sum_post /= n_batch_utt
                    batch_loss_sc_x_sum_post /= n_batch_utt
                    batch_loss_sc_magsp_x_sum /= n_batch_utt
                    batch_loss_sc_magsp_x_sum_post /= n_batch_utt
                    batch_loss_melsp_x_sum /= n_batch_utt
                    batch_loss_melsp_x_sum_dB /= n_batch_utt
                    batch_loss_melsp_x_sum_post /= n_batch_utt
                    batch_loss_melsp_x_sum_post_dB /= n_batch_utt
                    batch_loss_magsp_x_sum /= n_batch_utt
                    batch_loss_magsp_x_sum_dB /= n_batch_utt
                    batch_loss_magsp_x_sum_post /= n_batch_utt
                    batch_loss_magsp_x_sum_post_dB /= n_batch_utt
                    batch_loss_ms_norm_x_sum /= n_batch_utt
                    batch_loss_ms_err_x_sum /= n_batch_utt
                    batch_loss_ms_norm_x_sum_post /= n_batch_utt
                    batch_loss_ms_err_x_sum_post /= n_batch_utt
                    batch_loss_ms_norm_magsp_x_sum /= n_batch_utt
                    batch_loss_ms_err_magsp_x_sum /= n_batch_utt
                    batch_loss_ms_norm_magsp_x_sum_post /= n_batch_utt
                    batch_loss_ms_err_magsp_x_sum_post /= n_batch_utt

                    total_eval_loss["eval/loss_elbo"].append(batch_loss_elbo.item())
                    for i in range(args.n_enc):
                        total_eval_loss["eval/loss_pxz_post-%d"%(i+1)].append(batch_loss_pxz_post[i].item())
                        total_eval_loss["eval/loss_qzx_pz-%d"%(i+1)].append(batch_loss_qzx_pz[i].item())
                        loss_pxz_post[i].append(batch_loss_pxz_post[i].item())
                        loss_qzx_pz[i].append(batch_loss_qzx_pz[i].item())
                    total_eval_loss["eval/loss_pxz_sum_post"].append(batch_loss_pxz_sum_post.item())
                    loss_elbo.append(batch_loss_elbo.item())
                    loss_pxz_sum_post.append(batch_loss_pxz_sum_post.item())

                    for i in range(args.n_enc):
                        total_eval_loss["eval/loss_sc_z-%d"%(i+1)].append(batch_loss_sc_z[i].item())
                        total_eval_loss["eval/loss_sc_x-%d"%(i+1)].append(batch_loss_sc_x[i].item())
                        total_eval_loss["eval/loss_sc_x_post-%d"%(i+1)].append(batch_loss_sc_x_post[i].item())
                        loss_sc_z[i].append(batch_loss_sc_z[i].item())
                        loss_sc_x[i].append(batch_loss_sc_x[i].item())
                        loss_sc_x_post[i].append(batch_loss_sc_x_post[i].item())
                    total_eval_loss["eval/loss_sc_x_sum"].append(batch_loss_sc_x_sum.item())
                    total_eval_loss["eval/loss_sc_x_sum_post"].append(batch_loss_sc_x_sum_post.item())
                    total_eval_loss["eval/loss_sc_magsp_x_sum"].append(batch_loss_sc_magsp_x_sum.item())
                    total_eval_loss["eval/loss_sc_magsp_x_sum_post"].append(batch_loss_sc_magsp_x_sum_post.item())
                    loss_sc_x_sum.append(batch_loss_sc_x_sum.item())
                    loss_sc_x_sum_post.append(batch_loss_sc_x_sum_post.item())
                    loss_sc_magsp_x_sum.append(batch_loss_sc_magsp_x_sum.item())
                    loss_sc_magsp_x_sum_post.append(batch_loss_sc_magsp_x_sum_post.item())

                    for i in range(args.n_enc):
                        total_eval_loss["eval/loss_melsp_x_dB-%d"%(i+1)].append(batch_loss_melsp_x_dB[i].item())
                        total_eval_loss["eval/loss_melsp_x_post_dB-%d"%(i+1)].append(batch_loss_melsp_x_post_dB[i].item())
                        loss_melsp_x_dB[i].append(batch_loss_melsp_x_dB[i].item())
                        loss_melsp_x_post_dB[i].append(batch_loss_melsp_x_post_dB[i].item())
                    total_eval_loss["eval/loss_melsp_x_sum"].append(batch_loss_melsp_x_sum.item())
                    total_eval_loss["eval/loss_melsp_x_sum_dB"].append(batch_loss_melsp_x_sum_dB.item())
                    total_eval_loss["eval/loss_melsp_x_sum_post"].append(batch_loss_melsp_x_sum_post.item())
                    total_eval_loss["eval/loss_melsp_x_sum_post_dB"].append(batch_loss_melsp_x_sum_post_dB.item())
                    total_eval_loss["eval/loss_magsp_x_sum"].append(batch_loss_magsp_x_sum.item())
                    total_eval_loss["eval/loss_magsp_x_sum_dB"].append(batch_loss_magsp_x_sum_dB.item())
                    total_eval_loss["eval/loss_magsp_x_sum_post"].append(batch_loss_magsp_x_sum_post.item())
                    total_eval_loss["eval/loss_magsp_x_sum_post_dB"].append(batch_loss_magsp_x_sum_post_dB.item())
                    loss_melsp_x_sum.append(batch_loss_melsp_x_sum.item())
                    loss_melsp_x_sum_dB.append(batch_loss_melsp_x_sum_dB.item())
                    loss_melsp_x_sum_post.append(batch_loss_melsp_x_sum_post.item())
                    loss_melsp_x_sum_post_dB.append(batch_loss_melsp_x_sum_post_dB.item())
                    loss_magsp_x_sum.append(batch_loss_magsp_x_sum.item())
                    loss_magsp_x_sum_dB.append(batch_loss_magsp_x_sum_dB.item())
                    loss_magsp_x_sum_post.append(batch_loss_magsp_x_sum_post.item())
                    loss_magsp_x_sum_post_dB.append(batch_loss_magsp_x_sum_post_dB.item())

                    for i in range(args.n_enc):
                        total_eval_loss["eval/loss_ms_norm_x-%d"%(i+1)].append(batch_loss_ms_norm_x[i].item())
                        total_eval_loss["eval/loss_ms_err_x-%d"%(i+1)].append(batch_loss_ms_err_x[i].item())
                        total_eval_loss["eval/loss_ms_norm_x_post-%d"%(i+1)].append(batch_loss_ms_norm_x_post[i].item())
                        total_eval_loss["eval/loss_ms_err_x_post-%d"%(i+1)].append(batch_loss_ms_err_x_post[i].item())
                        loss_ms_norm_x[i].append(batch_loss_ms_norm_x[i].item())
                        loss_ms_err_x[i].append(batch_loss_ms_err_x[i].item())
                        loss_ms_norm_x_post[i].append(batch_loss_ms_norm_x_post[i].item())
                        loss_ms_err_x_post[i].append(batch_loss_ms_err_x_post[i].item())
                    total_eval_loss["eval/loss_ms_norm_x_sum"].append(batch_loss_ms_norm_x_sum.item())
                    total_eval_loss["eval/loss_ms_err_x_sum"].append(batch_loss_ms_err_x_sum.item())
                    total_eval_loss["eval/loss_ms_norm_x_sum_post"].append(batch_loss_ms_norm_x_sum_post.item())
                    total_eval_loss["eval/loss_ms_err_x_sum_post"].append(batch_loss_ms_err_x_sum_post.item())
                    total_eval_loss["eval/loss_ms_norm_magsp_x_sum"].append(batch_loss_ms_norm_magsp_x_sum.item())
                    total_eval_loss["eval/loss_ms_err_magsp_x_sum"].append(batch_loss_ms_err_magsp_x_sum.item())
                    total_eval_loss["eval/loss_ms_norm_magsp_x_sum_post"].append(batch_loss_ms_norm_magsp_x_sum_post.item())
                    total_eval_loss["eval/loss_ms_err_magsp_x_sum_post"].append(batch_loss_ms_err_magsp_x_sum_post.item())
                    loss_ms_norm_x_sum.append(batch_loss_ms_norm_x_sum.item())
                    loss_ms_err_x_sum.append(batch_loss_ms_err_x_sum.item())
                    loss_ms_norm_x_sum_post.append(batch_loss_ms_norm_x_sum_post.item())
                    loss_ms_err_x_sum_post.append(batch_loss_ms_err_x_sum_post.item())
                    loss_ms_norm_magsp_x_sum.append(batch_loss_ms_norm_magsp_x_sum.item())
                    loss_ms_err_magsp_x_sum.append(batch_loss_ms_err_magsp_x_sum.item())
                    loss_ms_norm_magsp_x_sum_post.append(batch_loss_ms_norm_magsp_x_sum_post.item())
                    loss_ms_err_magsp_x_sum_post.append(batch_loss_ms_err_magsp_x_sum_post.item())

                    text_log = "batch eval loss [%d] " % (c_idx+1)
                    text_log += "%.3f ; " % (batch_loss_elbo.item())
                    for i in range(args.n_enc):
                        text_log += "[%d] %.3f %.3f ; %.3f %.3f %.3f ; %.3f dB %.3f dB ; "\
                            "%.3f %.3f , %.3f %.3f ; " % (i+1,
                                batch_loss_pxz_post[i].item(), batch_loss_qzx_pz[i].item(),
                                batch_loss_sc_z[i].item(), batch_loss_sc_x[i].item(), batch_loss_sc_x_post[i].item(),
                                batch_loss_melsp_x_dB[i].item(), batch_loss_melsp_x_post_dB[i].item(),
                                batch_loss_ms_norm_x[i].item(), batch_loss_ms_err_x[i].item(),
                                batch_loss_ms_norm_x_post[i].item(), batch_loss_ms_err_x_post[i].item())
                    text_log += "[+] %.3f ; %.3f %.3f %.3f %.3f ; "\
                        "%.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB ; "\
                        "%.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ;; " % (
                                batch_loss_pxz_sum_post.item(),
                                batch_loss_sc_x_sum.item(), batch_loss_sc_x_sum_post.item(),
                                batch_loss_sc_magsp_x_sum.item(), batch_loss_sc_magsp_x_sum_post.item(),
                                batch_loss_melsp_x_sum.item(), batch_loss_melsp_x_sum_dB.item(),
                                batch_loss_melsp_x_sum_post.item(), batch_loss_melsp_x_sum_post_dB.item(),
                                batch_loss_magsp_x_sum.item(), batch_loss_magsp_x_sum_dB.item(),
                                batch_loss_magsp_x_sum_post.item(), batch_loss_magsp_x_sum_post_dB.item(),
                                batch_loss_ms_norm_x_sum.item(), batch_loss_ms_err_x_sum.item(),
                                batch_loss_ms_norm_x_sum_post.item(), batch_loss_ms_err_x_sum_post.item(),
                                batch_loss_ms_norm_magsp_x_sum.item(), batch_loss_ms_err_magsp_x_sum.item(),
                                batch_loss_ms_norm_magsp_x_sum_post.item(), batch_loss_ms_err_magsp_x_sum_post.item())
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            for i in range(args.n_enc):
                tmp_gv_1[i] = []
                tmp_gv_2[i] = []
                for j in range(n_spk):
                    if len(gv_x[i][j]) > 0:
                        tmp_gv_1[i].append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_x[i][j],
                                            axis=0))-np.log(gv_mean[j])))))
                    if len(gv_x_post[i][j]) > 0:
                        tmp_gv_2[i].append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_x_post[i][j],
                                            axis=0))-np.log(gv_mean[j])))))
            tmp_gv_3 = []
            tmp_gv_4 = []
            for j in range(n_spk):
                if len(gv_x_sum[j]) > 0:
                    tmp_gv_3.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_x_sum[j], \
                                        axis=0))-np.log(gv_mean[j])))))
                if len(gv_x_sum_post[j]) > 0:
                    tmp_gv_4.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_x_sum_post[j], \
                                        axis=0))-np.log(gv_mean[j])))))
            for i in range(args.n_enc):
                eval_loss_gv_x[i] = np.mean(tmp_gv_1[i])
                eval_loss_gv_x_post[i] = np.mean(tmp_gv_2[i])
                total_eval_loss["eval/loss_gv_x-%d"%(i+1)].append(eval_loss_gv_x[i])
                total_eval_loss["eval/loss_gv_x_post-%d"%(i+1)].append(eval_loss_gv_x_post[i])
            eval_loss_gv_x_sum = np.mean(tmp_gv_3)
            eval_loss_gv_x_sum_post = np.mean(tmp_gv_4)
            total_eval_loss["eval/loss_gv_x_sum"].append(eval_loss_gv_x_sum)
            total_eval_loss["eval/loss_gv_x_sum_post"].append(eval_loss_gv_x_sum_post)
            logging.info('sme %d' % (epoch_idx + 1))
            for key in total_eval_loss.keys():
                total_eval_loss[key] = np.mean(total_eval_loss[key])
                logging.info(f"(Steps: {iter_idx}) {key} = {total_eval_loss[key]:.4f}.")
            write_to_tensorboard(writer, iter_idx, total_eval_loss)
            total_eval_loss = defaultdict(list)
            eval_loss_elbo = np.mean(loss_elbo)
            eval_loss_elbo_std = np.std(loss_elbo)
            for i in range(args.n_enc):
                eval_loss_pxz_post[i] = np.mean(loss_pxz_post[i])
                eval_loss_pxz_post_std[i] = np.std(loss_pxz_post[i])
                eval_loss_qzx_pz[i] = np.mean(loss_qzx_pz[i])
                eval_loss_qzx_pz_std[i] = np.std(loss_qzx_pz[i])
                eval_loss_sc_z[i] = np.mean(loss_sc_z[i])
                eval_loss_sc_z_std[i] = np.std(loss_sc_z[i])
                eval_loss_sc_x[i] = np.mean(loss_sc_x[i])
                eval_loss_sc_x_std[i] = np.std(loss_sc_x[i])
                eval_loss_sc_x_post[i] = np.mean(loss_sc_x_post[i])
                eval_loss_sc_x_post_std[i] = np.std(loss_sc_x_post[i])
                eval_loss_melsp_x_dB[i] = np.mean(loss_melsp_x_dB[i])
                eval_loss_melsp_x_dB_std[i] = np.std(loss_melsp_x_dB[i])
                eval_loss_melsp_x_post_dB[i] = np.mean(loss_melsp_x_post_dB[i])
                eval_loss_melsp_x_post_dB_std[i] = np.std(loss_melsp_x_post_dB[i])
                eval_loss_ms_norm_x[i] = np.mean(loss_ms_norm_x[i])
                eval_loss_ms_norm_x_std[i] = np.std(loss_ms_norm_x[i])
                eval_loss_ms_err_x[i] = np.mean(loss_ms_err_x[i])
                eval_loss_ms_err_x_std[i] = np.std(loss_ms_err_x[i])
                eval_loss_ms_norm_x_post[i] = np.mean(loss_ms_norm_x_post[i])
                eval_loss_ms_norm_x_post_std[i] = np.std(loss_ms_norm_x_post[i])
                eval_loss_ms_err_x_post[i] = np.mean(loss_ms_err_x_post[i])
                eval_loss_ms_err_x_post_std[i] = np.std(loss_ms_err_x_post[i])
            eval_loss_pxz_sum_post = np.mean(loss_pxz_sum_post)
            eval_loss_pxz_sum_post_std = np.std(loss_pxz_sum_post)
            eval_loss_sc_x_sum = np.mean(loss_sc_x_sum)
            eval_loss_sc_x_sum_std = np.std(loss_sc_x_sum)
            eval_loss_sc_x_sum_post = np.mean(loss_sc_x_sum_post)
            eval_loss_sc_x_sum_post_std = np.std(loss_sc_x_sum_post)
            eval_loss_sc_magsp_x_sum = np.mean(loss_sc_magsp_x_sum)
            eval_loss_sc_magsp_x_sum_std = np.std(loss_sc_magsp_x_sum)
            eval_loss_sc_magsp_x_sum_post = np.mean(loss_sc_magsp_x_sum_post)
            eval_loss_sc_magsp_x_sum_post_std = np.std(loss_sc_magsp_x_sum_post)
            eval_loss_melsp_x_sum = np.mean(loss_melsp_x_sum)
            eval_loss_melsp_x_sum_std = np.std(loss_melsp_x_sum)
            eval_loss_melsp_x_sum_dB = np.mean(loss_melsp_x_sum_dB)
            eval_loss_melsp_x_sum_dB_std = np.std(loss_melsp_x_sum_dB)
            eval_loss_melsp_x_sum_post = np.mean(loss_melsp_x_sum_post)
            eval_loss_melsp_x_sum_post_std = np.std(loss_melsp_x_sum_post)
            eval_loss_melsp_x_sum_post_dB = np.mean(loss_melsp_x_sum_post_dB)
            eval_loss_melsp_x_sum_post_dB_std = np.std(loss_melsp_x_sum_post_dB)
            eval_loss_magsp_x_sum = np.mean(loss_magsp_x_sum)
            eval_loss_magsp_x_sum_std = np.std(loss_magsp_x_sum)
            eval_loss_magsp_x_sum_dB = np.mean(loss_magsp_x_sum_dB)
            eval_loss_magsp_x_sum_dB_std = np.std(loss_magsp_x_sum_dB)
            eval_loss_magsp_x_sum_post = np.mean(loss_magsp_x_sum_post)
            eval_loss_magsp_x_sum_post_std = np.std(loss_magsp_x_sum_post)
            eval_loss_magsp_x_sum_post_dB = np.mean(loss_magsp_x_sum_post_dB)
            eval_loss_magsp_x_sum_post_dB_std = np.std(loss_magsp_x_sum_post_dB)
            eval_loss_ms_norm_x_sum = np.mean(loss_ms_norm_x_sum)
            eval_loss_ms_norm_x_sum_std = np.std(loss_ms_norm_x_sum)
            eval_loss_ms_err_x_sum = np.mean(loss_ms_err_x_sum)
            eval_loss_ms_err_x_sum_std = np.std(loss_ms_err_x_sum)
            eval_loss_ms_norm_x_sum_post = np.mean(loss_ms_norm_x_sum_post)
            eval_loss_ms_norm_x_sum_post_std = np.std(loss_ms_norm_x_sum_post)
            eval_loss_ms_err_x_sum_post = np.mean(loss_ms_err_x_sum_post)
            eval_loss_ms_err_x_sum_post_std = np.std(loss_ms_err_x_sum_post)
            eval_loss_ms_norm_magsp_x_sum = np.mean(loss_ms_norm_magsp_x_sum)
            eval_loss_ms_norm_magsp_x_sum_std = np.std(loss_ms_norm_magsp_x_sum)
            eval_loss_ms_err_magsp_x_sum = np.mean(loss_ms_err_magsp_x_sum)
            eval_loss_ms_err_magsp_x_sum_std = np.std(loss_ms_err_magsp_x_sum)
            eval_loss_ms_norm_magsp_x_sum_post = np.mean(loss_ms_norm_magsp_x_sum_post)
            eval_loss_ms_norm_magsp_x_sum_post_std = np.std(loss_ms_norm_magsp_x_sum_post)
            eval_loss_ms_err_magsp_x_sum_post = np.mean(loss_ms_err_magsp_x_sum_post)
            eval_loss_ms_err_magsp_x_sum_post_std = np.std(loss_ms_err_magsp_x_sum_post)
            text_log = "(EPOCH:%d) average evaluation loss " % (epoch_idx + 1)
            text_log += "%.6f (+- %.6f) ; " % (eval_loss_elbo, eval_loss_elbo_std)
            for i in range(args.n_enc):
                text_log += "[%d] %.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                    "%.6f (+- %.6f) dB %.6f (+- %.6f) dB ; "\
                    "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                    "%.6f %.6f ;; " % (i+1,
                    eval_loss_pxz_post[i], eval_loss_pxz_post_std[i], eval_loss_qzx_pz[i], eval_loss_qzx_pz_std[i],
                    eval_loss_sc_z[i], eval_loss_sc_z_std[i], eval_loss_sc_x[i], eval_loss_sc_x_std[i], eval_loss_sc_x_post[i], eval_loss_sc_x_post_std[i],
                    eval_loss_melsp_x_dB[i], eval_loss_melsp_x_dB_std[i], eval_loss_melsp_x_post_dB[i], eval_loss_melsp_x_post_dB_std[i],
                    eval_loss_ms_norm_x[i], eval_loss_ms_norm_x_std[i], eval_loss_ms_err_x[i], eval_loss_ms_err_x_std[i],
                    eval_loss_ms_norm_x_post[i], eval_loss_ms_norm_x_post_std[i], eval_loss_ms_err_x_post[i], eval_loss_ms_err_x_post_std[i],
                    eval_loss_gv_x[i], eval_loss_gv_x_post[i])
            text_log += "[+] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                "%.6f %.6f ;; " % (                
                eval_loss_pxz_sum_post, eval_loss_pxz_sum_post_std,
                eval_loss_sc_x_sum, eval_loss_sc_x_sum_std, eval_loss_sc_x_sum_post, eval_loss_sc_x_sum_post_std,
                eval_loss_sc_magsp_x_sum, eval_loss_sc_magsp_x_sum_std, eval_loss_sc_magsp_x_sum_post, eval_loss_sc_magsp_x_sum_post_std,
                eval_loss_melsp_x_sum, eval_loss_melsp_x_sum_std, eval_loss_melsp_x_sum_dB, eval_loss_melsp_x_sum_dB_std,
                eval_loss_melsp_x_sum_post, eval_loss_melsp_x_sum_post_std, eval_loss_melsp_x_sum_post_dB, eval_loss_melsp_x_sum_post_dB_std,
                eval_loss_magsp_x_sum, eval_loss_magsp_x_sum_std, eval_loss_magsp_x_sum_dB, eval_loss_magsp_x_sum_dB_std,
                eval_loss_magsp_x_sum_post, eval_loss_magsp_x_sum_post_std, eval_loss_magsp_x_sum_post_dB, eval_loss_magsp_x_sum_post_dB_std,
                eval_loss_ms_norm_x_sum, eval_loss_ms_norm_x_sum_std, eval_loss_ms_err_x_sum, eval_loss_ms_err_x_sum_std,
                eval_loss_ms_norm_x_sum_post, eval_loss_ms_norm_x_sum_post_std, eval_loss_ms_err_x_sum_post, eval_loss_ms_err_x_sum_post_std,
                eval_loss_ms_norm_magsp_x_sum, eval_loss_ms_norm_magsp_x_sum_std, eval_loss_ms_err_magsp_x_sum, eval_loss_ms_err_magsp_x_sum_std,
                eval_loss_ms_norm_magsp_x_sum_post, eval_loss_ms_norm_magsp_x_sum_post_std, eval_loss_ms_err_magsp_x_sum_post, eval_loss_ms_err_magsp_x_sum_post_std,
                eval_loss_gv_x_sum, eval_loss_gv_x_sum_post)
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            if eval_loss_melsp_x_sum_post_dB <= min_eval_loss_melsp_x_sum_post_dB \
                or (eval_loss_melsp_x_sum_post_dB+eval_loss_melsp_x_sum_post_dB_std <= min_eval_loss_melsp_x_sum_post_dB+min_eval_loss_melsp_x_sum_post_dB_std) \
            or round(eval_loss_melsp_x_sum_post_dB,2) <= round(min_eval_loss_melsp_x_sum_post_dB,2) \
                or (round(eval_loss_melsp_x_sum_post_dB+eval_loss_melsp_x_sum_post_dB_std,2) <= round(min_eval_loss_melsp_x_sum_post_dB+min_eval_loss_melsp_x_sum_post_dB_std,2)) \
            or eval_loss_magsp_x_sum_post_dB <= min_eval_loss_magsp_x_sum_post_dB \
                or (eval_loss_magsp_x_sum_post_dB+eval_loss_magsp_x_sum_post_dB_std <= min_eval_loss_magsp_x_sum_post_dB+min_eval_loss_magsp_x_sum_post_dB_std) \
            or round(eval_loss_magsp_x_sum_post_dB,2) <= round(min_eval_loss_magsp_x_sum_post_dB,2) \
                or (round(eval_loss_magsp_x_sum_post_dB+eval_loss_magsp_x_sum_post_dB_std,2) <= round(min_eval_loss_magsp_x_sum_post_dB+min_eval_loss_magsp_x_sum_post_dB_std,2)):
                min_eval_loss_elbo = eval_loss_elbo
                min_eval_loss_elbo_std = eval_loss_elbo_std
                for i in range(args.n_enc):
                    min_eval_loss_pxz_post[i] = eval_loss_pxz_post[i]
                    min_eval_loss_pxz_post_std[i] = eval_loss_pxz_post_std[i]
                    min_eval_loss_qzx_pz[i] = eval_loss_qzx_pz[i]
                    min_eval_loss_qzx_pz_std[i] = eval_loss_qzx_pz_std[i]
                    min_eval_loss_sc_z[i] = eval_loss_sc_z[i]
                    min_eval_loss_sc_z_std[i] = eval_loss_sc_z_std[i]
                    min_eval_loss_sc_x[i] = eval_loss_sc_x[i]
                    min_eval_loss_sc_x_std[i] = eval_loss_sc_x_std[i]
                    min_eval_loss_sc_x_post[i] = eval_loss_sc_x_post[i]
                    min_eval_loss_sc_x_post_std[i] = eval_loss_sc_x_post_std[i]
                    min_eval_loss_melsp_x_dB[i] = eval_loss_melsp_x_dB[i]
                    min_eval_loss_melsp_x_dB_std[i] = eval_loss_melsp_x_dB_std[i]
                    min_eval_loss_melsp_x_post_dB[i] = eval_loss_melsp_x_post_dB[i]
                    min_eval_loss_melsp_x_post_dB_std[i] = eval_loss_melsp_x_post_dB_std[i]
                    min_eval_loss_ms_norm_x[i] = eval_loss_ms_norm_x[i]
                    min_eval_loss_ms_norm_x_std[i] = eval_loss_ms_norm_x_std[i]
                    min_eval_loss_ms_err_x[i] = eval_loss_ms_err_x[i]
                    min_eval_loss_ms_err_x_std[i] = eval_loss_ms_err_x_std[i]
                    min_eval_loss_ms_norm_x_post[i] = eval_loss_ms_norm_x_post[i]
                    min_eval_loss_ms_norm_x_post_std[i] = eval_loss_ms_norm_x_post_std[i]
                    min_eval_loss_ms_err_x_post[i] = eval_loss_ms_err_x_post[i]
                    min_eval_loss_ms_err_x_post_std[i] = eval_loss_ms_err_x_post_std[i]
                    min_eval_loss_gv_x[i] = eval_loss_gv_x[i]
                    min_eval_loss_gv_x_post[i] = eval_loss_gv_x_post[i]
                min_eval_loss_pxz_sum_post = eval_loss_pxz_sum_post
                min_eval_loss_pxz_sum_post_std = eval_loss_pxz_sum_post_std
                min_eval_loss_sc_x_sum = eval_loss_sc_x_sum
                min_eval_loss_sc_x_sum_std = eval_loss_sc_x_sum_std
                min_eval_loss_sc_x_sum_post = eval_loss_sc_x_sum_post
                min_eval_loss_sc_x_sum_post_std = eval_loss_sc_x_sum_post_std
                min_eval_loss_sc_magsp_x_sum = eval_loss_sc_magsp_x_sum
                min_eval_loss_sc_magsp_x_sum_std = eval_loss_sc_magsp_x_sum_std
                min_eval_loss_sc_magsp_x_sum_post = eval_loss_sc_magsp_x_sum_post
                min_eval_loss_sc_magsp_x_sum_post_std = eval_loss_sc_magsp_x_sum_post_std
                min_eval_loss_melsp_x_sum = eval_loss_melsp_x_sum
                min_eval_loss_melsp_x_sum_std = eval_loss_melsp_x_sum_std
                min_eval_loss_melsp_x_sum_dB = eval_loss_melsp_x_sum_dB
                min_eval_loss_melsp_x_sum_dB_std = eval_loss_melsp_x_sum_dB_std
                min_eval_loss_melsp_x_sum_post = eval_loss_melsp_x_sum_post
                min_eval_loss_melsp_x_sum_post_std = eval_loss_melsp_x_sum_post_std
                min_eval_loss_melsp_x_sum_post_dB = eval_loss_melsp_x_sum_post_dB
                min_eval_loss_melsp_x_sum_post_dB_std = eval_loss_melsp_x_sum_post_dB_std
                min_eval_loss_magsp_x_sum = eval_loss_magsp_x_sum
                min_eval_loss_magsp_x_sum_std = eval_loss_magsp_x_sum_std
                min_eval_loss_magsp_x_sum_dB = eval_loss_magsp_x_sum_dB
                min_eval_loss_magsp_x_sum_dB_std = eval_loss_magsp_x_sum_dB_std
                min_eval_loss_magsp_x_sum_post = eval_loss_magsp_x_sum_post
                min_eval_loss_magsp_x_sum_post_std = eval_loss_magsp_x_sum_post_std
                min_eval_loss_magsp_x_sum_post_dB = eval_loss_magsp_x_sum_post_dB
                min_eval_loss_magsp_x_sum_post_dB_std = eval_loss_magsp_x_sum_post_dB_std
                min_eval_loss_ms_norm_x_sum = eval_loss_ms_norm_x_sum
                min_eval_loss_ms_norm_x_sum_std = eval_loss_ms_norm_x_sum_std
                min_eval_loss_ms_err_x_sum = eval_loss_ms_err_x_sum
                min_eval_loss_ms_err_x_sum_std = eval_loss_ms_err_x_sum_std
                min_eval_loss_ms_norm_x_sum_post = eval_loss_ms_norm_x_sum_post
                min_eval_loss_ms_norm_x_sum_post_std = eval_loss_ms_norm_x_sum_post_std
                min_eval_loss_ms_err_x_sum_post = eval_loss_ms_err_x_sum_post
                min_eval_loss_ms_err_x_sum_post_std = eval_loss_ms_err_x_sum_post_std
                min_eval_loss_ms_norm_magsp_x_sum = eval_loss_ms_norm_magsp_x_sum
                min_eval_loss_ms_norm_magsp_x_sum_std = eval_loss_ms_norm_magsp_x_sum_std
                min_eval_loss_ms_err_magsp_x_sum = eval_loss_ms_err_magsp_x_sum
                min_eval_loss_ms_err_magsp_x_sum_std = eval_loss_ms_err_magsp_x_sum_std
                min_eval_loss_ms_norm_magsp_x_sum_post = eval_loss_ms_norm_magsp_x_sum_post
                min_eval_loss_ms_norm_magsp_x_sum_post_std = eval_loss_ms_norm_magsp_x_sum_post_std
                min_eval_loss_ms_err_magsp_x_sum_post = eval_loss_ms_err_magsp_x_sum_post
                min_eval_loss_ms_err_magsp_x_sum_post_std = eval_loss_ms_err_magsp_x_sum_post_std
                min_eval_loss_gv_x_sum = eval_loss_gv_x_sum
                min_eval_loss_gv_x_sum_post = eval_loss_gv_x_sum_post
                min_idx = epoch_idx
                #epoch_min_flag = True
                change_min_flag = True
            if change_min_flag:
                text_log = "min_eval_loss = "
                text_log += "%.6f (+- %.6f) ; " % (min_eval_loss_elbo, min_eval_loss_elbo_std)
                for i in range(args.n_enc):
                    text_log += "[%d] %.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                        "%.6f (+- %.6f) dB %.6f (+- %.6f) dB ; "\
                        "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                        "%.6f %.6f ;; " % (i+1,
                        min_eval_loss_pxz_post[i], min_eval_loss_pxz_post_std[i], min_eval_loss_qzx_pz[i], min_eval_loss_qzx_pz_std[i],
                        min_eval_loss_sc_z[i], min_eval_loss_sc_z_std[i], min_eval_loss_sc_x[i], min_eval_loss_sc_x_std[i], min_eval_loss_sc_x_post[i], min_eval_loss_sc_x_post_std[i],
                        min_eval_loss_melsp_x_dB[i], min_eval_loss_melsp_x_dB_std[i], min_eval_loss_melsp_x_post_dB[i], min_eval_loss_melsp_x_post_dB_std[i],
                        min_eval_loss_ms_norm_x[i], min_eval_loss_ms_norm_x_std[i], min_eval_loss_ms_err_x[i], min_eval_loss_ms_err_x_std[i],
                        min_eval_loss_ms_norm_x_post[i], min_eval_loss_ms_norm_x_post_std[i], min_eval_loss_ms_err_x_post[i], min_eval_loss_ms_err_x_post_std[i],
                        min_eval_loss_gv_x[i], min_eval_loss_gv_x_post[i])
                text_log += "[+] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                    "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                    "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                    "%.6f %.6f ;; " % (                
                    min_eval_loss_pxz_sum_post, min_eval_loss_pxz_sum_post_std,
                    min_eval_loss_sc_x_sum, min_eval_loss_sc_x_sum_std, min_eval_loss_sc_x_sum_post, min_eval_loss_sc_x_sum_post_std,
                    min_eval_loss_sc_magsp_x_sum, min_eval_loss_sc_magsp_x_sum_std, min_eval_loss_sc_magsp_x_sum_post, min_eval_loss_sc_magsp_x_sum_post_std,
                    min_eval_loss_melsp_x_sum, min_eval_loss_melsp_x_sum_std, min_eval_loss_melsp_x_sum_dB, min_eval_loss_melsp_x_sum_dB_std,
                    min_eval_loss_melsp_x_sum_post, min_eval_loss_melsp_x_sum_post_std, min_eval_loss_melsp_x_sum_post_dB, min_eval_loss_melsp_x_sum_post_dB_std,
                    min_eval_loss_magsp_x_sum, min_eval_loss_magsp_x_sum_std, min_eval_loss_magsp_x_sum_dB, min_eval_loss_magsp_x_sum_dB_std,
                    min_eval_loss_magsp_x_sum_post, min_eval_loss_magsp_x_sum_post_std, min_eval_loss_magsp_x_sum_post_dB, min_eval_loss_magsp_x_sum_post_dB_std,
                    min_eval_loss_ms_norm_x_sum, min_eval_loss_ms_norm_x_sum_std, min_eval_loss_ms_err_x_sum, min_eval_loss_ms_err_x_sum_std,
                    min_eval_loss_ms_norm_x_sum_post, min_eval_loss_ms_norm_x_sum_post_std, min_eval_loss_ms_err_x_sum_post, min_eval_loss_ms_err_x_sum_post_std,
                    min_eval_loss_ms_norm_magsp_x_sum, min_eval_loss_ms_norm_magsp_x_sum_std, min_eval_loss_ms_err_magsp_x_sum, min_eval_loss_ms_err_magsp_x_sum_std,
                    min_eval_loss_ms_norm_magsp_x_sum_post, min_eval_loss_ms_norm_magsp_x_sum_post_std, min_eval_loss_ms_err_magsp_x_sum_post, min_eval_loss_ms_err_magsp_x_sum_post_std,
                    min_eval_loss_gv_x_sum, min_eval_loss_gv_x_sum_post)
                logging.info("%s min_idx=%d" % (text_log, min_idx+1))
            #if ((epoch_idx + 1) % args.save_interval_epoch == 0) or (epoch_min_flag):
            if True:
                logging.info('save epoch:%d' % (epoch_idx+1))
                save_checkpoint(args.expdir, model_encoders, args.n_enc, model_decoder, model_post, model_classifier,
                    min_eval_loss_melsp_x_sum_post_dB, min_eval_loss_melsp_x_sum_post_dB_std,
                    min_eval_loss_magsp_x_sum_post_dB, min_eval_loss_magsp_x_sum_post_dB_std,
                    iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1)
            total = 0
            iter_count = 0
            loss_elbo = []
            for i in range(args.n_enc):
                loss_pxz_post[i] = []
                loss_qzx_pz[i] = []
                loss_sc_z[i] = []
                loss_sc_x[i] = []
                loss_sc_x_post[i] = []
                loss_melsp_x_dB[i] = []
                loss_melsp_x_post_dB[i] = []
                loss_ms_norm_x[i] = []
                loss_ms_err_x[i] = []
                loss_ms_norm_x_post[i] = []
                loss_ms_err_x_post[i] = []
            loss_pxz_sum_post = []
            loss_sc_x_sum = []
            loss_sc_x_sum_post = []
            loss_sc_magsp_x_sum = []
            loss_sc_magsp_x_sum_post = []
            loss_melsp_x_sum = []
            loss_melsp_x_sum_dB = []
            loss_melsp_x_sum_post = []
            loss_melsp_x_sum_post_dB = []
            loss_magsp_x_sum = []
            loss_magsp_x_sum_dB = []
            loss_magsp_x_sum_post = []
            loss_magsp_x_sum_post_dB = []
            loss_ms_norm_x_sum = []
            loss_ms_err_x_sum = []
            loss_ms_norm_x_sum_post = []
            loss_ms_err_x_sum_post = []
            loss_ms_norm_magsp_x_sum = []
            loss_ms_err_magsp_x_sum = []
            loss_ms_norm_magsp_x_sum_post = []
            loss_ms_err_magsp_x_sum_post = []
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            for i in range(args.n_enc):
                model_encoders[i].train()
            model_decoder.train()
            model_post.train()
            model_classifier.train()
            for i in range(args.n_enc):
                for param in model_encoders[i].parameters():
                    param.requires_grad = True
                for param in model_encoders[i].scale_in.parameters():
                    param.requires_grad = False
            # start next epoch
            if epoch_idx < args.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_feat, batch_feat_magsp, batch_sc, c_idx, utt_idx, featfile, f_bs, f_ss, flens, n_batch_utt, \
                    del_index_utt, max_flen, idx_select, idx_select_full, flens_acc = next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

            f_es = f_ss+f_bs
            logging.info(f'{f_ss} {f_bs} {f_es} {max_flen}')
            # handle first pad for input on melsp
            f_ss_first_pad_left = f_ss-first_pad_left
            f_es_first_pad_right = f_es+first_pad_right
            if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                batch_x_in = batch_feat[:,f_ss_first_pad_left:f_es_first_pad_right]
            elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                batch_x_in = F.pad(batch_feat[:,:f_es_first_pad_right].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
            elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                batch_x_in = F.pad(batch_feat[:,f_ss_first_pad_left:max_flen].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
            else: # pad left and right need additional replicate
                batch_x_in = F.pad(batch_feat[:,:max_flen].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
            batch_melsp = batch_feat[:,f_ss:f_es]
            batch_magsp = batch_feat_magsp[:,f_ss:f_es]
            batch_sc = batch_sc[:,f_ss:f_es]

            if f_ss > 0:
                if len(del_index_utt) > 0:
                    for i in range(args.n_enc):
                        h_z[i] = torch.FloatTensor(np.delete(h_z[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_x[i] = torch.FloatTensor(np.delete(h_x[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_x_post[i] = torch.FloatTensor(np.delete(h_x_post[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                    h_x_sum_post = torch.FloatTensor(np.delete(h_x_sum_post.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    for i in range(args.n_enc):
                        h_z_sc[i] = torch.FloatTensor(np.delete(h_z_sc[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_x_sc[i] = torch.FloatTensor(np.delete(h_x_sc[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                        h_x_post_sc[i] = torch.FloatTensor(np.delete(h_x_post_sc[i].cpu().data.numpy(),
                                                        del_index_utt, axis=1)).to(device)
                    h_x_sum_sc = torch.FloatTensor(np.delete(h_x_sum_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_x_sum_post_sc = torch.FloatTensor(np.delete(h_x_sum_post_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_magsp_x_sum_sc = torch.FloatTensor(np.delete(h_magsp_x_sum_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_magsp_x_sum_post_sc = torch.FloatTensor(np.delete(h_magsp_x_sum_post_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                ## lat. infer.
                idx_in = 0
                for i in range(args.n_enc):
                    batch_qzx[i], batch_z[i], h_z[i] = model_encoders[i](batch_x_in, outpad_right=outpad_rights[idx_in], h=h_z[i], do=True)
                    batch_qzx[i] = batch_qzx[i][:,outpad_lefts[idx_in]:batch_qzx[i].shape[1]-outpad_rights[idx_in]]
                ## reconst.
                idx_in += 1
                for i in range(args.n_enc):
                    batch_x[i], h_x[i] = model_decoder(batch_z[i], outpad_right=outpad_rights[idx_in], h=h_x[i], do=True)
                    if i > 0:
                        batch_x_sum += (batch_x[i].exp()-1)/10000
                    else:
                        batch_x_sum = (batch_x[i].exp()-1)/10000
                batch_x_sum = torch.log(torch.clamp(batch_x_sum, min=1e-13)*10000+1)
                idx_in_1 = idx_in-1
                for i in range(args.n_enc):
                    batch_z[i] = batch_z[i][:,outpad_lefts[idx_in_1]:batch_z[i].shape[1]-outpad_rights[idx_in_1]]
                    batch_z_sc[i], h_z_sc[i] = model_classifier(lat=batch_z[i], h=h_z_sc[i], do=True)
                ## post.
                idx_in += 1
                for i in range(args.n_enc):
                    batch_pxz_post[i], batch_x_post[i], h_x_post[i] = model_post(batch_x[i], outpad_right=outpad_rights[idx_in], h=h_x_post[i], do=True)
                batch_pxz_sum_post, batch_x_sum_post, h_x_sum_post = model_post(batch_x_sum, outpad_right=outpad_rights[idx_in], h=h_x_sum_post, do=True)
                if model_post.pad_right > 0:
                    batch_x_sum = batch_x_sum[:,model_post.pad_left:-model_post.pad_right]
                else:
                    batch_x_sum = batch_x_sum[:,model_post.pad_left:]
                idx_in_1 = idx_in-1
                for i in range(args.n_enc):
                    batch_x[i] = batch_x[i][:,outpad_lefts[idx_in_1]:batch_x[i].shape[1]-outpad_rights[idx_in_1]]
                    batch_x_sc[i], h_x_sc[i] = model_classifier(feat=batch_x[i], h=h_x_sc[i], do=True)
                    batch_x_post_sc[i], h_x_post_sc[i] = model_classifier(feat=batch_x_post[i], h=h_x_post_sc[i], do=True)
                batch_magsp_x_sum = torch.matmul((torch.exp(batch_x_sum)-1)/10000, melfb_t)
                batch_magsp_x_sum_post = torch.matmul((torch.exp(batch_x_sum_post)-1)/10000, melfb_t)
                batch_x_sum_sc, h_x_sum_sc = model_classifier(feat=batch_x_sum, h=h_x_sum_sc, do=True)
                batch_x_sum_post_sc, h_x_sum_post_sc = model_classifier(feat=batch_x_sum_post, h=h_x_sum_post_sc, do=True)
                batch_magsp_x_sum_sc, h_magsp_x_sum_sc = model_classifier(feat_aux=batch_magsp_x_sum, h=h_magsp_x_sum_sc, do=True)
                batch_magsp_x_sum_post_sc, h_magsp_x_sum_post_sc = model_classifier(feat_aux=batch_magsp_x_sum_post, h=h_magsp_x_sum_post_sc, do=True)
            else:
                ## lat. infer.
                idx_in = 0
                for i in range(args.n_enc):
                    batch_qzx[i], batch_z[i], h_z[i] = model_encoders[i](batch_x_in, outpad_right=outpad_rights[idx_in], do=True)
                    batch_qzx[i] = batch_qzx[i][:,outpad_lefts[idx_in]:batch_qzx[i].shape[1]-outpad_rights[idx_in]]
                ## reconst.
                idx_in += 1
                for i in range(args.n_enc):
                    batch_x[i], h_x[i] = model_decoder(batch_z[i], outpad_right=outpad_rights[idx_in], do=True)
                    if i > 0:
                        batch_x_sum += (batch_x[i].exp()-1)/10000
                    else:
                        batch_x_sum = (batch_x[i].exp()-1)/10000
                batch_x_sum = torch.log(torch.clamp(batch_x_sum, min=1e-13)*10000+1)
                idx_in_1 = idx_in-1
                for i in range(args.n_enc):
                    batch_z[i] = batch_z[i][:,outpad_lefts[idx_in_1]:batch_z[i].shape[1]-outpad_rights[idx_in_1]]
                    batch_z_sc[i], h_z_sc[i] = model_classifier(lat=batch_z[i], do=True)
                ## post.
                idx_in += 1
                for i in range(args.n_enc):
                    batch_pxz_post[i], batch_x_post[i], h_x_post[i] = model_post(batch_x[i], outpad_right=outpad_rights[idx_in], do=True)
                batch_pxz_sum_post, batch_x_sum_post, h_x_sum_post = model_post(batch_x_sum, outpad_right=outpad_rights[idx_in], do=True)
                if model_post.pad_right > 0:
                    batch_x_sum = batch_x_sum[:,model_post.pad_left:-model_post.pad_right]
                else:
                    batch_x_sum = batch_x_sum[:,model_post.pad_left:]
                idx_in_1 = idx_in-1
                for i in range(args.n_enc):
                    batch_x[i] = batch_x[i][:,outpad_lefts[idx_in_1]:batch_x[i].shape[1]-outpad_rights[idx_in_1]]
                    batch_x_sc[i], h_x_sc[i] = model_classifier(feat=batch_x[i], do=True)
                    batch_x_post_sc[i], h_x_post_sc[i] = model_classifier(feat=batch_x_post[i], do=True)
                batch_magsp_x_sum = torch.matmul((torch.exp(batch_x_sum)-1)/10000, melfb_t)
                batch_magsp_x_sum_post = torch.matmul((torch.exp(batch_x_sum_post)-1)/10000, melfb_t)
                batch_x_sum_sc, h_x_sum_sc = model_classifier(feat=batch_x_sum, do=True)
                batch_x_sum_post_sc, h_x_sum_post_sc = model_classifier(feat=batch_x_sum_post, do=True)
                batch_magsp_x_sum_sc, h_magsp_x_sum_sc = model_classifier(feat_aux=batch_magsp_x_sum, do=True)
                batch_magsp_x_sum_post_sc, h_magsp_x_sum_post_sc = model_classifier(feat_aux=batch_magsp_x_sum_post, do=True)

            # samples check
            with torch.no_grad():
                i = np.random.randint(0, batch_x[0].shape[0])
                logging.info("%d %s %d %d %d %d" % (i, \
                    os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                        f_ss, f_es, flens[i], max_flen))
                for j in range(args.n_enc):
                    logging.info(batch_x[j][i,:2,:4])
                    logging.info(batch_x_post[j][i,:2,:4])
                logging.info(batch_x_sum[i,:2,:4])
                logging.info(batch_x_sum_post[i,:2,:4])
                logging.info(batch_melsp[i,:2,:4])
                logging.info(batch_magsp_x_sum[i,:2,:4])
                logging.info(batch_magsp_x_sum_post[i,:2,:4])
                logging.info(batch_magsp[i,:2,:4])

            # Losses computation
            batch_loss = 0

            # handle short ending
            if len(idx_select) > 0:
                logging.info('len_idx_select: '+str(len(idx_select)))
                batch_loss_p_select = 0
                batch_loss_q_select = 0
                batch_loss_sc_select = 0
                for j in range(len(idx_select)):
                    k = idx_select[j]
                    flens_utt = flens_acc[k]
                    logging.info('%s %d' % (featfile[k], flens_utt))

                    melsp = batch_melsp[k,:flens_utt]
                    melsp_rest = (torch.exp(melsp)-1)/10000

                    melsp_x_sum = batch_x_sum[k,:flens_utt]
                    melsp_x_sum_rest = (torch.exp(melsp_x_sum)-1)/10000

                    melsp_x_sum_post = batch_x_sum_post[k,:flens_utt]
                    melsp_x_sum_post_rest = (torch.exp(melsp_x_sum_post)-1)/10000

                    magsp = batch_magsp[k,:flens_utt]
                    magsp_x_sum = batch_magsp_x_sum[k,:flens_utt]
                    magsp_x_sum_post = batch_magsp_x_sum_post[k,:flens_utt]

                    batch_pxz_sum_post_ = batch_pxz_sum_post[k,:flens_utt]
                    batch_loss_p_select += criterion_laplace(batch_pxz_sum_post_[:,:args.mel_dim], batch_pxz_sum_post_[:,args.mel_dim:], melsp).mean()
                    if flens_utt > 1:
                        batch_loss_p_select += torch.mean(torch.sum(criterion_l1(melsp_x_sum, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_x_sum, melsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(melsp_x_sum, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_x_sum, melsp), -1))) \
                                                + torch.mean(torch.sum(criterion_l1(melsp_x_sum_post, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_x_sum_post, melsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(melsp_x_sum_post, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_x_sum_post, melsp), -1)))
                        if iter_idx >= 50: #prevent early large losses
                            batch_loss_p_select += torch.mean(torch.sum(criterion_l1(magsp_x_sum, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_x_sum, magsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_x_sum, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_x_sum, magsp), -1))) \
                                                + torch.mean(torch.sum(criterion_l1(magsp_x_sum_post, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_x_sum_post, magsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_x_sum_post, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_x_sum_post, magsp), -1)))
                    else:
                        batch_loss_p_select += torch.mean(torch.sum(criterion_l1(melsp_x_sum, melsp), -1)) \
                                                + torch.mean(torch.mean(criterion_l1(melsp_x_sum, melsp), -1)) \
                                            + torch.mean(torch.sum(criterion_l1(melsp_x_sum_post, melsp), -1)) \
                                                + torch.mean(torch.mean(criterion_l1(melsp_x_sum_post, melsp), -1))
                        if iter_idx >= 50: #prevent early large losses
                            batch_loss_p_select += torch.mean(torch.sum(criterion_l1(magsp_x_sum, magsp), -1)) \
                                                    + torch.mean(torch.mean(criterion_l1(magsp_x_sum, magsp), -1)) \
                                                + torch.mean(torch.sum(criterion_l1(magsp_x_sum_post, magsp), -1)) \
                                                    + torch.mean(torch.mean(criterion_l1(magsp_x_sum_post, magsp), -1))

                    if iter_idx >= 50:
                        batch_loss_ms_norm_x__, batch_loss_ms_err_x__ = criterion_ms(melsp_x_sum_rest, melsp_rest)
                        if not torch.isinf(batch_loss_ms_norm_x__) and not torch.isnan(batch_loss_ms_norm_x__):
                            batch_loss_p_select += batch_loss_ms_norm_x__
                        if not torch.isinf(batch_loss_ms_err_x__) and not torch.isnan(batch_loss_ms_err_x__):
                            batch_loss_p_select += batch_loss_ms_err_x__

                        batch_loss_ms_norm_x__, batch_loss_ms_err_x__ = criterion_ms(melsp_x_sum_post_rest, melsp_rest)
                        if not torch.isinf(batch_loss_ms_norm_x__) and not torch.isnan(batch_loss_ms_norm_x__):
                            batch_loss_p_select += batch_loss_ms_norm_x__
                        if not torch.isinf(batch_loss_ms_err_x__) and not torch.isnan(batch_loss_ms_err_x__):
                            batch_loss_p_select += batch_loss_ms_err_x__

                        batch_loss_ms_norm_x__, batch_loss_ms_err_x__ = criterion_ms(magsp_x_sum, magsp)
                        if not torch.isinf(batch_loss_ms_norm_x__) and not torch.isnan(batch_loss_ms_norm_x__):
                            batch_loss_p_select += batch_loss_ms_norm_x__
                        if not torch.isinf(batch_loss_ms_err_x__) and not torch.isnan(batch_loss_ms_err_x__):
                            batch_loss_p_select += batch_loss_ms_err_x__

                        batch_loss_ms_norm_x__, batch_loss_ms_err_x__ = criterion_ms(magsp_x_sum_post, magsp)
                        if not torch.isinf(batch_loss_ms_norm_x__) and not torch.isnan(batch_loss_ms_norm_x__):
                            batch_loss_p_select += batch_loss_ms_norm_x__
                        if not torch.isinf(batch_loss_ms_err_x__) and not torch.isnan(batch_loss_ms_err_x__):
                            batch_loss_p_select += batch_loss_ms_err_x__

                    for i in range(args.n_enc):
                        batch_loss_q_select += torch.mean(torch.sum(kl_laplace(batch_qzx[i][k,:flens_utt]), -1))

                    batch_sc_ = batch_sc[k,:flens_utt]
                    batch_loss_sc_select += torch.mean(criterion_ce(batch_x_sum_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_x_sum_post_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_magsp_x_sum_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_magsp_x_sum_post_sc[k,:flens_utt], batch_sc_))

                batch_loss += batch_loss_p_select + batch_loss_q_select + batch_loss_sc_select
                if len(idx_select_full) > 0:
                    logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                    batch_melsp = torch.index_select(batch_melsp,0,idx_select_full)
                    batch_magsp = torch.index_select(batch_magsp,0,idx_select_full)
                    batch_sc = torch.index_select(batch_sc,0,idx_select_full)
                    for i in range(args.n_enc):
                        batch_x[i] = torch.index_select(batch_x[i],0,idx_select_full)
                        batch_x_post[i] = torch.index_select(batch_x_post[i],0,idx_select_full)
                        batch_pxz_post[i] = torch.index_select(batch_pxz_post[i],0,idx_select_full)
                        batch_qzx[i] = torch.index_select(batch_qzx[i],0,idx_select_full)
                        batch_z_sc[i] = torch.index_select(batch_z_sc[i],0,idx_select_full)
                        batch_x_sc[i] = torch.index_select(batch_x_sc[i],0,idx_select_full)
                        batch_x_post_sc[i] = torch.index_select(batch_x_post_sc[i],0,idx_select_full)
                    batch_x_sum = torch.index_select(batch_x_sum,0,idx_select_full)
                    batch_x_sum_post = torch.index_select(batch_x_sum_post,0,idx_select_full)
                    batch_magsp_x_sum = torch.index_select(batch_magsp_x_sum,0,idx_select_full)
                    batch_magsp_x_sum_post = torch.index_select(batch_magsp_x_sum_post,0,idx_select_full)
                    batch_pxz_sum_post = torch.index_select(batch_pxz_sum_post,0,idx_select_full)
                    batch_x_sum_sc = torch.index_select(batch_x_sum_sc,0,idx_select_full)
                    batch_x_sum_post_sc = torch.index_select(batch_x_sum_post_sc,0,idx_select_full)
                    batch_magsp_x_sum_sc = torch.index_select(batch_magsp_x_sum_sc,0,idx_select_full)
                    batch_magsp_x_sum_post_sc = torch.index_select(batch_magsp_x_sum_post_sc,0,idx_select_full)
                    n_batch_utt = batch_x[0].shape[0]
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

            ## melsp density
            for i in range(args.n_enc):
                batch_loss_pxz_post[i] = criterion_laplace(batch_pxz_post[i][:,:,:args.mel_dim], batch_pxz_post[i][:,:,args.mel_dim:], batch_melsp).mean()

            batch_loss_pxz_sum_post_ = criterion_laplace(batch_pxz_sum_post[:,:,:args.mel_dim], batch_pxz_sum_post[:,:,args.mel_dim:], batch_melsp)
            batch_loss_pxz_sum_post = batch_loss_pxz_sum_post_.mean()

            ## melsp
            melsp_rest = (torch.exp(batch_melsp)-1)/10000
            for i in range(args.n_enc):
                melsp_x_rest[i] = (torch.exp(batch_x[i])-1)/10000
                melsp_x_post_rest[i] = (torch.exp(batch_x_post[i])-1)/10000
            melsp_x_sum_rest = (torch.exp(batch_x_sum)-1)/10000
            melsp_x_sum_post_rest = (torch.exp(batch_x_sum_post)-1)/10000

            ## melsp acc
            for i in range(args.n_enc):
                batch_loss_melsp_x_dB[i] = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_rest[i], min=1e-16))
                                                        -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)), -1))
                batch_loss_melsp_x_post_dB[i] = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_post_rest[i], min=1e-16))
                                                        -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)), -1))

            batch_loss_melsp_x_sum_ = torch.mean(torch.sum(criterion_l1(batch_x_sum, batch_melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(batch_x_sum, batch_melsp), -1), -1)) \
                                    + torch.mean(torch.mean(criterion_l1(batch_x_sum, batch_melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_x_sum, batch_melsp), -1), -1))
            batch_loss_melsp_x_sum = batch_loss_melsp_x_sum_.mean()
            batch_loss_melsp_x_sum_dB = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_sum_rest, min=1e-16))
                                                    -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)), -1))

            batch_loss_melsp_x_sum_post_ = torch.mean(torch.sum(criterion_l1(batch_x_sum_post, batch_melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(batch_x_sum_post, batch_melsp), -1), -1)) \
                                    + torch.mean(torch.mean(criterion_l1(batch_x_sum_post, batch_melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_x_sum_post, batch_melsp), -1), -1))
            batch_loss_melsp_x_sum_post = batch_loss_melsp_x_sum_post_.mean()
            batch_loss_melsp_x_sum_post_dB = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_sum_post_rest, min=1e-16))
                                                    -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)), -1))

            ## magsp acc
            batch_loss_magsp_x_sum_ = torch.mean(torch.sum(criterion_l1(batch_magsp_x_sum, batch_magsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(batch_magsp_x_sum, batch_magsp), -1), -1)) \
                                    + torch.mean(torch.mean(criterion_l1(batch_magsp_x_sum, batch_magsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_magsp_x_sum, batch_magsp), -1), -1))
            batch_loss_magsp_x_sum = batch_loss_magsp_x_sum_.mean()
            batch_loss_magsp_x_sum_dB = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(batch_magsp_x_sum, min=1e-16))
                                                    -torch.log10(torch.clamp(batch_magsp, min=1e-16))))**2, -1)), -1))

            batch_loss_magsp_x_sum_post_ = torch.mean(torch.sum(criterion_l1(batch_magsp_x_sum_post, batch_magsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(batch_magsp_x_sum_post, batch_magsp), -1), -1)) \
                                    + torch.mean(torch.mean(criterion_l1(batch_magsp_x_sum_post, batch_magsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_magsp_x_sum_post, batch_magsp), -1), -1))
            batch_loss_magsp_x_sum_post = batch_loss_magsp_x_sum_post_.mean()
            batch_loss_magsp_x_sum_post_dB = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(batch_magsp_x_sum_post, min=1e-16))
                                                    -torch.log10(torch.clamp(batch_magsp, min=1e-16))))**2, -1)), -1))

            batch_loss_p = batch_loss_pxz_sum_post_.sum() \
                            + batch_loss_melsp_x_sum_.sum() + batch_loss_melsp_x_sum_post_.sum()
            if iter_idx >= 50: #prevent early large losses
                batch_loss_p += batch_loss_magsp_x_sum_.sum() + batch_loss_magsp_x_sum_post_.sum()

            ## melsp ms
            for i in range(args.n_enc):
                batch_loss_ms_norm_x__, batch_loss_ms_err_x__ = criterion_ms(melsp_x_rest[i], melsp_rest)
                batch_loss_ms_norm_x[i] = batch_loss_ms_norm_x__.mean()
                batch_loss_ms_err_x[i] = batch_loss_ms_err_x__.mean()
                batch_loss_ms_norm_x_post__, batch_loss_ms_err_x_post__ = criterion_ms(melsp_x_post_rest[i], melsp_rest)
                batch_loss_ms_norm_x_post[i] = batch_loss_ms_norm_x_post__.mean()
                batch_loss_ms_err_x_post[i] = batch_loss_ms_err_x_post__.mean()

            batch_loss_ms_norm_x_sum_, batch_loss_ms_err_x_sum_ = criterion_ms(melsp_x_sum_rest, melsp_rest)
            batch_loss_ms_norm_x_sum = batch_loss_ms_norm_x_sum_.mean()
            batch_loss_ms_err_x_sum = batch_loss_ms_err_x_sum_.mean()
            if iter_idx >= 50:
                if not torch.isinf(batch_loss_ms_norm_x_sum) and not torch.isnan(batch_loss_ms_norm_x_sum):
                    batch_loss_p += batch_loss_ms_norm_x_sum_.sum()
                if not torch.isinf(batch_loss_ms_err_x_sum) and not torch.isnan(batch_loss_ms_err_x_sum):
                    batch_loss_p += batch_loss_ms_err_x_sum_.sum()

            batch_loss_ms_norm_x_sum_post_, batch_loss_ms_err_x_sum_post_ = criterion_ms(melsp_x_sum_post_rest, melsp_rest)
            batch_loss_ms_norm_x_sum_post = batch_loss_ms_norm_x_sum_post_.mean()
            batch_loss_ms_err_x_sum_post = batch_loss_ms_err_x_sum_post_.mean()
            if iter_idx >= 50:
                if not torch.isinf(batch_loss_ms_norm_x_sum_post) and not torch.isnan(batch_loss_ms_norm_x_sum_post):
                    batch_loss_p += batch_loss_ms_norm_x_sum_post_.sum()
                if not torch.isinf(batch_loss_ms_err_x_sum_post) and not torch.isnan(batch_loss_ms_err_x_sum_post):
                    batch_loss_p += batch_loss_ms_err_x_sum_post_.sum()

            ## magsp ms
            batch_loss_ms_norm_magsp_x_sum_, batch_loss_ms_err_magsp_x_sum_ = criterion_ms(batch_magsp_x_sum, batch_magsp)
            batch_loss_ms_norm_magsp_x_sum = batch_loss_ms_norm_magsp_x_sum_.mean()
            batch_loss_ms_err_magsp_x_sum = batch_loss_ms_err_magsp_x_sum_.mean()
            if iter_idx >= 50:
                if not torch.isinf(batch_loss_ms_norm_magsp_x_sum) and not torch.isnan(batch_loss_ms_norm_magsp_x_sum):
                    batch_loss_p += batch_loss_ms_norm_magsp_x_sum_.sum()
                if not torch.isinf(batch_loss_ms_err_magsp_x_sum) and not torch.isnan(batch_loss_ms_err_magsp_x_sum):
                    batch_loss_p += batch_loss_ms_err_magsp_x_sum_.sum()

            batch_loss_ms_norm_magsp_x_sum_post_, batch_loss_ms_err_magsp_x_sum_post_ = criterion_ms(batch_magsp_x_sum_post, batch_magsp)
            batch_loss_ms_norm_magsp_x_sum_post = batch_loss_ms_norm_magsp_x_sum_post_.mean()
            batch_loss_ms_err_magsp_x_sum_post = batch_loss_ms_err_magsp_x_sum_post_.mean()
            if iter_idx >= 50:
                if not torch.isinf(batch_loss_ms_norm_magsp_x_sum_post) and not torch.isnan(batch_loss_ms_norm_magsp_x_sum_post):
                    batch_loss_p += batch_loss_ms_norm_magsp_x_sum_post_.sum()
                if not torch.isinf(batch_loss_ms_err_magsp_x_sum_post) and not torch.isnan(batch_loss_ms_err_magsp_x_sum_post):
                    batch_loss_p += batch_loss_ms_err_magsp_x_sum_post_.sum()

            # KL-div. lat.
            for i in range(args.n_enc):
                batch_loss_qzx_pz__ = torch.mean(torch.sum(kl_laplace(batch_qzx[i]), -1), -1)
                batch_loss_qzx_pz[i] = batch_loss_qzx_pz__.mean()
                if i > 0:
                    batch_loss_q += batch_loss_qzx_pz__.sum()
                else:
                    batch_loss_q = batch_loss_qzx_pz__.sum()

            # lat/melsp/magsp cls
            batch_sc_ = batch_sc.reshape(-1)

            for i in range(args.n_enc):
                batch_loss_sc_z[i] = torch.mean(criterion_ce(batch_z_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1))
                batch_loss_sc_x[i] = torch.mean(criterion_ce(batch_x_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1))
                batch_loss_sc_x_post[i] = torch.mean(criterion_ce(batch_x_post_sc[i].reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1))

            batch_loss_sc_x_sum_ = torch.mean(criterion_ce(batch_x_sum_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_x_sum = batch_loss_sc_x_sum_.mean()

            batch_loss_sc_x_sum_post_ = torch.mean(criterion_ce(batch_x_sum_post_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_x_sum_post = batch_loss_sc_x_sum_post_.mean()

            batch_loss_sc_magsp_x_sum_ = torch.mean(criterion_ce(batch_magsp_x_sum_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_magsp_x_sum = batch_loss_sc_magsp_x_sum_.mean()

            batch_loss_sc_magsp_x_sum_post_ = torch.mean(criterion_ce(batch_magsp_x_sum_post_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_magsp_x_sum_post = batch_loss_sc_magsp_x_sum_post_.mean()

            batch_loss_sc = batch_loss_sc_x_sum_.sum() + batch_loss_sc_x_sum_post_.sum() \
                            + batch_loss_sc_magsp_x_sum_.sum() + batch_loss_sc_magsp_x_sum_post_.sum()

            # elbo
            batch_loss_elbo = batch_loss_p + batch_loss_q + batch_loss_sc
            batch_loss += batch_loss_elbo

            total_train_loss["train/loss_elbo"].append(batch_loss_elbo.item())
            for i in range(args.n_enc):
                total_train_loss["train/loss_pxz_post-%d"%(i+1)].append(batch_loss_pxz_post[i].item())
                total_train_loss["train/loss_qzx_pz-%d"%(i+1)].append(batch_loss_qzx_pz[i].item())
                loss_pxz_post[i].append(batch_loss_pxz_post[i].item())
                loss_qzx_pz[i].append(batch_loss_qzx_pz[i].item())
            total_train_loss["train/loss_pxz_sum_post"].append(batch_loss_pxz_sum_post.item())
            loss_elbo.append(batch_loss_elbo.item())
            loss_pxz_sum_post.append(batch_loss_pxz_sum_post.item())

            for i in range(args.n_enc):
                total_train_loss["train/loss_sc_z-%d"%(i+1)].append(batch_loss_sc_z[i].item())
                total_train_loss["train/loss_sc_x-%d"%(i+1)].append(batch_loss_sc_x[i].item())
                total_train_loss["train/loss_sc_x_post-%d"%(i+1)].append(batch_loss_sc_x_post[i].item())
                loss_sc_z[i].append(batch_loss_sc_z[i].item())
                loss_sc_x[i].append(batch_loss_sc_x[i].item())
                loss_sc_x_post[i].append(batch_loss_sc_x_post[i].item())
            total_train_loss["train/loss_sc_x_sum"].append(batch_loss_sc_x_sum.item())
            total_train_loss["train/loss_sc_x_sum_post"].append(batch_loss_sc_x_sum_post.item())
            total_train_loss["train/loss_sc_magsp_x_sum"].append(batch_loss_sc_magsp_x_sum.item())
            total_train_loss["train/loss_sc_magsp_x_sum_post"].append(batch_loss_sc_magsp_x_sum_post.item())
            loss_sc_x_sum.append(batch_loss_sc_x_sum.item())
            loss_sc_x_sum_post.append(batch_loss_sc_x_sum_post.item())
            loss_sc_magsp_x_sum.append(batch_loss_sc_magsp_x_sum.item())
            loss_sc_magsp_x_sum_post.append(batch_loss_sc_magsp_x_sum_post.item())

            for i in range(args.n_enc):
                total_train_loss["train/loss_melsp_x_dB-%d"%(i+1)].append(batch_loss_melsp_x_dB[i].item())
                total_train_loss["train/loss_melsp_x_post_dB-%d"%(i+1)].append(batch_loss_melsp_x_post_dB[i].item())
                loss_melsp_x_dB[i].append(batch_loss_melsp_x_dB[i].item())
                loss_melsp_x_post_dB[i].append(batch_loss_melsp_x_post_dB[i].item())
            total_train_loss["train/loss_melsp_x_sum"].append(batch_loss_melsp_x_sum.item())
            total_train_loss["train/loss_melsp_x_sum_dB"].append(batch_loss_melsp_x_sum_dB.item())
            total_train_loss["train/loss_melsp_x_sum_post"].append(batch_loss_melsp_x_sum_post.item())
            total_train_loss["train/loss_melsp_x_sum_post_dB"].append(batch_loss_melsp_x_sum_post_dB.item())
            total_train_loss["train/loss_magsp_x_sum"].append(batch_loss_magsp_x_sum.item())
            total_train_loss["train/loss_magsp_x_sum_dB"].append(batch_loss_magsp_x_sum_dB.item())
            total_train_loss["train/loss_magsp_x_sum_post"].append(batch_loss_magsp_x_sum_post.item())
            total_train_loss["train/loss_magsp_x_sum_post_dB"].append(batch_loss_magsp_x_sum_post_dB.item())
            loss_melsp_x_sum.append(batch_loss_melsp_x_sum.item())
            loss_melsp_x_sum_dB.append(batch_loss_melsp_x_sum_dB.item())
            loss_melsp_x_sum_post.append(batch_loss_melsp_x_sum_post.item())
            loss_melsp_x_sum_post_dB.append(batch_loss_melsp_x_sum_post_dB.item())
            loss_magsp_x_sum.append(batch_loss_magsp_x_sum.item())
            loss_magsp_x_sum_dB.append(batch_loss_magsp_x_sum_dB.item())
            loss_magsp_x_sum_post.append(batch_loss_magsp_x_sum_post.item())
            loss_magsp_x_sum_post_dB.append(batch_loss_magsp_x_sum_post_dB.item())

            for i in range(args.n_enc):
                total_train_loss["train/loss_ms_norm_x-%d"%(i+1)].append(batch_loss_ms_norm_x[i].item())
                total_train_loss["train/loss_ms_err_x-%d"%(i+1)].append(batch_loss_ms_err_x[i].item())
                total_train_loss["train/loss_ms_norm_x_post-%d"%(i+1)].append(batch_loss_ms_norm_x_post[i].item())
                total_train_loss["train/loss_ms_err_x_post-%d"%(i+1)].append(batch_loss_ms_err_x_post[i].item())
                loss_ms_norm_x[i].append(batch_loss_ms_norm_x[i].item())
                loss_ms_err_x[i].append(batch_loss_ms_err_x[i].item())
                loss_ms_norm_x_post[i].append(batch_loss_ms_norm_x_post[i].item())
                loss_ms_err_x_post[i].append(batch_loss_ms_err_x_post[i].item())
            total_train_loss["train/loss_ms_norm_x_sum"].append(batch_loss_ms_norm_x_sum.item())
            total_train_loss["train/loss_ms_err_x_sum"].append(batch_loss_ms_err_x_sum.item())
            total_train_loss["train/loss_ms_norm_x_sum_post"].append(batch_loss_ms_norm_x_sum_post.item())
            total_train_loss["train/loss_ms_err_x_sum_post"].append(batch_loss_ms_err_x_sum_post.item())
            total_train_loss["train/loss_ms_norm_magsp_x_sum"].append(batch_loss_ms_norm_magsp_x_sum.item())
            total_train_loss["train/loss_ms_err_magsp_x_sum"].append(batch_loss_ms_err_magsp_x_sum.item())
            total_train_loss["train/loss_ms_norm_magsp_x_sum_post"].append(batch_loss_ms_norm_magsp_x_sum_post.item())
            total_train_loss["train/loss_ms_err_magsp_x_sum_post"].append(batch_loss_ms_err_magsp_x_sum_post.item())
            loss_ms_norm_x_sum.append(batch_loss_ms_norm_x_sum.item())
            loss_ms_err_x_sum.append(batch_loss_ms_err_x_sum.item())
            loss_ms_norm_x_sum_post.append(batch_loss_ms_norm_x_sum_post.item())
            loss_ms_err_x_sum_post.append(batch_loss_ms_err_x_sum_post.item())
            loss_ms_norm_magsp_x_sum.append(batch_loss_ms_norm_magsp_x_sum.item())
            loss_ms_err_magsp_x_sum.append(batch_loss_ms_err_magsp_x_sum.item())
            loss_ms_norm_magsp_x_sum_post.append(batch_loss_ms_norm_magsp_x_sum_post.item())
            loss_ms_err_magsp_x_sum_post.append(batch_loss_ms_err_magsp_x_sum_post.item())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            text_log = "batch loss [%d] %d %d " % (c_idx+1, f_ss, f_bs)
            text_log += "%.3f ; " % (batch_loss_elbo.item())
            for i in range(args.n_enc):
                text_log += "[%d] %.3f %.3f ; %.3f %.3f %.3f ; %.3f dB %.3f dB ; "\
                    "%.3f %.3f , %.3f %.3f ; " % (i+1,
                        batch_loss_pxz_post[i].item(), batch_loss_qzx_pz[i].item(),
                        batch_loss_sc_z[i].item(), batch_loss_sc_x[i].item(), batch_loss_sc_x_post[i].item(),
                        batch_loss_melsp_x_dB[i].item(), batch_loss_melsp_x_post_dB[i].item(),
                        batch_loss_ms_norm_x[i].item(), batch_loss_ms_err_x[i].item(),
                        batch_loss_ms_norm_x_post[i].item(), batch_loss_ms_err_x_post[i].item())
            text_log += "[+] %.3f ; %.3f %.3f %.3f %.3f ; "\
                "%.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB ; "\
                "%.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ;; " % (
                        batch_loss_pxz_sum_post.item(),
                        batch_loss_sc_x_sum.item(), batch_loss_sc_x_sum_post.item(),
                        batch_loss_sc_magsp_x_sum.item(), batch_loss_sc_magsp_x_sum_post.item(),
                        batch_loss_melsp_x_sum.item(), batch_loss_melsp_x_sum_dB.item(),
                        batch_loss_melsp_x_sum_post.item(), batch_loss_melsp_x_sum_post_dB.item(),
                        batch_loss_magsp_x_sum.item(), batch_loss_magsp_x_sum_dB.item(),
                        batch_loss_magsp_x_sum_post.item(), batch_loss_magsp_x_sum_post_dB.item(),
                        batch_loss_ms_norm_x_sum.item(), batch_loss_ms_err_x_sum.item(),
                        batch_loss_ms_norm_x_sum_post.item(), batch_loss_ms_err_x_sum_post.item(),
                        batch_loss_ms_norm_magsp_x_sum.item(), batch_loss_ms_err_magsp_x_sum.item(),
                        batch_loss_ms_norm_magsp_x_sum_post.item(), batch_loss_ms_err_magsp_x_sum_post.item())
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

    logging.info("maximum epoch reached, please see the optimum index from the development set acc.")

if __name__ == "__main__":
    main()
