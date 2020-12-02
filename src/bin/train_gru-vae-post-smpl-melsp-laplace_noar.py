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
from vcneuvoco import kl_laplace, ModulationSpectrumLoss, LaplaceLoss

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


def save_checkpoint(checkpoint_dir, model_encoder, model_decoder, model_post, model_classifier,
        min_eval_loss_melsp_x_post_dB, min_eval_loss_melsp_x_post_dB_std,
        min_eval_loss_magsp_x_post_dB, min_eval_loss_magsp_x_post_dB_std,
        iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, iterations, model_spkidtr=None):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    model_encoder.cpu()
    model_decoder.cpu()
    model_post.cpu()
    model_classifier.cpu()
    checkpoint = {
        "model_encoder": model_encoder.state_dict(),
        "model_decoder": model_decoder.state_dict(),
        "model_post": model_post.state_dict(),
        "model_classifier": model_classifier.state_dict(),
        "min_eval_loss_melsp_x_post_dB": min_eval_loss_melsp_x_post_dB,
        "min_eval_loss_melsp_x_post_dB_std": min_eval_loss_melsp_x_post_dB_std,
        "min_eval_loss_magsp_x_post_dB": min_eval_loss_melsp_x_post_dB,
        "min_eval_loss_magsp_x_post_dB_std": min_eval_loss_melsp_x_post_dB_std,
        "iter_idx": iter_idx,
        "min_idx": min_idx,
        "optimizer": optimizer.state_dict(),
        "numpy_random_state": numpy_random_state,
        "torch_random_state": torch_random_state,
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    model_encoder.cuda()
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
    parser.add_argument("--pretrained", default=None,
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
    model_encoder = GRU_VAE_ENCODER(
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
    logging.info(model_encoder)
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
        model_encoder.cuda()
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

    model_encoder.train()
    model_decoder.train()
    model_post.train()
    model_classifier.train()

    if model_encoder.use_weight_norm:
        torch.nn.utils.remove_weight_norm(model_encoder.scale_in)
    if model_decoder.use_weight_norm:
        torch.nn.utils.remove_weight_norm(model_decoder.scale_out)
    if model_post.use_weight_norm:
        torch.nn.utils.remove_weight_norm(model_post.scale_in)
        torch.nn.utils.remove_weight_norm(model_post.scale_out)

    model_encoder.scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/scale_stats.data),2))
    model_encoder.scale_in.bias = torch.nn.Parameter(-(mean_stats.data/scale_stats.data))
    model_decoder.scale_out.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(scale_stats.data),2))
    model_decoder.scale_out.bias = torch.nn.Parameter(mean_stats.data)
    model_post.scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/scale_stats.data),2))
    model_post.scale_in.bias = torch.nn.Parameter(-(mean_stats.data/scale_stats.data))
    model_post.scale_out.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(scale_stats.data),2))
    model_post.scale_out.bias = torch.nn.Parameter(mean_stats.data)

    if model_encoder.use_weight_norm:
        torch.nn.utils.weight_norm(model_encoder.scale_in)
    if model_decoder.use_weight_norm:
        torch.nn.utils.weight_norm(model_decoder.scale_out)
    if model_post.use_weight_norm:
        torch.nn.utils.weight_norm(model_post.scale_in)
        torch.nn.utils.weight_norm(model_post.scale_out)

    parameters = filter(lambda p: p.requires_grad, model_encoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_decoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_post.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (post): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_classifier.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (classifier): %.3f million' % parameters)

    for param in model_encoder.parameters():
        param.requires_grad = True
    for param in model_encoder.scale_in.parameters():
        param.requires_grad = False
    for param in model_decoder.parameters():
        param.requires_grad = True
    for param in model_decoder.scale_out.parameters():
        param.requires_grad = False
    for param in model_post.parameters():
        param.requires_grad = True
    for param in model_post.scale_in.parameters():
        param.requires_grad = False
    for param in model_post.scale_out.parameters():
        param.requires_grad = False

    module_list = list(model_encoder.conv.parameters())
    module_list += list(model_encoder.gru.parameters()) + list(model_encoder.out.parameters())

    module_list += list(model_decoder.conv.parameters())
    module_list += list(model_decoder.gru.parameters()) + list(model_decoder.out.parameters())

    module_list += list(model_post.conv.parameters())
    module_list += list(model_post.gru.parameters()) + list(model_post.out.parameters())

    module_list += list(model_classifier.conv_lat.parameters()) + list(model_classifier.conv_feat.parameters())
    module_list += list(model_classifier.gru.parameters()) + list(model_classifier.out.parameters())

    # model = ...
    optimizer = optim.RAdam(
        module_list,
        lr= args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    # resume
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained)
        model_encoder.load_state_dict(checkpoint["model_encoder"])
        model_decoder.load_state_dict(checkpoint["model_decoder"])
        model_post.load_state_dict(checkpoint["model_post"])
        model_classifier.load_state_dict(checkpoint["model_classifier"])
        epoch_idx = checkpoint["iterations"]
        logging.info("pretrained from %d-iter checkpoint." % epoch_idx)
        epoch_idx = 0
    elif args.resume is not None:
        checkpoint = torch.load(args.resume)
        model_encoder.load_state_dict(checkpoint["model_encoder"])
        model_decoder.load_state_dict(checkpoint["model_decoder"])
        model_post.load_state_dict(checkpoint["model_post"])
        model_classifier.load_state_dict(checkpoint["model_classifier"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)
    else:
        epoch_idx = 0

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
    enc_pad_left = model_encoder.pad_left
    enc_pad_right = model_encoder.pad_right
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
    total = 0
    iter_count = 0
    loss_elbo = []
    loss_pxz_post = []
    loss_qzx_pz = []
    loss_sc_z = []
    loss_sc_x_in = []
    loss_sc_x = []
    loss_sc_x_post = []
    loss_sc_magsp_x_in = []
    loss_sc_magsp_x = []
    loss_sc_magsp_x_post = []
    loss_melsp_x = []
    loss_melsp_x_dB = []
    loss_melsp_x_post = []
    loss_melsp_x_post_dB = []
    loss_magsp_x = []
    loss_magsp_x_dB = []
    loss_magsp_x_post = []
    loss_magsp_x_post_dB = []
    loss_ms_norm_x = []
    loss_ms_err_x = []
    loss_ms_norm_x_post = []
    loss_ms_err_x_post = []
    loss_ms_norm_magsp_x = []
    loss_ms_err_magsp_x = []
    loss_ms_norm_magsp_x_post = []
    loss_ms_err_magsp_x_post = []
    gv_x = [None]*n_spk
    gv_x_post = [None]*n_spk
    min_eval_loss_melsp_x_post_dB = 99999999.99
    min_eval_loss_melsp_x_post_dB_std = 99999999.99
    min_eval_loss_magsp_x_post_dB = 99999999.99
    min_eval_loss_magsp_x_post_dB_std = 99999999.99
    iter_idx = 0
    min_idx = -1
    change_min_flag = False
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        min_eval_loss_melsp_x_post_dB = checkpoint["min_eval_loss_melsp_x_post_dB"]
        min_eval_loss_melsp_x_post_dB_std = checkpoint["min_eval_loss_melsp_x_post_dB_std"]
        min_eval_loss_magsp_x_post_dB = checkpoint["min_eval_loss_magsp_x_post_dB"]
        min_eval_loss_magsp_x_post_dB_std = checkpoint["min_eval_loss_magsp_x_post_dB_std"]
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
            text_log += "%.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (
                np.mean(loss_elbo), np.std(loss_elbo), np.mean(loss_pxz_post), np.std(loss_pxz_post), np.mean(loss_qzx_pz), np.std(loss_qzx_pz),
                np.mean(loss_sc_z), np.std(loss_sc_z),
                np.mean(loss_sc_x_in), np.std(loss_sc_x_in), np.mean(loss_sc_x), np.std(loss_sc_x), np.mean(loss_sc_x_post), np.std(loss_sc_x_post),
                np.mean(loss_sc_magsp_x_in), np.std(loss_sc_magsp_x_in), np.mean(loss_sc_magsp_x), np.std(loss_sc_magsp_x), np.mean(loss_sc_magsp_x_post), np.std(loss_sc_magsp_x_post),
                np.mean(loss_melsp_x), np.std(loss_melsp_x), np.mean(loss_melsp_x_dB), np.std(loss_melsp_x_dB),
                np.mean(loss_melsp_x_post), np.std(loss_melsp_x_post), np.mean(loss_melsp_x_post_dB), np.std(loss_melsp_x_post_dB),
                np.mean(loss_magsp_x), np.std(loss_magsp_x), np.mean(loss_magsp_x_dB), np.std(loss_magsp_x_dB),
                np.mean(loss_magsp_x_post), np.std(loss_magsp_x_post), np.mean(loss_magsp_x_post_dB), np.std(loss_magsp_x_post_dB),
                np.mean(loss_ms_norm_x), np.std(loss_ms_norm_x), np.mean(loss_ms_err_x), np.std(loss_ms_err_x),
                np.mean(loss_ms_norm_x_post), np.std(loss_ms_norm_x_post), np.mean(loss_ms_err_x_post), np.std(loss_ms_err_x_post),
                np.mean(loss_ms_norm_magsp_x), np.std(loss_ms_norm_magsp_x), np.mean(loss_ms_err_magsp_x), np.std(loss_ms_err_magsp_x),
                np.mean(loss_ms_norm_magsp_x_post), np.std(loss_ms_norm_magsp_x_post), np.mean(loss_ms_err_magsp_x_post), np.std(loss_ms_err_magsp_x_post))
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            logging.info("estimated time until max. epoch = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * total))))
            # compute loss in evaluation data
            total = 0
            iter_count = 0
            loss_elbo = []
            loss_pxz_post = []
            loss_qzx_pz = []
            loss_sc_z = []
            loss_sc_x = []
            loss_sc_x_in = []
            loss_sc_x_post = []
            loss_sc_magsp_x = []
            loss_sc_magsp_x_in = []
            loss_sc_magsp_x_post = []
            loss_melsp_x = []
            loss_melsp_x_dB = []
            loss_melsp_x_post = []
            loss_melsp_x_post_dB = []
            loss_magsp_x = []
            loss_magsp_x_dB = []
            loss_magsp_x_post = []
            loss_magsp_x_post_dB = []
            loss_ms_norm_x = []
            loss_ms_err_x = []
            loss_ms_norm_x_post = []
            loss_ms_err_x_post = []
            loss_ms_norm_magsp_x = []
            loss_ms_err_magsp_x = []
            loss_ms_norm_magsp_x_post = []
            loss_ms_err_magsp_x_post = []
            for i in range(n_spk):
                gv_x[i] = []
                gv_x_post[i] = []
            model_encoder.eval()
            model_decoder.eval()
            model_post.eval()
            model_classifier.eval()
            for param in model_encoder.parameters():
                param.requires_grad = False
            for param in model_decoder.parameters():
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
                    batch_melsp, batch_magsp, batch_sc, c_idx, utt_idx, featfile, flens, n_batch_utt, max_flen = next(generator_eval)
                    if c_idx < 0:
                        break

                    # handle first pad for input on melsp
                    batch_x_in = F.pad(batch_melsp.transpose(1,2), (first_pad_left,first_pad_right), "replicate").transpose(1,2)

                    batch_x_in_sc, _ = model_classifier(feat=batch_melsp)
                    batch_magsp_x_in_sc, _ = model_classifier(feat_aux=batch_magsp)
                    ## lat. infer.
                    idx_in = 0
                    batch_qzx, batch_z, _ = model_encoder(batch_x_in, outpad_right=outpad_rights[idx_in], sampling=False)
                    batch_qzx = batch_qzx[:,outpad_lefts[idx_in]:batch_qzx.shape[1]-outpad_rights[idx_in]]
                    ## reconst.
                    idx_in += 1
                    batch_x, _ = model_decoder(batch_z, outpad_right=outpad_rights[idx_in])
                    idx_in_1 = idx_in-1
                    batch_z = batch_z[:,outpad_lefts[idx_in_1]:batch_z.shape[1]-outpad_rights[idx_in_1]]
                    batch_z_sc, _ = model_classifier(lat=batch_z)
                    ## post.
                    idx_in += 1
                    batch_pxz_post, batch_x_post, _ = model_post(batch_x, outpad_right=outpad_rights[idx_in])
                    idx_in_1 = idx_in-1
                    batch_x = batch_x[:,outpad_lefts[idx_in_1]:batch_x.shape[1]-outpad_rights[idx_in_1]]
                    batch_magsp_x = torch.matmul((torch.exp(batch_x)-1)/10000, melfb_t)
                    batch_magsp_x_post = torch.matmul((torch.exp(batch_x_post)-1)/10000, melfb_t)
                    batch_x_sc, _ = model_classifier(feat=batch_x)
                    batch_magsp_x_sc, _ = model_classifier(feat_aux=batch_magsp_x)
                    batch_x_post_sc, _ = model_classifier(feat=batch_x_post)
                    batch_magsp_x_post_sc, _ = model_classifier(feat_aux=batch_magsp_x_post)
                   
                    # samples check
                    i = np.random.randint(0, batch_x.shape[0])
                    logging.info("%d %s %d %d" % (i, \
                        os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                            flens[i], max_flen))
                    logging.info(batch_x[i,:2,:4])
                    logging.info(batch_x_post[i,:2,:4])
                    logging.info(batch_melsp[i,:2,:4])
                    logging.info(batch_magsp_x[i,:2,:4])
                    logging.info(batch_magsp_x_post[i,:2,:4])
                    logging.info(batch_magsp[i,:2,:4])

                    for k in range(n_batch_utt):
                        flens_utt = flens[k]
                        ## melsp
                        melsp = batch_melsp[k,:flens_utt]
                        melsp_rest = (torch.exp(melsp)-1)/10000

                        melsp_x = batch_x[k,:flens_utt]
                        melsp_x_rest = (torch.exp(melsp_x)-1)/10000

                        melsp_x_post = batch_x_post[k,:flens_utt]
                        melsp_x_post_rest = (torch.exp(melsp_x_post)-1)/10000

                        ## GV stat
                        spk_src_idx = spk_list.index(os.path.basename(os.path.dirname(featfile[k])))
                        gv_x[spk_src_idx].append(torch.var(melsp_x_rest, 0).cpu().data.numpy())
                        gv_x_post[spk_src_idx].append(torch.var(melsp_x_post_rest, 0).cpu().data.numpy())

                        ## melsp density
                        batch_pxz_post_ = batch_pxz_post[k,:flens_utt]
                        batch_loss_pxz_post_ = criterion_laplace(batch_pxz_post_[:,:args.mel_dim], batch_pxz_post_[:,args.mel_dim:], melsp).mean()
                        if k > 0:
                            batch_loss_pxz_post += batch_loss_pxz_post_
                            batch_loss_p += batch_loss_pxz_post_
                        else:
                            batch_loss_pxz_post = batch_loss_pxz_post_
                            batch_loss_p = batch_loss_pxz_post_

                        ## melsp acc
                        batch_loss_melsp_x_ = torch.mean(torch.sum(criterion_l1(melsp_x, melsp), -1)) \
                                                + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_x, melsp), -1))) \
                                            + torch.mean(torch.mean(criterion_l1(melsp_x, melsp), -1)) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_x, melsp), -1)))
                        batch_loss_melsp_x_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_rest, min=1e-16))
                                                                -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)))
                        batch_loss_melsp_x_post_ = torch.mean(torch.sum(criterion_l1(melsp_x_post, melsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_x_post, melsp), -1))) \
                                                    + torch.mean(torch.mean(criterion_l1(melsp_x_post, melsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_x_post, melsp), -1)))
                        batch_loss_melsp_x_post_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_post_rest, min=1e-16))
                                                                -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)))
                        if k > 0:
                            batch_loss_melsp_x += batch_loss_melsp_x_
                            batch_loss_melsp_x_dB += batch_loss_melsp_x_dB_
                            batch_loss_melsp_x_post += batch_loss_melsp_x_post_
                            batch_loss_melsp_x_post_dB += batch_loss_melsp_x_post_dB_
                        else:
                            batch_loss_melsp_x = batch_loss_melsp_x_
                            batch_loss_melsp_x_dB = batch_loss_melsp_x_dB_
                            batch_loss_melsp_x_post = batch_loss_melsp_x_post_
                            batch_loss_melsp_x_post_dB = batch_loss_melsp_x_post_dB_

                        ## magsp acc
                        magsp = batch_magsp[k,:flens_utt]
                        magsp_x = batch_magsp_x[k,:flens_utt]
                        magsp_x_post = batch_magsp_x_post[k,:flens_utt]
                        batch_loss_magsp_x_ = torch.mean(torch.sum(criterion_l1(magsp_x, magsp), -1)) \
                                                + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_x, magsp), -1))) \
                                            + torch.mean(torch.mean(criterion_l1(magsp_x, magsp), -1)) \
                                                + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_x, magsp), -1)))
                        batch_loss_magsp_x_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_x, min=1e-16))
                                                                -torch.log10(torch.clamp(magsp, min=1e-16))))**2, -1)))
                        batch_loss_magsp_x_post_ = torch.mean(torch.sum(criterion_l1(magsp_x_post, magsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_x_post, magsp), -1))) \
                                                    + torch.mean(torch.mean(criterion_l1(magsp_x_post, magsp), -1)) \
                                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_x_post, magsp), -1)))
                        batch_loss_magsp_x_post_dB_ = torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(magsp_x_post, min=1e-16))
                                                                -torch.log10(torch.clamp(magsp, min=1e-16))))**2, -1)))
                        if k > 0:
                            batch_loss_magsp_x += batch_loss_magsp_x_
                            batch_loss_magsp_x_dB += batch_loss_magsp_x_dB_
                            batch_loss_magsp_x_post += batch_loss_magsp_x_post_
                            batch_loss_magsp_x_post_dB += batch_loss_magsp_x_post_dB_
                        else:
                            batch_loss_magsp_x = batch_loss_magsp_x_
                            batch_loss_magsp_x_dB = batch_loss_magsp_x_dB_
                            batch_loss_magsp_x_post = batch_loss_magsp_x_post_
                            batch_loss_magsp_x_post_dB = batch_loss_magsp_x_post_dB_

                        batch_loss_p += batch_loss_melsp_x_ + batch_loss_melsp_x_post_ \
                                        + batch_loss_magsp_x_ + batch_loss_magsp_x_post_

                        ## melsp ms
                        batch_loss_ms_norm_x_, batch_loss_ms_err_x_ = criterion_ms(melsp_x_rest, melsp_rest)
                        if not torch.isinf(batch_loss_ms_norm_x_) and not torch.isnan(batch_loss_ms_norm_x_):
                            batch_loss_p += batch_loss_ms_norm_x_
                        if not torch.isinf(batch_loss_ms_err_x_) and not torch.isnan(batch_loss_ms_err_x_):
                            batch_loss_p += batch_loss_ms_err_x_
                        batch_loss_ms_norm_x_post_, batch_loss_ms_err_x_post_ = criterion_ms(melsp_x_post_rest, melsp_rest)
                        if not torch.isinf(batch_loss_ms_norm_x_post_) and not torch.isnan(batch_loss_ms_norm_x_post_):
                            batch_loss_p += batch_loss_ms_norm_x_post_
                        if not torch.isinf(batch_loss_ms_err_x_post_) and not torch.isnan(batch_loss_ms_err_x_post_):
                            batch_loss_p += batch_loss_ms_err_x_post_
                        batch_loss_ms_norm_magsp_x_, batch_loss_ms_err_magsp_x_ = criterion_ms(magsp_x, magsp)
                        if not torch.isinf(batch_loss_ms_norm_magsp_x_) and not torch.isnan(batch_loss_ms_norm_magsp_x_):
                            batch_loss_p += batch_loss_ms_norm_magsp_x_
                        if not torch.isinf(batch_loss_ms_err_magsp_x_) and not torch.isnan(batch_loss_ms_err_magsp_x_):
                            batch_loss_p += batch_loss_ms_err_magsp_x_
                        batch_loss_ms_norm_magsp_x_post_, batch_loss_ms_err_magsp_x_post_ = criterion_ms(magsp_x_post, magsp)
                        if not torch.isinf(batch_loss_ms_norm_magsp_x_post_) and not torch.isnan(batch_loss_ms_norm_magsp_x_post_):
                            batch_loss_p += batch_loss_ms_norm_magsp_x_post_
                        if not torch.isinf(batch_loss_ms_err_magsp_x_post_) and not torch.isnan(batch_loss_ms_err_magsp_x_post_):
                            batch_loss_p += batch_loss_ms_err_magsp_x_post_
                        if k > 0:
                            batch_loss_ms_norm_x += batch_loss_ms_norm_x_
                            batch_loss_ms_err_x += batch_loss_ms_norm_x_
                            batch_loss_ms_norm_x_post += batch_loss_ms_norm_x_post_
                            batch_loss_ms_err_x_post += batch_loss_ms_norm_x_post_
                            batch_loss_ms_norm_magsp_x += batch_loss_ms_norm_magsp_x_
                            batch_loss_ms_err_magsp_x += batch_loss_ms_norm_magsp_x_
                            batch_loss_ms_norm_magsp_x_post += batch_loss_ms_norm_magsp_x_post_
                            batch_loss_ms_err_magsp_x_post += batch_loss_ms_norm_magsp_x_post_
                        else:
                            batch_loss_ms_norm_x = batch_loss_ms_norm_x_
                            batch_loss_ms_err_x = batch_loss_ms_norm_x_
                            batch_loss_ms_norm_x_post = batch_loss_ms_norm_x_post_
                            batch_loss_ms_err_x_post = batch_loss_ms_norm_x_post_
                            batch_loss_ms_norm_magsp_x = batch_loss_ms_norm_magsp_x_
                            batch_loss_ms_err_magsp_x = batch_loss_ms_norm_magsp_x_
                            batch_loss_ms_norm_magsp_x_post = batch_loss_ms_norm_magsp_x_post_
                            batch_loss_ms_err_magsp_x_post = batch_loss_ms_norm_magsp_x_post_

                        # KL-div. lat.
                        batch_loss_qzx_pz_ = torch.mean(torch.sum(kl_laplace(batch_qzx[k,:flens_utt]), -1))
                        if k > 0:
                            batch_loss_qzx_pz += batch_loss_qzx_pz_
                            batch_loss_q += batch_loss_qzx_pz_
                        else:
                            batch_loss_qzx_pz = batch_loss_qzx_pz_
                            batch_loss_q = batch_loss_qzx_pz_

                        # lat/melsp/magsp cls
                        batch_sc_ = batch_sc[k,:flens_utt]
                        batch_loss_sc_z_ = torch.mean(criterion_ce(batch_z_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_x_in_ = torch.mean(criterion_ce(batch_x_in_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_x_ = torch.mean(criterion_ce(batch_x_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_x_post_ = torch.mean(criterion_ce(batch_x_post_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_magsp_x_in_ = torch.mean(criterion_ce(batch_magsp_x_in_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_magsp_x_ = torch.mean(criterion_ce(batch_magsp_x_sc[k,:flens_utt], batch_sc_))
                        batch_loss_sc_magsp_x_post_ = torch.mean(criterion_ce(batch_magsp_x_post_sc[k,:flens_utt], batch_sc_))
                        if k > 0:
                            batch_loss_sc_z += batch_loss_sc_z_
                            batch_loss_sc_x_in += batch_loss_sc_x_in_
                            batch_loss_sc_x += batch_loss_sc_x_
                            batch_loss_sc_x_post += batch_loss_sc_x_post_
                            batch_loss_sc_magsp_x_in += batch_loss_sc_magsp_x_in_
                            batch_loss_sc_magsp_x += batch_loss_sc_magsp_x_
                            batch_loss_sc_magsp_x_post += batch_loss_sc_magsp_x_post_
                            batch_loss_sc += batch_loss_sc_z_ \
                                            + batch_loss_sc_x_in_ + batch_loss_sc_x_ + batch_loss_sc_x_post_ \
                                                + batch_loss_sc_magsp_x_in_ + batch_loss_sc_magsp_x_ + batch_loss_sc_magsp_x_post_
                        else:
                            batch_loss_sc_z = batch_loss_sc_z_
                            batch_loss_sc_x_in = batch_loss_sc_x_in_
                            batch_loss_sc_x = batch_loss_sc_x_
                            batch_loss_sc_x_post = batch_loss_sc_x_post_
                            batch_loss_sc_magsp_x_in = batch_loss_sc_magsp_x_in_
                            batch_loss_sc_magsp_x = batch_loss_sc_magsp_x_
                            batch_loss_sc_magsp_x_post = batch_loss_sc_magsp_x_post_
                            batch_loss_sc = batch_loss_sc_z_ \
                                            + batch_loss_sc_x_in_ + batch_loss_sc_x_ + batch_loss_sc_x_post_ \
                                                + batch_loss_sc_magsp_x_in_ + batch_loss_sc_magsp_x_ + batch_loss_sc_magsp_x_post_

                        # elbo
                        if k > 0:
                            batch_loss_elbo += batch_loss_p + batch_loss_q + batch_loss_sc
                        else:
                            batch_loss_elbo = batch_loss_p + batch_loss_q + batch_loss_sc

                    batch_loss_pxz_post /= n_batch_utt
                    batch_loss_qzx_pz /= n_batch_utt
                    batch_loss_sc_z /= n_batch_utt
                    batch_loss_sc_x_in /= n_batch_utt
                    batch_loss_sc_x /= n_batch_utt
                    batch_loss_sc_x_post /= n_batch_utt
                    batch_loss_sc_magsp_x_in /= n_batch_utt
                    batch_loss_sc_magsp_x /= n_batch_utt
                    batch_loss_sc_magsp_x_post /= n_batch_utt
                    batch_loss_melsp_x /= n_batch_utt
                    batch_loss_melsp_x_dB /= n_batch_utt
                    batch_loss_melsp_x_post /= n_batch_utt
                    batch_loss_melsp_x_post_dB /= n_batch_utt
                    batch_loss_magsp_x /= n_batch_utt
                    batch_loss_magsp_x_dB /= n_batch_utt
                    batch_loss_magsp_x_post /= n_batch_utt
                    batch_loss_magsp_x_post_dB /= n_batch_utt
                    batch_loss_ms_norm_x /= n_batch_utt
                    batch_loss_ms_err_x /= n_batch_utt
                    batch_loss_ms_norm_x_post /= n_batch_utt
                    batch_loss_ms_err_x_post /= n_batch_utt
                    batch_loss_ms_norm_magsp_x /= n_batch_utt
                    batch_loss_ms_err_magsp_x /= n_batch_utt
                    batch_loss_ms_norm_magsp_x_post /= n_batch_utt
                    batch_loss_ms_err_magsp_x_post /= n_batch_utt

                    total_eval_loss["eval/loss_elbo"].append(batch_loss_elbo.item())
                    total_eval_loss["eval/loss_pxz_post"].append(batch_loss_pxz_post.item())
                    total_eval_loss["eval/loss_qzx_pz"].append(batch_loss_qzx_pz.item())
                    loss_elbo.append(batch_loss_elbo.item())
                    loss_pxz_post.append(batch_loss_pxz_post.item())
                    loss_qzx_pz.append(batch_loss_qzx_pz.item())

                    total_eval_loss["eval/loss_sc_z"].append(batch_loss_sc_z.item())
                    total_eval_loss["eval/loss_sc_x_in"].append(batch_loss_sc_x_in.item())
                    total_eval_loss["eval/loss_sc_x"].append(batch_loss_sc_x.item())
                    total_eval_loss["eval/loss_sc_x_post"].append(batch_loss_sc_x_post.item())
                    total_eval_loss["eval/loss_sc_magsp_x_in"].append(batch_loss_sc_magsp_x_in.item())
                    total_eval_loss["eval/loss_sc_magsp_x"].append(batch_loss_sc_magsp_x.item())
                    total_eval_loss["eval/loss_sc_magsp_x_post"].append(batch_loss_sc_magsp_x_post.item())
                    loss_sc_z.append(batch_loss_sc_z.item())
                    loss_sc_x_in.append(batch_loss_sc_x_in.item())
                    loss_sc_x.append(batch_loss_sc_x.item())
                    loss_sc_x_post.append(batch_loss_sc_x_post.item())
                    loss_sc_magsp_x_in.append(batch_loss_sc_magsp_x_in.item())
                    loss_sc_magsp_x.append(batch_loss_sc_magsp_x.item())
                    loss_sc_magsp_x_post.append(batch_loss_sc_magsp_x_post.item())

                    total_eval_loss["eval/loss_melsp_x"].append(batch_loss_melsp_x.item())
                    total_eval_loss["eval/loss_melsp_x_dB"].append(batch_loss_melsp_x_dB.item())
                    total_eval_loss["eval/loss_melsp_x_post"].append(batch_loss_melsp_x_post.item())
                    total_eval_loss["eval/loss_melsp_x_post_dB"].append(batch_loss_melsp_x_post_dB.item())
                    total_eval_loss["eval/loss_magsp_x"].append(batch_loss_magsp_x.item())
                    total_eval_loss["eval/loss_magsp_x_dB"].append(batch_loss_magsp_x_dB.item())
                    total_eval_loss["eval/loss_magsp_x_post"].append(batch_loss_magsp_x_post.item())
                    total_eval_loss["eval/loss_magsp_x_post_dB"].append(batch_loss_magsp_x_post_dB.item())
                    loss_melsp_x.append(batch_loss_melsp_x.item())
                    loss_melsp_x_dB.append(batch_loss_melsp_x_dB.item())
                    loss_melsp_x_post.append(batch_loss_melsp_x_post.item())
                    loss_melsp_x_post_dB.append(batch_loss_melsp_x_post_dB.item())
                    loss_magsp_x.append(batch_loss_magsp_x.item())
                    loss_magsp_x_dB.append(batch_loss_magsp_x_dB.item())
                    loss_magsp_x_post.append(batch_loss_magsp_x_post.item())
                    loss_magsp_x_post_dB.append(batch_loss_magsp_x_post_dB.item())

                    total_eval_loss["eval/loss_ms_norm_x"].append(batch_loss_ms_norm_x.item())
                    total_eval_loss["eval/loss_ms_err_x"].append(batch_loss_ms_err_x.item())
                    total_eval_loss["eval/loss_ms_norm_x_post"].append(batch_loss_ms_norm_x_post.item())
                    total_eval_loss["eval/loss_ms_err_x_post"].append(batch_loss_ms_err_x_post.item())
                    total_eval_loss["eval/loss_ms_norm_magsp_x"].append(batch_loss_ms_norm_magsp_x.item())
                    total_eval_loss["eval/loss_ms_err_magsp_x"].append(batch_loss_ms_err_magsp_x.item())
                    total_eval_loss["eval/loss_ms_norm_magsp_x_post"].append(batch_loss_ms_norm_magsp_x_post.item())
                    total_eval_loss["eval/loss_ms_err_magsp_x_post"].append(batch_loss_ms_err_magsp_x_post.item())
                    loss_ms_norm_x.append(batch_loss_ms_norm_x.item())
                    loss_ms_err_x.append(batch_loss_ms_err_x.item())
                    loss_ms_norm_x_post.append(batch_loss_ms_norm_x_post.item())
                    loss_ms_err_x_post.append(batch_loss_ms_err_x_post.item())
                    loss_ms_norm_magsp_x.append(batch_loss_ms_norm_magsp_x.item())
                    loss_ms_err_magsp_x.append(batch_loss_ms_err_magsp_x.item())
                    loss_ms_norm_magsp_x_post.append(batch_loss_ms_norm_magsp_x_post.item())
                    loss_ms_err_magsp_x_post.append(batch_loss_ms_err_magsp_x_post.item())

                    text_log = "batch eval loss [%d] " % (c_idx+1)
                    text_log += "%.3f %.3f %.3f ; %.3f , %.3f %.3f %.3f , %.3f %.3f %.3f ; "\
                        "%.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB ; "\
                        "%.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ;; " % (
                        batch_loss_elbo.item(), batch_loss_pxz_post.item(), batch_loss_qzx_pz.item(),
                                batch_loss_sc_z.item(),
                                batch_loss_sc_x_in.item(), batch_loss_sc_x.item(), batch_loss_sc_x_post.item(),
                                batch_loss_sc_magsp_x_in.item(), batch_loss_sc_magsp_x.item(), batch_loss_sc_magsp_x_post.item(),
                                batch_loss_melsp_x.item(), batch_loss_melsp_x_dB.item(), batch_loss_melsp_x_post.item(), batch_loss_melsp_x_post_dB.item(),
                                batch_loss_magsp_x.item(), batch_loss_magsp_x_dB.item(), batch_loss_magsp_x_post.item(), batch_loss_magsp_x_post_dB.item(),
                                batch_loss_ms_norm_x.item(), batch_loss_ms_err_x.item(), batch_loss_ms_norm_x_post.item(), batch_loss_ms_err_x_post.item(),
                                batch_loss_ms_norm_magsp_x.item(), batch_loss_ms_err_magsp_x.item(),
                                batch_loss_ms_norm_magsp_x_post.item(), batch_loss_ms_err_magsp_x_post.item())
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            tmp_gv_1 = []
            tmp_gv_2 = []
            for j in range(n_spk):
                if len(gv_x[j]) > 0:
                    tmp_gv_1.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_x[j], \
                                        axis=0))-np.log(gv_mean[j])))))
                if len(gv_x_post[j]) > 0:
                    tmp_gv_2.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_x_post[j], \
                                        axis=0))-np.log(gv_mean[j])))))
            eval_loss_gv_x = np.mean(tmp_gv_1)
            eval_loss_gv_x_post = np.mean(tmp_gv_2)
            total_eval_loss["eval/loss_gv_x"].append(eval_loss_gv_x)
            total_eval_loss["eval/loss_gv_x_post"].append(eval_loss_gv_x_post)
            logging.info('sme %d' % (epoch_idx + 1))
            for key in total_eval_loss.keys():
                total_eval_loss[key] = np.mean(total_eval_loss[key])
                logging.info(f"(Steps: {iter_idx}) {key} = {total_eval_loss[key]:.4f}.")
            write_to_tensorboard(writer, iter_idx, total_eval_loss)
            total_eval_loss = defaultdict(list)
            eval_loss_elbo = np.mean(loss_elbo)
            eval_loss_elbo_std = np.std(loss_elbo)
            eval_loss_pxz_post = np.mean(loss_pxz_post)
            eval_loss_pxz_post_std = np.std(loss_pxz_post)
            eval_loss_qzx_pz = np.mean(loss_qzx_pz)
            eval_loss_qzx_pz_std = np.std(loss_qzx_pz)
            eval_loss_sc_z = np.mean(loss_sc_z)
            eval_loss_sc_z_std = np.std(loss_sc_z)
            eval_loss_sc_x_in = np.mean(loss_sc_x_in)
            eval_loss_sc_x_in_std = np.std(loss_sc_x_in)
            eval_loss_sc_x = np.mean(loss_sc_x)
            eval_loss_sc_x_std = np.std(loss_sc_x)
            eval_loss_sc_x_post = np.mean(loss_sc_x_post)
            eval_loss_sc_x_post_std = np.std(loss_sc_x_post)
            eval_loss_sc_magsp_x_in = np.mean(loss_sc_magsp_x_in)
            eval_loss_sc_magsp_x_in_std = np.std(loss_sc_magsp_x_in)
            eval_loss_sc_magsp_x = np.mean(loss_sc_magsp_x)
            eval_loss_sc_magsp_x_std = np.std(loss_sc_magsp_x)
            eval_loss_sc_magsp_x_post = np.mean(loss_sc_magsp_x_post)
            eval_loss_sc_magsp_x_post_std = np.std(loss_sc_magsp_x_post)
            eval_loss_melsp_x = np.mean(loss_melsp_x)
            eval_loss_melsp_x_std = np.std(loss_melsp_x)
            eval_loss_melsp_x_dB = np.mean(loss_melsp_x_dB)
            eval_loss_melsp_x_dB_std = np.std(loss_melsp_x_dB)
            eval_loss_melsp_x_post = np.mean(loss_melsp_x_post)
            eval_loss_melsp_x_post_std = np.std(loss_melsp_x_post)
            eval_loss_melsp_x_post_dB = np.mean(loss_melsp_x_post_dB)
            eval_loss_melsp_x_post_dB_std = np.std(loss_melsp_x_post_dB)
            eval_loss_magsp_x = np.mean(loss_magsp_x)
            eval_loss_magsp_x_std = np.std(loss_magsp_x)
            eval_loss_magsp_x_dB = np.mean(loss_magsp_x_dB)
            eval_loss_magsp_x_dB_std = np.std(loss_magsp_x_dB)
            eval_loss_magsp_x_post = np.mean(loss_magsp_x_post)
            eval_loss_magsp_x_post_std = np.std(loss_magsp_x_post)
            eval_loss_magsp_x_post_dB = np.mean(loss_magsp_x_post_dB)
            eval_loss_magsp_x_post_dB_std = np.std(loss_magsp_x_post_dB)
            eval_loss_ms_norm_x = np.mean(loss_ms_norm_x)
            eval_loss_ms_norm_x_std = np.std(loss_ms_norm_x)
            eval_loss_ms_err_x = np.mean(loss_ms_err_x)
            eval_loss_ms_err_x_std = np.std(loss_ms_err_x)
            eval_loss_ms_norm_x_post = np.mean(loss_ms_norm_x_post)
            eval_loss_ms_norm_x_post_std = np.std(loss_ms_norm_x_post)
            eval_loss_ms_err_x_post = np.mean(loss_ms_err_x_post)
            eval_loss_ms_err_x_post_std = np.std(loss_ms_err_x_post)
            eval_loss_ms_norm_magsp_x = np.mean(loss_ms_norm_magsp_x)
            eval_loss_ms_norm_magsp_x_std = np.std(loss_ms_norm_magsp_x)
            eval_loss_ms_err_magsp_x = np.mean(loss_ms_err_magsp_x)
            eval_loss_ms_err_magsp_x_std = np.std(loss_ms_err_magsp_x)
            eval_loss_ms_norm_magsp_x_post = np.mean(loss_ms_norm_magsp_x_post)
            eval_loss_ms_norm_magsp_x_post_std = np.std(loss_ms_norm_magsp_x_post)
            eval_loss_ms_err_magsp_x_post = np.mean(loss_ms_err_magsp_x_post)
            eval_loss_ms_err_magsp_x_post_std = np.std(loss_ms_err_magsp_x_post)
            text_log = "(EPOCH:%d) average evaluation loss = " % (epoch_idx + 1)
            text_log += "%.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                "%.6f %.6f ;; " % (
                eval_loss_elbo, eval_loss_elbo_std, eval_loss_pxz_post, eval_loss_pxz_post_std, eval_loss_qzx_pz, eval_loss_qzx_pz_std,
                eval_loss_sc_z, eval_loss_sc_z_std,
                eval_loss_sc_x_in, eval_loss_sc_x_in_std, eval_loss_sc_x, eval_loss_sc_x_std, eval_loss_sc_x_post, eval_loss_sc_x_post_std,
                eval_loss_sc_magsp_x_in, eval_loss_sc_magsp_x_in_std, eval_loss_sc_magsp_x, eval_loss_sc_magsp_x_std, eval_loss_sc_magsp_x_post, eval_loss_sc_magsp_x_post_std,
                eval_loss_melsp_x, eval_loss_melsp_x_std, eval_loss_melsp_x_dB, eval_loss_melsp_x_dB_std,
                eval_loss_melsp_x_post, eval_loss_melsp_x_post_std, eval_loss_melsp_x_post_dB, eval_loss_melsp_x_post_dB_std,
                eval_loss_magsp_x, eval_loss_magsp_x_std, eval_loss_magsp_x_dB, eval_loss_magsp_x_dB_std,
                eval_loss_magsp_x_post, eval_loss_magsp_x_post_std, eval_loss_magsp_x_post_dB, eval_loss_magsp_x_post_dB_std,
                eval_loss_ms_norm_x, eval_loss_ms_norm_x_std, eval_loss_ms_err_x, eval_loss_ms_err_x_std,
                eval_loss_ms_norm_x_post, eval_loss_ms_norm_x_post_std, eval_loss_ms_err_x_post, eval_loss_ms_err_x_post_std,
                eval_loss_ms_norm_magsp_x, eval_loss_ms_norm_magsp_x_std, eval_loss_ms_err_magsp_x, eval_loss_ms_err_magsp_x_std,
                eval_loss_ms_norm_magsp_x_post, eval_loss_ms_norm_magsp_x_post_std, eval_loss_ms_err_magsp_x_post, eval_loss_ms_err_magsp_x_post_std,
                eval_loss_gv_x, eval_loss_gv_x_post)
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            if eval_loss_melsp_x_post_dB <= min_eval_loss_melsp_x_post_dB \
                or (eval_loss_melsp_x_post_dB+eval_loss_melsp_x_post_dB_std <= min_eval_loss_melsp_x_post_dB+min_eval_loss_melsp_x_post_dB_std) \
            or round(eval_loss_melsp_x_post_dB,2) <= round(min_eval_loss_melsp_x_post_dB,2) \
                or (round(eval_loss_melsp_x_post_dB+eval_loss_melsp_x_post_dB_std,2) <= round(min_eval_loss_melsp_x_post_dB+min_eval_loss_melsp_x_post_dB_std,2)) \
            or eval_loss_magsp_x_post_dB <= min_eval_loss_magsp_x_post_dB \
                or (eval_loss_magsp_x_post_dB+eval_loss_magsp_x_post_dB_std <= min_eval_loss_magsp_x_post_dB+min_eval_loss_magsp_x_post_dB_std) \
            or round(eval_loss_magsp_x_post_dB,2) <= round(min_eval_loss_magsp_x_post_dB,2) \
                or (round(eval_loss_magsp_x_post_dB+eval_loss_magsp_x_post_dB_std,2) <= round(min_eval_loss_magsp_x_post_dB+min_eval_loss_magsp_x_post_dB_std,2)):
                min_eval_loss_elbo = eval_loss_elbo
                min_eval_loss_elbo_std = eval_loss_elbo_std
                min_eval_loss_pxz_post = eval_loss_pxz_post
                min_eval_loss_pxz_post_std = eval_loss_pxz_post_std
                min_eval_loss_qzx_pz = eval_loss_qzx_pz
                min_eval_loss_qzx_pz_std = eval_loss_qzx_pz_std
                min_eval_loss_sc_z = eval_loss_sc_z
                min_eval_loss_sc_z_std = eval_loss_sc_z_std
                min_eval_loss_sc_x_in = eval_loss_sc_x_in
                min_eval_loss_sc_x_in_std = eval_loss_sc_x_in_std
                min_eval_loss_sc_x = eval_loss_sc_x
                min_eval_loss_sc_x_std = eval_loss_sc_x_std
                min_eval_loss_sc_x_post = eval_loss_sc_x_post
                min_eval_loss_sc_x_post_std = eval_loss_sc_x_post_std
                min_eval_loss_sc_magsp_x_in = eval_loss_sc_magsp_x_in
                min_eval_loss_sc_magsp_x_in_std = eval_loss_sc_magsp_x_in_std
                min_eval_loss_sc_magsp_x = eval_loss_sc_magsp_x
                min_eval_loss_sc_magsp_x_std = eval_loss_sc_magsp_x_std
                min_eval_loss_sc_magsp_x_post = eval_loss_sc_magsp_x_post
                min_eval_loss_sc_magsp_x_post_std = eval_loss_sc_magsp_x_post_std
                min_eval_loss_melsp_x = eval_loss_melsp_x
                min_eval_loss_melsp_x_std = eval_loss_melsp_x_std
                min_eval_loss_melsp_x_dB = eval_loss_melsp_x_dB
                min_eval_loss_melsp_x_dB_std = eval_loss_melsp_x_dB_std
                min_eval_loss_melsp_x_post = eval_loss_melsp_x_post
                min_eval_loss_melsp_x_post_std = eval_loss_melsp_x_post_std
                min_eval_loss_melsp_x_post_dB = eval_loss_melsp_x_post_dB
                min_eval_loss_melsp_x_post_dB_std = eval_loss_melsp_x_post_dB_std
                min_eval_loss_magsp_x = eval_loss_magsp_x
                min_eval_loss_magsp_x_std = eval_loss_magsp_x_std
                min_eval_loss_magsp_x_dB = eval_loss_magsp_x_dB
                min_eval_loss_magsp_x_dB_std = eval_loss_magsp_x_dB_std
                min_eval_loss_magsp_x_post = eval_loss_magsp_x_post
                min_eval_loss_magsp_x_post_std = eval_loss_magsp_x_post_std
                min_eval_loss_magsp_x_post_dB = eval_loss_magsp_x_post_dB
                min_eval_loss_magsp_x_post_dB_std = eval_loss_magsp_x_post_dB_std
                min_eval_loss_ms_norm_x = eval_loss_ms_norm_x
                min_eval_loss_ms_norm_x_std = eval_loss_ms_norm_x_std
                min_eval_loss_ms_err_x = eval_loss_ms_err_x
                min_eval_loss_ms_err_x_std = eval_loss_ms_err_x_std
                min_eval_loss_ms_norm_x_post = eval_loss_ms_norm_x_post
                min_eval_loss_ms_norm_x_post_std = eval_loss_ms_norm_x_post_std
                min_eval_loss_ms_err_x_post = eval_loss_ms_err_x_post
                min_eval_loss_ms_err_x_post_std = eval_loss_ms_err_x_post_std
                min_eval_loss_ms_norm_magsp_x = eval_loss_ms_norm_magsp_x
                min_eval_loss_ms_norm_magsp_x_std = eval_loss_ms_norm_magsp_x_std
                min_eval_loss_ms_err_magsp_x = eval_loss_ms_err_magsp_x
                min_eval_loss_ms_err_magsp_x_std = eval_loss_ms_err_magsp_x_std
                min_eval_loss_ms_norm_magsp_x_post = eval_loss_ms_norm_magsp_x_post
                min_eval_loss_ms_norm_magsp_x_post_std = eval_loss_ms_norm_magsp_x_post_std
                min_eval_loss_ms_err_magsp_x_post = eval_loss_ms_err_magsp_x_post
                min_eval_loss_ms_err_magsp_x_post_std = eval_loss_ms_err_magsp_x_post_std
                min_eval_loss_gv_x = eval_loss_gv_x
                min_eval_loss_gv_x_post = eval_loss_gv_x_post
                min_idx = epoch_idx
                #epoch_min_flag = True
                change_min_flag = True
            if change_min_flag:
                text_log = "min_eval_loss = "
                text_log += "%.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                    "%.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB , %.6f (+- %.6f) %.6f (+- %.6f) dB ; "\
                    "%.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) ; "\
                    "%.6f %.6f ;; " % (
                    min_eval_loss_elbo, min_eval_loss_elbo_std, min_eval_loss_pxz_post, min_eval_loss_pxz_post_std, min_eval_loss_qzx_pz, min_eval_loss_qzx_pz_std,
                    min_eval_loss_sc_z, min_eval_loss_sc_z_std,
                    min_eval_loss_sc_x_in, min_eval_loss_sc_x_in_std, min_eval_loss_sc_x, min_eval_loss_sc_x_std, min_eval_loss_sc_x_post, min_eval_loss_sc_x_post_std,
                    min_eval_loss_sc_magsp_x_in, min_eval_loss_sc_magsp_x_in_std, min_eval_loss_sc_magsp_x, min_eval_loss_sc_magsp_x_std, min_eval_loss_sc_magsp_x_post, min_eval_loss_sc_magsp_x_post_std,
                    min_eval_loss_melsp_x, min_eval_loss_melsp_x_std, min_eval_loss_melsp_x_dB, min_eval_loss_melsp_x_dB_std,
                    min_eval_loss_melsp_x_post, min_eval_loss_melsp_x_post_std, min_eval_loss_melsp_x_post_dB, min_eval_loss_melsp_x_post_dB_std,
                    min_eval_loss_magsp_x, min_eval_loss_magsp_x_std, min_eval_loss_magsp_x_dB, min_eval_loss_magsp_x_dB_std,
                    min_eval_loss_magsp_x_post, min_eval_loss_magsp_x_post_std, min_eval_loss_magsp_x_post_dB, min_eval_loss_magsp_x_post_dB_std,
                    min_eval_loss_ms_norm_x, min_eval_loss_ms_norm_x_std, min_eval_loss_ms_err_x, min_eval_loss_ms_err_x_std,
                    min_eval_loss_ms_norm_x_post, min_eval_loss_ms_norm_x_post_std, min_eval_loss_ms_err_x_post, min_eval_loss_ms_err_x_post_std,
                    min_eval_loss_ms_norm_magsp_x, min_eval_loss_ms_norm_magsp_x_std, min_eval_loss_ms_err_magsp_x, min_eval_loss_ms_err_magsp_x_std,
                    min_eval_loss_ms_norm_magsp_x_post, min_eval_loss_ms_norm_magsp_x_post_std, min_eval_loss_ms_err_magsp_x_post, min_eval_loss_ms_err_magsp_x_post_std,
                    min_eval_loss_gv_x, min_eval_loss_gv_x_post)
                logging.info("%s min_idx=%d" % (text_log, min_idx+1))
            #if ((epoch_idx + 1) % args.save_interval_epoch == 0) or (epoch_min_flag):
            if True:
                logging.info('save epoch:%d' % (epoch_idx+1))
                save_checkpoint(args.expdir, model_encoder, model_decoder, model_post, model_classifier,
                    min_eval_loss_melsp_x_post_dB, min_eval_loss_melsp_x_post_dB_std,
                    min_eval_loss_magsp_x_post_dB, min_eval_loss_magsp_x_post_dB_std,
                    iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1)
            total = 0
            iter_count = 0
            loss_elbo = []
            loss_pxz_post = []
            loss_qzx_pz = []
            loss_sc_z = []
            loss_sc_x_in = []
            loss_sc_x = []
            loss_sc_x_post = []
            loss_sc_magsp_x_in = []
            loss_sc_magsp_x = []
            loss_sc_magsp_x_post = []
            loss_melsp_x = []
            loss_melsp_x_dB = []
            loss_melsp_x_post = []
            loss_melsp_x_post_dB = []
            loss_magsp_x = []
            loss_magsp_x_dB = []
            loss_magsp_x_post = []
            loss_magsp_x_post_dB = []
            loss_ms_norm_x = []
            loss_ms_err_x = []
            loss_ms_norm_x_post = []
            loss_ms_err_x_post = []
            loss_ms_norm_magsp_x = []
            loss_ms_err_magsp_x = []
            loss_ms_norm_magsp_x_post = []
            loss_ms_err_magsp_x_post = []
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model_encoder.train()
            model_decoder.train()
            model_post.train()
            model_classifier.train()
            for param in model_encoder.parameters():
                param.requires_grad = True
            for param in model_encoder.scale_in.parameters():
                param.requires_grad = False
            for param in model_decoder.parameters():
                param.requires_grad = True
            for param in model_decoder.scale_out.parameters():
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
                    h_x_in_sc = torch.FloatTensor(np.delete(h_x_in_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_magsp_x_in_sc = torch.FloatTensor(np.delete(h_magsp_x_in_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_z = torch.FloatTensor(np.delete(h_z.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_x = torch.FloatTensor(np.delete(h_x.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_x_post = torch.FloatTensor(np.delete(h_x_post.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_z_sc = torch.FloatTensor(np.delete(h_z_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_x_sc = torch.FloatTensor(np.delete(h_x_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_magsp_x_sc = torch.FloatTensor(np.delete(h_magsp_x_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_x_post_sc = torch.FloatTensor(np.delete(h_x_post_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                    h_magsp_x_post_sc = torch.FloatTensor(np.delete(h_magsp_x_post_sc.cpu().data.numpy(),
                                                    del_index_utt, axis=1)).to(device)
                batch_x_in_sc, h_x_in_sc = model_classifier(feat=batch_melsp, h=h_x_in_sc, do=True)
                batch_magsp_x_in_sc, h_magsp_x_in_sc = model_classifier(feat_aux=batch_magsp, h=h_magsp_x_in_sc, do=True)
                ## lat. infer.
                idx_in = 0
                batch_qzx, batch_z, h_z = model_encoder(batch_x_in, outpad_right=outpad_rights[idx_in], h=h_z, do=True)
                batch_qzx = batch_qzx[:,outpad_lefts[idx_in]:batch_qzx.shape[1]-outpad_rights[idx_in]]
                ## reconst.
                idx_in += 1
                batch_x, h_x = model_decoder(batch_z, outpad_right=outpad_rights[idx_in], h=h_x, do=True)
                idx_in_1 = idx_in-1
                batch_z = batch_z[:,outpad_lefts[idx_in_1]:batch_z.shape[1]-outpad_rights[idx_in_1]]
                batch_z_sc, h_z_sc = model_classifier(lat=batch_z, h=h_z_sc, do=True)
                ## post.
                idx_in += 1
                batch_pxz_post, batch_x_post, h_x_post = model_post(batch_x, outpad_right=outpad_rights[idx_in], h=h_x_post, do=True)
                idx_in_1 = idx_in-1
                batch_x = batch_x[:,outpad_lefts[idx_in_1]:batch_x.shape[1]-outpad_rights[idx_in_1]]
                batch_magsp_x = torch.matmul((torch.exp(batch_x)-1)/10000, melfb_t)
                batch_magsp_x_post = torch.matmul((torch.exp(batch_x_post)-1)/10000, melfb_t)
                batch_x_sc, h_x_sc = model_classifier(feat=batch_x, h=h_x_sc, do=True)
                batch_magsp_x_sc, h_magsp_x_sc = model_classifier(feat_aux=batch_magsp_x, h=h_magsp_x_sc, do=True)
                batch_x_post_sc, h_x_post_sc = model_classifier(feat=batch_x_post, h=h_x_post_sc, do=True)
                batch_magsp_x_post_sc, h_magsp_x_post_sc = model_classifier(feat_aux=batch_magsp_x_post, h=h_magsp_x_post_sc, do=True)
            else:
                batch_x_in_sc, h_x_in_sc = model_classifier(feat=batch_melsp, do=True)
                batch_magsp_x_in_sc, h_magsp_x_in_sc = model_classifier(feat_aux=batch_magsp, do=True)
                ## lat. infer.
                idx_in = 0
                batch_qzx, batch_z, h_z = model_encoder(batch_x_in, outpad_right=outpad_rights[idx_in], do=True)
                batch_qzx = batch_qzx[:,outpad_lefts[idx_in]:batch_qzx.shape[1]-outpad_rights[idx_in]]
                ## reconst.
                idx_in += 1
                batch_x, h_x = model_decoder(batch_z, outpad_right=outpad_rights[idx_in], do=True)
                idx_in_1 = idx_in-1
                batch_z = batch_z[:,outpad_lefts[idx_in_1]:batch_z.shape[1]-outpad_rights[idx_in_1]]
                batch_z_sc, h_z_sc = model_classifier(lat=batch_z, do=True)
                ## post.
                idx_in += 1
                batch_pxz_post, batch_x_post, h_x_post = model_post(batch_x, outpad_right=outpad_rights[idx_in], do=True)
                idx_in_1 = idx_in-1
                batch_x = batch_x[:,outpad_lefts[idx_in_1]:batch_x.shape[1]-outpad_rights[idx_in_1]]
                batch_magsp_x = torch.matmul((torch.exp(batch_x)-1)/10000, melfb_t)
                batch_magsp_x_post = torch.matmul((torch.exp(batch_x_post)-1)/10000, melfb_t)
                batch_x_sc, h_x_sc = model_classifier(feat=batch_x, do=True)
                batch_magsp_x_sc, h_magsp_x_sc = model_classifier(feat_aux=batch_magsp_x, do=True)
                batch_x_post_sc, h_x_post_sc = model_classifier(feat=batch_x_post, do=True)
                batch_magsp_x_post_sc, h_magsp_x_post_sc = model_classifier(feat_aux=batch_magsp_x_post, do=True)

            # samples check
            with torch.no_grad():
                i = np.random.randint(0, batch_x.shape[0])
                logging.info("%d %s %d %d %d %d" % (i, \
                    os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                        f_ss, f_es, flens[i], max_flen))
                logging.info(batch_x[i,:2,:4])
                logging.info(batch_x_post[i,:2,:4])
                logging.info(batch_melsp[i,:2,:4])
                logging.info(batch_magsp_x[i,:2,:4])
                logging.info(batch_magsp_x_post[i,:2,:4])
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

                    melsp_x = batch_x[k,:flens_utt]
                    melsp_x_rest = (torch.exp(melsp_x)-1)/10000

                    melsp_x_post = batch_x_post[k,:flens_utt]
                    melsp_x_post_rest = (torch.exp(melsp_x_post)-1)/10000

                    magsp = batch_magsp[k,:flens_utt]
                    magsp_x = batch_magsp_x[k,:flens_utt]
                    magsp_x_post = batch_magsp_x_post[k,:flens_utt]

                    batch_pxz_post_ = batch_pxz_post[k,:flens_utt]
                    batch_loss_p_select += criterion_laplace(batch_pxz_post_[:,:args.mel_dim], batch_pxz_post_[:,args.mel_dim:], melsp).mean()
                    if flens_utt > 1:
                        batch_loss_p_select += torch.mean(torch.sum(criterion_l1(melsp_x, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_x, melsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(melsp_x, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_x, melsp), -1))) \
                                                + torch.mean(torch.sum(criterion_l1(melsp_x_post, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(melsp_x_post, melsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(melsp_x_post, melsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(melsp_x_post, melsp), -1)))
                        if iter_idx >= 50: #prevent early large losses
                            batch_loss_p_select += torch.mean(torch.sum(criterion_l1(magsp_x, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_x, magsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_x, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_x, magsp), -1))) \
                                                + torch.mean(torch.sum(criterion_l1(magsp_x_post, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(magsp_x_post, magsp), -1))) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_x_post, magsp), -1)) \
                                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(magsp_x_post, magsp), -1)))
                    else:
                        batch_loss_p_select += torch.mean(torch.sum(criterion_l1(melsp_x, melsp), -1)) \
                                                + torch.mean(torch.mean(criterion_l1(melsp_x, melsp), -1)) \
                                                + torch.mean(torch.sum(criterion_l1(melsp_x_post, melsp), -1)) \
                                                + torch.mean(torch.mean(criterion_l1(melsp_x_post, melsp), -1))
                        if iter_idx >= 50: #prevent early large losses
                            batch_loss_p_select += torch.mean(torch.sum(criterion_l1(magsp_x, magsp), -1)) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_x, magsp), -1)) \
                                                + torch.mean(torch.sum(criterion_l1(magsp_x_post, magsp), -1)) \
                                                + torch.mean(torch.mean(criterion_l1(magsp_x_post, magsp), -1))

                    if iter_idx >= 50:
                        batch_loss_ms_norm_x_, batch_loss_ms_err_x_ = criterion_ms(melsp_x_rest, melsp_rest)
                        if not torch.isinf(batch_loss_ms_norm_x_) and not torch.isnan(batch_loss_ms_norm_x_):
                            batch_loss_p_select += batch_loss_ms_norm_x_
                        if not torch.isinf(batch_loss_ms_err_x_) and not torch.isnan(batch_loss_ms_err_x_):
                            batch_loss_p_select += batch_loss_ms_err_x_

                        batch_loss_ms_norm_x_, batch_loss_ms_err_x_ = criterion_ms(melsp_x_post_rest, melsp_rest)
                        if not torch.isinf(batch_loss_ms_norm_x_) and not torch.isnan(batch_loss_ms_norm_x_):
                            batch_loss_p_select += batch_loss_ms_norm_x_
                        if not torch.isinf(batch_loss_ms_err_x_) and not torch.isnan(batch_loss_ms_err_x_):
                            batch_loss_p_select += batch_loss_ms_err_x_

                        batch_loss_ms_norm_x_, batch_loss_ms_err_x_ = criterion_ms(magsp_x, magsp)
                        if not torch.isinf(batch_loss_ms_norm_x_) and not torch.isnan(batch_loss_ms_norm_x_):
                            batch_loss_p_select += batch_loss_ms_norm_x_
                        if not torch.isinf(batch_loss_ms_err_x_) and not torch.isnan(batch_loss_ms_err_x_):
                            batch_loss_p_select += batch_loss_ms_err_x_

                        batch_loss_ms_norm_x_, batch_loss_ms_err_x_ = criterion_ms(magsp_x_post, magsp)
                        if not torch.isinf(batch_loss_ms_norm_x_) and not torch.isnan(batch_loss_ms_norm_x_):
                            batch_loss_p_select += batch_loss_ms_norm_x_
                        if not torch.isinf(batch_loss_ms_err_x_) and not torch.isnan(batch_loss_ms_err_x_):
                            batch_loss_p_select += batch_loss_ms_err_x_

                    batch_loss_q_select += torch.mean(torch.sum(kl_laplace(batch_qzx[k,:flens_utt]), -1))

                    batch_sc_ = batch_sc[k,:flens_utt]
                    batch_loss_sc_select += torch.mean(criterion_ce(batch_z_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_x_in_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_x_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_x_post_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_magsp_x_in_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_magsp_x_sc[k,:flens_utt], batch_sc_)) \
                                            + torch.mean(criterion_ce(batch_magsp_x_post_sc[k,:flens_utt], batch_sc_))

                batch_loss += batch_loss_p_select + batch_loss_q_select + batch_loss_sc_select
                if len(idx_select_full) > 0:
                    logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                    batch_melsp = torch.index_select(batch_melsp,0,idx_select_full)
                    batch_magsp = torch.index_select(batch_magsp,0,idx_select_full)
                    batch_sc = torch.index_select(batch_sc,0,idx_select_full)
                    batch_x = torch.index_select(batch_x,0,idx_select_full)
                    batch_x_post = torch.index_select(batch_x_post,0,idx_select_full)
                    batch_magsp_x = torch.index_select(batch_magsp_x,0,idx_select_full)
                    batch_magsp_x_post = torch.index_select(batch_magsp_x_post,0,idx_select_full)
                    batch_pxz_post = torch.index_select(batch_pxz_post,0,idx_select_full)
                    batch_qzx = torch.index_select(batch_qzx,0,idx_select_full)
                    batch_z_sc = torch.index_select(batch_z_sc,0,idx_select_full)
                    batch_x_in_sc = torch.index_select(batch_x_in_sc,0,idx_select_full)
                    batch_x_sc = torch.index_select(batch_x_sc,0,idx_select_full)
                    batch_x_post_sc = torch.index_select(batch_x_post_sc,0,idx_select_full)
                    batch_magsp_x_in_sc = torch.index_select(batch_magsp_x_in_sc,0,idx_select_full)
                    batch_magsp_x_sc = torch.index_select(batch_magsp_x_sc,0,idx_select_full)
                    batch_magsp_x_post_sc = torch.index_select(batch_magsp_x_post_sc,0,idx_select_full)
                    n_batch_utt = batch_x.shape[0]
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
            batch_loss_pxz_post_ = criterion_laplace(batch_pxz_post[:,:,:args.mel_dim], batch_pxz_post[:,:,args.mel_dim:], batch_melsp)
            batch_loss_pxz_post = batch_loss_pxz_post_.mean()

            ## melsp
            melsp_rest = (torch.exp(batch_melsp)-1)/10000
            melsp_x_rest = (torch.exp(batch_x)-1)/10000
            melsp_x_post_rest = (torch.exp(batch_x_post)-1)/10000

            ## melsp acc
            batch_loss_melsp_x_ = torch.mean(torch.sum(criterion_l1(batch_x, batch_melsp), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(batch_x, batch_melsp), -1), -1)) \
                                + torch.mean(torch.mean(criterion_l1(batch_x, batch_melsp), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_x, batch_melsp), -1), -1))
            batch_loss_melsp_x = batch_loss_melsp_x_.mean()
            batch_loss_melsp_x_dB = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_rest, min=1e-16))
                                                    -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)), -1))

            batch_loss_melsp_x_post_ = torch.mean(torch.sum(criterion_l1(batch_x_post, batch_melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(batch_x_post, batch_melsp), -1), -1)) \
                                    + torch.mean(torch.mean(criterion_l1(batch_x_post, batch_melsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_x_post, batch_melsp), -1), -1))
            batch_loss_melsp_x_post = batch_loss_melsp_x_post_.mean()
            batch_loss_melsp_x_post_dB = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(melsp_x_post_rest, min=1e-16))
                                                    -torch.log10(torch.clamp(melsp_rest, min=1e-16))))**2, -1)), -1))

            ## magsp acc
            batch_loss_magsp_x_ = torch.mean(torch.sum(criterion_l1(batch_magsp_x, batch_magsp), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.sum(criterion_l2(batch_magsp_x, batch_magsp), -1), -1)) \
                                + torch.mean(torch.mean(criterion_l1(batch_magsp_x, batch_magsp), -1), -1) \
                                    + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_magsp_x, batch_magsp), -1), -1))
            batch_loss_magsp_x = batch_loss_magsp_x_.mean()
            batch_loss_magsp_x_dB = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(batch_magsp_x, min=1e-16))
                                                    -torch.log10(torch.clamp(batch_magsp, min=1e-16))))**2, -1)), -1))

            batch_loss_magsp_x_post_ = torch.mean(torch.sum(criterion_l1(batch_magsp_x_post, batch_magsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.sum(criterion_l2(batch_magsp_x_post, batch_magsp), -1), -1)) \
                                    + torch.mean(torch.mean(criterion_l1(batch_magsp_x_post, batch_magsp), -1), -1) \
                                        + torch.sqrt(torch.mean(torch.mean(criterion_l2(batch_magsp_x_post, batch_magsp), -1), -1))
            batch_loss_magsp_x_post = batch_loss_magsp_x_post_.mean()
            batch_loss_magsp_x_post_dB = torch.mean(torch.mean(torch.sqrt(torch.mean((20*(torch.log10(torch.clamp(batch_magsp_x_post, min=1e-16))
                                                    -torch.log10(torch.clamp(batch_magsp, min=1e-16))))**2, -1)), -1))

            batch_loss_p = batch_loss_pxz_post_.sum() + batch_loss_melsp_x_.sum() + batch_loss_melsp_x_post_.sum()
            if iter_idx >= 50: #prevent early large losses
                batch_loss_p += batch_loss_magsp_x_.sum() + batch_loss_magsp_x_post_.sum()

            ## melsp ms
            batch_loss_ms_norm_x_, batch_loss_ms_err_x_ = criterion_ms(melsp_x_rest, melsp_rest)
            batch_loss_ms_norm_x = batch_loss_ms_norm_x_.mean()
            batch_loss_ms_err_x = batch_loss_ms_err_x_.mean()
            if iter_idx >= 50:
                if not torch.isinf(batch_loss_ms_norm_x) and not torch.isnan(batch_loss_ms_norm_x):
                    batch_loss_p += batch_loss_ms_norm_x_.sum()
                if not torch.isinf(batch_loss_ms_err_x) and not torch.isnan(batch_loss_ms_err_x):
                    batch_loss_p += batch_loss_ms_err_x_.sum()

            batch_loss_ms_norm_x_post_, batch_loss_ms_err_x_post_ = criterion_ms(melsp_x_post_rest, melsp_rest)
            batch_loss_ms_norm_x_post = batch_loss_ms_norm_x_post_.mean()
            batch_loss_ms_err_x_post = batch_loss_ms_err_x_post_.mean()
            if iter_idx >= 50:
                if not torch.isinf(batch_loss_ms_norm_x_post) and not torch.isnan(batch_loss_ms_norm_x_post):
                    batch_loss_p += batch_loss_ms_norm_x_post_.sum()
                if not torch.isinf(batch_loss_ms_err_x_post) and not torch.isnan(batch_loss_ms_err_x_post):
                    batch_loss_p += batch_loss_ms_err_x_post_.sum()

            ## magsp ms
            batch_loss_ms_norm_magsp_x_, batch_loss_ms_err_magsp_x_ = criterion_ms(batch_magsp_x, batch_magsp)
            batch_loss_ms_norm_magsp_x = batch_loss_ms_norm_magsp_x_.mean()
            batch_loss_ms_err_magsp_x = batch_loss_ms_err_magsp_x_.mean()
            if iter_idx >= 50:
                if not torch.isinf(batch_loss_ms_norm_magsp_x) and not torch.isnan(batch_loss_ms_norm_magsp_x):
                    batch_loss_p += batch_loss_ms_norm_magsp_x_.sum()
                if not torch.isinf(batch_loss_ms_err_magsp_x) and not torch.isnan(batch_loss_ms_err_magsp_x):
                    batch_loss_p += batch_loss_ms_err_magsp_x_.sum()

            batch_loss_ms_norm_magsp_x_post_, batch_loss_ms_err_magsp_x_post_ = criterion_ms(batch_magsp_x_post, batch_magsp)
            batch_loss_ms_norm_magsp_x_post = batch_loss_ms_norm_magsp_x_post_.mean()
            batch_loss_ms_err_magsp_x_post = batch_loss_ms_err_magsp_x_post_.mean()
            if iter_idx >= 50:
                if not torch.isinf(batch_loss_ms_norm_magsp_x_post) and not torch.isnan(batch_loss_ms_norm_magsp_x_post):
                    batch_loss_p += batch_loss_ms_norm_magsp_x_post_.sum()
                if not torch.isinf(batch_loss_ms_err_magsp_x_post) and not torch.isnan(batch_loss_ms_err_magsp_x_post):
                    batch_loss_p += batch_loss_ms_err_magsp_x_post_.sum()

            # KL-div. lat.
            batch_loss_qzx_pz_ = torch.mean(torch.sum(kl_laplace(batch_qzx), -1), -1)
            batch_loss_qzx_pz = batch_loss_qzx_pz_.mean()

            batch_loss_q = batch_loss_qzx_pz_.sum()

            # lat/melsp/magsp cls
            batch_sc_ = batch_sc.reshape(-1)

            batch_loss_sc_z_ = torch.mean(criterion_ce(batch_z_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_z = batch_loss_sc_z_.mean()

            batch_loss_sc_x_in_ = torch.mean(criterion_ce(batch_x_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_x_in = batch_loss_sc_x_in_.mean()

            batch_loss_sc_x_ = torch.mean(criterion_ce(batch_x_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_x = batch_loss_sc_x_.mean()

            batch_loss_sc_x_post_ = torch.mean(criterion_ce(batch_x_post_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_x_post = batch_loss_sc_x_post_.mean()

            batch_loss_sc_magsp_x_in_ = torch.mean(criterion_ce(batch_magsp_x_in_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_magsp_x_in = batch_loss_sc_magsp_x_in_.mean()

            batch_loss_sc_magsp_x_ = torch.mean(criterion_ce(batch_magsp_x_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_magsp_x = batch_loss_sc_magsp_x_.mean()

            batch_loss_sc_magsp_x_post_ = torch.mean(criterion_ce(batch_magsp_x_post_sc.reshape(-1, n_spk), batch_sc_).reshape(n_batch_utt, -1), -1)
            batch_loss_sc_magsp_x_post = batch_loss_sc_magsp_x_post_.mean()

            batch_loss_sc = batch_loss_sc_z_.sum() \
                            + batch_loss_sc_x_in_.sum() + batch_loss_sc_x_.sum() + batch_loss_sc_x_post_.sum() \
                                + batch_loss_sc_magsp_x_in_.sum() + batch_loss_sc_magsp_x_.sum() + batch_loss_sc_magsp_x_post_.sum() \

            # elbo
            batch_loss_elbo = batch_loss_p + batch_loss_q + batch_loss_sc
            batch_loss += batch_loss_elbo

            total_train_loss["train/loss_elbo"].append(batch_loss_elbo.item())
            total_train_loss["train/loss_pxz_post"].append(batch_loss_pxz_post.item())
            total_train_loss["train/loss_qzx_pz"].append(batch_loss_qzx_pz.item())
            loss_elbo.append(batch_loss_elbo.item())
            loss_pxz_post.append(batch_loss_pxz_post.item())
            loss_qzx_pz.append(batch_loss_qzx_pz.item())

            total_train_loss["train/loss_sc_z"].append(batch_loss_sc_z.item())
            total_train_loss["train/loss_sc_x_in"].append(batch_loss_sc_x_in.item())
            total_train_loss["train/loss_sc_x"].append(batch_loss_sc_x.item())
            total_train_loss["train/loss_sc_x_post"].append(batch_loss_sc_x_post.item())
            total_train_loss["train/loss_sc_magsp_x_in"].append(batch_loss_sc_magsp_x_in.item())
            total_train_loss["train/loss_sc_magsp_x"].append(batch_loss_sc_magsp_x.item())
            total_train_loss["train/loss_sc_magsp_x_post"].append(batch_loss_sc_magsp_x_post.item())
            loss_sc_z.append(batch_loss_sc_z.item())
            loss_sc_x_in.append(batch_loss_sc_x_in.item())
            loss_sc_x.append(batch_loss_sc_x.item())
            loss_sc_x_post.append(batch_loss_sc_x_post.item())
            loss_sc_magsp_x_in.append(batch_loss_sc_magsp_x_in.item())
            loss_sc_magsp_x.append(batch_loss_sc_magsp_x.item())
            loss_sc_magsp_x_post.append(batch_loss_sc_magsp_x_post.item())

            total_train_loss["train/loss_melsp_x"].append(batch_loss_melsp_x.item())
            total_train_loss["train/loss_melsp_x_dB"].append(batch_loss_melsp_x_dB.item())
            total_train_loss["train/loss_melsp_x_post"].append(batch_loss_melsp_x_post.item())
            total_train_loss["train/loss_melsp_x_post_dB"].append(batch_loss_melsp_x_post_dB.item())
            total_train_loss["train/loss_magsp_x"].append(batch_loss_magsp_x.item())
            total_train_loss["train/loss_magsp_x_dB"].append(batch_loss_magsp_x_dB.item())
            total_train_loss["train/loss_magsp_x_post"].append(batch_loss_magsp_x_post.item())
            total_train_loss["train/loss_magsp_x_post_dB"].append(batch_loss_magsp_x_post_dB.item())
            loss_melsp_x.append(batch_loss_melsp_x.item())
            loss_melsp_x_dB.append(batch_loss_melsp_x_dB.item())
            loss_melsp_x_post.append(batch_loss_melsp_x_post.item())
            loss_melsp_x_post_dB.append(batch_loss_melsp_x_post_dB.item())
            loss_magsp_x.append(batch_loss_magsp_x.item())
            loss_magsp_x_dB.append(batch_loss_magsp_x_dB.item())
            loss_magsp_x_post.append(batch_loss_magsp_x_post.item())
            loss_magsp_x_post_dB.append(batch_loss_magsp_x_post_dB.item())

            total_train_loss["train/loss_ms_norm_x"].append(batch_loss_ms_norm_x.item())
            total_train_loss["train/loss_ms_err_x"].append(batch_loss_ms_err_x.item())
            total_train_loss["train/loss_ms_norm_x_post"].append(batch_loss_ms_norm_x_post.item())
            total_train_loss["train/loss_ms_err_x_post"].append(batch_loss_ms_err_x_post.item())
            total_train_loss["train/loss_ms_norm_magsp_x"].append(batch_loss_ms_norm_magsp_x.item())
            total_train_loss["train/loss_ms_err_magsp_x"].append(batch_loss_ms_err_magsp_x.item())
            total_train_loss["train/loss_ms_norm_magsp_x_post"].append(batch_loss_ms_norm_magsp_x_post.item())
            total_train_loss["train/loss_ms_err_magsp_x_post"].append(batch_loss_ms_err_magsp_x_post.item())
            loss_ms_norm_x.append(batch_loss_ms_norm_x.item())
            loss_ms_err_x.append(batch_loss_ms_err_x.item())
            loss_ms_norm_x_post.append(batch_loss_ms_norm_x_post.item())
            loss_ms_err_x_post.append(batch_loss_ms_err_x_post.item())
            loss_ms_norm_magsp_x.append(batch_loss_ms_norm_magsp_x.item())
            loss_ms_err_magsp_x.append(batch_loss_ms_err_magsp_x.item())
            loss_ms_norm_magsp_x_post.append(batch_loss_ms_norm_magsp_x_post.item())
            loss_ms_err_magsp_x_post.append(batch_loss_ms_err_magsp_x_post.item())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            text_log = "batch loss [%d] %d %d " % (c_idx+1, f_ss, f_bs)
            text_log += "%.3f %.3f %.3f ; %.3f , %.3f %.3f %.3f , %.3f %.3f %.3f ; "\
                "%.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB , %.3f %.3f dB ; "\
                "%.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ;; " % (
                batch_loss_elbo.item(), batch_loss_pxz_post.item(), batch_loss_qzx_pz.item(),
                        batch_loss_sc_z.item(),
                        batch_loss_sc_x_in.item(), batch_loss_sc_x.item(), batch_loss_sc_x_post.item(),
                        batch_loss_sc_magsp_x_in.item(), batch_loss_sc_magsp_x.item(), batch_loss_sc_magsp_x_post.item(),
                        batch_loss_melsp_x.item(), batch_loss_melsp_x_dB.item(), batch_loss_melsp_x_post.item(), batch_loss_melsp_x_post_dB.item(),
                        batch_loss_magsp_x.item(), batch_loss_magsp_x_dB.item(), batch_loss_magsp_x_post.item(), batch_loss_magsp_x_post_dB.item(),
                        batch_loss_ms_norm_x.item(), batch_loss_ms_err_x.item(), batch_loss_ms_norm_x_post.item(), batch_loss_ms_err_x_post.item(),
                        batch_loss_ms_norm_magsp_x.item(), batch_loss_ms_err_magsp_x.item(),
                        batch_loss_ms_norm_magsp_x_post.item(), batch_loss_ms_err_magsp_x_post.item())
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


    # save final model
    model_encoder.cpu()
    model_decoder.cpu()
    model_post.cpu()
    model_classifier.cpu()
    torch.save({"model_encoder": model_encoder.state_dict(),
                "model_decoder": model_decoder.state_dict(),
                "model_post": model_post.state_dict(),
                "model_classifier": model_classifier.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
