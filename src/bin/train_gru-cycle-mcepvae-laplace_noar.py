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
from vcneuvoco import GRU_VAE_ENCODER, GRU_SPEC_DECODER, GRU_LAT_FEAT_CLASSIFIER, RevGrad
from vcneuvoco import kl_laplace, ModulationSpectrumLoss, kl_categorical_categorical_logits
#from radam import RAdam
import torch_optimizer as optim

from dataset import FeatureDatasetCycMceplf0WavVAE, FeatureDatasetEvalCycMceplf0WavVAE, padding

from dtw_c import dtw_c as dtw

#from sklearn.cluster import KMeans

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
                yield feat, sc, sc_cv, feat_cv, c_idx, idx, featfiles, f_bs, f_ss, flens, \
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

        yield [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], []


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
                if spcidx:
                    yield feat, feat_trg, sc, sc_cv, feat_cv, c_idx, idx, featfiles, f_bs, f_ss, flens, \
                        n_batch_utt, del_index_utt, max_flen, spk_cv, file_src_trg_flag, spcidx_src, \
                            spcidx_src_trg, flens_spc_src, flens_spc_src_trg, feat_full, sc_full, sc_cv_full
                else:
                    yield feat, feat_trg, sc, sc_cv, feat_cv, c_idx, idx, featfiles, f_bs, f_ss, flens, \
                        n_batch_utt, del_index_utt, max_flen, spk_cv, file_src_trg_flag, spcidx_src, \
                            spcidx_src_trg, flens_spc_src, flens_spc_src_trg
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
            yield [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        else:
            yield [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], []


def save_checkpoint(checkpoint_dir, model_encoder, model_decoder, model_classifier,
        min_eval_loss_mcep, min_eval_loss_mcep_std, min_eval_loss_mcep_cv,
        min_eval_loss_mcep_rec, min_eval_loss_mcd_src_trg, min_eval_loss_mcd_src_trg_std,
        iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, iterations):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    model_encoder.cpu()
    model_decoder.cpu()
    model_classifier.cpu()
    checkpoint = {
        "model_encoder": model_encoder.state_dict(),
        "model_decoder": model_decoder.state_dict(),
        "model_classifier": model_classifier.state_dict(),
        "min_eval_loss_mcep": min_eval_loss_mcep,
        "min_eval_loss_mcep_std": min_eval_loss_mcep_std,
        "min_eval_loss_mcep_cv": min_eval_loss_mcep_cv,
        "min_eval_loss_mcep_rec": min_eval_loss_mcep_rec,
        "min_eval_loss_mcd_src_trg": min_eval_loss_mcd_src_trg,
        "min_eval_loss_mcd_src_trg_std": min_eval_loss_mcd_src_trg_std,
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
                        type=str, help="directory of training feat files")
    parser.add_argument("--feats_eval_list", required=True,
                        type=str, help="list of evaluation feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="directory of joint statistics")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model and logs")
    # network structure setting
    parser.add_argument("--hidden_units_enc", default=1024,
                        type=int, help="hidden units gru encoder")
    parser.add_argument("--hidden_layers_enc", default=1,
                        type=int, help="hidden layers gru encoder")
    parser.add_argument("--hidden_units_dec", default=1024,
                        type=int, help="hidden units gru decoder")
    parser.add_argument("--hidden_layers_dec", default=1,
                        type=int, help="hidden layers gru decoder")
    parser.add_argument("--kernel_size_enc", default=7,
                        type=int, help="kernel size of input conv. encoder")
    parser.add_argument("--dilation_size_enc", default=1,
                        type=int, help="dilation size of input conv. encoder")
    parser.add_argument("--kernel_size_dec", default=7,
                        type=int, help="kernel size of input conv. decoder")
    parser.add_argument("--dilation_size_dec", default=1,
                        type=int, help="dilation size of input conv. decoder")
    parser.add_argument("--right_size_enc", default=2,
                        type=int, help="lookup frame in case of input skewed conv. (if 0, it's a balanced conv.)")
    parser.add_argument("--right_size_dec", default=0,
                        type=int, help="lookup frame in case of input skewed conv. (if 0, it's a balanced conv.)")
    parser.add_argument("--spk_list", required=True,
                        type=str, help="list of speakers")
    parser.add_argument("--stats_list", required=True,
                        type=str, help="list of stats files")
    parser.add_argument("--lat_dim", default=32,
                        type=int, help="number of latent dimension")
    parser.add_argument("--mcep_dim", default=50,
                        type=int, help="number of mel-cep dimension including 0th power")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=30,
                        type=int, help="batch sequence size in frames")
    parser.add_argument("--epoch_count", default=120,
                        type=int, help="number of maximum training epochs")
    parser.add_argument("--do_prob", default=0,
                        type=float, help="dropout probability")
    parser.add_argument("--batch_size_utt", default=5,
                        type=int, help="batch size for train data")
    parser.add_argument("--batch_size_utt_eval", default=5,
                        type=int, help="batch size for eval data")
    parser.add_argument("--n_workers", default=2,
                        type=int, help="number of workers for dataset loading")
    parser.add_argument("--n_half_cyc", default=2,
                        type=int, help="number of half cycles, number of cycles = half cycles // 2")
    parser.add_argument("--causal_conv_enc", default=False,
                        type=strtobool, help="causal input conv. gru encoder")
    parser.add_argument("--causal_conv_dec", default=False,
                        type=strtobool, help="causal input conv. gru decoder")
    # other setting
    parser.add_argument("--pad_len", default=3000,
                        type=int, help="zero pad length for batch loading in frames")
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
                        type=str, help="pretrained model path")
    #parser.add_argument("--string_path", required=True,
    #                    type=str, help="h5 path of features")
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

    mean_stats = torch.FloatTensor(read_hdf5(args.stats, "/mean_feat_mceplf0cap"))
    scale_stats = torch.FloatTensor(read_hdf5(args.stats, "/scale_feat_mceplf0cap"))

    args.cap_dim = mean_stats.shape[0]-(args.mcep_dim+3)
    args.excit_dim = 2+1+args.cap_dim

    # save args as conf
    args.fftsize = 2 ** (len(bin(args.batch_size)) - 2 + 1)
    args.string_path = "/feat_mceplf0cap"
    torch.save(args, args.expdir + "/model.conf")

    # define network
    model_encoder = GRU_VAE_ENCODER(
        in_dim=args.mcep_dim+args.excit_dim,
        n_spk=n_spk,
        lat_dim=args.lat_dim,
        hidden_layers=args.hidden_layers_enc,
        hidden_units=args.hidden_units_enc,
        kernel_size=args.kernel_size_enc,
        dilation_size=args.dilation_size_enc,
        causal_conv=args.causal_conv_enc,
        bi=False,
        ar=False,
        pad_first=True,
        right_size=args.right_size_enc,
        do_prob=args.do_prob)
    logging.info(model_encoder)
    model_decoder = GRU_SPEC_DECODER(
        feat_dim=args.lat_dim,
        out_dim=args.mcep_dim,
        n_spk=n_spk,
        hidden_layers=args.hidden_layers_dec,
        hidden_units=args.hidden_units_dec,
        kernel_size=args.kernel_size_dec,
        dilation_size=args.dilation_size_dec,
        causal_conv=args.causal_conv_dec,
        bi=False,
        ar=False,
        pad_first=True,
        right_size=args.right_size_dec,
        do_prob=args.do_prob)
    logging.info(model_decoder)
    model_classifier = GRU_LAT_FEAT_CLASSIFIER(
        lat_dim=args.lat_dim,
        feat_dim=args.mcep_dim,
        n_spk=n_spk,
        hidden_units=16,
        hidden_layers=1)
    logging.info(model_classifier)
    criterion_ms = ModulationSpectrumLoss(args.fftsize)
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    revgrad = RevGrad()
    criterion_l1 = torch.nn.L1Loss(reduction='none')
    criterion_l2 = torch.nn.MSELoss(reduction='none')

    p_spk = torch.ones(n_spk)/n_spk

    # send to gpu
    if torch.cuda.is_available():
        model_encoder.cuda()
        model_decoder.cuda()
        model_classifier.cuda()
        criterion_ms.cuda()
        criterion_ce.cuda()
        revgrad.cuda()
        criterion_l1.cuda()
        criterion_l2.cuda()
        mean_stats = mean_stats.cuda()
        scale_stats = scale_stats.cuda()
        p_spk = p_spk.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)
    logits_p_spk = torch.log(p_spk)

    logging.info(p_spk)
    logging.info(logits_p_spk)

    model_encoder.train()
    model_decoder.train()

    if model_encoder.use_weight_norm:
        torch.nn.utils.remove_weight_norm(model_encoder.scale_in)
    if model_decoder.use_weight_norm:
        torch.nn.utils.remove_weight_norm(model_decoder.scale_out)

    model_encoder.scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/scale_stats.data),2))
    model_encoder.scale_in.bias = torch.nn.Parameter(-(mean_stats.data/scale_stats.data))
    model_decoder.scale_out.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(scale_stats[args.excit_dim:].data),2))
    model_decoder.scale_out.bias = torch.nn.Parameter(mean_stats[args.excit_dim:].data)

    if model_encoder.use_weight_norm:
        torch.nn.utils.weight_norm(model_encoder.scale_in)
    if model_decoder.use_weight_norm:
        torch.nn.utils.weight_norm(model_decoder.scale_out)

    parameters = filter(lambda p: p.requires_grad, model_encoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_decoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder): %.3f million' % parameters)
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

    module_list = list(model_encoder.conv.parameters())
    module_list += list(model_encoder.gru.parameters()) + list(model_encoder.out.parameters())

    module_list += list(model_decoder.conv.parameters())
    module_list += list(model_decoder.gru.parameters()) + list(model_decoder.out.parameters())

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
    #optimizer = RAdam(module_list, lr=args.lr)
    #optimizer = torch.optim.Adam(module_list, lr=args.lr)

    # resume
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained)
        model_encoder.load_state_dict(checkpoint["model_encoder"])
        model_decoder.load_state_dict(checkpoint["model_decoder"])
        model_classifier.load_state_dict(checkpoint["model_classifier"])
        epoch_idx = checkpoint["iterations"]
        logging.info("pretrained from %d-iter checkpoint." % epoch_idx)
        epoch_idx = 0
    elif args.resume is not None:
        checkpoint = torch.load(args.resume)
        model_encoder.load_state_dict(checkpoint["model_encoder"])
        model_decoder.load_state_dict(checkpoint["model_decoder"])
        model_classifier.load_state_dict(checkpoint["model_classifier"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)
    else:
        epoch_idx = 0

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
    dataset = FeatureDatasetCycMceplf0WavVAE(feat_list, pad_feat_transform, spk_list, stats_list, \
                    args.n_half_cyc, args.string_path, excit_dim=args.excit_dim, uvcap_flag=False)
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
                    stats_list, args.string_path, excit_dim=args.excit_dim, uvcap_flag=False)
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
        gv_mean[i] = read_hdf5(stats_list[i], "/gv_range_mean")[1:]

    # train
    eps_1 = torch.finfo(mean_stats.dtype).eps-1
    logging.info(eps_1)
    mcd_constant = 6.1418514637137542748551098286418
    logging.info(f'n_half_cyc: {args.n_half_cyc}')
    logging.info(f'n_rec: {n_rec}')
    logging.info(f'n_cv: {n_cv}')
    enc_pad_left = model_encoder.pad_left
    enc_pad_right = model_encoder.pad_right
    logging.info(f'enc_pad_left: {enc_pad_left}')
    logging.info(f'enc_pad_right: {enc_pad_right}')
    dec_pad_left = model_decoder.pad_left
    dec_pad_right = model_decoder.pad_right
    logging.info(f'dec_pad_left: {dec_pad_left}')
    logging.info(f'dec_pad_right: {dec_pad_right}')
    first_pad_left = (enc_pad_left + dec_pad_left)*args.n_half_cyc
    first_pad_right = (enc_pad_right + dec_pad_right)*args.n_half_cyc
    if args.n_half_cyc == 1:
        first_pad_left += enc_pad_left
        first_pad_right += enc_pad_right
    logging.info(f'first_pad_left: {first_pad_left}')
    logging.info(f'first_pad_right: {first_pad_right}')
    if args.n_half_cyc > 1:
        outpad_lefts = [None]*args.n_half_cyc*2
        outpad_rights = [None]*args.n_half_cyc*2
    else:
        outpad_lefts = [None]*(args.n_half_cyc*2+1)
        outpad_rights = [None]*(args.n_half_cyc*2+1)
    outpad_lefts[0] = first_pad_left-enc_pad_left
    outpad_rights[0] = first_pad_right-enc_pad_right
    for i in range(1,args.n_half_cyc*2):
        if i % 2 == 1:
            outpad_lefts[i] = outpad_lefts[i-1]-dec_pad_left
            outpad_rights[i] = outpad_rights[i-1]-dec_pad_right
        else:
            outpad_lefts[i] = outpad_lefts[i-1]-enc_pad_left
            outpad_rights[i] = outpad_rights[i-1]-enc_pad_right
    if args.n_half_cyc == 1:
        outpad_lefts[len(outpad_lefts)-1] = outpad_lefts[len(outpad_lefts)-2]-enc_pad_left
        outpad_rights[len(outpad_rights)-1] = outpad_rights[len(outpad_rights)-2]-enc_pad_right
    logging.info(outpad_lefts)
    logging.info(outpad_rights)
    batch_feat_in = [None]*args.n_half_cyc*2
    batch_sc_in = [None]*args.n_half_cyc*2
    batch_sc_cv_in = [None]*n_cv
    batch_feat_cv_in = [None]*n_cv
    total = 0
    iter_count = 0
    batch_sc_cv = [None]*n_cv
    batch_feat_cv = [None]*n_cv
    qy_logits = [None]*n_rec
    qz_alpha = [None]*n_rec
    z = [None]*n_rec
    batch_z_sc = [None]*n_rec
    batch_mcep_rec = [None]*n_rec
    batch_mcep_rec_sc = [None]*n_rec
    batch_mcep_cv = [None]*n_cv
    batch_mcep_cv_sc = [None]*n_cv
    h_z = [None]*n_rec
    h_z_sc = [None]*n_rec
    h_mcep = [None]*n_rec
    h_mcep_sc = [None]*n_rec
    h_mcep_cv = [None]*n_cv
    h_mcep_cv_sc = [None]*n_cv
    loss_elbo = [None]*args.n_half_cyc
    loss_px = [None]*args.n_half_cyc
    loss_qy_py = [None]*n_rec
    loss_qy_py_err = [None]*n_rec
    loss_qz_pz = [None]*n_rec
    loss_powmcep = [None]*args.n_half_cyc
    loss_mcep = [None]*args.n_half_cyc
    loss_mcep_cv = [None]*args.n_half_cyc
    loss_mcep_rec = [None]*args.n_half_cyc
    loss_mcdpow_src_trg = []
    loss_mcd_src_trg = []
    loss_lat_dist_rmse = []
    loss_lat_dist_cossim = []
    for i in range(args.n_half_cyc):
        loss_elbo[i] = []
        loss_px[i] = []
        loss_qy_py[i] = []
        loss_qy_py_err[i] = []
        loss_qz_pz[i] = []
        if args.n_half_cyc == 1:
            loss_qy_py[i+1] = []
            loss_qy_py_err[i+1] = []
            loss_qz_pz[i+1] = []
        loss_powmcep[i] = []
        loss_mcep[i] = []
        loss_mcep_cv[i] = []
        loss_mcep_rec[i] = []
    batch_loss_powmcep = [None]*args.n_half_cyc
    batch_loss_mcep = [None]*args.n_half_cyc
    batch_loss_sc_mcep = [None]*args.n_half_cyc
    batch_loss_sc_mcep_rev = [None]*args.n_half_cyc
    batch_loss_mcep_rec = [None]*args.n_half_cyc
    batch_loss_mcep_cv = [None]*n_cv
    batch_loss_sc_mcep_cv = [None]*n_cv
    batch_loss_sc_mcep_cv_rev = [None]*n_cv
    batch_loss_px = [None]*args.n_half_cyc
    batch_loss_ms_norm = [None]*args.n_half_cyc
    batch_loss_ms_err = [None]*args.n_half_cyc
    batch_loss_qy_py = [None]*n_rec
    batch_loss_qy_py_rev = [None]*n_rec
    batch_loss_qy_py_err = [None]*n_rec
    batch_loss_qz_pz = [None]*n_rec
    batch_loss_sc_z = [None]*n_rec
    batch_loss_sc_z_rev = [None]*n_rec
    batch_loss_sc_z_cv_rev = [None]*n_cv
    batch_loss_elbo = [None]*args.n_half_cyc
    n_half_cyc_eval = min(2,args.n_half_cyc)
    n_rec_eval = n_half_cyc_eval + n_half_cyc_eval%2
    n_cv_eval = int(n_half_cyc_eval/2+n_half_cyc_eval%2)
    first_pad_left_eval = (enc_pad_left + dec_pad_left)*n_half_cyc_eval
    first_pad_right_eval = (enc_pad_right + dec_pad_right)*n_half_cyc_eval
    if n_half_cyc_eval == 1:
        first_pad_left_eval += enc_pad_left
        first_pad_right_eval += enc_pad_right
    logging.info(f'first_pad_left_eval: {first_pad_left_eval}')
    logging.info(f'first_pad_right_eval: {first_pad_right_eval}')
    if n_half_cyc_eval > 1:
        outpad_lefts_eval = [None]*n_half_cyc_eval*2
        outpad_rights_eval = [None]*n_half_cyc_eval*2
    else:
        outpad_lefts_eval = [None]*(n_half_cyc_eval*2+1)
        outpad_rights_eval = [None]*(n_half_cyc_eval*2+1)
    outpad_lefts_eval[0] = first_pad_left_eval-enc_pad_left
    outpad_rights_eval[0] = first_pad_right_eval-enc_pad_right
    for i in range(1,n_half_cyc_eval*2):
        if i % 2 == 1:
            outpad_lefts_eval[i] = outpad_lefts_eval[i-1]-dec_pad_left
            outpad_rights_eval[i] = outpad_rights_eval[i-1]-dec_pad_right
        else:
            outpad_lefts_eval[i] = outpad_lefts_eval[i-1]-enc_pad_left
            outpad_rights_eval[i] = outpad_rights_eval[i-1]-enc_pad_right
    if n_half_cyc_eval == 1:
        outpad_lefts_eval[len(outpad_lefts_eval)-1] = outpad_lefts_eval[len(outpad_lefts_eval)-2]-enc_pad_left
        outpad_rights_eval[len(outpad_rights_eval)-1] = outpad_rights_eval[len(outpad_rights_eval)-2]-enc_pad_right
    logging.info(outpad_lefts_eval)
    logging.info(outpad_rights_eval)
    first_pad_left_eval_utt = enc_pad_left + dec_pad_left
    first_pad_right_eval_utt = enc_pad_right + dec_pad_right
    logging.info(f'first_pad_left_eval_utt: {first_pad_left_eval_utt}')
    logging.info(f'first_pad_right_eval_utt: {first_pad_right_eval_utt}')
    gv_src_src = [None]*n_spk
    gv_src_trg = [None]*n_spk
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
    eval_loss_powmcep = [None]*n_half_cyc_eval
    eval_loss_powmcep_std = [None]*n_half_cyc_eval
    eval_loss_mcep = [None]*n_half_cyc_eval
    eval_loss_mcep_std = [None]*n_half_cyc_eval
    eval_loss_mcep_rec = [None]*n_half_cyc_eval
    eval_loss_mcep_rec_std = [None]*n_half_cyc_eval
    eval_loss_mcep_cv = [None]*n_cv_eval
    eval_loss_mcep_cv_std = [None]*n_cv_eval
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
    min_eval_loss_powmcep = [None]*n_half_cyc_eval
    min_eval_loss_powmcep_std = [None]*n_half_cyc_eval
    min_eval_loss_mcep = [None]*n_half_cyc_eval
    min_eval_loss_mcep_std = [None]*n_half_cyc_eval
    min_eval_loss_mcep_rec = [None]*n_half_cyc_eval
    min_eval_loss_mcep_rec_std = [None]*n_half_cyc_eval
    min_eval_loss_mcep_cv = [None]*n_cv_eval
    min_eval_loss_mcep_cv_std = [None]*n_cv_eval
    min_eval_loss_mcep[0] = 99999999.99
    min_eval_loss_mcep_std[0] = 99999999.99
    min_eval_loss_mcep_cv[0] = 99999999.99
    min_eval_loss_mcep_rec[0] = 99999999.99
    min_eval_loss_mcd_src_trg = 99999999.99
    min_eval_loss_mcd_src_trg_std = 99999999.99
    iter_idx = 0
    min_idx = -1
    change_min_flag = False
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        min_eval_loss_mcep[0] = checkpoint["min_eval_loss_mcep"]
        min_eval_loss_mcep_std[0] = checkpoint["min_eval_loss_mcep_std"]
        min_eval_loss_mcep_cv[0] = checkpoint["min_eval_loss_mcep_cv"]
        min_eval_loss_mcep_rec[0] = checkpoint["min_eval_loss_mcep_rec"]
        min_eval_loss_mcd_src_trg = checkpoint["min_eval_loss_mcd_src_trg"]
        min_eval_loss_mcd_src_trg_std = checkpoint["min_eval_loss_mcd_src_trg_std"]
        iter_idx = checkpoint["iter_idx"]
        min_idx = checkpoint["min_idx"]
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        batch_feat, batch_sc, batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
            f_bs, f_ss, flens, n_batch_utt, del_index_utt, max_flen, spk_cv, idx_select, idx_select_full, flens_acc = next(generator)
        if c_idx < 0: # summarize epoch
            # save current epoch model
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # report current epoch
            text_log = "(EPOCH:%d) average optimization loss " % (epoch_idx + 1)
            for i in range(args.n_half_cyc):
                if i % 2 == 0:
                    if args.n_half_cyc > 1:
                        text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ;; " % (i+1, \
                            np.mean(loss_elbo[i]), np.std(loss_elbo[i]), np.mean(loss_px[i]), np.std(loss_px[i]), \
                            np.mean(loss_qy_py[i]), np.std(loss_qy_py[i]), np.mean(loss_qy_py_err[i]), np.std(loss_qy_py_err[i]), np.mean(loss_qz_pz[i]), np.std(loss_qz_pz[i]), \
                            np.mean(loss_mcep_rec[i]), np.std(loss_mcep_rec[i]), np.mean(loss_mcep_cv[i//2]), np.std(loss_mcep_cv[i//2]), \
                            np.mean(loss_powmcep[i]), np.std(loss_powmcep[i]), np.mean(loss_mcep[i]), np.std(loss_mcep[i]))
                    else:
                        text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ;; " % (i+1, \
                            np.mean(loss_elbo[i]), np.std(loss_elbo[i]), np.mean(loss_px[i]), np.std(loss_px[i]), \
                            np.mean(loss_qy_py[i]), np.std(loss_qy_py[i]), np.mean(loss_qy_py_err[i]), np.std(loss_qy_py_err[i]), np.mean(loss_qz_pz[i]), np.std(loss_qz_pz[i]), \
                            np.mean(loss_qy_py[i+1]), np.std(loss_qy_py[i+1]), np.mean(loss_qy_py_err[i+1]), np.std(loss_qy_py_err[i+1]), np.mean(loss_qz_pz[i+1]), np.std(loss_qz_pz[i+1]), \
                            np.mean(loss_mcep_rec[i]), np.std(loss_mcep_rec[i]), np.mean(loss_mcep_cv[i//2]), np.std(loss_mcep_cv[i//2]), \
                            np.mean(loss_powmcep[i]), np.std(loss_powmcep[i]), np.mean(loss_mcep[i]), np.std(loss_mcep[i]))
                else:
                    text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                        "%.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ;; " % (i+1, \
                        np.mean(loss_elbo[i]), np.std(loss_elbo[i]),
                        np.mean(loss_px[i]), np.std(loss_px[i]), np.mean(loss_qy_py[i]), np.std(loss_qy_py[i]), np.mean(loss_qy_py_err[i]), np.std(loss_qy_py_err[i]), \
                        np.mean(loss_qz_pz[i]), np.std(loss_qz_pz[i]), np.mean(loss_mcep_rec[i]), np.std(loss_mcep_rec[i]), \
                        np.mean(loss_powmcep[i]), np.std(loss_powmcep[i]), np.mean(loss_mcep[i]), np.std(loss_mcep[i]))
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            logging.info("estimated time until max. epoch = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * total))))
            # compute loss in evaluation data
            total = 0
            iter_count = 0
            loss_mcdpow_src_trg = []
            loss_mcd_src_trg = []
            loss_lat_dist_rmse = []
            loss_lat_dist_cossim = []
            for i in range(n_spk):
                gv_src_src[i] = []
                gv_src_trg[i] = []
            for i in range(args.n_half_cyc):
                loss_elbo[i] = []
                loss_px[i] = []
                loss_qy_py[i] = []
                loss_qy_py_err[i] = []
                loss_qz_pz[i] = []
                if args.n_half_cyc == 1:
                    loss_qy_py[i+1] = []
                    loss_qy_py_err[i+1] = []
                    loss_qz_pz[i+1] = []
                loss_powmcep[i] = []
                loss_mcep[i] = []
                loss_mcep_cv[i] = []
                loss_mcep_rec[i] = []
            model_encoder.eval()
            model_decoder.eval()
            model_classifier.eval()
            for param in model_encoder.parameters():
                param.requires_grad = False
            for param in model_decoder.parameters():
                param.requires_grad = False
            for param in model_classifier.parameters():
                param.requires_grad = False
            pair_exist = False
            logging.info("Evaluation data")
            while True:
                with torch.no_grad():
                    start = time.time()
                    batch_feat_data, batch_feat_trg_data, batch_sc_data, batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
                        f_bs, f_ss, flens, n_batch_utt, del_index_utt, max_flen, spk_cv, src_trg_flag, \
                            spcidx_src, spcidx_src_trg, flens_spc_src, flens_spc_src_trg, \
                                batch_feat_data_full, batch_sc_data_full, batch_sc_cv_data_full = next(generator_eval)
                    if c_idx < 0:
                        break

                    f_es = f_ss+f_bs
                    logging.info(f'{f_ss} {f_bs} {f_es} {max_flen}')
                    # handle first pad for input
                    flag_cv = True
                    i_cv = 0
                    flag_cv_enc = False
                    i_cv_enc = 0
                    f_ss_first_pad_left = f_ss-first_pad_left_eval
                    f_es_first_pad_right = f_es+first_pad_right_eval
                    i_end = n_half_cyc_eval*2
                    if n_half_cyc_eval == 1:
                        i_end += 1
                    for i in range(i_end):
                        #logging.info(f'{f_ss_first_pad_left} {f_es_first_pad_right}')
                        if i % 2 == 0: #enc
                            if not flag_cv_enc:
                                if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                                    batch_feat_in[i] = batch_feat_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                                elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                                    batch_feat_in[i] = F.pad(batch_feat_data[:,:f_es_first_pad_right].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
                                elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                                    batch_feat_in[i] = F.pad(batch_feat_data[:,f_ss_first_pad_left:max_flen].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                                else: # pad left and right need additional replicate
                                    batch_feat_in[i] = F.pad(batch_feat_data[:,:max_flen].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                                flag_cv_enc = True
                            else:
                                if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                                    batch_feat_cv_in[i_cv_enc] = batch_feat_cv_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                                elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                                    batch_feat_cv_in[i_cv_enc] = F.pad(batch_feat_cv_data[:,:f_es_first_pad_right].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
                                elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                                    batch_feat_cv_in[i_cv_enc] = F.pad(batch_feat_cv_data[:,f_ss_first_pad_left:max_flen].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                                else: # pad left and right need additional replicate
                                    batch_feat_cv_in[i_cv_enc] = F.pad(batch_feat_cv_data[:,:max_flen].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                                i_cv_enc += 1
                                flag_cv_enc = False
                            f_ss_first_pad_left += enc_pad_left
                            f_es_first_pad_right -= enc_pad_right
                        else: #dec
                            if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                                batch_sc_in[i] = batch_sc_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                                if flag_cv:
                                    batch_sc_cv_in[i_cv] = batch_sc_cv_data[:,f_ss_first_pad_left:f_es_first_pad_right]
                                    i_cv += 1
                                    flag_cv = False
                                else:
                                    flag_cv = True
                            elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                                batch_sc_in[i] = F.pad(batch_sc_data[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv] = F.pad(batch_sc_cv_data[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                                    i_cv += 1
                                    flag_cv = False
                                else:
                                    flag_cv = True
                            elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                                diff_pad = f_es_first_pad_right - max_flen
                                batch_sc_in[i] = F.pad(batch_sc_data[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv] = F.pad(batch_sc_cv_data[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                                    i_cv += 1
                                    flag_cv = False
                                else:
                                    flag_cv = True
                            else: # pad left and right need additional replicate
                                diff_pad = f_es_first_pad_right - max_flen
                                batch_sc_in[i] = F.pad(batch_sc_data[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                                if flag_cv:
                                    batch_sc_cv_in[i_cv] = F.pad(batch_sc_cv_data[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                                    i_cv += 1
                                    flag_cv = False
                                else:
                                    flag_cv = True
                            f_ss_first_pad_left += dec_pad_left
                            f_es_first_pad_right -= dec_pad_right
                    batch_feat = batch_feat_data[:,f_ss:f_es]
                    batch_sc = batch_sc_data[:,f_ss:f_es]
                    batch_sc_cv[0] = batch_sc_cv_data[:,f_ss:f_es]
        
                    if f_ss > 0:
                        idx_in = 0
                        for i in range(0,n_half_cyc_eval,2):
                            i_cv = i//2
                            j = i+1
                            if len(del_index_utt) > 0:
                                if i == 0:
                                    h_mcep_in_sc = torch.FloatTensor(np.delete(h_mcep_in_sc.cpu().data.numpy(), \
                                                                    del_index_utt, axis=1)).to(device)
                                h_z[i] = torch.FloatTensor(np.delete(h_z[i].cpu().data.numpy(), \
                                                                del_index_utt, axis=1)).to(device)
                                h_z_sc[i] = torch.FloatTensor(np.delete(h_z_sc[i].cpu().data.numpy(), \
                                                                del_index_utt, axis=1)).to(device)
                                h_mcep[i] = torch.FloatTensor(np.delete(h_mcep[i].cpu().data.numpy(), \
                                                                del_index_utt, axis=1)).to(device)
                                h_mcep_sc[i] = torch.FloatTensor(np.delete(h_mcep_sc[i].cpu().data.numpy(), \
                                                                del_index_utt, axis=1)).to(device)
                                h_mcep_cv[i_cv] = torch.FloatTensor(np.delete(h_mcep_cv[i_cv].cpu().data.numpy(), \
                                                                del_index_utt, axis=1)).to(device)
                                h_mcep_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_mcep_cv_sc[i_cv].cpu().data.numpy(), \
                                                                del_index_utt, axis=1)).to(device)
                                h_z[j] = torch.FloatTensor(np.delete(h_z[j].cpu().data.numpy(), \
                                                                del_index_utt, axis=1)).to(device)
                                h_z_sc[j] = torch.FloatTensor(np.delete(h_z_sc[j].cpu().data.numpy(), \
                                                                del_index_utt, axis=1)).to(device)
                                if n_half_cyc_eval > 1:
                                    h_mcep[j] = torch.FloatTensor(np.delete(h_mcep[j].cpu().data.numpy(), \
                                                                    del_index_utt, axis=1)).to(device)
                                    h_mcep_sc[j] = torch.FloatTensor(np.delete(h_mcep_sc[j].cpu().data.numpy(), \
                                                                    del_index_utt, axis=1)).to(device)
                            if i > 0:
                                idx_in += 1
                                qy_logits[i], qz_alpha[i], z[i], h_z[i] \
                                    = model_encoder(torch.cat((batch_feat_in[idx_in][:,:,:args.excit_dim], batch_mcep_rec[i-1]), 2), h=h_z[i], outpad_right=outpad_rights_eval[idx_in], sampling=False)
                                i_1 = i-1
                                idx_in_1 = idx_in-1
                                feat_len = batch_mcep_rec[i_1].shape[1]
                                batch_mcep_rec[i_1] = batch_mcep_rec[i_1][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                                batch_mcep_rec_sc[i_1], h_mcep_sc[i_1] = model_classifier(feat=batch_mcep_rec[i_1], h=h_mcep_sc[i_1])
                            else:
                                qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder(batch_feat_in[idx_in], h=h_z[i], outpad_right=outpad_rights_eval[idx_in], sampling=False)
                                batch_mcep_in_sc, h_mcep_in_sc = model_classifier(feat=batch_feat[:,:,args.excit_dim:], h=h_mcep_in_sc)
                            idx_in += 1
                            batch_mcep_rec[i], h_mcep[i] = model_decoder(batch_sc_in[idx_in], z[i], h=h_mcep[i], outpad_right=outpad_rights_eval[idx_in])
                            batch_mcep_cv[i_cv], h_mcep_cv[i_cv] = model_decoder(batch_sc_cv_in[i_cv], z[i], h=h_mcep_cv[i_cv], outpad_right=outpad_rights_eval[idx_in])
                            feat_len = qy_logits[i].shape[1]
                            idx_in_1 = idx_in-1
                            z[i] = z[i][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            batch_z_sc[i], h_z_sc[i] = model_classifier(lat=z[i], h=h_z_sc[i])
                            qy_logits[i] = qy_logits[i][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            qz_alpha[i] = qz_alpha[i][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            idx_in += 1
                            qy_logits[j], qz_alpha[j], z[j], h_z[j] \
                                = model_encoder(torch.cat((batch_feat_cv_in[i_cv], batch_mcep_cv[i_cv]), 2), h=h_z[j], outpad_right=outpad_rights_eval[idx_in], sampling=False)
                            feat_len = batch_mcep_rec[i].shape[1]
                            idx_in_1 = idx_in-1
                            batch_mcep_rec[i] = batch_mcep_rec[i][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            batch_mcep_rec_sc[i], h_mcep_sc[i] = model_classifier(feat=batch_mcep_rec[i], h=h_mcep_sc[i])
                            batch_mcep_cv[i_cv] = batch_mcep_cv[i_cv][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            batch_mcep_cv_sc[i_cv], h_mcep_cv_sc[i_cv] = model_classifier(feat=batch_mcep_cv[i_cv], h=h_mcep_cv_sc[i_cv])
                            if n_half_cyc_eval > 1:
                                idx_in += 1
                                batch_mcep_rec[j], h_mcep[j] = model_decoder(batch_sc_in[idx_in], z[j], h=h_mcep[j], outpad_right=outpad_rights_eval[idx_in])
                                feat_len = qy_logits[j].shape[1]
                                idx_in_1 = idx_in-1
                                z[j] = z[j][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                                batch_z_sc[j], h_z_sc[j] = model_classifier(lat=z[j], h=h_z_sc[j])
                                qy_logits[j] = qy_logits[j][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                                qz_alpha[j] = qz_alpha[j][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                                if j+1 == n_half_cyc_eval:
                                    batch_mcep_rec[j] = batch_mcep_rec[j][:,outpad_lefts_eval[idx_in]:batch_mcep_rec[j].shape[1]-outpad_rights_eval[idx_in]]
                                    batch_mcep_rec_sc[j], h_mcep_sc[j] = model_classifier(feat=batch_mcep_rec[j], h=h_mcep_sc[j])
                            else:
                                qy_logits[j] = qy_logits[j][:,outpad_lefts_eval[idx_in]:feat_len-outpad_rights_eval[idx_in]]
                                qz_alpha[j] = qz_alpha[j][:,outpad_lefts_eval[idx_in]:feat_len-outpad_rights_eval[idx_in]]
                    else:
                        pair_flag = False
                        for k in range(n_batch_utt):
                            if src_trg_flag[k]:
                                pair_flag = True
                                pair_exist = True
                                break
                        batch_feat_data_full = F.pad(batch_feat_data_full.transpose(1,2), (first_pad_left_eval_utt,first_pad_right_eval_utt), "replicate").transpose(1,2)
                        _, _, trj_lat_src, _ = model_encoder(batch_feat_data_full, sampling=False)
                        batch_sc_data_full = F.pad(batch_sc_data_full.unsqueeze(1).float(), (dec_pad_left,dec_pad_right), "replicate").squeeze(1).long()
                        batch_sc_cv_data_full = F.pad(batch_sc_cv_data_full.unsqueeze(1).float(), (dec_pad_left,dec_pad_right), "replicate").squeeze(1).long()
                        trj_src_src, _ = model_decoder(batch_sc_data_full, trj_lat_src)
                        trj_src_trg, _ = model_decoder(batch_sc_cv_data_full, trj_lat_src)
                        if dec_pad_right > 0:
                            trj_lat_src = trj_lat_src[:,dec_pad_left:-dec_pad_right]
                        else:
                            trj_lat_src = trj_lat_src[:,dec_pad_left:]
                        for k in range(n_batch_utt):
                            spk_src = os.path.basename(os.path.dirname(featfile[k]))
                            #GV stat of reconstructed
                            gv_src_src[spk_list.index(spk_src)].append(torch.var(\
                                trj_src_src[k,:flens[k],1:], 0).cpu().data.numpy())
                            spk_src_trg = spk_cv[k] # find target pair
                            #GV stat of converted
                            gv_src_trg[spk_list.index(spk_src_trg)].append(torch.var(\
                                trj_src_trg[k,:flens[k],1:], 0).cpu().data.numpy())
                        if pair_flag:
                            batch_feat_trg_data_in = F.pad(batch_feat_trg_data.transpose(1,2), (enc_pad_left,enc_pad_right), "replicate").transpose(1,2)
                            _, _, trj_lat_trg, _ = model_encoder(batch_feat_trg_data_in, sampling=False)
                            for k in range(n_batch_utt):
                                if src_trg_flag[k]:
                                    trj_lat_src_ = np.array(torch.index_select(trj_lat_src[k],0,spcidx_src[k,:flens_spc_src[k]]).cpu().data.numpy(), dtype=np.float64)
                                    trj_lat_trg_ = np.array(torch.index_select(trj_lat_trg[k],0,spcidx_src_trg[k,:flens_spc_src_trg[k]]).cpu().data.numpy(), dtype=np.float64)
                                    trj_src_trg_ = torch.index_select(trj_src_trg[k],0,spcidx_src[k,:flens_spc_src[k]])
                                    trj_trg_ = torch.index_select(batch_feat_trg_data[k],0,spcidx_src_trg[k,:flens_spc_src_trg[k]])
                                    trj_trg_ = trj_trg_[:,args.excit_dim:]
                                    # MCD of spectral with 0th power
                                    _, _, batch_mcdpow_src_trg, _ = dtw.dtw_org_to_trg(\
                                        np.array(trj_src_trg_.cpu().data.numpy(), dtype=np.float64), \
                                        np.array(trj_trg_.cpu().data.numpy(), dtype=np.float64))
                                    # MCD of spectral w/o 0th power, i.e., [:,1:]
                                    _, twf_mcep, batch_mcd_src_trg, _ = dtw.dtw_org_to_trg(\
                                        np.array(trj_src_trg_[:,1:].cpu().data.numpy(), dtype=np.float64), \
                                        np.array(trj_trg_[:,1:].cpu().data.numpy(), dtype=np.float64))
                                    twf_mcep = torch.LongTensor(twf_mcep[:,0]).cuda()
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
                                    loss_mcdpow_src_trg.append(batch_mcdpow_src_trg)
                                    loss_mcd_src_trg.append(batch_mcd_src_trg)
                                    loss_lat_dist_rmse.append(batch_lat_dist_rmse)
                                    loss_lat_dist_cossim.append(batch_lat_dist_cossim)
                                    total_eval_loss["eval/loss_mcdpow_src_trg"].append(batch_mcdpow_src_trg)
                                    total_eval_loss["eval/loss_mcd_src_trg"].append(batch_mcd_src_trg)
                                    total_eval_loss["eval/loss_lat_dist_rmse"].append(batch_lat_dist_rmse)
                                    total_eval_loss["eval/loss_lat_dist_cossim"].append(batch_lat_dist_cossim)
                                    logging.info('acc cv %s %s %.3f dB %.3f dB %.3f %.3f' % (featfile[k], \
                                        spk_cv[k], batch_mcdpow_src_trg, batch_mcd_src_trg, \
                                            batch_lat_dist_rmse, batch_lat_dist_cossim))
                        idx_in = 0
                        for i in range(0,n_half_cyc_eval,2):
                            i_cv = i//2
                            j = i+1
                            if i > 0:
                                idx_in += 1
                                qy_logits[i], qz_alpha[i], z[i], h_z[i] \
                                    = model_encoder(torch.cat((batch_feat_in[idx_in][:,:,:args.excit_dim], batch_mcep_rec[i-1]), 2), outpad_right=outpad_rights_eval[idx_in], sampling=False)
                                i_1 = i-1
                                idx_in_1 = idx_in-1
                                feat_len = batch_mcep_rec[i_1].shape[1]
                                batch_mcep_rec[i_1] = batch_mcep_rec[i_1][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                                batch_mcep_rec_sc[i_1], h_mcep_sc[i_1] = model_classifier(feat=batch_mcep_rec[i_1])
                            else:
                                qy_logits[i], qz_alpha[i], z[i], h_z[i] = model_encoder(batch_feat_in[idx_in], outpad_right=outpad_rights_eval[idx_in], sampling=False)
                                batch_mcep_in_sc, h_mcep_in_sc = model_classifier(feat=batch_feat[:,:,args.excit_dim:])
                            idx_in += 1
                            batch_mcep_rec[i], h_mcep[i] = model_decoder(batch_sc_in[idx_in], z[i], outpad_right=outpad_rights_eval[idx_in])
                            batch_mcep_cv[i_cv], h_mcep_cv[i_cv] = model_decoder(batch_sc_cv_in[i_cv], z[i], outpad_right=outpad_rights_eval[idx_in])
                            feat_len = qy_logits[i].shape[1]
                            idx_in_1 = idx_in-1
                            z[i] = z[i][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            batch_z_sc[i], h_z_sc[i] = model_classifier(lat=z[i])
                            qy_logits[i] = qy_logits[i][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            qz_alpha[i] = qz_alpha[i][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            idx_in += 1
                            qy_logits[j], qz_alpha[j], z[j], h_z[j] \
                                = model_encoder(torch.cat((batch_feat_cv_in[i_cv], batch_mcep_cv[i_cv]), 2), outpad_right=outpad_rights_eval[idx_in], sampling=False)
                            feat_len = batch_mcep_rec[i].shape[1]
                            idx_in_1 = idx_in-1
                            batch_mcep_rec[i] = batch_mcep_rec[i][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            batch_mcep_rec_sc[i], h_mcep_sc[i] = model_classifier(feat=batch_mcep_rec[i])
                            batch_mcep_cv[i_cv] = batch_mcep_cv[i_cv][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                            batch_mcep_cv_sc[i_cv], h_mcep_cv_sc[i_cv] = model_classifier(feat=batch_mcep_cv[i_cv])
                            if n_half_cyc_eval > 1:
                                idx_in += 1
                                batch_mcep_rec[j], h_mcep[j] = model_decoder(batch_sc_in[idx_in], z[j], outpad_right=outpad_rights_eval[idx_in])
                                feat_len = qy_logits[j].shape[1]
                                idx_in_1 = idx_in-1
                                z[j] = z[j][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                                batch_z_sc[j], h_z_sc[j] = model_classifier(lat=z[j])
                                qy_logits[j] = qy_logits[j][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                                qz_alpha[j] = qz_alpha[j][:,outpad_lefts_eval[idx_in_1]:feat_len-outpad_rights_eval[idx_in_1]]
                                if j+1 == n_half_cyc_eval:
                                    batch_mcep_rec[j] = batch_mcep_rec[j][:,outpad_lefts_eval[idx_in]:batch_mcep_rec[j].shape[1]-outpad_rights_eval[idx_in]]
                                    batch_mcep_rec_sc[j], h_mcep_sc[j] = model_classifier(feat=batch_mcep_rec[j])
                            else:
                                qy_logits[j] = qy_logits[j][:,outpad_lefts_eval[idx_in]:feat_len-outpad_rights_eval[idx_in]]
                                qz_alpha[j] = qz_alpha[j][:,outpad_lefts_eval[idx_in]:feat_len-outpad_rights_eval[idx_in]]
 
                    # samples check
                    i = np.random.randint(0, batch_mcep_rec[0].shape[0])
                    logging.info("%d %s %d %d %d %d %s" % (i, \
                        os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                            f_ss, f_es, flens[i], max_flen, spk_cv[i]))
                    logging.info(batch_mcep_rec[0][i,:2,:4])
                    if n_half_cyc_eval > 1:
                        logging.info(batch_mcep_rec[1][i,:2,:4])
                    logging.info(batch_feat[i,:2,args.excit_dim:args.excit_dim+4])
                    logging.info(batch_mcep_cv[0][i,:2,:4])
                    #logging.info(qy_logits[0][i,:2])
                    #logging.info(batch_sc[i,0])
                    #logging.info(qy_logits[1][i,:2])
                    #logging.info(batch_sc_cv[0][i,0])

                    # loss_compute
                    powmcep = batch_feat[:,:,args.excit_dim:]
                    mcep = batch_feat[:,:,args.excit_dim+1:]
                    sc_onehot = F.one_hot(batch_sc, num_classes=n_spk).float()
                    batch_loss_sc_mcep_in_ = torch.mean(criterion_ce(batch_mcep_in_sc.reshape(-1, n_spk), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                    batch_loss_sc_mcep_in = batch_loss_sc_mcep_in_.mean()
                    for i in range(n_half_cyc_eval):
                        mcep_est = batch_mcep_rec[i]
                        if i % 2 == 0:
                            mcep_cv = batch_mcep_cv[i//2]
                            sc_cv_onehot = F.one_hot(batch_sc_cv[i//2], num_classes=n_spk).float()

                        ## mcep acc.
                        batch_loss_powmcep_ = torch.mean(mcd_constant*torch.sqrt(\
                                                            torch.sum(criterion_l2(mcep_est, powmcep), -1)), -1)
                        batch_loss_powmcep[i] = batch_loss_powmcep_.mean()
                        batch_loss_mcep[i] = torch.mean(torch.mean(mcd_constant*torch.sqrt(\
                                                            torch.sum(criterion_l2(mcep_est[:,:,1:], mcep), -1)), -1))
                        batch_loss_px_mcep_ = torch.mean(mcd_constant*torch.sum(criterion_l1(mcep_est, powmcep), -1), -1)
                        batch_loss_mcep_rec[i] = batch_loss_px_mcep_.mean()
                        if i % 2 == 0:
                            batch_loss_mcep_cv[i//2] = torch.mean(torch.mean(mcd_constant*torch.sum(criterion_l1(mcep_cv, powmcep),-1), -1))
                        batch_loss_px_sum = batch_loss_px_mcep_.sum()

                        batch_loss_px_ms_norm_, batch_loss_px_ms_err_ = criterion_ms(mcep_est, powmcep)
                        batch_loss_ms_norm[i] = batch_loss_px_ms_norm_.mean()
                        if not torch.isinf(batch_loss_ms_norm[i]) and not torch.isnan(batch_loss_ms_norm[i]):
                            batch_loss_px_sum += batch_loss_px_ms_norm_.sum()
                        batch_loss_ms_err[i] = batch_loss_px_ms_err_.mean()
                        if not torch.isinf(batch_loss_ms_err[i]) and not torch.isnan(batch_loss_ms_err[i]):
                            batch_loss_px_sum += batch_loss_px_ms_err_.sum()

                        batch_loss_px[i] = batch_loss_mcep_rec[i] + batch_loss_ms_norm[i] + batch_loss_ms_err[i]

                        # KL div
                        batch_loss_sc_mcep_ = torch.mean(criterion_ce(batch_mcep_rec_sc[i].reshape(-1, n_spk), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                        batch_loss_sc_mcep[i] = batch_loss_sc_mcep_.mean()
                        batch_loss_sc_mcep_rev_ = torch.mean(criterion_ce(revgrad(batch_mcep_rec_sc[i].reshape(-1, n_spk)), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1)
                        batch_loss_sc_mcep_rev[i] = batch_loss_sc_mcep_rev_.mean()
                        batch_loss_sc_z_ = torch.mean(kl_categorical_categorical_logits(p_spk, logits_p_spk, batch_z_sc[i]), -1)
                        batch_loss_sc_z[i] = batch_loss_sc_z_.mean()
                        batch_loss_sc_z_rev_ = torch.mean(criterion_ce(revgrad(batch_z_sc[i].reshape(-1, n_spk)), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                        batch_loss_sc_z_rev[i] = batch_loss_sc_z_rev_.mean()
                        if i % 2 == 0:
                            batch_loss_qy_py_ = torch.mean(criterion_ce(qy_logits[i].reshape(-1, n_spk), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                            batch_loss_qy_py[i] = batch_loss_qy_py_.mean()
                            batch_loss_qy_py_err_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i], dim=-1), sc_onehot), -1), -1)
                            batch_loss_qy_py_err[i] = batch_loss_qy_py_err_.mean()
                            batch_loss_sc_mcep_cv_ = torch.mean(criterion_ce(batch_mcep_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1)
                            batch_loss_sc_mcep_cv[i//2] = batch_loss_sc_mcep_cv_.mean()
                            batch_loss_sc_mcep_cv_rev_ = torch.mean(criterion_ce(revgrad(batch_mcep_cv_sc[i//2].reshape(-1, n_spk)), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                            batch_loss_sc_mcep_cv_rev[i//2] = batch_loss_sc_mcep_cv_rev_.mean()
                            if n_half_cyc_eval == 1:
                                batch_loss_qy_py[i+1] = torch.mean(criterion_ce(qy_logits[i+1].reshape(-1, n_spk), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1).mean()
                                batch_loss_qy_py_err[i+1] = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i+1], dim=-1), F.one_hot(batch_sc_cv[i//2], num_classes=n_spk).float()), -1), -1).mean()
                                batch_loss_qz_pz[i+1] = torch.mean(torch.sum(kl_laplace(qz_alpha[i+1]), -1), -1).mean()
                            batch_loss_sc_mcep_kl = batch_loss_sc_mcep_.sum() + batch_loss_sc_mcep_cv_.sum() + batch_loss_sc_mcep_cv_rev_.sum()
                            batch_loss_sc_z_kl = batch_loss_sc_z_.sum() + batch_loss_sc_z_rev_.sum()
                            batch_loss_qy_py_ce = batch_loss_qy_py_.sum()
                        else:
                            batch_loss_qy_py_ = torch.mean(criterion_ce(qy_logits[i].reshape(-1, n_spk), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1)
                            batch_loss_qy_py[i] = batch_loss_qy_py_.mean()
                            batch_loss_qy_py_rev_ = torch.mean(criterion_ce(revgrad(qy_logits[i].reshape(-1, n_spk)), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                            batch_loss_qy_py_rev[i] = batch_loss_qy_py_rev_.mean()
                            batch_loss_qy_py_err_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i], dim=-1), F.one_hot(batch_sc_cv[i//2], num_classes=n_spk).float()), -1), -1)
                            batch_loss_qy_py_err[i] = batch_loss_qy_py_err_.mean()
                            batch_loss_sc_mcep_kl = batch_loss_sc_mcep_.sum()
                            batch_loss_sc_z_cv_rev_ = torch.mean(criterion_ce(revgrad(batch_z_sc[i].reshape(-1, n_spk)), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1)
                            batch_loss_sc_z_cv_rev[i//2] = batch_loss_sc_z_cv_rev_.mean()
                            #batch_loss_sc_z_kl = batch_loss_sc_z_.sum() + batch_loss_sc_z_cv_rev_.sum()
                            batch_loss_sc_z_kl = batch_loss_sc_z_.sum() + batch_loss_sc_z_rev_.sum() + batch_loss_sc_z_cv_rev_.sum()
                            batch_loss_qy_py_ce = batch_loss_qy_py_.sum() + batch_loss_qy_py_rev_.sum()
                        batch_loss_qz_pz_ = torch.mean(torch.sum(kl_laplace(qz_alpha[i]), -1), -1)
                        batch_loss_qz_pz[i] = batch_loss_qz_pz_.mean()
                        batch_loss_qz_pz_kl = batch_loss_qz_pz_.sum()

                        # elbo
                        batch_loss_elbo[i] = batch_loss_px_sum + batch_loss_qy_py_ce + batch_loss_qz_pz_kl \
                                                    + batch_loss_sc_mcep_kl + batch_loss_sc_z_kl

                        total_eval_loss["eval/loss_elbo-%d"%(i+1)].append(batch_loss_elbo[i].item())
                        total_eval_loss["eval/loss_px-%d"%(i+1)].append(batch_loss_px[i].item())
                        total_eval_loss["eval/loss_qy_py-%d"%(i+1)].append(batch_loss_qy_py[i].item())
                        if i % 2 != 0:
                            total_eval_loss["eval/loss_qy_py_rev-%d"%(i+1)].append(batch_loss_qy_py_rev[i].item())
                        total_eval_loss["eval/loss_qy_py_err-%d"%(i+1)].append(batch_loss_qy_py_err[i].item())
                        total_eval_loss["eval/loss_qz_pz-%d"%(i+1)].append(batch_loss_qz_pz[i].item())
                        total_eval_loss["eval/loss_sc_z-%d"%(i+1)].append(batch_loss_sc_z[i].item())
                        total_eval_loss["eval/loss_sc_z_rev-%d"%(i+1)].append(batch_loss_sc_z_rev[i].item())
                        total_eval_loss["eval/loss_sc_mcep-%d"%(i+1)].append(batch_loss_sc_mcep[i].item())
                        total_eval_loss["eval/loss_sc_mcep_rev-%d"%(i+1)].append(batch_loss_sc_mcep_rev[i].item())
                        if i == 0:
                            total_eval_loss["eval/loss_sc_mcep_in-%d"%(i+1)].append(batch_loss_sc_mcep_in.item())
                        total_eval_loss["eval/loss_ms_norm-%d"%(i+1)].append(batch_loss_ms_norm[i].item())
                        total_eval_loss["eval/loss_ms_err-%d"%(i+1)].append(batch_loss_ms_err[i].item())
                        total_eval_loss["eval/loss_powmcep-%d"%(i+1)].append(batch_loss_powmcep[i].item())
                        total_eval_loss["eval/loss_mcep-%d"%(i+1)].append(batch_loss_mcep[i].item())
                        total_eval_loss["eval/loss_mcep_rec-%d"%(i+1)].append(batch_loss_mcep_rec[i].item())
                        if i % 2 == 0:
                            total_eval_loss["eval/loss_sc_mcep_cv-%d"%(i+1)].append(batch_loss_sc_mcep_cv[i//2].item())
                            total_eval_loss["eval/loss_sc_mcep_cv_rev-%d"%(i+1)].append(batch_loss_sc_mcep_cv_rev[i//2].item())
                            total_eval_loss["eval/loss_mcep_cv-%d"%(i+1)].append(batch_loss_mcep_cv[i//2].item())
                            loss_mcep_cv[i//2].append(batch_loss_mcep_cv[i//2].item())
                        else:
                            total_eval_loss["eval/loss_sc_z_cv_rev-%d"%(i+1)].append(batch_loss_sc_z_cv_rev[i//2].item())
                        loss_elbo[i].append(batch_loss_elbo[i].item())
                        loss_px[i].append(batch_loss_px[i].item())
                        loss_qy_py[i].append(batch_loss_qy_py[i].item())
                        loss_qy_py_err[i].append(batch_loss_qy_py_err[i].item())
                        loss_qz_pz[i].append(batch_loss_qz_pz[i].item())
                        loss_powmcep[i].append(batch_loss_powmcep[i].item())
                        loss_mcep[i].append(batch_loss_mcep[i].item())
                        loss_mcep_rec[i].append(batch_loss_mcep_rec[i].item())
                        if n_half_cyc_eval == 1:
                            total_eval_loss["eval/loss_qy_py-%d"%(i+2)].append(batch_loss_qy_py[i+1].item())
                            total_eval_loss["eval/loss_qy_py_err-%d"%(i+2)].append(batch_loss_qy_py_err[i+1].item())
                            total_eval_loss["eval/loss_qz_pz-%d"%(i+2)].append(batch_loss_qz_pz[i+1].item())
                            loss_qy_py[i+1].append(batch_loss_qy_py[i+1].item())
                            loss_qy_py_err[i+1].append(batch_loss_qy_py_err[i+1].item())
                            loss_qz_pz[i+1].append(batch_loss_qz_pz[i+1].item())

                    text_log = "batch eval loss [%d] %d %d %.3f " % (c_idx+1, f_ss, f_bs, batch_loss_sc_mcep_in.item())
                    for i in range(n_half_cyc_eval):
                        if i % 2 == 0:
                            if n_half_cyc_eval > 1:
                                text_log += "[%ld] %.3f ; %.3f %.3f %.3f %% %.3f ; %.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; %.3f %.3f , %.3f dB %.3f dB ;; " % (
                                    i+1, batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(),
                                        batch_loss_qz_pz[i].item(), batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                                            batch_loss_sc_z[i].item(), batch_loss_sc_z_rev[i].item(),
                                            batch_loss_sc_mcep[i].item(), batch_loss_sc_mcep_rev[i].item(),
                                            batch_loss_sc_mcep_cv[i//2].item(), batch_loss_sc_mcep_cv_rev[i//2].item(),
                                            batch_loss_mcep_rec[i].item(), batch_loss_mcep_cv[i//2].item(),
                                            batch_loss_powmcep[i].item(), batch_loss_mcep[i].item())
                            else:
                                text_log += "[%ld] %.3f ; %.3f %.3f %.3f %% %.3f , %.3f %.3f %% %.3f ; %.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; %.3f %.3f , %.3f dB %.3f dB ;; " % (
                                    i+1, batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(), \
                                        batch_loss_qz_pz[i].item(), batch_loss_qy_py[i+1].item(), batch_loss_qy_py_err[i+1].item(),
                                        batch_loss_qz_pz[i+1].item(), batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                                            batch_loss_sc_z[i].item(), batch_loss_sc_z_rev[i].item(),
                                            batch_loss_sc_mcep[i].item(), batch_loss_sc_mcep_rev[i].item(),
                                            batch_loss_sc_mcep_cv[i//2].item(), batch_loss_sc_mcep_cv_rev[i//2].item(),
                                            batch_loss_mcep_rec[i].item(), batch_loss_mcep_cv[i//2].item(),
                                            batch_loss_powmcep[i].item(), batch_loss_mcep[i].item())
                        else:
                            text_log += "[%ld] %.3f ; %.3f %.3f %.3f %.3f %% %.3f ; %.3f %.3f , %.3f %.3f %.3f , %.3f %.3f ; %.3f , %.3f dB %.3f dB ;; " % (
                                i+1, batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_qy_py[i].item(), batch_loss_qy_py_rev[i].item(), batch_loss_qy_py_err[i].item(),
                                    batch_loss_qz_pz[i].item(), batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                                    batch_loss_sc_z[i].item(), batch_loss_sc_z_rev[i].item(), batch_loss_sc_z_cv_rev[i//2].item(),
                                    batch_loss_sc_mcep[i].item(), batch_loss_sc_mcep_rev[i].item(),
                                    batch_loss_mcep_rec[i].item(), batch_loss_powmcep[i].item(), batch_loss_mcep[i].item())
                    logging.info("%s (%.3f sec)" % (text_log, time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            tmp_gv_1 = []
            tmp_gv_2 = []
            for j in range(n_spk):
                if len(gv_src_src[j]) > 0:
                    tmp_gv_1.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_src[j], \
                                        axis=0))-np.log(gv_mean[j])))))
                if len(gv_src_trg[j]) > 0:
                    tmp_gv_2.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg[j], \
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
            if pair_exist:
                eval_loss_mcdpow_src_trg = np.mean(loss_mcdpow_src_trg)
                eval_loss_mcdpow_src_trg_std = np.std(loss_mcdpow_src_trg)
                eval_loss_mcd_src_trg = np.mean(loss_mcd_src_trg)
                eval_loss_mcd_src_trg_std = np.std(loss_mcd_src_trg)
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
                if n_half_cyc_eval == 1:
                    eval_loss_qy_py[i+1] = np.mean(loss_qy_py[i+1])
                    eval_loss_qy_py_std[i+1] = np.std(loss_qy_py[i+1])
                    eval_loss_qy_py_err[i+1] = np.mean(loss_qy_py_err[i+1])
                    eval_loss_qy_py_err_std[i+1] = np.std(loss_qy_py_err[i+1])
                    eval_loss_qz_pz[i+1] = np.mean(loss_qz_pz[i+1])
                    eval_loss_qz_pz_std[i+1] = np.std(loss_qz_pz[i+1])
                eval_loss_powmcep[i] = np.mean(loss_powmcep[i])
                eval_loss_powmcep_std[i] = np.std(loss_powmcep[i])
                eval_loss_mcep[i] = np.mean(loss_mcep[i])
                eval_loss_mcep_std[i] = np.std(loss_mcep[i])
                eval_loss_mcep_rec[i] = np.mean(loss_mcep_rec[i])
                eval_loss_mcep_rec_std[i] = np.std(loss_mcep_rec[i])
                if i % 2 == 0:
                    eval_loss_mcep_cv[i//2] = np.mean(loss_mcep_cv[i//2])
                    eval_loss_mcep_cv_std[i//2] = np.std(loss_mcep_cv[i//2])
            text_log = "(EPOCH:%d) average evaluation loss = " % (epoch_idx + 1)
            for i in range(n_half_cyc_eval):
                if i % 2 == 0:
                    if n_half_cyc_eval > 1:
                        if pair_exist:
                            text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ; " \
                                "%.6f %.6f %.6f (+- %.6f) dB %.6f (+- %.6f) dB %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (i+1, \
                                eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_px[i], eval_loss_px_std[i], \
                                eval_loss_qy_py[i], eval_loss_qy_py_std[i], eval_loss_qy_py_err[i], eval_loss_qy_py_err_std[i], \
                                eval_loss_qz_pz[i], eval_loss_qz_pz_std[i], \
                                eval_loss_mcep_rec[i], eval_loss_mcep_rec_std[i], eval_loss_mcep_cv[i//2], eval_loss_mcep_cv_std[i//2], \
                                eval_loss_powmcep[i], eval_loss_powmcep_std[i], eval_loss_mcep[i], eval_loss_mcep_std[i], \
                                eval_loss_gv_src_src, eval_loss_gv_src_trg, eval_loss_mcdpow_src_trg, eval_loss_mcdpow_src_trg_std, \
                                eval_loss_mcd_src_trg, eval_loss_mcd_src_trg_std, eval_loss_lat_dist_rmse, eval_loss_lat_dist_rmse_std, \
                                eval_loss_lat_dist_cossim, eval_loss_lat_dist_cossim_std)
                        else:
                            text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ; " \
                                "%.6f %.6f n/a (+- n/a) dB n/a (+- n/a) dB n/a (+- n/a) n/a (+- n/a) ;; " % (i+1, \
                                eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_px[i], eval_loss_px_std[i], \
                                eval_loss_qy_py[i], eval_loss_qy_py_std[i], eval_loss_qy_py_err[i], eval_loss_qy_py_err_std[i], \
                                eval_loss_qz_pz[i], eval_loss_qz_pz_std[i], \
                                eval_loss_mcep_rec[i], eval_loss_mcep_rec_std[i], eval_loss_mcep_cv[i//2], eval_loss_mcep_cv_std[i//2], \
                                eval_loss_powmcep[i], eval_loss_powmcep_std[i], eval_loss_mcep[i], eval_loss_mcep_std[i], \
                                eval_loss_gv_src_src, eval_loss_gv_src_trg)
                    else:
                        if pair_exist:
                            text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ; " \
                                "%.6f %.6f %.6f (+- %.6f) dB %.6f (+- %.6f) dB %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (i+1, \
                                eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_px[i], eval_loss_px_std[i], \
                                eval_loss_qy_py[i], eval_loss_qy_py_std[i], eval_loss_qy_py_err[i], eval_loss_qy_py_err_std[i], \
                                eval_loss_qz_pz[i], eval_loss_qz_pz_std[i], \
                                eval_loss_qy_py[i+1], eval_loss_qy_py_std[i+1], eval_loss_qy_py_err[i+1], eval_loss_qy_py_err_std[i+1], \
                                eval_loss_qz_pz[i+1], eval_loss_qz_pz_std[i+1], \
                                eval_loss_mcep_rec[i], eval_loss_mcep_rec_std[i], eval_loss_mcep_cv[i//2], eval_loss_mcep_cv_std[i//2], \
                                eval_loss_powmcep[i], eval_loss_powmcep_std[i], eval_loss_mcep[i], eval_loss_mcep_std[i], \
                                eval_loss_gv_src_src, eval_loss_gv_src_trg, eval_loss_mcdpow_src_trg, eval_loss_mcdpow_src_trg_std, \
                                eval_loss_mcd_src_trg, eval_loss_mcd_src_trg_std, eval_loss_lat_dist_rmse, eval_loss_lat_dist_rmse_std, \
                                eval_loss_lat_dist_cossim, eval_loss_lat_dist_cossim_std)
                        else:
                            text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ; " \
                                "%.6f %.6f n/a (+- n/a) dB n/a (+- n/a) dB n/a (+- n/a) n/a (+- n/a) ;; " % (i+1, \
                                eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_px[i], eval_loss_px_std[i], \
                                eval_loss_qy_py[i], eval_loss_qy_py_std[i], eval_loss_qy_py_err[i], eval_loss_qy_py_err_std[i], \
                                eval_loss_qz_pz[i], eval_loss_qz_pz_std[i], \
                                eval_loss_qy_py[i+1], eval_loss_qy_py_std[i+1], eval_loss_qy_py_err[i+1], eval_loss_qy_py_err_std[i+1], \
                                eval_loss_qz_pz[i+1], eval_loss_qz_pz_std[i+1], \
                                eval_loss_mcep_rec[i], eval_loss_mcep_rec_std[i], eval_loss_mcep_cv[i//2], eval_loss_mcep_cv_std[i//2], \
                                eval_loss_powmcep[i], eval_loss_powmcep_std[i], eval_loss_mcep[i], eval_loss_mcep_std[i], \
                                eval_loss_gv_src_src, eval_loss_gv_src_trg)
                else:
                    text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                        "%.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ;; " % (i+1, \
                        eval_loss_elbo[i], eval_loss_elbo_std[i], eval_loss_px[i], eval_loss_px_std[i], \
                        eval_loss_qy_py[i], eval_loss_qy_py_std[i], \
                        eval_loss_qy_py_err[i], eval_loss_qy_py_err_std[i], eval_loss_qz_pz[i], eval_loss_qz_pz_std[i], \
                        eval_loss_mcep_rec[i], eval_loss_mcep_rec_std[i], \
                        eval_loss_powmcep[i], eval_loss_powmcep_std[i], eval_loss_mcep[i], eval_loss_mcep_std[i])
            logging.info("%s (%.3f min., %.3f sec / batch)" % (text_log, total / 60.0, total / iter_count))
            if (pair_exist and (eval_loss_mcd_src_trg+eval_loss_mcd_src_trg_std) <= (min_eval_loss_mcd_src_trg+min_eval_loss_mcd_src_trg_std)) \
                or (pair_exist and eval_loss_mcd_src_trg <= min_eval_loss_mcd_src_trg) \
                    or (not pair_exist and (eval_loss_mcep_cv[0]-eval_loss_mcep_rec[0]) >= (min_eval_loss_mcep_cv[0]-min_eval_loss_mcep_rec[0])) \
                        or (not pair_exist and (eval_loss_mcep[0]+eval_loss_mcep_std[0]) <= (min_eval_loss_mcep[0]-min_eval_loss_mcep_std[0])) \
                            or (not pair_exist and eval_loss_mcep[0] <= min_eval_loss_mcep[0]):
                min_eval_loss_gv_src_src = eval_loss_gv_src_src
                min_eval_loss_gv_src_trg = eval_loss_gv_src_trg
                if pair_exist:
                    min_eval_loss_mcdpow_src_trg = eval_loss_mcdpow_src_trg
                    min_eval_loss_mcdpow_src_trg_std = eval_loss_mcdpow_src_trg_std
                    min_eval_loss_mcd_src_trg = eval_loss_mcd_src_trg
                    min_eval_loss_mcd_src_trg_std = eval_loss_mcd_src_trg_std
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
                    if n_half_cyc_eval == 1:
                        min_eval_loss_qy_py[i+1] = eval_loss_qy_py[i+1]
                        min_eval_loss_qy_py_std[i+1] = eval_loss_qy_py_std[i+1]
                        min_eval_loss_qy_py_err[i+1] = eval_loss_qy_py_err[i+1]
                        min_eval_loss_qy_py_err_std[i+1] = eval_loss_qy_py_err_std[i+1]
                        min_eval_loss_qz_pz[i+1] = eval_loss_qz_pz[i+1]
                        min_eval_loss_qz_pz_std[i+1] = eval_loss_qz_pz_std[i+1]
                    min_eval_loss_powmcep[i] = eval_loss_powmcep[i]
                    min_eval_loss_powmcep_std[i] = eval_loss_powmcep_std[i]
                    min_eval_loss_mcep[i] = eval_loss_mcep[i]
                    min_eval_loss_mcep_std[i] = eval_loss_mcep_std[i]
                    min_eval_loss_mcep_rec[i] = eval_loss_mcep_rec[i]
                    min_eval_loss_mcep_rec_std[i] = eval_loss_mcep_rec_std[i]
                    if i % 2 == 0:
                        min_eval_loss_mcep_cv[i//2] = eval_loss_mcep_cv[i//2]
                        min_eval_loss_mcep_cv_std[i//2] = eval_loss_mcep_cv_std[i//2]
                min_idx = epoch_idx
                epoch_min_flag = True
                change_min_flag = True
            if args.resume is None or (args.resume is not None and change_min_flag):
                text_log = "min_eval_loss = "
                for i in range(n_half_cyc_eval):
                    if i % 2 == 0:
                        if n_half_cyc_eval > 1:
                            if pair_exist:
                                text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                    "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ; " \
                                    "%.6f %.6f %.6f (+- %.6f) dB %.6f (+- %.6f) dB %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (i+1, \
                                    min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_px[i], min_eval_loss_px_std[i], \
                                    min_eval_loss_qy_py[i], min_eval_loss_qy_py_std[i], \
                                    min_eval_loss_qy_py_err[i], min_eval_loss_qy_py_err_std[i], min_eval_loss_qz_pz[i], min_eval_loss_qz_pz_std[i], \
                                    min_eval_loss_mcep_rec[i], min_eval_loss_mcep_rec_std[i], min_eval_loss_mcep_cv[i//2], min_eval_loss_mcep_cv_std[i//2], \
                                    min_eval_loss_powmcep[i], min_eval_loss_powmcep_std[i], min_eval_loss_mcep[i], min_eval_loss_mcep_std[i], \
                                    min_eval_loss_gv_src_src, min_eval_loss_gv_src_trg, min_eval_loss_mcdpow_src_trg, min_eval_loss_mcdpow_src_trg_std, \
                                    min_eval_loss_mcd_src_trg, min_eval_loss_mcd_src_trg_std, min_eval_loss_lat_dist_rmse, min_eval_loss_lat_dist_rmse_std, \
                                    min_eval_loss_lat_dist_cossim, min_eval_loss_lat_dist_cossim_std)
                            else:
                                text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                    "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ; " \
                                    "%.6f %.6f n/a (+- n/a) dB n/a (+- n/a) dB n/a n/a (+- n/a) n/a (+- n/a) ;; " % (i+1, \
                                    min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_px[i], min_eval_loss_px_std[i], \
                                    min_eval_loss_qy_py[i], min_eval_loss_qy_py_std[i], \
                                    min_eval_loss_qy_py_err[i], min_eval_loss_qy_py_err_std[i], min_eval_loss_qz_pz[i], min_eval_loss_qz_pz_std[i], \
                                    min_eval_loss_mcep_rec[i], min_eval_loss_mcep_rec_std[i], min_eval_loss_mcep_cv[i//2], min_eval_loss_mcep_cv_std[i//2], \
                                    min_eval_loss_powmcep[i], min_eval_loss_powmcep_std[i], min_eval_loss_mcep[i], min_eval_loss_mcep_std[i], \
                                    min_eval_loss_gv_src_src, min_eval_loss_gv_src_trg)
                        else:
                            if pair_exist:
                                text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                    "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ; " \
                                    "%.6f %.6f %.6f (+- %.6f) dB %.6f (+- %.6f) dB %.6f (+- %.6f) %.6f (+- %.6f) ;; " % (i+1, \
                                    min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_px[i], min_eval_loss_px_std[i], \
                                    min_eval_loss_qy_py[i], min_eval_loss_qy_py_std[i], \
                                    min_eval_loss_qy_py_err[i], min_eval_loss_qy_py_err_std[i], min_eval_loss_qz_pz[i], min_eval_loss_qz_pz_std[i], \
                                    min_eval_loss_qy_py[i+1], min_eval_loss_qy_py_std[i+1], \
                                    min_eval_loss_qy_py_err[i+1], min_eval_loss_qy_py_err_std[i+1], min_eval_loss_qz_pz[i+1], min_eval_loss_qz_pz_std[i+1], \
                                    min_eval_loss_mcep_rec[i], min_eval_loss_mcep_rec_std[i], min_eval_loss_mcep_cv[i//2], min_eval_loss_mcep_cv_std[i//2], \
                                    min_eval_loss_powmcep[i], min_eval_loss_powmcep_std[i], min_eval_loss_mcep[i], min_eval_loss_mcep_std[i], \
                                    min_eval_loss_gv_src_src, min_eval_loss_gv_src_trg, min_eval_loss_mcdpow_src_trg, min_eval_loss_mcdpow_src_trg_std, \
                                    min_eval_loss_mcd_src_trg, min_eval_loss_mcd_src_trg_std, min_eval_loss_lat_dist_rmse, min_eval_loss_lat_dist_rmse_std, \
                                    min_eval_loss_lat_dist_cossim, min_eval_loss_lat_dist_cossim_std)
                            else:
                                text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) , %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                                    "%.6f (+- %.6f) %.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ; " \
                                    "%.6f %.6f n/a (+- n/a) dB n/a (+- n/a) dB n/a n/a (+- n/a) n/a (+- n/a) ;; " % (i+1, \
                                    min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_px[i], min_eval_loss_px_std[i], \
                                    min_eval_loss_qy_py[i], min_eval_loss_qy_py_std[i], \
                                    min_eval_loss_qy_py_err[i], min_eval_loss_qy_py_err_std[i], min_eval_loss_qz_pz[i], min_eval_loss_qz_pz_std[i], \
                                    min_eval_loss_qy_py[i+1], min_eval_loss_qy_py_std[i+1], \
                                    min_eval_loss_qy_py_err[i+1], min_eval_loss_qy_py_err_std[i+1], min_eval_loss_qz_pz[i+1], min_eval_loss_qz_pz_std[i+1], \
                                    min_eval_loss_mcep_rec[i], min_eval_loss_mcep_rec_std[i], min_eval_loss_mcep_cv[i//2], min_eval_loss_mcep_cv_std[i//2], \
                                    min_eval_loss_powmcep[i], min_eval_loss_powmcep_std[i], min_eval_loss_mcep[i], min_eval_loss_mcep_std[i], \
                                    min_eval_loss_gv_src_src, min_eval_loss_gv_src_trg)
                    else:
                        text_log += "[%ld] %.6f (+- %.6f) ; %.6f (+- %.6f) %.6f (+- %.6f) %.6f (+- %.6f) %% %.6f (+- %.6f) ; "\
                            "%.6f (+- %.6f) ; %.6f (+- %.6f) dB %.6f (+- %.6f) dB ;; " % (i+1, \
                            min_eval_loss_elbo[i], min_eval_loss_elbo_std[i], min_eval_loss_px[i], min_eval_loss_px_std[i], \
                            min_eval_loss_qy_py[i], min_eval_loss_qy_py_std[i], \
                            min_eval_loss_qy_py_err[i], min_eval_loss_qy_py_err_std[i], min_eval_loss_qz_pz[i], min_eval_loss_qz_pz_std[i], \
                            min_eval_loss_mcep_rec[i], min_eval_loss_mcep_rec_std[i], \
                            min_eval_loss_powmcep[i], min_eval_loss_powmcep_std[i], min_eval_loss_mcep[i], min_eval_loss_mcep_std[i])
                logging.info("%s min_idx=%d" % (text_log, min_idx+1))
            #if ((epoch_idx + 1) % args.save_interval_epoch == 0) or (epoch_min_flag):
            if True:
                logging.info('save epoch:%d' % (epoch_idx+1))
                save_checkpoint(args.expdir, model_encoder, model_decoder, model_classifier,
                    min_eval_loss_mcep[0], min_eval_loss_mcep_std[0], min_eval_loss_mcep_cv[0],
                    min_eval_loss_mcep_rec[0], min_eval_loss_mcd_src_trg, min_eval_loss_mcd_src_trg_std,
                    iter_idx, min_idx, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1)
            total = 0
            iter_count = 0
            for i in range(args.n_half_cyc):
                loss_elbo[i] = []
                loss_px[i] = []
                loss_qy_py[i] = []
                loss_qy_py_err[i] = []
                loss_qz_pz[i] = []
                if args.n_half_cyc == 1:
                    loss_qy_py[i+1] = []
                    loss_qy_py_err[i+1] = []
                    loss_qz_pz[i+1] = []
                loss_powmcep[i] = []
                loss_mcep[i] = []
                loss_mcep_cv[i] = []
                loss_mcep_rec[i] = []
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model_encoder.train()
            model_decoder.train()
            model_classifier.train()
            for param in model_encoder.parameters():
                param.requires_grad = True
            for param in model_encoder.scale_in.parameters():
                param.requires_grad = False
            for param in model_decoder.parameters():
                param.requires_grad = True
            for param in model_decoder.scale_out.parameters():
                param.requires_grad = False
            for param in model_classifier.parameters():
                param.requires_grad = True
            # start next epoch
            if epoch_idx < args.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_feat, batch_sc, batch_sc_cv_data, batch_feat_cv_data, c_idx, utt_idx, featfile, \
                    f_bs, f_ss, flens, n_batch_utt, del_index_utt, max_flen, spk_cv, idx_select, idx_select_full, flens_acc = next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

            f_es = f_ss+f_bs
            logging.info(f'{f_ss} {f_bs} {f_es} {max_flen}')
            # handle first pad for input
            flag_cv = True
            i_cv = 0
            flag_cv_enc = False
            i_cv_enc = 0
            f_ss_first_pad_left = f_ss-first_pad_left
            f_es_first_pad_right = f_es+first_pad_right
            i_end = args.n_half_cyc*2
            if args.n_half_cyc == 1:
                i_end += 1
            for i in range(i_end):
                #logging.info(f'{f_ss_first_pad_left} {f_es_first_pad_right}')
                if i % 2 == 0: #enc
                    if not flag_cv_enc:
                        if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                            batch_feat_in[i] = batch_feat[:,f_ss_first_pad_left:f_es_first_pad_right]
                        elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                            batch_feat_in[i] = F.pad(batch_feat[:,:f_es_first_pad_right].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
                        elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                            batch_feat_in[i] = F.pad(batch_feat[:,f_ss_first_pad_left:max_flen].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                        else: # pad left and right need additional replicate
                            batch_feat_in[i] = F.pad(batch_feat[:,:max_flen].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                        flag_cv_enc = True
                    else:
                        if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                            batch_feat_cv_in[i_cv_enc] = batch_feat_cv_data[i_cv_enc][:,f_ss_first_pad_left:f_es_first_pad_right]
                        elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                            batch_feat_cv_in[i_cv_enc] = F.pad(batch_feat_cv_data[i_cv_enc][:,:f_es_first_pad_right].transpose(1,2), (-f_ss_first_pad_left,0), "replicate").transpose(1,2)
                        elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                            batch_feat_cv_in[i_cv_enc] = F.pad(batch_feat_cv_data[i_cv_enc][:,f_ss_first_pad_left:max_flen].transpose(1,2), (0,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                        else: # pad left and right need additional replicate
                            batch_feat_cv_in[i_cv_enc] = F.pad(batch_feat_cv_data[i_cv_enc][:,:max_flen].transpose(1,2), (-f_ss_first_pad_left,f_es_first_pad_right-max_flen), "replicate").transpose(1,2)
                        i_cv_enc += 1
                        flag_cv_enc = False
                    f_ss_first_pad_left += enc_pad_left
                    f_es_first_pad_right -= enc_pad_right
                else: #dec
                    if f_ss_first_pad_left >= 0 and f_es_first_pad_right <= max_flen: # pad left and right available
                        batch_sc_in[i] = batch_sc[:,f_ss_first_pad_left:f_es_first_pad_right]
                        if flag_cv:
                            batch_sc_cv_in[i_cv] = batch_sc_cv_data[i_cv][:,f_ss_first_pad_left:f_es_first_pad_right]
                            i_cv += 1
                            flag_cv = False
                        else:
                            flag_cv = True
                    elif f_es_first_pad_right <= max_flen: # pad right available, left need additional replicate
                        batch_sc_in[i] = F.pad(batch_sc[:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv] = F.pad(batch_sc_cv_data[i_cv][:,:f_es_first_pad_right].unsqueeze(1).float(), (-f_ss_first_pad_left,0), "replicate").squeeze(1).long()
                            i_cv += 1
                            flag_cv = False
                        else:
                            flag_cv = True
                    elif f_ss_first_pad_left >= 0: # pad left available, right need additional replicate
                        diff_pad = f_es_first_pad_right - max_flen
                        batch_sc_in[i] = F.pad(batch_sc[:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv] = F.pad(batch_sc_cv_data[i_cv][:,f_ss_first_pad_left:max_flen].unsqueeze(1).float(), (0,diff_pad), "replicate").squeeze(1).long()
                            i_cv += 1
                            flag_cv = False
                        else:
                            flag_cv = True
                    else: # pad left and right need additional replicate
                        diff_pad = f_es_first_pad_right - max_flen
                        batch_sc_in[i] = F.pad(batch_sc[:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                        if flag_cv:
                            batch_sc_cv_in[i_cv] = F.pad(batch_sc_cv_data[i_cv][:,:max_flen].unsqueeze(1).float(), (-f_ss_first_pad_left,diff_pad), "replicate").squeeze(1).long()
                            i_cv += 1
                            flag_cv = False
                        else:
                            flag_cv = True
                    f_ss_first_pad_left += dec_pad_left
                    f_es_first_pad_right -= dec_pad_right
            # for target optimization
            batch_feat = batch_feat[:,f_ss:f_es]
            batch_sc = batch_sc[:,f_ss:f_es]
            for i in range(n_cv):
                batch_sc_cv[i] = batch_sc_cv_data[i][:,f_ss:f_es]

            if f_ss > 0:
                idx_in = 0
                for i in range(0,args.n_half_cyc,2):
                    i_cv = i//2
                    j = i+1
                    if len(del_index_utt) > 0:
                        if i == 0:
                            h_mcep_in_sc = torch.FloatTensor(np.delete(h_mcep_in_sc.cpu().data.numpy(), \
                                                            del_index_utt, axis=1)).to(device)
                        h_z[i] = torch.FloatTensor(np.delete(h_z[i].cpu().data.numpy(), \
                                                        del_index_utt, axis=1)).to(device)
                        h_z_sc[i] = torch.FloatTensor(np.delete(h_z_sc[i].cpu().data.numpy(), \
                                                        del_index_utt, axis=1)).to(device)
                        h_mcep[i] = torch.FloatTensor(np.delete(h_mcep[i].cpu().data.numpy(), \
                                                        del_index_utt, axis=1)).to(device)
                        h_mcep_sc[i] = torch.FloatTensor(np.delete(h_mcep_sc[i].cpu().data.numpy(), \
                                                        del_index_utt, axis=1)).to(device)
                        h_mcep_cv[i_cv] = torch.FloatTensor(np.delete(h_mcep_cv[i_cv].cpu().data.numpy(), \
                                                        del_index_utt, axis=1)).to(device)
                        h_mcep_cv_sc[i_cv] = torch.FloatTensor(np.delete(h_mcep_cv_sc[i_cv].cpu().data.numpy(), \
                                                        del_index_utt, axis=1)).to(device)
                        h_z[j] = torch.FloatTensor(np.delete(h_z[j].cpu().data.numpy(), \
                                                        del_index_utt, axis=1)).to(device)
                        h_z_sc[j] = torch.FloatTensor(np.delete(h_z_sc[j].cpu().data.numpy(), \
                                                        del_index_utt, axis=1)).to(device)
                        if args.n_half_cyc > 1:
                            h_mcep[j] = torch.FloatTensor(np.delete(h_mcep[j].cpu().data.numpy(), \
                                                            del_index_utt, axis=1)).to(device)
                            h_mcep_sc[j] = torch.FloatTensor(np.delete(h_mcep_sc[j].cpu().data.numpy(), \
                                                            del_index_utt, axis=1)).to(device)
                    if i > 0:
                        idx_in += 1
                        qy_logits[i], qz_alpha[i], h_z[i] \
                            = model_encoder(torch.cat((batch_feat_in[idx_in][:,:,:args.excit_dim], batch_mcep_rec[i-1].detach()), 2), h=h_z[i], outpad_right=outpad_rights[idx_in], do=True)
                        i_1 = i-1
                        idx_in_1 = idx_in-1
                        feat_len = batch_mcep_rec[i_1].shape[1]
                        batch_mcep_rec[i_1] = batch_mcep_rec[i_1][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                        batch_mcep_rec_sc[i_1], h_mcep_sc[i_1] = model_classifier(feat=batch_mcep_rec[i_1], h=h_mcep_sc[i_1])
                    else:
                        qy_logits[i], qz_alpha[i], h_z[i] = model_encoder(batch_feat_in[idx_in], h=h_z[i], outpad_right=outpad_rights[idx_in], do=True)
                        batch_mcep_in_sc, h_mcep_in_sc = model_classifier(feat=batch_feat[:,:,args.excit_dim:], h=h_mcep_in_sc)
                    eps = torch.empty_like(qz_alpha[i][:,:,:args.lat_dim])
                    eps.uniform_(eps_1,1)
                    z[i] = qz_alpha[i][:,:,:args.lat_dim] - torch.exp(qz_alpha[i][:,:,args.lat_dim:]) * eps.sign() * torch.log1p(-eps.abs()) # sampling laplace
                    idx_in += 1
                    batch_mcep_rec[i], h_mcep[i] = model_decoder(batch_sc_in[idx_in], z[i], h=h_mcep[i], outpad_right=outpad_rights[idx_in], do=True)
                    batch_mcep_cv[i_cv], h_mcep_cv[i_cv] = model_decoder(batch_sc_cv_in[i_cv], z[i], h=h_mcep_cv[i_cv], outpad_right=outpad_rights[idx_in], do=True)
                    feat_len = qy_logits[i].shape[1]
                    idx_in_1 = idx_in-1
                    z[i] = z[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_z_sc[i], h_z_sc[i] = model_classifier(lat=z[i], h=h_z_sc[i])
                    qy_logits[i] = qy_logits[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    qz_alpha[i] = qz_alpha[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    idx_in += 1
                    qy_logits[j], qz_alpha[j], h_z[j] \
                        = model_encoder(torch.cat((batch_feat_cv_in[i_cv], batch_mcep_cv[i_cv].detach()), 2), h=h_z[j], outpad_right=outpad_rights[idx_in], do=True)
                    feat_len = batch_mcep_rec[i].shape[1]
                    idx_in_1 = idx_in-1
                    batch_mcep_rec[i] = batch_mcep_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_mcep_rec_sc[i], h_mcep_sc[i] = model_classifier(feat=batch_mcep_rec[i], h=h_mcep_sc[i])
                    batch_mcep_cv[i_cv] = batch_mcep_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_mcep_cv_sc[i_cv], h_mcep_cv_sc[i_cv] = model_classifier(feat=batch_mcep_cv[i_cv], h=h_mcep_cv_sc[i_cv])
                    if args.n_half_cyc > 1:
                        eps = torch.empty_like(qz_alpha[j][:,:,:args.lat_dim])
                        eps.uniform_(eps_1,1)
                        z[j] = qz_alpha[j][:,:,:args.lat_dim] - torch.exp(qz_alpha[j][:,:,args.lat_dim:]) * eps.sign() * torch.log1p(-eps.abs()) # sampling laplace
                        idx_in += 1
                        batch_mcep_rec[j], h_mcep[j] = model_decoder(batch_sc_in[idx_in], z[j], h=h_mcep[j], outpad_right=outpad_rights[idx_in], do=True)
                        feat_len = qy_logits[j].shape[1]
                        idx_in_1 = idx_in-1
                        z[j] = z[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                        batch_z_sc[j], h_z_sc[j] = model_classifier(lat=z[j], h=h_z_sc[j])
                        qy_logits[j] = qy_logits[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                        qz_alpha[j] = qz_alpha[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                        if j+1 == args.n_half_cyc:
                            batch_mcep_rec[j] = batch_mcep_rec[j][:,outpad_lefts[idx_in]:batch_mcep_rec[j].shape[1]-outpad_rights[idx_in]]
                            batch_mcep_rec_sc[j], h_mcep_sc[j] = model_classifier(feat=batch_mcep_rec[j], h=h_mcep_sc[j])
                    else:
                        qy_logits[j] = qy_logits[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                        qz_alpha[j] = qz_alpha[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
            else:
                idx_in = 0
                for i in range(0,args.n_half_cyc,2):
                    i_cv = i//2
                    j = i+1
                    if i > 0:
                        idx_in += 1
                        qy_logits[i], qz_alpha[i], h_z[i] \
                            = model_encoder(torch.cat((batch_feat_in[idx_in][:,:,:args.excit_dim], batch_mcep_rec[i-1].detach()), 2), outpad_right=outpad_rights[idx_in], do=True)
                        i_1 = i-1
                        idx_in_1 = idx_in-1
                        feat_len = batch_mcep_rec[i_1].shape[1]
                        batch_mcep_rec[i_1] = batch_mcep_rec[i_1][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                        batch_mcep_rec_sc[i_1], h_mcep_sc[i_1] = model_classifier(feat=batch_mcep_rec[i_1])
                    else:
                        qy_logits[i], qz_alpha[i], h_z[i] = model_encoder(batch_feat_in[idx_in], outpad_right=outpad_rights[idx_in], do=True)
                        batch_mcep_in_sc, h_mcep_in_sc = model_classifier(feat=batch_feat[:,:,args.excit_dim:])
                    eps = torch.empty_like(qz_alpha[i][:,:,:args.lat_dim])
                    eps.uniform_(eps_1,1)
                    z[i] = qz_alpha[i][:,:,:args.lat_dim] - torch.exp(qz_alpha[i][:,:,args.lat_dim:]) * eps.sign() * torch.log1p(-eps.abs()) # sampling laplace
                    idx_in += 1
                    batch_mcep_rec[i], h_mcep[i] = model_decoder(batch_sc_in[idx_in], z[i], outpad_right=outpad_rights[idx_in], do=True)
                    batch_mcep_cv[i_cv], h_mcep_cv[i_cv] = model_decoder(batch_sc_cv_in[i_cv], z[i], outpad_right=outpad_rights[idx_in], do=True)
                    feat_len = qy_logits[i].shape[1]
                    idx_in_1 = idx_in-1
                    z[i] = z[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_z_sc[i], h_z_sc[i] = model_classifier(lat=z[i])
                    qy_logits[i] = qy_logits[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    qz_alpha[i] = qz_alpha[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    idx_in += 1
                    qy_logits[j], qz_alpha[j], h_z[j] \
                        = model_encoder(torch.cat((batch_feat_cv_in[i_cv], batch_mcep_cv[i_cv].detach()), 2), outpad_right=outpad_rights[idx_in], do=True)
                    feat_len = batch_mcep_rec[i].shape[1]
                    idx_in_1 = idx_in-1
                    batch_mcep_rec[i] = batch_mcep_rec[i][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_mcep_rec_sc[i], h_mcep_sc[i] = model_classifier(feat=batch_mcep_rec[i])
                    batch_mcep_cv[i_cv] = batch_mcep_cv[i_cv][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                    batch_mcep_cv_sc[i_cv], h_mcep_cv_sc[i_cv] = model_classifier(feat=batch_mcep_cv[i_cv])
                    if args.n_half_cyc > 1:
                        eps = torch.empty_like(qz_alpha[j][:,:,:args.lat_dim])
                        eps.uniform_(eps_1,1)
                        z[j] = qz_alpha[j][:,:,:args.lat_dim] - torch.exp(qz_alpha[j][:,:,args.lat_dim:]) * eps.sign() * torch.log1p(-eps.abs()) # sampling laplace
                        idx_in += 1
                        batch_mcep_rec[j], h_mcep[j] = model_decoder(batch_sc_in[idx_in], z[j], outpad_right=outpad_rights[idx_in], do=True)
                        feat_len = qy_logits[j].shape[1]
                        idx_in_1 = idx_in-1
                        z[j] = z[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                        batch_z_sc[j], h_z_sc[j] = model_classifier(lat=z[j])
                        qy_logits[j] = qy_logits[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                        qz_alpha[j] = qz_alpha[j][:,outpad_lefts[idx_in_1]:feat_len-outpad_rights[idx_in_1]]
                        if j+1 == args.n_half_cyc:
                            batch_mcep_rec[j] = batch_mcep_rec[j][:,outpad_lefts[idx_in]:batch_mcep_rec[j].shape[1]-outpad_rights[idx_in]]
                            batch_mcep_rec_sc[j], h_mcep_sc[j] = model_classifier(feat=batch_mcep_rec[j])
                    else:
                        qy_logits[j] = qy_logits[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]
                        qz_alpha[j] = qz_alpha[j][:,outpad_lefts[idx_in]:feat_len-outpad_rights[idx_in]]

            # samples check
            with torch.no_grad():
                i = np.random.randint(0, batch_mcep_rec[0].shape[0])
                logging.info("%d %s %d %d %d %d %s" % (i, \
                    os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i])), \
                        f_ss, f_es, flens[i], max_flen, spk_cv[0][i]))
                logging.info(batch_mcep_rec[0][i,:2,:4])
                if args.n_half_cyc > 1:
                    logging.info(batch_mcep_rec[1][i,:2,:4])
                logging.info(batch_feat[i,:2,args.excit_dim:args.excit_dim+4])
                logging.info(batch_mcep_cv[0][i,:2,:4])
                #logging.info(qy_logits[0][i,:2])
                #logging.info(batch_sc[i,0])
                #logging.info(qy_logits[1][i,:2])
                #logging.info(batch_sc_cv[0][i,0])

            # loss
            batch_loss = 0

            if len(idx_select) > 0:
                logging.info('len_idx_select: '+str(len(idx_select)))
                batch_loss_px_select = 0
                batch_loss_px_ms_norm_select = 0
                batch_loss_px_ms_err_select = 0
                batch_loss_qz_pz_kl_select = 0
                batch_loss_qy_py_ce_select = 0
                batch_loss_sc_mcep_in_kl_select = 0
                batch_loss_sc_mcep_kl_select = 0
                batch_loss_sc_z_kl_select = 0
                for j in range(len(idx_select)):
                    k = idx_select[j]
                    flens_utt = flens_acc[k]
                    logging.info('%s %d' % (featfile[k], flens_utt))
                    powmcep = batch_feat[k,:flens_utt,args.excit_dim:]

                    batch_sc_ = batch_sc[k,:flens_utt]
                    batch_loss_sc_mcep_in_kl_select += torch.mean(criterion_ce(batch_mcep_in_sc[k,:flens_utt], batch_sc_))
                    for i in range(args.n_half_cyc):
                        batch_loss_px_select += torch.mean(mcd_constant*torch.sum(criterion_l1(batch_mcep_rec[i][k,:flens_utt], powmcep), -1))
                        batch_loss_px_ms_norm_, batch_loss_px_ms_err_ = criterion_ms(batch_mcep_rec[i][k,:flens_utt], powmcep)
                        if not torch.isinf(batch_loss_px_ms_norm_) and not torch.isnan(batch_loss_px_ms_norm_):
                            batch_loss_px_ms_norm_select += batch_loss_px_ms_norm_
                        if not torch.isinf(batch_loss_px_ms_err_) and not torch.isnan(batch_loss_px_ms_err_):
                            batch_loss_px_ms_err_select += batch_loss_px_ms_err_

                        qy_logits_select_ = qy_logits[i][k,:flens_utt]
                        batch_mcep_rec_sc_ = batch_mcep_rec_sc[i][k,:flens_utt]
                        batch_z_sc_ = batch_z_sc[i][k,:flens_utt]
                        batch_sc_cv_ = batch_sc_cv[i//2][k,:flens_utt]
                        #batch_loss_sc_z_kl_select += torch.mean(kl_categorical_categorical_logits(p_spk, logits_p_spk, batch_z_sc_))
                        batch_loss_sc_z_kl_select += torch.mean(kl_categorical_categorical_logits(p_spk, logits_p_spk, batch_z_sc_)) \
                                                        + torch.mean(criterion_ce(revgrad(batch_z_sc_), batch_sc_))
                        if i % 2 == 0:
                            batch_loss_qy_py_ce_select += torch.mean(criterion_ce(qy_logits_select_, batch_sc_))
                            batch_mcep_cv_sc_ = batch_mcep_cv_sc[i//2][k,:flens_utt]
                            batch_loss_sc_mcep_kl_select += torch.mean(criterion_ce(batch_mcep_rec_sc_, batch_sc_)) \
                                                                + torch.mean(criterion_ce(batch_mcep_cv_sc_, batch_sc_cv_)) \
                                                                + torch.mean(criterion_ce(revgrad(batch_mcep_cv_sc_), batch_sc_))
                            batch_loss_sc_z_kl_select += torch.mean(criterion_ce(revgrad(batch_z_sc_), batch_sc_))
                        else:
                            batch_loss_qy_py_ce_select += torch.mean(criterion_ce(qy_logits_select_, batch_sc_cv_)) \
                                                            + torch.mean(criterion_ce(revgrad(qy_logits_select_), batch_sc_))
                            batch_loss_sc_mcep_kl_select += torch.mean(criterion_ce(batch_mcep_rec_sc_, batch_sc_))
                            #batch_loss_sc_z_kl_select += torch.mean(criterion_ce(revgrad(batch_z_sc_), batch_sc_cv_))

                        batch_loss_qz_pz_kl_select += torch.mean(torch.sum(kl_laplace(qz_alpha[i][k,:flens_utt]), -1))
                batch_loss += batch_loss_px_select + batch_loss_px_ms_norm_select + batch_loss_px_ms_err_select \
                                + batch_loss_qz_pz_kl_select + batch_loss_qy_py_ce_select \
                                    + batch_loss_sc_mcep_in_kl_select + batch_loss_sc_mcep_kl_select + batch_loss_sc_z_kl_select
                if len(idx_select_full) > 0:
                    logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                    batch_feat = torch.index_select(batch_feat,0,idx_select_full)
                    batch_sc = torch.index_select(batch_sc,0,idx_select_full)
                    batch_mcep_in_sc = torch.index_select(batch_mcep_in_sc,0,idx_select_full)
                    for i in range(args.n_half_cyc):
                        batch_mcep_rec[i] = torch.index_select(batch_mcep_rec[i],0,idx_select_full)
                        batch_mcep_rec_sc[i] = torch.index_select(batch_mcep_rec_sc[i],0,idx_select_full)
                        batch_z_sc[i] = torch.index_select(batch_z_sc[i],0,idx_select_full)
                        qz_alpha[i] = torch.index_select(qz_alpha[i],0,idx_select_full)
                        qy_logits[i] = torch.index_select(qy_logits[i],0,idx_select_full)
                        if i % 2 == 0:
                            batch_mcep_cv[i//2] = torch.index_select(batch_mcep_cv[i//2],0,idx_select_full)
                            batch_mcep_cv_sc[i//2] = torch.index_select(batch_mcep_cv_sc[i//2],0,idx_select_full)
                            batch_sc_cv[i//2] = torch.index_select(batch_sc_cv[i//2],0,idx_select_full)
                            if args.n_half_cyc == 1:
                                qz_alpha[i+1] = torch.index_select(qz_alpha[i+1],0,idx_select_full)
                                qy_logits[i+1] = torch.index_select(qy_logits[i+1],0,idx_select_full)
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
            powmcep = batch_feat[:,:,args.excit_dim:]
            mcep = batch_feat[:,:,args.excit_dim+1:]
            sc_onehot = F.one_hot(batch_sc, num_classes=n_spk).float()
            batch_loss_sc_mcep_in_ = torch.mean(criterion_ce(batch_mcep_in_sc.reshape(-1, n_spk), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
            batch_loss_sc_mcep_in = batch_loss_sc_mcep_in_.mean()
            batch_loss += batch_loss_sc_mcep_in_.sum()
            for i in range(args.n_half_cyc):
                mcep_est = batch_mcep_rec[i]
                if i % 2 == 0:
                    mcep_cv = batch_mcep_cv[i//2]

                ## mcep acc.
                batch_loss_powmcep_ = torch.mean(mcd_constant*torch.sqrt(\
                                                    torch.sum(criterion_l2(mcep_est, powmcep), -1)), -1)
                batch_loss_powmcep[i] = batch_loss_powmcep_.mean()
                batch_loss_mcep[i] = torch.mean(torch.mean(mcd_constant*torch.sqrt(\
                                                    torch.sum(criterion_l2(mcep_est[:,:,1:], mcep), -1)), -1))
                batch_loss_px_mcep_ = torch.mean(mcd_constant*torch.sum(criterion_l1(mcep_est, powmcep), -1), -1)
                batch_loss_mcep_rec[i] = batch_loss_px_mcep_.mean()
                if i % 2 == 0:
                    batch_loss_mcep_cv[i//2] = torch.mean(torch.mean(mcd_constant*torch.sum(criterion_l1(mcep_cv, powmcep),-1), -1))
                batch_loss_px_sum = batch_loss_px_mcep_.sum()

                batch_loss_px_ms_norm_, batch_loss_px_ms_err_ = criterion_ms(mcep_est, powmcep)
                batch_loss_ms_norm[i] = batch_loss_px_ms_norm_.mean()
                if not torch.isinf(batch_loss_ms_norm[i]) and not torch.isnan(batch_loss_ms_norm[i]):
                    batch_loss_px_sum += batch_loss_px_ms_norm_.sum()
                batch_loss_ms_err[i] = batch_loss_px_ms_err_.mean()
                if not torch.isinf(batch_loss_ms_err[i]) and not torch.isnan(batch_loss_ms_err[i]):
                    batch_loss_px_sum += batch_loss_px_ms_err_.sum()

                batch_loss_px[i] = batch_loss_mcep_rec[i] + batch_loss_ms_norm[i] + batch_loss_ms_err[i]

                # KL div
                batch_loss_sc_mcep_ = torch.mean(criterion_ce(batch_mcep_rec_sc[i].reshape(-1, n_spk), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                batch_loss_sc_mcep[i] = batch_loss_sc_mcep_.mean()
                batch_loss_sc_mcep_rev_ = torch.mean(criterion_ce(revgrad(batch_mcep_rec_sc[i].reshape(-1, n_spk)), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1)
                batch_loss_sc_mcep_rev[i] = batch_loss_sc_mcep_rev_.mean()
                batch_loss_sc_z_ = torch.mean(kl_categorical_categorical_logits(p_spk, logits_p_spk, batch_z_sc[i]), -1)
                batch_loss_sc_z[i] = batch_loss_sc_z_.mean()
                batch_loss_sc_z_rev_ = torch.mean(criterion_ce(revgrad(batch_z_sc[i].reshape(-1, n_spk)), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                batch_loss_sc_z_rev[i] = batch_loss_sc_z_rev_.mean()
                if i % 2 == 0:
                    batch_loss_qy_py_ = torch.mean(criterion_ce(qy_logits[i].reshape(-1, n_spk), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                    batch_loss_qy_py[i] = batch_loss_qy_py_.mean()
                    batch_loss_qy_py_err_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i], dim=-1), sc_onehot), -1), -1)
                    batch_loss_qy_py_err[i] = batch_loss_qy_py_err_.mean()
                    batch_loss_sc_mcep_cv_ = torch.mean(criterion_ce(batch_mcep_cv_sc[i//2].reshape(-1, n_spk), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1)
                    batch_loss_sc_mcep_cv[i//2] = batch_loss_sc_mcep_cv_.mean()
                    batch_loss_sc_mcep_cv_rev_ = torch.mean(criterion_ce(revgrad(batch_mcep_cv_sc[i//2].reshape(-1, n_spk)), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                    batch_loss_sc_mcep_cv_rev[i//2] = batch_loss_sc_mcep_cv_rev_.mean()
                    if args.n_half_cyc == 1:
                        batch_loss_qy_py[i+1] = torch.mean(criterion_ce(qy_logits[i+1].reshape(-1, n_spk), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1).mean()
                        batch_loss_qy_py_err[i+1] = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i+1], dim=-1), F.one_hot(batch_sc_cv[i//2], num_classes=n_spk).float()), -1), -1).mean()
                        batch_loss_qz_pz[i+1] = torch.mean(torch.sum(kl_laplace(qz_alpha[i+1]), -1), -1).mean()
                    batch_loss_sc_mcep_kl = batch_loss_sc_mcep_.sum() + batch_loss_sc_mcep_cv_.sum() + batch_loss_sc_mcep_cv_rev_.sum()
                    batch_loss_sc_z_kl = batch_loss_sc_z_.sum() + batch_loss_sc_z_rev_.sum()
                    batch_loss_qy_py_ce = batch_loss_qy_py_.sum()
                else:
                    batch_loss_qy_py_ = torch.mean(criterion_ce(qy_logits[i].reshape(-1, n_spk), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1)
                    batch_loss_qy_py[i] = batch_loss_qy_py_.mean()
                    batch_loss_qy_py_rev_ = torch.mean(criterion_ce(revgrad(qy_logits[i].reshape(-1, n_spk)), batch_sc.reshape(-1)).reshape(batch_sc.shape[0], -1), -1)
                    batch_loss_qy_py_rev[i] = batch_loss_qy_py_rev_.mean()
                    batch_loss_qy_py_err_ = torch.mean(100*torch.sum(criterion_l1(F.softmax(qy_logits[i], dim=-1), F.one_hot(batch_sc_cv[i//2], num_classes=n_spk).float()), -1), -1)
                    batch_loss_qy_py_err[i] = batch_loss_qy_py_err_.mean()
                    batch_loss_sc_mcep_kl = batch_loss_sc_mcep_.sum()
                    batch_loss_sc_z_cv_rev_ = torch.mean(criterion_ce(revgrad(batch_z_sc[i].reshape(-1, n_spk)), batch_sc_cv[i//2].reshape(-1)).reshape(batch_sc_cv[i//2].shape[0], -1), -1)
                    batch_loss_sc_z_cv_rev[i//2] = batch_loss_sc_z_cv_rev_.mean()
                    #batch_loss_sc_z_kl = batch_loss_sc_z_.sum() + batch_loss_sc_z_cv_rev_.sum()
                    batch_loss_sc_z_kl = batch_loss_sc_z_.sum() + batch_loss_sc_z_rev_.sum() + batch_loss_sc_z_cv_rev_.sum()
                    batch_loss_qy_py_ce = batch_loss_qy_py_.sum() + batch_loss_qy_py_rev_.sum()
                batch_loss_qz_pz_ = torch.mean(torch.sum(kl_laplace(qz_alpha[i]), -1), -1)
                batch_loss_qz_pz[i] = batch_loss_qz_pz_.mean()
                batch_loss_qz_pz_kl = batch_loss_qz_pz_.sum()

                # elbo
                batch_loss_elbo[i] = batch_loss_px_sum + batch_loss_qy_py_ce + batch_loss_qz_pz_kl \
                                            + batch_loss_sc_mcep_kl + batch_loss_sc_z_kl

                batch_loss += batch_loss_elbo[i]

                total_train_loss["train/loss_elbo-%d"%(i+1)].append(batch_loss_elbo[i].item())
                total_train_loss["train/loss_px-%d"%(i+1)].append(batch_loss_px[i].item())
                total_train_loss["train/loss_qy_py-%d"%(i+1)].append(batch_loss_qy_py[i].item())
                if i % 2 != 0:
                    total_train_loss["train/loss_qy_py_rev-%d"%(i+1)].append(batch_loss_qy_py_rev[i].item())
                total_train_loss["train/loss_qy_py_err-%d"%(i+1)].append(batch_loss_qy_py_err[i].item())
                total_train_loss["train/loss_qz_pz-%d"%(i+1)].append(batch_loss_qz_pz[i].item())
                total_train_loss["train/loss_sc_z-%d"%(i+1)].append(batch_loss_sc_z[i].item())
                total_train_loss["train/loss_sc_z_rev-%d"%(i+1)].append(batch_loss_sc_z_rev[i].item())
                total_train_loss["train/loss_sc_mcep-%d"%(i+1)].append(batch_loss_sc_mcep[i].item())
                total_train_loss["train/loss_sc_mcep_rev-%d"%(i+1)].append(batch_loss_sc_mcep_rev[i].item())
                if i == 0:
                    total_train_loss["train/loss_sc_mcep_in-%d"%(i+1)].append(batch_loss_sc_mcep_in.item())
                total_train_loss["train/loss_ms_norm-%d"%(i+1)].append(batch_loss_ms_norm[i].item())
                total_train_loss["train/loss_ms_err-%d"%(i+1)].append(batch_loss_ms_err[i].item())
                total_train_loss["train/loss_powmcep-%d"%(i+1)].append(batch_loss_powmcep[i].item())
                total_train_loss["train/loss_mcep-%d"%(i+1)].append(batch_loss_mcep[i].item())
                total_train_loss["train/loss_mcep_rec-%d"%(i+1)].append(batch_loss_mcep_rec[i].item())
                if i % 2 == 0:
                    total_train_loss["train/loss_sc_mcep_cv-%d"%(i+1)].append(batch_loss_sc_mcep_cv[i//2].item())
                    total_train_loss["train/loss_sc_mcep_cv_rev-%d"%(i+1)].append(batch_loss_sc_mcep_cv_rev[i//2].item())
                    total_train_loss["train/loss_mcep_cv-%d"%(i+1)].append(batch_loss_mcep_cv[i//2].item())
                    loss_mcep_cv[i//2].append(batch_loss_mcep_cv[i//2].item())
                else:
                    total_train_loss["train/loss_sc_z_cv_rev-%d"%(i+1)].append(batch_loss_sc_z_cv_rev[i//2].item())
                loss_elbo[i].append(batch_loss_elbo[i].item())
                loss_px[i].append(batch_loss_px[i].item())
                loss_qy_py[i].append(batch_loss_qy_py[i].item())
                loss_qy_py_err[i].append(batch_loss_qy_py_err[i].item())
                loss_qz_pz[i].append(batch_loss_qz_pz[i].item())
                loss_powmcep[i].append(batch_loss_powmcep[i].item())
                loss_mcep[i].append(batch_loss_mcep[i].item())
                loss_mcep_rec[i].append(batch_loss_mcep_rec[i].item())
                if args.n_half_cyc == 1:
                    total_train_loss["train/loss_qy_py-%d"%(i+2)].append(batch_loss_qy_py[i+1].item())
                    total_train_loss["train/loss_qy_py_err-%d"%(i+2)].append(batch_loss_qy_py_err[i+1].item())
                    total_train_loss["train/loss_qz_pz-%d"%(i+2)].append(batch_loss_qz_pz[i+1].item())
                    loss_qy_py[i+1].append(batch_loss_qy_py[i+1].item())
                    loss_qy_py_err[i+1].append(batch_loss_qy_py_err[i+1].item())
                    loss_qz_pz[i+1].append(batch_loss_qz_pz[i+1].item())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            text_log = "batch loss [%d] %d %d %.3f " % (c_idx+1, f_ss, f_bs, batch_loss_sc_mcep_in.item())
            for i in range(args.n_half_cyc):
                if i % 2 == 0:
                    if args.n_half_cyc > 1:
                        text_log += "[%ld] %.3f ; %.3f %.3f %.3f %% %.3f ; %.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; %.3f %.3f , %.3f dB %.3f dB ;; " % (
                            i+1, batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(),
                                batch_loss_qz_pz[i].item(), batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                                    batch_loss_sc_z[i].item(), batch_loss_sc_z_rev[i].item(),
                                    batch_loss_sc_mcep[i].item(), batch_loss_sc_mcep_rev[i].item(),
                                    batch_loss_sc_mcep_cv[i//2].item(), batch_loss_sc_mcep_cv_rev[i//2].item(),
                                    batch_loss_mcep_rec[i].item(), batch_loss_mcep_cv[i//2].item(),
                                    batch_loss_powmcep[i].item(), batch_loss_mcep[i].item())
                    else:
                        text_log += "[%ld] %.3f ; %.3f %.3f %.3f %% %.3f , %.3f %.3f %% %.3f ; %.3f %.3f , %.3f %.3f , %.3f %.3f , %.3f %.3f ; %.3f %.3f , %.3f dB %.3f dB ;; " % (
                            i+1, batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_qy_py[i].item(), batch_loss_qy_py_err[i].item(), \
                                batch_loss_qz_pz[i].item(), batch_loss_qy_py[i+1].item(), batch_loss_qy_py_err[i+1].item(),
                                batch_loss_qz_pz[i+1].item(), batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                                    batch_loss_sc_z[i].item(), batch_loss_sc_z_rev[i].item(),
                                    batch_loss_sc_mcep[i].item(), batch_loss_sc_mcep_rev[i].item(),
                                    batch_loss_sc_mcep_cv[i//2].item(), batch_loss_sc_mcep_cv_rev[i//2].item(),
                                    batch_loss_mcep_rec[i].item(), batch_loss_mcep_cv[i//2].item(),
                                    batch_loss_powmcep[i].item(), batch_loss_mcep[i].item())
                else:
                    text_log += "[%ld] %.3f ; %.3f %.3f %.3f %.3f %% %.3f ; %.3f %.3f , %.3f %.3f %.3f , %.3f %.3f ; %.3f , %.3f dB %.3f dB ;; " % (
                        i+1, batch_loss_elbo[i].item(), batch_loss_px[i].item(), batch_loss_qy_py[i].item(), batch_loss_qy_py_rev[i].item(), batch_loss_qy_py_err[i].item(),
                            batch_loss_qz_pz[i].item(), batch_loss_ms_norm[i].item(), batch_loss_ms_err[i].item(),
                            batch_loss_sc_z[i].item(), batch_loss_sc_z_rev[i].item(), batch_loss_sc_z_cv_rev[i//2].item(),
                            batch_loss_sc_mcep[i].item(), batch_loss_sc_mcep_rev[i].item(),
                            batch_loss_mcep_rec[i].item(), batch_loss_powmcep[i].item(), batch_loss_mcep[i].item())
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


    # save final model
    model_encoder.cpu()
    model_decoder.cpu()
    model_classifier.cpu()
    torch.save({"model_encoder": model_encoder.state_dict(),
                "model_decoder": model_decoder.state_dict(),
                "model_classifier": model_decoder.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
