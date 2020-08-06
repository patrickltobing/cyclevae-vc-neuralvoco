#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
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
from vcneuvoco import DSWNV, encode_mu_law
from radam import RAdam

from dataset import FeatureDatasetNeuVoco, padding

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)


def train_generator(dataloader, device, batch_size, upsampling_factor, limit_count=None):
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
            x_ss = 0
            f_ss = 0
            x_bs = batch_size*upsampling_factor
            f_bs = batch_size
            delta = batch_size*upsampling_factor
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
                    xs = torch.LongTensor(np.delete(xs.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
                    feat = torch.FloatTensor(np.delete(feat.cpu().data.numpy(), del_index_utt, axis=0)).to(device)
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
                yield xs, feat, c_idx, idx, featfiles, x_bs, f_bs, x_ss, f_ss, n_batch_utt, del_index_utt, max_slen, \
                    idx_select, idx_select_full, slens_acc
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

        yield [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], []


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
    parser.add_argument("--hid_chn", default=256,
                        type=int, help="depth of dilation")
    parser.add_argument("--skip_chn", default=256,
                        type=int, help="depth of dilation")
    parser.add_argument("--dilation_depth", default=3,
                        type=int, help="depth of dilation")
    parser.add_argument("--dilation_repeat", default=2,
                        type=int, help="depth of dilation")
    parser.add_argument("--kernel_size", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--kernel_size_wave", default=7,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dilation_size_wave", default=1,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--mcep_dim", default=50,
                        type=int, help="kernel size of dilated causal convolution")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=30,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--epoch_count", default=4000,
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
    parser.add_argument("--bi_wave", default=True,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--causal_conv_wave", default=False,
                        type=strtobool, help="batch size (if set 0, utterance batch will be used)")
    # other setting
    parser.add_argument("--init", default=False,
                        type=strtobool, help="seed number")
    parser.add_argument("--pad_len", default=3000,
                        type=int, help="seed number")
    ##parser.add_argument("--save_interval_iter", default=5000,
    #parser.add_argument("--save_interval_iter", default=3000,
    #                    type=int, help="interval steps to logr")
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
    parser.add_argument("--preconf", default=None,
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

    if args.pretrained is None:
        if 'mel' in args.string_path:
            mean_stats = torch.FloatTensor(read_hdf5(args.stats, "/mean_melsp"))
            scale_stats = torch.FloatTensor(read_hdf5(args.stats, "/scale_melsp"))
            args.excit_dim = 0
        else:
            mean_stats = torch.FloatTensor(read_hdf5(args.stats, "/mean_feat_mceplf0cap"))
            scale_stats = torch.FloatTensor(read_hdf5(args.stats, "/scale_feat_mceplf0cap"))
            args.cap_dim = mean_stats.shape[0]-(args.mcep_dim+3)
            args.excit_dim = 2+1+args.cap_dim
    else:
        config = torch.load(args.preconf)
        args.excit_dim = config.excit_dim
        args.cap_dim = config.cap_dim

    # save args as conf
    torch.save(args, args.expdir + "/model.conf")

    # define network
    model_waveform = DSWNV(
        n_aux=args.mcep_dim+args.excit_dim,
        upsampling_factor=args.upsampling_factor,
        hid_chn=args.hid_chn,
        skip_chn=args.skip_chn,
        kernel_size=args.kernel_size,
        aux_kernel_size=args.kernel_size_wave,
        aux_dilation_size=args.dilation_size_wave,
        dilation_depth=args.dilation_depth,
        dilation_repeat=args.dilation_repeat,
        n_quantize=args.n_quantize,
        do_prob=args.do_prob)
    logging.info(model_waveform)
    shift_rec_field = model_waveform.receptive_field
    logging.info(shift_rec_field)
    if shift_rec_field % args.upsampling_factor > 0:
        shift_rec_field_frm = shift_rec_field//args.upsampling_factor + 1
    else:
        shift_rec_field_frm = shift_rec_field//args.upsampling_factor
    shift_rec_field = shift_rec_field_frm*args.upsampling_factor
    logging.info(shift_rec_field)
    logging.info(shift_rec_field_frm)
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_l1 = torch.nn.L1Loss(reduction='none')

    # send to gpu
    if torch.cuda.is_available():
        model_waveform.cuda()
        criterion_ce.cuda()
        criterion_l1.cuda()
        if args.pretrained is None:
            mean_stats = mean_stats.cuda()
            scale_stats = scale_stats.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)

    model_waveform.train()

    if args.pretrained is None:
        model_waveform.scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/scale_stats.data),2))
        model_waveform.scale_in.bias = torch.nn.Parameter(-(mean_stats.data/scale_stats.data))

    #if args.pretrained is not None:
    #    checkpoint = torch.load(args.pretrained)
    #    #model_waveform.remove_weight_norm()
    #    #model_waveform.load_state_dict(checkpoint["model"])
    #    model_waveform.load_state_dict(checkpoint["model_waveform"])
    #    epoch_idx = checkpoint["iterations"]
    #    logging.info("pretrained from %d-iter checkpoint." % epoch_idx)
    #    epoch_idx = 0
    #    #model_waveform.apply_weight_norm()
    #    #torch.nn.utils.remove_weight_norm(model_waveform.scale_in)

    for param in model_waveform.parameters():
        param.requires_grad = True
    for param in model_waveform.scale_in.parameters():
        param.requires_grad = False

    parameters = filter(lambda p: p.requires_grad, model_waveform.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (waveform): %.3f million' % parameters)

    module_list = list(model_waveform.conv_aux.parameters()) + list(model_waveform.upsampling.parameters())
    if model_waveform.wav_conv_flag:
        module_list += list(model_waveform.wav_conv.parameters())
    module_list += list(model_waveform.causal.parameters())
    module_list += list(model_waveform.in_x.parameters()) + list(model_waveform.dil_h.parameters())
    module_list += list(model_waveform.out_skip.parameters())
    module_list += list(model_waveform.out_1.parameters()) + list(model_waveform.out_2.parameters())

    optimizer = RAdam(module_list, lr=args.lr)
    #optimizer = torch.optim.Adam(module_list, lr=args.lr)

    # resume
    if args.pretrained is not None and args.resume is None:
        checkpoint = torch.load(args.pretrained)
        model_waveform.load_state_dict(checkpoint["model_waveform"])
    #    optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("pretrained from %d-iter checkpoint." % epoch_idx)
        epoch_idx = 0
    elif args.resume is not None:
    #if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model_waveform.load_state_dict(checkpoint["model_waveform"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)
    #    epoch_idx = 2
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size_utt, shuffle=True, num_workers=args.n_workers)
    #generator = train_generator(dataloader, device, args.batch_size, args.upsampling_factor, limit_count=1)
    generator = train_generator(dataloader, device, args.batch_size, args.upsampling_factor, limit_count=None)
    #generator = train_generator(dataloader, device, args.batch_size, args.upsampling_factor, limit_count=1, resume_c_idx=1426, max_c_idx=(len(feat_list)//args.batch_size_utt))
    #generator = train_generator(dataloader, device, args.batch_size, args.upsampling_factor, limit_count=None, resume_c_idx=1426, max_c_idx=(len(feat_list)//args.batch_size_utt))

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
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size_utt_eval, shuffle=False, num_workers=args.n_workers)
    ##generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=1)
    #generator_eval = eval_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=None)
    #generator_eval = train_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=1)
    generator_eval = train_generator(dataloader_eval, device, args.batch_size, args.upsampling_factor, limit_count=None)

    writer = SummaryWriter(args.expdir)
    total_train_loss = defaultdict(list)
    total_eval_loss = defaultdict(list)

    # train
    logging.info(args.string_path)
    total = 0
    iter_count = 0
    loss_ce = []
    loss_err = []
    min_eval_loss_err = 99999999.99
    min_eval_loss_err_std = 99999999.99
    min_eval_loss_ce = 99999999.99
    min_eval_loss_ce_std = 99999999.99
    iter_idx = 0
    min_idx = -1
    #min_eval_loss_ce = 1.575400
    #min_eval_loss_ce_std = 0.645726
    #iter_idx = 8098898
    #min_idx = 68 #resume70
    change_min_flag = False
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        batch_x, batch_feat, c_idx, utt_idx, featfile, x_bs, f_bs, x_ss, f_ss, n_batch_utt, \
            del_index_utt, max_slen, idx_select, idx_select_full, slens_acc = next(generator)
        if args.init:
            c_idx = -1
        if c_idx < 0: # summarize epoch
            if not args.init:
                # save current epoch model
                numpy_random_state = np.random.get_state()
                torch_random_state = torch.get_rng_state()
                # report current epoch
                logging.info("(EPOCH:%d) average optimization loss = %.6f (+- %.6f) %.6f (+- %.6f) %% ;; "\
                    "(%.3f min., %.3f sec / batch)" % (epoch_idx + 1, np.mean(loss_ce), np.std(loss_ce), \
                        np.mean(loss_err), np.std(loss_err), total / 60.0, total / iter_count))
                logging.info("estimated time until max. epoch = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
                "{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * total))))
            # compute loss in evaluation data
            total = 0
            iter_count = 0
            loss_ce = []
            loss_err = []
            model_waveform.eval()
            for param in model_waveform.parameters():
                param.requires_grad = False
            pair_exist = False
            logging.info("Evaluation data")
            while True:
                with torch.no_grad():
                    start = time.time()
                    batch_x, batch_feat, c_idx, utt_idx, featfile, x_bs, f_bs, x_ss, f_ss, n_batch_utt, \
                        del_index_utt, max_slen, idx_select, idx_select_full, slens_acc = next(generator_eval)
                    if c_idx < 0:
                        break

                    x_es = x_ss+x_bs
                    f_es = f_ss+f_bs
                    logging.info(f'{x_ss} {x_bs} {x_es} {f_ss} {f_bs} {f_es} {max_slen}')
                    if x_ss > 0:
                        if x_es <= max_slen:
                            batch_x_prev = batch_x[:,x_ss-shift_rec_field-1:x_es-1]
                            batch_feat = batch_feat[:,f_ss-shift_rec_field_frm:f_es]
                            batch_x = batch_x[:,x_ss:x_es]
                        else:
                            batch_x_prev = batch_x[:,x_ss-shift_rec_field-1:-1]
                            batch_feat = batch_feat[:,f_ss-shift_rec_field_frm:]
                            batch_x = batch_x[:,x_ss:]
                    #    assert((batch_x_prev[:,shift_rec_field+1:] == batch_x[:,:-1]).all())
                    else:
                        batch_x_prev = F.pad(batch_x[:,:x_es-1], (model_waveform.receptive_field+1, 0), "constant", args.n_quantize // 2)
                        batch_feat = batch_feat[:,:f_es]
                        batch_x = batch_x[:,:x_es]
                    #    assert((batch_x_prev[:,model_waveform.receptive_field+1:] == batch_x[:,:-1]).all())

                    if x_ss > 0:
                        batch_x_output = model_waveform(batch_feat, batch_x_prev)[:, shift_rec_field:]
                    else:
                        batch_x_output = model_waveform(batch_feat, batch_x_prev, first=True)[:, model_waveform.receptive_field:]

                    # samples check
                    i = np.random.randint(0, batch_x_output.shape[0])
                    logging.info("%s" % (os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i]))))
                    #check_samples = batch_x[i,5:10].long()
                    #logging.info(torch.index_select(F.softmax(batch_x_output[i,5:10], dim=-1), 1, check_samples))
                    #logging.info(check_samples)

                    # handle short ending
                    batch_loss = 0
                    if len(idx_select) > 0:
                        logging.info('len_idx_select: '+str(len(idx_select)))
                        batch_loss_ce = 0
                        batch_loss_err = 0
                        for j in range(len(idx_select)):
                            k = idx_select[j]
                            slens_utt = slens_acc[k]
                            logging.info('%s %d' % (featfile[k], slens_utt))
                            batch_x_output_k = batch_x_output[k,:slens_utt]
                            batch_x_k = batch_x[k,:slens_utt]
                            batch_loss_ce += torch.mean(criterion_ce(batch_x_output_k, batch_x_k))
                            batch_loss_err += torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_output_k, dim=-1), F.one_hot(batch_x_k, num_classes=args.n_quantize).float()), -1))
                        batch_loss += batch_loss_ce
                        batch_loss_ce /= len(idx_select)
                        batch_loss_err /= len(idx_select)
                        total_eval_loss["eval/loss_ce"].append(batch_loss_ce.item())
                        total_eval_loss["eval/loss_err"].append(batch_loss_err.item())
                        loss_ce.append(batch_loss_ce.item())
                        loss_err.append(batch_loss_err.item())
                        if len(idx_select_full) > 0:
                            logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                            batch_x = torch.index_select(batch_x,0,idx_select_full)
                            batch_x_output = torch.index_select(batch_x_output,0,idx_select_full)
                        else:
                            logging.info("batch eval loss select %.3f %.3f (%.3f sec)" % (batch_loss_ce.item(), batch_loss_err.item(), time.time() - start))
                            iter_count += 1
                            total += time.time() - start
                            continue

                    batch_loss_ce_ = torch.mean(criterion_ce(batch_x_output.reshape(-1, args.n_quantize), batch_x.reshape(-1)).reshape(batch_x_output.shape[0], -1), -1)
                    batch_loss_err_ = torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_output, dim=-1), F.one_hot(batch_x, num_classes=args.n_quantize).float()), -1), -1)

                    batch_loss_ce = batch_loss_ce_.mean()
                    batch_loss_err = batch_loss_err_.mean()
                    total_eval_loss["eval/loss_ce"].append(batch_loss_ce.item())
                    total_eval_loss["eval/loss_err"].append(batch_loss_err.item())
                    loss_ce.append(batch_loss_ce.item())
                    loss_err.append(batch_loss_err.item())

                    logging.info("batch eval loss [%d] %d %d %d %d %d : %.3f %.3f %% (%.3f sec)" % (c_idx+1, max_slen, \
                        x_ss, x_bs, f_ss, f_bs, batch_loss_ce.item(), batch_loss_err.item(), time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            logging.info('sme')
            for key in total_eval_loss.keys():
                total_eval_loss[key] = np.mean(total_eval_loss[key])
                logging.info(f"(Steps: {iter_idx}) {key} = {total_eval_loss[key]:.4f}.")
            write_to_tensorboard(writer, iter_idx, total_eval_loss)
            total_eval_loss = defaultdict(list)
            eval_loss_ce = np.mean(loss_ce)
            eval_loss_ce_std = np.std(loss_ce)
            eval_loss_err = np.mean(loss_err)
            eval_loss_err_std = np.std(loss_err)
            logging.info("(EPOCH:%d) average evaluation loss = %.6f (+- %.6f) %.6f (+- %.6f) %% ;; (%.3f min., "\
                "%.3f sec / batch)" % (epoch_idx + 1, eval_loss_ce, eval_loss_ce_std, \
                    eval_loss_err, eval_loss_err_std, total / 60.0, total / iter_count))
            if (eval_loss_ce+eval_loss_ce_std) <= (min_eval_loss_ce+min_eval_loss_ce_std) \
                or (eval_loss_ce <= min_eval_loss_ce):
                min_eval_loss_ce = eval_loss_ce
                min_eval_loss_ce_std = eval_loss_ce_std
                min_eval_loss_err = eval_loss_err
                min_eval_loss_err_std = eval_loss_err_std
                min_idx = epoch_idx
                change_min_flag = True
            #else:
            #    epoch_min_flag = False
            if change_min_flag:
                logging.info("min_eval_loss = %.6f (+- %.6f) %.6f (+- %.6f) %% min_idx=%d" % (min_eval_loss_ce, \
                    min_eval_loss_ce_std, min_eval_loss_err, min_eval_loss_err_std, min_idx+1))
            #if ((epoch_idx + 1) % args.save_interval_epoch == 0) or (epoch_min_flag):
            #    logging.info('save epoch:%d' % (epoch_idx+1))
            #    save_checkpoint(args.expdir, model_waveform, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1)
            if args.init:
               exit()
            logging.info('save epoch:%d' % (epoch_idx+1))
            save_checkpoint(args.expdir, model_waveform, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1)
            total = 0
            iter_count = 0
            loss_ce = []
            loss_err = []
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
                    del_index_utt, max_slen, idx_select, idx_select_full, slens_acc = next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

            x_es = x_ss+x_bs
            f_es = f_ss+f_bs
            logging.info(f'{x_ss} {x_bs} {x_es} {f_ss} {f_bs} {f_es} {max_slen}')
            if x_ss > 0:
                if x_es <= max_slen:
                    batch_x_prev = batch_x[:,x_ss-shift_rec_field-1:x_es-1]
                    batch_feat = batch_feat[:,f_ss-shift_rec_field_frm:f_es]
                    batch_x = batch_x[:,x_ss:x_es]
                else:
                    batch_x_prev = batch_x[:,x_ss-shift_rec_field-1:-1]
                    batch_feat = batch_feat[:,f_ss-shift_rec_field_frm:]
                    batch_x = batch_x[:,x_ss:]
            #    assert((batch_x_prev[:,shift_rec_field+1:] == batch_x[:,:-1]).all())
            else:
                batch_x_prev = F.pad(batch_x[:,:x_es-1], (model_waveform.receptive_field+1, 0), "constant", args.n_quantize // 2)
                batch_feat = batch_feat[:,:f_es]
                batch_x = batch_x[:,:x_es]
            #    assert((batch_x_prev[:,model_waveform.receptive_field+1:] == batch_x[:,:-1]).all())

            if x_ss > 0:
                batch_x_output = model_waveform(batch_feat, batch_x_prev, do=True)[:, shift_rec_field:]
            else:
                batch_x_output = model_waveform(batch_feat, batch_x_prev, first=True, do=True)[:, model_waveform.receptive_field:]

            # samples check
            i = np.random.randint(0, batch_x_output.shape[0])
            logging.info("%s" % (os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i]))))
            #with torch.no_grad():
            #    i = np.random.randint(0, batch_x_output.shape[0])
            #    logging.info("%s" % (os.path.join(os.path.basename(os.path.dirname(featfile[i])),os.path.basename(featfile[i]))))
            #    check_samples = batch_x[i,5:10].long()
            #    logging.info(torch.index_select(F.softmax(batch_x_output[i,5:10], dim=-1), 1, check_samples))
            #    logging.info(check_samples)

            # handle short ending
            batch_loss = 0
            if len(idx_select) > 0:
                logging.info('len_idx_select: '+str(len(idx_select)))
                batch_loss_ce = 0
                batch_loss_err = 0
                for j in range(len(idx_select)):
                    k = idx_select[j]
                    slens_utt = slens_acc[k]
                    logging.info('%s %d' % (featfile[k], slens_utt))
                    batch_x_output_k = batch_x_output[k,:slens_utt]
                    batch_x_k = batch_x[k,:slens_utt]
                    batch_loss_ce += torch.mean(criterion_ce(batch_x_output_k, batch_x_k))
                    batch_loss_err += torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_output_k, dim=-1), F.one_hot(batch_x_k, num_classes=args.n_quantize).float()), -1))
                batch_loss += batch_loss_ce
                batch_loss_ce /= len(idx_select)
                batch_loss_err /= len(idx_select)
                total_train_loss["train/loss_ce"].append(batch_loss_ce.item())
                total_train_loss["train/loss_err"].append(batch_loss_err.item())
                loss_ce.append(batch_loss_ce.item())
                loss_err.append(batch_loss_err.item())
                if len(idx_select_full) > 0:
                    logging.info('len_idx_select_full: '+str(len(idx_select_full)))
                    batch_x = torch.index_select(batch_x,0,idx_select_full)
                    batch_x_output = torch.index_select(batch_x_output,0,idx_select_full)
                else:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_waveform.parameters(), 10)
                    optimizer.step()

                    logging.info("batch loss select %.3f %.3f (%.3f sec)" % (batch_loss_ce.item(), batch_loss_err.item(), time.time() - start))
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

            # loss
            batch_loss_ce_ = torch.mean(criterion_ce(batch_x_output.reshape(-1, args.n_quantize), batch_x.reshape(-1)).reshape(batch_x_output.shape[0], -1), -1)
            batch_loss_err_ = torch.mean(torch.sum(100*criterion_l1(F.softmax(batch_x_output, dim=-1), F.one_hot(batch_x, num_classes=args.n_quantize).float()), -1), -1)

            batch_loss_ce = batch_loss_ce_.mean()
            batch_loss_err = batch_loss_err_.mean()
            total_train_loss["train/loss_ce"].append(batch_loss_ce.item())
            total_train_loss["train/loss_err"].append(batch_loss_err.item())
            loss_ce.append(batch_loss_ce.item())
            loss_err.append(batch_loss_err.item())

            batch_loss += batch_loss_ce_.sum()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_waveform.parameters(), 10)
            optimizer.step()

            logging.info("batch loss [%d] %d %d %d %d %d : %.3f %.3f %% (%.3f sec)" % (c_idx+1, max_slen, x_ss, x_bs, \
                f_ss, f_bs, batch_loss_ce.item(), batch_loss_err.item(), time.time() - start))
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


    # save final model
    model_waveform.cpu()
    torch.save({"model_waveform": model_waveform.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()

