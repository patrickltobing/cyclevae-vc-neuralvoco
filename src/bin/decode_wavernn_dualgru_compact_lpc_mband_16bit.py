#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

from distutils.util import strtobool
import argparse
import logging
import math
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy.io import wavfile
import torch
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

from utils import find_files
from utils import read_txt, read_hdf5, shape_hdf5
from vcneuvoco import GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF

import torch.nn.functional as F

from pqmf import PQMF

#import warnings
#warnings.filterwarnings('ignore')

#from torch.distributions.one_hot_categorical import OneHotCategorical
#import torch.nn.functional as F


def pad_list(batch_list, pad_value=0.0):
    """FUNCTION TO PAD VALUE

    Args:
        batch_list (list): list of batch, where the shape of i-th batch (T_i, C)
        pad_value (float): value to pad

    Return:
        (ndarray): padded batch with the shape (B, T_max, C)

    """
    batch_size = len(batch_list)
    maxlen = max([batch.shape[0] for batch in batch_list])
    #if len(batch_list[0].shape) > 1:
    #    n_feats = batch_list[0].shape[-1]
    #    batch_pad = np.zeros((batch_size, maxlen, n_feats))
    #else:
    #    batch_pad = np.zeros((batch_size, maxlen))
    #for idx, batch in enumerate(batch_list):
    #    batch_pad[idx, :batch.shape[0]] = batch
    #logging.info(maxlen)
    for idx, batch in enumerate(batch_list):
        if idx > 0:
            batch_pad = np.r_[batch_pad, np.expand_dims(np.pad(batch_list[idx], ((0, maxlen-batch_list[idx].shape[0]), (0, 0)), 'edge'), axis=0)]
        else:
            batch_pad = np.expand_dims(np.pad(batch_list[idx], ((0, maxlen-batch_list[idx].shape[0]), (0, 0)), 'edge'), axis=0)
    #    logging.info(batch_list[idx].shape)
    #    logging.info(np.expand_dims(np.pad(batch_list[idx], ((0, maxlen-batch_list[idx].shape[0]), (0, 0)), 'edge'), axis=0).shape)
    #    logging.info(batch_pad.shape)

    return batch_pad


def decode_generator(feat_list, upsampling_factor=120, string_path='/feat_mceplf0cap', batch_size=1, excit_dim=0, n_enc=None):
    """DECODE BATCH GENERATOR

    Args:
        wav_list (str): list including wav files
        batch_size (int): batch size in decoding
        upsampling_factor (int): upsampling factor

    Return:
        (object): generator instance
    """
    with torch.no_grad():
        shape_list = [shape_hdf5(f, string_path)[0] for f in feat_list]
        idx = np.argsort(shape_list)
        feat_list = [feat_list[i] for i in idx]

        # divide into batch list
        n_batch = math.ceil(len(feat_list) / batch_size)
        batch_feat_lists = np.array_split(feat_list, n_batch)
        batch_feat_lists = [f.tolist() for f in batch_feat_lists]

        if n_enc is not None:
            feats = [None]*n_enc
        for batch_feat_list in batch_feat_lists:
            batch_feat = []
            n_samples_list = []
            feat_ids = []
            for featfile in batch_feat_list:
                ## load waveform
                if n_enc is not None:
                    for i in range(n_enc):
                        feats[i] = read_hdf5(featfile, string_path+"-%d"%(i+1))
                    feat_sum = read_hdf5(featfile, string_path+"_sum")
                elif 'mel' in string_path:
                    if excit_dim > 0:
                        feat = np.c_[read_hdf5(featfile, '/feat_mceplf0cap')[:,:excit_dim], read_hdf5(featfile, string_path)]
                    else:
                        feat = read_hdf5(featfile, string_path)
                else:
                    feat = read_hdf5(featfile, string_path)

                # append to list
                if n_enc is not None:
                    for i in range(n_enc):
                        batch_feat += [feats[i]]
                    batch_feat += [feat_sum]
                else:
                    batch_feat += [feat]
                if n_enc is not None:
                    for i in range(n_enc):
                        n_samples_list += [feats[i].shape[0]*upsampling_factor]
                    n_samples_list += [feat_sum.shape[0]*upsampling_factor]
                else:
                    n_samples_list += [feat.shape[0]*upsampling_factor]
                if n_enc is not None:
                    for i in range(n_enc):
                        feat_ids += [os.path.basename(featfile).replace(".h5", "_%d"%(i+1))]
                    feat_ids += [os.path.basename(featfile).replace(".h5", "_sum")]
                else:
                    feat_ids += [os.path.basename(featfile).replace(".h5", "")]
                logging.info(feat_ids)

            # convert list to ndarray
            batch_feat = pad_list(batch_feat)

            # convert to torch variable
            batch_feat = torch.FloatTensor(batch_feat)
            if torch.cuda.is_available():
                batch_feat = batch_feat.cuda()

            yield feat_ids, (batch_feat, n_samples_list)


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of wav files")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="model file")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--fs", default=22050,
                        type=int, help="sampling rate")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="number of batch size in decoding")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    # other setting
    parser.add_argument("--string_path", default=None,
                        type=str, help="log interval")
    parser.add_argument("--intervals", default=4410,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--GPU_device", default=None,
                        type=int, help="selection of GPU device")
    parser.add_argument("--GPU_device_str", default=None,
                        type=str, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    parser.add_argument("--n_enc", default=None,
                        type=int, help="number of encoder in case of source-sep. net")
    args = parser.parse_args()

    if args.GPU_device is not None or args.GPU_device_str is not None:
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
        if args.GPU_device_str is None:
            os.environ["CUDA_VISIBLE_DEVICES"]  = str(args.GPU_device)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]  = args.GPU_device_str

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load config
    config = torch.load(args.config)
    logging.info(config)

    # get file list
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

    # define gpu decode function
    def gpu_decode(feat_list, gpu):
        with torch.cuda.device(gpu):
            with torch.no_grad():
                model_waveform = GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF(
                    feat_dim=config.mcep_dim+config.excit_dim,
                    upsampling_factor=config.upsampling_factor,
                    hidden_units=config.hidden_units_wave,
                    hidden_units_2=config.hidden_units_wave_2,
                    kernel_size=config.kernel_size_wave,
                    dilation_size=config.dilation_size_wave,
                    n_quantize=config.n_quantize,
                    causal_conv=config.causal_conv_wave,
                    right_size=config.right_size,
                    n_bands=config.n_bands,
                    pad_first=True,
                    lpc=config.lpc)
                logging.info(model_waveform)
                model_waveform.cuda()
                model_waveform.load_state_dict(torch.load(args.checkpoint)["model_waveform"])
                model_waveform.remove_weight_norm()
                model_waveform.eval()
                for param in model_waveform.parameters():
                    param.requires_grad = False
                torch.backends.cudnn.benchmark = True

                # define generator
                if args.string_path is None:
                    string_path = config.string_path
                else:
                    string_path = args.string_path
                logging.info(string_path)
                generator = decode_generator(
                    feat_list,
                    batch_size=args.batch_size,
                    upsampling_factor=config.upsampling_factor,
                    excit_dim=config.excit_dim,
                    n_enc=args.n_enc,
                    string_path=string_path)

                # decode
                time_sample = []
                n_samples = []
                n_samples_t = []
                count = 0
                pqmf = PQMF(config.n_bands).cuda()
                print(f'{pqmf.subbands} {pqmf.A} {pqmf.taps} {pqmf.cutoff_ratio} {pqmf.beta}')
                for feat_ids, (batch_feat, n_samples_list) in generator:
                    logging.info("decoding start")
                    start = time.time()
                    logging.info(batch_feat.shape)

                    #batch_feat = F.pad(batch_feat.transpose(1,2), (model_waveform.pad_left,model_waveform.pad_right), "replicate").transpose(1,2)
                    samples = model_waveform.generate(batch_feat)
                    logging.info(samples.shape) # B x n_bands x T//n_bands
                    samples = pqmf.synthesis(samples)[:,0].cpu().data.numpy() # B x 1 x T --> B x T
                    logging.info(samples.shape)

                    samples_list = samples

                    time_sample.append(time.time()-start)
                    n_samples.append(max(n_samples_list))
                    n_samples_t.append(max(n_samples_list)*len(n_samples_list))

                    for feat_id, samples, samples_len in zip(feat_ids, samples_list, n_samples_list):
                        #wav = np.clip(samples[:samples_len], -1, 1)
                        wav = np.clip(samples[:samples_len], -1, 0.999969482421875)
                        outpath = os.path.join(args.outdir, feat_id+".wav")
                        sf.write(outpath, wav, args.fs, "PCM_16")
                        logging.info("wrote %s." % (outpath))
                    #break

                    #figname = os.path.join(args.outdir, feat_id+"_wav.png")
                    #plt.subplot(2, 1, 1)
                    #plt.plot(wav_src)
                    #plt.title("source wave")
                    #plt.subplot(2, 1, 2)
                    #plt.plot(wav)
                    #plt.title("generated wave")
                    #plt.tight_layout()
                    #plt.savefig(figname)
                    #plt.close()
                        
                    count += 1
                    #if count >= 3:
                    #if count >= 6:
                    #if count >= 1:
                    #    break

                logging.info("average time / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % (\
                    sum(time_sample)/sum(n_samples), sum(n_samples), sum(n_samples)/(1000*sum(time_sample))))
                logging.info("average throughput / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % (\
                sum(time_sample)/sum(n_samples_t), sum(n_samples_t), sum(n_samples_t)/(1000*sum(time_sample))))

    # parallel decode
    processes = []
    gpu = 0
    for i, feat_list in enumerate(feat_lists):
        p = mp.Process(target=gpu_decode, args=(feat_list, gpu,))
        p.start()
        processes.append(p)
        gpu += 1
        if (i + 1) % args.n_gpus == 0:
            gpu = 0

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
