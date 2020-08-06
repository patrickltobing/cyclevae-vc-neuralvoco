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
import librosa

from utils import find_files
from utils import read_txt, read_hdf5, shape_hdf5
from vcneuvoco import DSWNV, decode_mu_law

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
    if len(batch_list[0].shape) > 1:
        n_feats = batch_list[0].shape[-1]
        batch_pad = np.zeros((batch_size, maxlen, n_feats))
    else:
        batch_pad = np.zeros((batch_size, maxlen))
    for idx, batch in enumerate(batch_list):
        batch_pad[idx, :batch.shape[0]] = batch

    return batch_pad


#def decode_generator(wav_list, feat_list, upsampling_factor=120, string_path='/feat_mceplf0cap', batch_size=1):
def decode_generator(feat_list, upsampling_factor=120, string_path='/feat_mceplf0cap', batch_size=1):
    """DECODE BATCH GENERATOR

    Args:
        wav_list (str): list including wav files
        batch_size (int): batch size in decoding
        upsampling_factor (int): upsampling factor

    Return:
        (object): generator instance
    """
    with torch.no_grad():
        # sort with the wav length
        #shape_list = [length_wav(f) for f in wav_list]
        shape_list = [shape_hdf5(f, string_path)[0] for f in feat_list]
        idx = np.argsort(shape_list)
        #wav_list = [wav_list[i] for i in idx]
        feat_list = [feat_list[i] for i in idx]

        # divide into batch list
        #n_batch = math.ceil(len(wav_list) / batch_size)
        n_batch = math.ceil(len(feat_list) / batch_size)
        #batch_wav_lists = np.array_split(wav_list, n_batch)
        #batch_wav_lists = [f.tolist() for f in batch_wav_lists]
        batch_feat_lists = np.array_split(feat_list, n_batch)
        batch_feat_lists = [f.tolist() for f in batch_feat_lists]

        #for batch_wav_list, batch_feat_list in zip(batch_wav_lists, batch_feat_lists):
        for batch_feat_list in batch_feat_lists:
            #batch_x = []
            batch_feat = []
            n_samples_list = []
            feat_ids = []
            #for wav_file, featfile in zip(batch_wav_list, batch_feat_list):
            for featfile in batch_feat_list:
                # load waveform
            #    x, _ = sf.read(wav_file, dtype=np.float32)
                #_, x = wavfile.read(wav_file)
                #x = np.array(x, dtype=np.float64)
            #    x = x[:x.shape[0]-(x.shape[0]%upsampling_factor)]
                feat = read_hdf5(featfile, string_path)
                logging.info(feat[100])
                logging.info(featfile)
                #feat = np.c_[feat[:,0:2], feat[:,2:3]*(-np.exp(feat[:,3:6])), feat[:,6:]]
                #logging.info(feat[100])

                # append to list
            #    batch_x += [x]
                batch_feat += [feat]
                n_samples_list += [feat.shape[0]*upsampling_factor]
                feat_ids += [os.path.basename(featfile).replace(".h5", "")]

            # convert list to ndarray
            #batch_x = pad_list(batch_x)
            batch_feat = pad_list(batch_feat)

            # convert to torch variable
            #batch_x = torch.FloatTensor(batch_x)
            batch_feat = torch.FloatTensor(batch_feat)
            if torch.cuda.is_available():
                batch_feat = batch_feat.cuda()

            #yield feat_ids, (batch_x, batch_feat, n_samples_list)
            yield feat_ids, (batch_feat, n_samples_list)


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    #parser.add_argument("--waveforms", required=True,
    #                    type=str, help="list or directory of wav files")
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
    #if os.path.isdir(args.waveforms):
    #    wav_list = sorted(find_files(args.waveforms, "*.wav"))
    #elif os.path.isfile(args.waveforms):
    #    wav_list = read_txt(args.waveforms)
    #else:
    #    logging.error("--waveforms should be directory or list.")
    #    sys.exit(1)
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    #wav_lists = np.array_split(wav_list, args.n_gpus)
    #wav_lists = [f_list.tolist() for f_list in wav_lists]
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]

    # define gpu decode function
    #def gpu_decode(wav_list, feat_list, gpu):
    def gpu_decode(feat_list, gpu):
        with torch.cuda.device(gpu):
            with torch.no_grad():
                if 'mel' in config.string_path:
                    n_aux=config.mcep_dim
                else:
                    n_aux=config.mcep_dim+config.excit_dim
                model_waveform = DSWNV(
                    n_aux=n_aux,
                    upsampling_factor=config.upsampling_factor,
                    hid_chn=config.hid_chn,
                    skip_chn=config.skip_chn,
                    kernel_size=config.kernel_size,
                    aux_kernel_size=config.kernel_size_wave,
                    aux_dilation_size=config.dilation_size_wave,
                    dilation_depth=config.dilation_depth,
                    dilation_repeat=config.dilation_repeat,
                    n_quantize=config.n_quantize)
                #model_waveform = DSWNV(
                #    n_quantize=config.n_quantize,
                #    n_aux=config.n_aux,
                #    hid_chn=config.hid_chn,
                #    skip_chn=config.skip_chn,
                #    dilation_depth=config.dilation_depth,
                #    dilation_repeat=config.dilation_repeat,
                #    kernel_size=config.kernel_size,
                #    aux_kernel_size=config.aux_kernel_size,
                #    aux_dilation_size=config.aux_dilation_size,
                #    audio_in_flag=config.audio_in,
                #    wav_conv_flag=config.wav_conv_flag,
                #    upsampling_factor=config.upsampling_factor)
                logging.info(model_waveform)
                model_waveform.cuda()
                #model_waveform.load_state_dict(torch.load(args.checkpoint)["model"])
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
                    string_path=string_path)
                    #wav_list,

                # decode
                time_sample = []
                n_samples = []
                n_samples_t = []
                count = 0
                #for feat_ids, (batch_x, batch_feat, n_samples_list) in generator:
                for feat_ids, (batch_feat, n_samples_list) in generator:
                    logging.info("decoding start")
                    start = time.time()
                    #logging.info(batch_x.shape)
                    logging.info(batch_feat.shape)

                    batch_x_prev = torch.zeros((batch_feat.shape[0], 1)).cuda().fill_(config.n_quantize//2).long()
                    logging.info(batch_x_prev)

                    samples = model_waveform.batch_fast_generate(batch_x_prev, batch_feat, n_samples_list)
                    #samples = model_waveform.batch_fast_generate(batch_x_prev, batch_feat.transpose(1,2), n_samples_list)
                    #logging.info(samples.shape)

                    #samples_src_list = batch_x.data.numpy()
                    samples_list = samples

                    time_sample.append(time.time()-start)
                    n_samples.append(max(n_samples_list))
                    n_samples_t.append(max(n_samples_list)*len(n_samples_list))

                    #for feat_id, samples_src, samples, samples_len in zip(feat_ids, samples_src_list, samples_list, n_samples_list):
                    for feat_id, samples, samples_len in zip(feat_ids, samples_list, n_samples_list):
                        #wav_src = samples_src[:samples_len]
                        wav = np.clip(decode_mu_law(samples[:samples_len], config.n_quantize), -1, 1)
                        #outpath = args.outdir + "/" + feat_id + "_src.wav"
                        #sf.write(outpath, wav_src, args.fs, "PCM_16")
                        #logging.info("wrote %s." % (outpath))
                        #outpath = args.outdir + "/" + feat_id + "_gen.wav"
                        outpath = args.outdir + "/" + feat_id + ".wav"
                        sf.write(outpath, wav, args.fs, "PCM_16")
                        logging.info("wrote %s." % (outpath))

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
    #for i, (wav_list, feat_list) in enumerate(zip(wav_lists, feat_lists)):
    for i, feat_list in enumerate(feat_lists):
        #p = mp.Process(target=gpu_decode, args=(wav_list, feat_list, gpu,))
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
