#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import math
import os
import sys
from distutils.util import strtobool

import matplotlib
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.nn.functional as F

from utils import find_files
from utils import read_hdf5
from utils import read_txt
from utils import check_hdf5
from utils import write_hdf5

import matplotlib.pyplot as plt

from vcneuvoco import SPKID_TRANSFORM_LAYER

matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

MIN_CLAMP = -103
MAX_CLAMP = 85

VERBOSE = 1


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--model", required=True,
                        type=str, help="GRU_RNN model file")
    parser.add_argument("--config", required=True,
                        type=str, help="GRU_RNN configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    # other setting
    #parser.add_argument("--GPU_device", default=None,
    #                    type=int, help="selection of GPU device")
    #parser.add_argument("--GPU_device_str", default=None,
    #                    type=str, help="selection of GPU device")
    parser.add_argument("--verbose", default=VERBOSE,
                        type=int, help="log level")
    args = parser.parse_args()

    #if args.GPU_device is not None or args.GPU_device_str is not None:
    #    os.environ["CUDA_DEVICE_ORDER"]		= "PCI_BUS_ID"
    #    if args.GPU_device_str is None:
    #        os.environ["CUDA_VISIBLE_DEVICES"]	= str(args.GPU_device)
    #    else:
    #        os.environ["CUDA_VISIBLE_DEVICES"]	= args.GPU_device_str
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S', filemode='w',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S', filemode='w',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S', filemode='w',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # load config
    config = torch.load(args.config)

    spk_list = config.spk_list.split('@')
    n_spk = len(spk_list)

    model_epoch = os.path.basename(args.model).split('.')[0].split('-')[1]
    logging.info('epoch: '+model_epoch)

    device = torch.device("cpu")
    #with torch.cuda.device(0):
    # define model and load parameters
    with torch.no_grad():
        model_spkidtr = SPKID_TRANSFORM_LAYER(
            n_spk=n_spk,
            emb_dim=config.emb_spk_dim_ti,
            n_weight_emb=config.n_weight_emb,
            conv_emb_flag=True,
            spkidtr_dim=config.spkidtr_dim)
        logging.info(model_spkidtr)
        model_spkidtr.load_state_dict(torch.load(args.model, map_location=device)["model_spkidtr"])
        model_spkidtr.eval()
        for param in model_spkidtr.parameters():
            param.requires_grad = False

        #feat = torch.LongTensor(np.arange(n_spk)).cuda().unsqueeze(0)
        feat = torch.LongTensor(np.arange(n_spk)).unsqueeze(0)
        logging.info(feat)

        logging.info(spk_list)

        colormap = np.array(['b', 'r'])
        #colormap = np.array(['b'])
        male = ['bdl', 'p237', 'p245', 'p251', 'p252', 'p259', 'p274', 'p304', 'p311', 'p326', 'p345', 'p360', 'p363', \
                'p226', 'p227', 'p232', 'p237', 'p241', 'p243', 'p245', 'p246', 'p247', 'p251', 'p252', 'p254', 'p255', 'p256', 'p258', 'p259', 'p260', 'p263', 'p270', 'p271', 'p272', 'p273', 'p274', 'p275', 'p278', 'p279', 'p281', 'p284', 'p285', 'p286', 'p287', 'p292', 'p298', 'p302', 'p304', 'p311', 'p315', 'p316', 'p326', \
                    'SEM1', 'SEM2', 'TFM1', 'TGM1', 'TMM1', 'TEM1', 'TEM2', \
                        'VCC2SM1', 'VCC2SM2', 'VCC2SM3', 'VCC2TM1', 'VCC2TM2', 'VCC2SM4']
        female = ['slt', 'p231', 'p238', 'p248', 'p253', 'p264', 'p265', 'p266', 'p276', 'p305', 'p308', 'p318', 'p335', \
                    'p225', 'p228', 'p229', 'p230', 'p231', 'p233', 'p234', 'p236', 'p238', 'p239', 'p240', 'p244', 'p248', 'p249', 'p250', 'p253', 'p257', 'p261', 'p262', 'p264', 'p265', 'p266', 'p267', 'p268', 'p269', 'p276', 'p277', 'p282', 'p283', 'p288', 'p293', 'p294', 'p295', 'p297', 'p299', 'p300', 'p301', 'p303', 'p305', \
                    'SEF1', 'SEF2', 'TEF1', 'TEF2', 'TFF1', 'TGF1', 'TMF1', \
                        'VCC2SF1', 'VCC2SF2', 'VCC2SF3', 'VCC2TF1', 'VCC2TF2', 'VCC2SF4']
        gender = []
        for i in range(n_spk):
            #gender.append(0)
            if spk_list[i] in male:
                gender.append(0)
            elif spk_list[i] in female:
                gender.append(1)
            else:
                logging.info('error %s not in gender list' % (spk_list[i]))
                exit()

        z = F.tanhshrink(torch.clamp(model_spkidtr.conv(model_spkidtr.conv_emb(F.one_hot(feat, num_classes=n_spk).float().transpose(1,2))), min=MIN_CLAMP, max=MAX_CLAMP)).transpose(1,2).cpu().data.numpy().astype(np.float64)
        #z = F.tanhshrink(model_spkidtr.conv(F.one_hot(feat, num_classes=n_spk).float().transpose(1,2))).transpose(1,2).cpu().data.numpy().astype(np.float64)
        #z = F.tanhshrink(torch.clamp(model_spkidtr.conv(F.one_hot(feat, num_classes=self.n_spk).float().transpose(1,2)), min=-32, max=32)).transpose(1,2)
        logging.info(z)

        logging.info(args.outdir)

        #plt.rcParams["figure.figsize"] = (20,11.25) #1920x1080
        plt.rcParams["figure.figsize"] = (14.229166667,11.25) #1366x1080
        #plt.rcParams["figure.figsize"] = (11.25,11.25) #1080x1080
        #plt.rcParams["figure.figsize"] = (14.229166667,14.229166667) #1366x1366

        logging.info("spk-id spk-name x-coord y-coord")
        for i in range(n_spk):
            logging.info("%d %s %lf %lf", i+1, spk_list[i], z[0,i,0], z[0,i,1])

        #z = z.cpu().data.numpy()
        #z = z.data.numpy()
        logging.info(z.shape)
        x = z[0,:,0]
        y = z[0,:,1]
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=40, c=colormap[gender])
        #ax.scatter(x, y, s=80, c=colormap[gender])
        #ax.scatter(x, y, s=100, c=colormap[gender])
        #ax.scatter(x, y, s=120, c=colormap[gender])
        #ax.scatter(x, y, s=160, c=colormap[gender])
        for i, txt in enumerate(spk_list):
            #ax.annotate(txt, (x[i], y[i]))
            #ax.annotate(txt, (x[i], y[i]), weight='bold', size=16)
            ax.annotate(txt, (x[i], y[i]), size=14)
        #plt.xlim([-0.35, 0.05])
        plt.savefig(os.path.join(args.outdir, 'spk_map.png'))
        plt.close()

        #z_e = model_decoder_excit.spkidtr_conv(F.one_hot(feat, num_classes=n_spk).float().transpose(1,2)).transpose(1,2)
        ##z_e_rec = model_decoder_excit.spkidtr_deconv(z_e.transpose(1,2)).transpose(1,2)
        #logging.info(z_e)

        ##z_e = z_e.cpu().data.numpy()
        #z_e = z_e.data.numpy()
        #x = z_e[0,:,0]
        #y = z_e[0,:,1]
        #fig, ax = plt.subplots()
        #ax.scatter(x, y, s=40, c=colormap[gender])
        #for i, txt in enumerate(spk_list):
        #    ax.annotate(txt, (x[i], y[i]))
        #plt.savefig(os.path.join(args.outdir, 'excit.png'))
        #plt.close()

 
if __name__ == "__main__":
    main()
