#!/usr/bin/env python
'''Copyright (c) 2017-2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
'''
    based on dump_lpcnet.py
    modified for 16-bit output multiband wavernn with data-driven LPC
    by: Patrick Lumban Tobing (Nagoya University) on October 2020
'''

import argparse
import os
import sys

import torch
from mwdlpnet import GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF
from pqmf import PQMF

import numpy as np

#import h5py
#import re


def printVector(f, vector, name, dtype='float'):
    v = np.reshape(vector, (-1));
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    f.write('static const {} {}[{}] = {{\n   '.format(dtype, name, len(v)))
    for i in range(0, len(v)):
        f.write('{}'.format(v[i]))
        if (i!=len(v)-1):
            f.write(',')
        else:
            break;
        if (i%8==7):
            f.write("\n   ")
        else:
            f.write(" ")
    #print(v, file=f)
    f.write('\n};\n\n')
    return;

def printSparseVector(f, A, name):
    N = A.shape[0]
    W = np.zeros((0,))
    diag = np.concatenate([np.diag(A[:,:N]), np.diag(A[:,N:2*N]), np.diag(A[:,2*N:])])
    A[:,:N] = A[:,:N] - np.diag(np.diag(A[:,:N]))
    A[:,N:2*N] = A[:,N:2*N] - np.diag(np.diag(A[:,N:2*N]))
    A[:,2*N:] = A[:,2*N:] - np.diag(np.diag(A[:,2*N:]))
    printVector(f, diag, name + '_diag')
    idx = np.zeros((0,), dtype='int')
    for i in range(3*N//16):
        pos = idx.shape[0]
        idx = np.append(idx, -1)
        nb_nonzero = 0
        for j in range(N):
            if np.sum(np.abs(A[j, i*16:(i+1)*16])) > 1e-10:
                nb_nonzero = nb_nonzero + 1
                idx = np.append(idx, j)
                W = np.concatenate([W, A[j, i*16:(i+1)*16]])
        idx[pos] = nb_nonzero
    printVector(f, W, name)
    #idx = np.tile(np.concatenate([np.array([N]), np.arange(N)]), 3*N//16)
    printVector(f, idx, name + '_idx', dtype='int')
    return;


def main():
    parser = argparse.ArgumentParser()
    # mandatory arguments
    parser.add_argument("config_cycvae", metavar="string",
                        type=str, help="path of model config")
    parser.add_argument("model_cycvae", metavar="string",
                        type=str, help="path of model file")
    parser.add_argument("config", metavar="string",
                        type=str, help="path of model config")
    parser.add_argument("model", metavar="string",
                        type=str, help="path of model file")
    # optional arguments
    parser.add_argument("--c_out_file", "-cf", default="nnet_data.c", metavar="string",
                        type=str, help="c out file; default is nnet_data.c")
    parser.add_argument("--h_out_file", "-hf", default="nnet_data.h", metavar="string",
                        type=str, help="header out file; default is nnet_data.h")
    parser.add_argument("--c_cycvae_out_file", "-cf", default="nnet_cv_data.c", metavar="string",
                        type=str, help="c out file; default is nnet_cv_data.c")
    parser.add_argument("--h_cycvae_out_file", "-hf", default="nnet_cv_data.h", metavar="string",
                        type=str, help="header out file; default is nnet_cv_data.h")
    args = parser.parse_args()

    #set config and model
    config_cycvae = torch.load(args.config_cycvae)
    print(config_cycvae)
    spk_list = config_cycvae.spk_list.split('@')
    n_spk = len(spk_list)
    print(spk_list)
    config = torch.load(args.config)
    print(config)

    model_encoder_melsp = GRU_VAE_ENCODER(
        in_dim=config_cycvae.mel_dim,
        n_spk=n_spk,
        lat_dim=config_cycvae.lat_dim,
        hidden_layers=config_cycvae.hidden_layers_enc,
        hidden_units=config_cycvae.hidden_units_enc,
        kernel_size=config_cycvae.kernel_size_enc,
        dilation_size=config_cycvae.dilation_size_enc,
        causal_conv=config_cycvae.causal_conv_enc,
        pad_first=True,
        right_size=config_cycvae.right_size_enc)
    print.info(model_encoder_melsp)
    model_decoder_melsp = GRU_SPEC_DECODER(
        feat_dim=config_cycvae.lat_dim+config_cycvae.lat_dim_e,
        excit_dim=config_cycvae.excit_dim,
        out_dim=config_cycvae.mel_dim,
        n_spk=n_spk,
        aux_dim=n_spk,
        hidden_layers=config_cycvae.hidden_layers_dec,
        hidden_units=config_cycvae.hidden_units_dec,
        kernel_size=config_cycvae.kernel_size_dec,
        dilation_size=config_cycvae.dilation_size_dec,
        causal_conv=config_cycvae.causal_conv_dec,
        pad_first=True,
        right_size=config_cycvae.right_size_dec)
    print(model_decoder_melsp)
    model_encoder_excit = GRU_VAE_ENCODER(
        in_dim=config_cycvae.mel_dim,
        n_spk=n_spk,
        lat_dim=config_cycvae.lat_dim_e,
        hidden_layers=config_cycvae.hidden_layers_enc,
        hidden_units=config_cycvae.hidden_units_enc,
        kernel_size=config_cycvae.kernel_size_enc,
        dilation_size=config_cycvae.dilation_size_enc,
        causal_conv=config_cycvae.causal_conv_enc,
        pad_first=True,
        right_size=config_cycvae.right_size_enc)
    print(model_encoder_excit)
    model_decoder_excit = GRU_EXCIT_DECODER(
        feat_dim=config_cycvae.lat_dim_e,
        cap_dim=config_cycvae.cap_dim,
        n_spk=n_spk,
        aux_dim=n_spk,
        hidden_layers=config_cycvae.hidden_layers_lf0,
        hidden_units=config_cycvae.hidden_units_lf0,
        kernel_size=config_cycvae.kernel_size_lf0,
        dilation_size=config_cycvae.dilation_size_lf0,
        causal_conv=config_cycvae.causal_conv_lf0,
        pad_first=True,
        right_size=config_cycvae.right_size_lf0)
    print(model_decoder_excit)
    if (config_cycvae.spkidtr_dim > 0):
        model_spkidtr = SPKID_TRANSFORM_LAYER(
            n_spk=n_spk,
            spkidtr_dim=config_cycvae.spkidtr_dim)
        print(model_spkidtr)
    model_spk = GRU_SPK(
        n_spk=n_spk,
        feat_dim=config_cycvae.lat_dim+config_cycvae.lat_dim_e,
        hidden_units=32)
    print(model_spk)
    model_post = GRU_POST_NET(
        spec_dim=config_cycvae.mel_dim,
        excit_dim=config_cycvae.excit_dim+config_cycvae.cap_dim+1,
        n_spk=n_spk,
        aux_dim=n_spk,
        hidden_layers=config_cycvae.hidden_layers_post,
        hidden_units=config_cycvae.hidden_units_post,
        kernel_size=config_cycvae.kernel_size_post,
        dilation_size=config_cycvae.dilation_size_post,
        causal_conv=config_cycvae.causal_conv_post,
        pad_first=True,
        right_size=config_cycvae.right_size_post,
        res=True,
        laplace=True)
    print(model_post)
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
        mid_out_flag=False,
        lpc=config.lpc)
    print(model)
    model_encoder_melsp.load_state_dict(torch.load(args.model_cycvae)["model_encoder_melsp"])
    model_decoder_melsp.load_state_dict(torch.load(args.model_cycvae)["model_decoder_melsp"])
    model_encoder_excit.load_state_dict(torch.load(args.model_cycvae)["model_encoder_excit"])
    model_decoder_excit.load_state_dict(torch.load(args.model_cycvae)["model_decoder_excit"])
    if (config.spkidtr_dim > 0):
        model_spkidtr.load_state_dict(torch.load(args.model_cycvae)["model_spkidtr"])
    model_spk.load_state_dict(torch.load(args.model_cycvae)["model_spk"])
    model_post.load_state_dict(torch.load(args.model_cycvae)["model_post"])
    model.load_state_dict(torch.load(args.model)["model_waveform"])
    model_encoder_melsp.remove_weight_norm()
    model_decoder_melsp.remove_weight_norm()
    model_encoder_excit.remove_weight_norm()
    model_decoder_excit.remove_weight_norm()
    if (config.spkidtr_dim > 0):
        model_spkidtr.remove_weight_norm()
    model_spk.remove_weight_norm()
    model_post.remove_weight_norm()
    model.remove_weight_norm()
    model_encoder_melsp.eval()
    model_decoder_melsp.eval()
    model_encoder_excit.eval()
    model_decoder_excit.eval()
    if (config.spkidtr_dim > 0):
        model_spkidtr.eval()
    model_spk.eval()
    model_post.eval()
    model.eval()
    for param in model_encoder_melsp.parameters():
        param.requires_grad = False
    for param in model_decoder_melsp.parameters():
        param.requires_grad = False
    for param in model_encoder_excit.parameters():
        param.requires_grad = False
    for param in model_decoder_excit.parameters():
        param.requires_grad = False
    if (config.spkidtr_dim > 0):
        for param in model_spkidtr.parameters():
            param.requires_grad = False
    for param in model_spk.parameters():
        param.requires_grad = False
    for param in model_post.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        param.requires_grad = False

    ## Multiband WaveRNN with data-driven LPC (MWDLP)
    cfile = args.c_out_file
    hfile = args.h_out_file
    
    f = open(cfile, 'w')
    hf = open(hfile, 'w')
    
    f.write('/*This file is automatically generated from a PyTorch model*/\n\n')
    f.write('#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif\n\n#include "nnet.h"\n#include "{}"\n\n'.format(hfile))
    
    hf.write('/*This file is automatically generated from a PyTorch model*/\n\n')
    hf.write('#ifndef RNN_MWDLP_DATA_H\n#define RNN_MWDLP_DATA_H\n\n#include "nnet.h"\n\n')
    
    cond_size = model.s_dim
    #PyTorch & Keras = (emb_dict_size,emb_size)
    embed_size = model.wav_dim
    embed_size_bands = model.wav_dim_bands
    
    max_rnn_neurons = 1
    #PyTorch = (hidden_dim*3,in_dim*3)
    #Keras = (in_dim*3,hidden_dim*3)

    #embedding coarse and fine
    E_coarse = model.embed_c_wav.weight.numpy()
    E_fine = model.embed_f_wav.weight.numpy()

    #gru_main weight_input
    W = model.gru.weight_ih_l0.permute(1,0).numpy()
    #dump coarse_embed pre-computed input_weight contribution for all classes
    name = 'gru_a_embed_coarse'
    print("printing layer " + name)
    W_bands = W[cond_size:-embed_size_bands]
    # n_bands x embed_dict_size x hidden_size
    weights = np.expand_dims(np.dot(E_coarse, W_bands[:embed_size]), axis=0)
    for i in range(1,model.n_bands):
        weights = np.r_[weights, np.expand_dims(np.dot(E_coarse, W_bands[embed_size*i:embed_size*(i+1)]), axis=0)]
    printVector(f, weights, name + '_weights')
    f.write('const EmbeddingLayer {} = {{\n   {}_weights,\n   {}, {}\n}};\n\n'
            .format(name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const EmbeddingLayer {};\n\n'.format(name));
    #dump coarse_fine pre-computed input_weight contribution for all classes
    name = 'gru_a_embed_fine'
    print("printing layer " + name)
    W_bands = W[-embed_size_bands:]
    # n_bands x embed_dict_size x hidden_size
    weights = np.expand_dims(np.dot(E_fine, W_bands[:embed_size]), axis=0)
    for i in range(1,model.n_bands):
        weights = np.r_[weights, np.expand_dims(np.dot(E_fine, W_bands[embed_size*i:embed_size*(i+1)]), axis=0)]
    printVector(f, weights, name + '_weights')
    f.write('const EmbeddingLayer {} = {{\n   {}_weights,\n   {}, {}\n}};\n\n'
            .format(name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const EmbeddingLayer {};\n\n'.format(name));
    #dump input cond-part weight and input bias
    name = 'gru_a_dense_feature'
    print("printing layer " + name)
    weights = W[:cond_size]
    bias = model.gru.bias_ih_l0.numpy()
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_LINEAR\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));

    #dump gru_coarse input weight cond-part and input bias
    name = 'gru_b_dense_feature'
    print("printing layer " + name)
    W = model.gru_2.weight_ih_l0.permute(1,0).numpy()
    weights = W[:cond_size]
    bias = model.gru_2.bias_ih_l0.numpy()
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_LINEAR\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));
    #dump gru_coarse input weight state-part
    name = 'gru_b_dense_feature_state'
    print("printing layer " + name)
    weights = W[cond_size:]
    bias = np.zeros(W.shape[1])
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_LINEAR\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));

    #gru_fine weight_input
    W = model.gru_f.weight_ih_l0.permute(1,0).numpy()
    #dump coarse_embed pre-computed input_weight contribution for all classes
    name = 'gru_c_embed_coarse'
    print("printing layer " + name)
    W_bands = W[cond_size:-model.hidden_units_2]
    # n_bands x embed_dict_size x hidden_size
    weights = np.expand_dims(np.dot(E_coarse, W_bands[:embed_size]), axis=0)
    for i in range(1,model.n_bands):
        weights = np.r_[weights, np.expand_dims(np.dot(E_coarse, W_bands[embed_size*i:embed_size*(i+1)]), axis=0)]
    printVector(f, weights, name + '_weights')
    f.write('const EmbeddingLayer {} = {{\n   {}_weights,\n   {}, {}\n}};\n\n'
            .format(name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const EmbeddingLayer {};\n\n'.format(name));
    #dump input cond-part weight and input bias
    name = 'gru_c_dense_feature'
    print("printing layer " + name)
    weights = W[:cond_size]
    bias = model.gru_f.bias_ih_l0.numpy()
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_LINEAR\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));
    #dump input state-part weight
    name = 'gru_c_dense_feature_state'
    print("printing layer " + name)
    weights = W[-model.hidden_units_2:]
    bias = np.zeros(W.shape[1])
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_LINEAR\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));
  
    #PyTorch = (out,in,ks) / (out,in)
    #Keras = (ks,in,out) / (in,out)

    #dump scale_in
    name = 'feature_norm'
    print("printing layer " + name + " of type " + model.scale_in.__class__.__name__)
    weights = model.scale_in.weight.permute(2,1,0)[0].numpy() #it's defined as conv1d with ks=1 on the model
    bias = model.scale_in.bias.numpy()
    std = 1.0/np.diag(weights) #in training script, diagonal square weights matrix defined as 1/std
    mean = (-bias)*std #in training script, bias defined as -mean/std
    printVector(f, mean, name + '_mean')
    printVector(f, std, name + '_std')
    f.write('const DenseLayer {} = {{\n   {}_mean,\n   {}_std,\n   {}\n}};\n\n'
            .format(name, name, name, bias.shape[0]))
    hf.write('extern const NormLayer {};\n\n'.format(name));

    #dump segmental_conv
    name = "feature_conv"
    #FIXME: make model format without sequential for two-sided/causal conv
    if model.right_size <= 0:
        print("printing layer " + name + " of type " + model.conv.conv[0].__class__.__name__)
        weights = model.conv.conv[0].weight.permute(2,1,0).numpy()
        bias = model.conv.conv[0].bias.numpy()
    else:
        print("printing layer " + name + " of type " + model.conv.conv.__class__.__name__)
        weights = model.conv.conv.weight.permute(2,1,0).numpy()
        bias = model.conv.conv.bias.numpy()
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const Conv1DLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, {}, ACTIVATION_LINEAR\n}};\n\n'
            .format(name, name, name, weights.shape[1], weights.shape[0], weights.shape[2]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[2]))
    hf.write('#define {}_STATE_SIZE ({}*{})\n'.format(name.upper(), weights.shape[1],
        model.pad_left+1+model.pad_right-1))
    hf.write('#define {}_DELAY {}\n'.format(name.upper(), model.pad_right))
    hf.write('extern const Conv1DLayer {};\n\n'.format(name));

    #dump dense_relu
    name = 'feature_dense'
    print("printing layer " + name + " of type " + model.conv_s_c[0].__class__.__name__)
    weights = model.conv_s_c[0].weight.permute(2,1,0)[0].numpy() #it's defined as conv1d with ks=1 on the model
    bias = model.conv_s_c[0].bias.numpy()
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_RELU\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));

    #dump sparse_main_gru
    name = 'sparse_gru_a'
    print("printing layer " + name + " of type sparse " + model.gru.__class__.__name__)
    weights = model.gru.weight_hh_l0.transpose(0,1).numpy()
    bias = model.gru.bias_hh_l0.numpy()
    printSparseVector(f, weights, name + '_recurrent_weights')
    printVector(f, bias, name + '_bias')
    activation = 'TANH'
    reset_after = 1
    neurons = weights.shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const SparseGRULayer {} = {{\n   {}_bias,\n   {}_recurrent_weights_diag,\n   {}_recurrent_weights,\n   '\
        '{}_recurrent_weights_idx,\n   {}, ACTIVATION_{}, {}\n}};\n\n'.format(name, name, name, name, name,
            weights.shape[1]//3, activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), weights.shape[1]//3))
    hf.write('extern const SparseGRULayer {};\n\n'.format(name));

    #dump dense_gru_coarse
    name = "gru_b"
    print("printing layer " + name + " of type " + model.gru_2.__class__.__name__)
    weights_ih = model.gru_2.weight_ih_l0.transpose(0,1)[cond_size:].numpy()
    weights_hh = model.gru_2.weight_hh_l0.transpose(0,1).numpy()
    bias = model.gru_2.bias_hh_l0
    printVector(f, weights_ih, name + '_weights')
    printVector(f, weights_hh, name + '_recurrent_weights')
    printVector(f, bias, name + '_bias')
    activation = 'TANH'
    reset_after = 1
    neurons = weights_hh.shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const GRULayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_recurrent_weights,\n   {}, {}, ACTIVATION_{}, '\
        '{}\n}};\n\n'.format(name, name, name, name, weights_ih.shape[0], weights_hh.shape[1]//3,
            activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights_hh.shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), weights_hh.shape[1]//3))
    hf.write('extern const GRULayer {};\n\n'.format(name));

    #dump dense_gru_fine
    name = "gru_c"
    print("printing layer " + name + " of type " + model.gru_f.__class__.__name__)
    weights_ih = model.gru_f.weight_ih_l0.transpose(0,1)[-model.hidden_units_2:].numpy()
    weights_hh = model.gru_f.weight_hh_l0.transpose(0,1).numpy()
    bias = model.gru_f.bias_hh_l0
    printVector(f, weights_ih, name + '_weights')
    printVector(f, weights_hh, name + '_recurrent_weights')
    printVector(f, bias, name + '_bias')
    activation = 'TANH'
    reset_after = 1
    neurons = weights_hh.shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const GRULayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_recurrent_weights,\n   {}, {}, ACTIVATION_{}, '\
        '{}\n}};\n\n'.format(name, name, name, name, weights_ih.shape[0], weights_hh.shape[1]//3,
            activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights_hh.shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), weights_hh.shape[1]//3))
    hf.write('extern const GRULayer {};\n\n'.format(name));

    #dump dual_fc_coarse
    name = "dual_fc_coarse"
    print("printing layer " + name)
    weights = model.out.conv.weight.permute(2,1,0)[0].numpy()
    bias = model.out.conv.bias.numpy()
    factors = (0.5*torch.exp(model.out.fact.weight[0])).numpy()
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    ## Previous implementation (as in ICASSP 2021) uses shared factors between bands for signs and mags [for data-driven LPC],
    ## though the mid-output uses band-dependent factors.
    ## These factors should be made band-dependent for all signs, mags, and mid-output for proper modeling as current implementation.
    ## Further, with 10-bit mu-law coarse-fine output, there is no need to use mid-output because
    ## this layer can directly generates the band-dependent 32-dim (sqrt(1024)) logits outputs
    ## instead of 256-dim (16-bit coarse-fine) or 512-dim (9-bit mu-law) which needs smaller mid-output for reducing computational cost.
    printVector(f, factors[:model.out.lpc2bands], name + '_factor_signs')
    printVector(f, factors[model.out.lpc2bands:model.out.lpc4bands], name + '_factor_mags')
    printVector(f, factors[model.out.lpc4bands:], name + '_factor_outs')
    f.write('const MDenseLayerMWDLP10 {} = {{\n   {}_bias,\n   {}_weights,\n   {}_factor_signs,\n   {}_factor_mags,\n   '\
        '{}_factor_outs,\n   ACTIVATION_TANH, ACTIVATION_EXP, ACTIVATION_TANHSHRINK\n}};\n\n'.format(name, name, name,
            name, name, name))
    hf.write('extern const MDenseLayerMWDLP10 {};\n\n'.format(name));

    #dump dual_fc_fine
    name = "dual_fc_fine"
    print("printing layer " + name)
    weights = model.out_f.conv.weight.permute(2,1,0)[0].numpy()
    bias = model.out_f.conv.bias.numpy()
    factors = (0.5*torch.exp(model.out_f.fact.weight[0])).numpy()
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    ## Previous implementation (as in ICASSP 2021) uses shared factors between bands for signs and mags [for data-driven LPC],
    ## though the mid-output uses band-dependent factors.
    ## These factors should be made band-dependent for all signs, mags, and mid-output for proper modeling as current implementation.
    ## Further, with 10-bit mu-law coarse-fine output, there is no need to use mid-output because
    ## this layer can directly generates the band-dependent 32-dim (sqrt(1024)) logits outputs
    ## instead of 256-dim (16-bit coarse-fine) or 512-dim (9-bit mu-law) which needs smaller mid-output for reducing computational cost.
    printVector(f, factors[:model.out_f.lpc2bands], name + '_factor_signs')
    printVector(f, factors[model.out_f.lpc2bands:model.out_f.lpc4bands], name + '_factor_mags')
    printVector(f, factors[model.out_f.lpc4bands:], name + '_factor_outs')
    f.write('const MDenseLayerMWDLP10 {} = {{\n   {}_bias,\n   {}_weights,\n   {}_factor_signs,\n   {}_factor_mags,\n   '\
        '{}_factor_outs,\n   ACTIVATION_TANH, ACTIVATION_EXP, ACTIVATION_TANHSHRINK\n}};\n\n'.format(name, name, name,
            name, name, name))
    hf.write('extern const MDenseLayerMWDLP10 {};\n\n'.format(name));

    #dump pqmf_synthesis filt
    name = "pqmf_synthesis"
    print("printing layer " + name)
    pqmf = PQMF(model.n_bands)
    pqmf_order = pqmf.taps
    pqmf_delay = pqmf_order // 2
    weights = pqmf.synthesis_filter.permute(2,1,0).numpy()
    bias = np.zeros(1)
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const Conv1DLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, {}, ACTIVATION_LINEAR\n}};\n\n'
            .format(name, name, name, weights.shape[1], weights.shape[0], weights.shape[2]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[2]))
    hf.write('#define {}_STATE_SIZE ({}*{})\n'.format(name.upper(), weights.shape[1], pqmf_delay+1))
    hf.write('#define {}_DELAY {}\n'.format(name.upper(), pqmf_delay))
    hf.write('extern const Conv1DLayer {};\n\n'.format(name));

    hf.write('#define MAX_RNN_NEURONS {}\n\n'.format(max_rnn_neurons))
    hf.write('#define RNN_MAIN_NEURONS {}\n\n'.format(model.hidden_units))
    hf.write('#define RNN_SUB_NEURONS {}\n\n'.format(model.hidden_units_2))
    hf.write('#define N_MBANDS {}\n\n'.format(model.n_bands))
    hf.write('#define DLPC_ORDER {}\n\n'.format(model.lpc))
    hf.write('#define PQMF_ORDER {}\n\n'.format(pqmf_order))
    hf.write('#define SQRT_QUANTIZE {}\n\n'.format(model.cf_dim))
    hf.write('#define N_SAMPLE_BANDS {}\n\n'.format(model.upsampling_factor))
    hf.write('#define FEATURES_DIM {}\n\n'.format(model.in_dim))

    hf.write('typedef struct {\n')
    hf.write('  float feature_conv_state[FEATURE_CONV_STATE_SIZE];\n')
    hf.write('  float gru_a_state[SPARSE_GRU_A_STATE_SIZE];\n')
    hf.write('  float gru_b_state[GRU_B_STATE_SIZE];\n')
    hf.write('  float gru_c_state[GRU_C_STATE_SIZE];\n')
    hf.write('} MWDLP10NNetState;\n')
    
    hf.write('\n\n#endif\n')
    
    f.close()
    hf.close()

    ## CycleVAE+PostNet+SpkNet for Mel-Spectrogram conversion with intermediate excitation estimation
    cfile = args.c_cycvae_out_file
    hfile = args.h_cycvae_out_file
    
    f = open(cfile, 'w')
    hf = open(hfile, 'w')
    
    f.write('/*This file is automatically generated from a PyTorch model*/\n\n')
    f.write('#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif\n\n#include "nnet.h"\n#include "{}"\n\n'.format(hfile))
    
    hf.write('/*This file is automatically generated from a PyTorch model*/\n\n')
    hf.write('#ifndef RNN_CYCVAE_DATA_H\n#define RNN_CYCVAE_DATA_H\n\n#include "nnet.h"\n\n')

    ## Dump melsp_norm, uvf0_norm, uvcap_norm
    ## Dump conv_in enc_melsp
    ## Dump conv_in enc_excit
    ## Dump conv_in dec_excit
    ## Dump conv_in dec_melsp
    ## Dump conv_in post
    ## Dump gru_layer enc_melsp
    ## Dump gru_layer enc_excit
    ## Dump gru_layer spk
    ## Dump gru_layer dec_excit
    ## Dump gru_layer dec_melsp
    ## Dump gru_layer post
    ## Dump out_layer enc_melsp
    ## Dump out_layer enc_excit
    ## Dump out_layer spk
    ## Dump out_layer dec_excit
    ## Dump out_layer dec_melsp
    ## Dump out_layer post

    hf.write('typedef struct {\n')
    hf.write('  float feature_enc_melsp_conv_state[FEATURE_ENC_MELSP_CONV_STATE_SIZE];\n')
    hf.write('  float feature_enc_excit_conv_state[FEATURE_ENC_EXCIT_CONV_STATE_SIZE];\n')
    hf.write('  float feature_dec_excit_conv_state[FEATURE_DEC_EXCIT_CONV_STATE_SIZE];\n')
    hf.write('  float feature_dec_melsp_conv_state[FEATURE_DEC_MELSP_CONV_STATE_SIZE];\n')
    hf.write('  float feature_post_conv_state[FEATURE_POST_CONV_STATE_SIZE];\n')
    hf.write('  float gru_enc_melsp_state[GRU_ENC_MELSP_STATE_SIZE];\n')
    hf.write('  float gru_enc_excit_state[GRU_ENC_EXCIT_STATE_SIZE];\n')
    hf.write('  float gru_spk_state[GRU_SPK_STATE_SIZE];\n')
    hf.write('  float gru_dec_excit_state[GRU_DEC_EXCIT_STATE_SIZE];\n')
    hf.write('  float gru_dec_melsp_state[GRU_DEC_MELSP_STATE_SIZE];\n')
    hf.write('  float gru_post_state[GRU_POST_STATE_SIZE];\n')
    hf.write('  short kernel_size_dec_excit KERNEL_SIZE_DEC_EXCIT;\n')
    hf.write('  short kernel_size_dec_melsp KERNEL_SIZE_DEC_MELSP;\n')
    hf.write('  short kernel_size_post KERNEL_SIZE_POST;\n')
    hf.write('  short nb_in_dec_excit NB_IN_DEC_EXCIT;\n')
    hf.write('  short nb_in_dec_melsp NB_IN_DEC_MELSP;\n')
    hf.write('  short nb_in_post NB_IN_POST;\n')
    hf.write('} CycleVAEPostMelspExcitSpkNNetState;\n')

    hf.write('\n\n#endif\n')

    f.close()
    hf.close()

    ## Dump high-pass filter coeffs, half hanning-window coeffs, and mel-filterbank here
    ## hpassfilt.h, halfwin.h, melfb.h


if __name__ == "__main__":
    main()
