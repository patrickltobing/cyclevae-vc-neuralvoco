# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import logging
import sys
import time
import math

import torch
import torch.nn.functional as F
import torch.fft
from torch import nn
from torch.autograd import Function

from torch.distributions.one_hot_categorical import OneHotCategorical

import numpy as np

CLIP_1E16 = -14.162084148244246758816564788835


def initialize(m):
    """FUNCTION TO INITILIZE CONV WITH XAVIER

    Arg:
        m (torch.nn.Module): torch nn module instance
    """
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    else:
        for name, param in m.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


def encode_mu_law(x, mu=256):
    """FUNCTION TO PERFORM MU-LAW ENCODING

    Args:
        x (ndarray): audio signal with the range from -1 to 1
        mu (int): quantized level

    Return:
        (ndarray): quantized audio signal with the range from 0 to mu - 1
    """
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64)


def encode_mu_law_torch(x, mu=256):
    """FUNCTION TO PERFORM MU-LAW ENCODING

    Args:
        x (ndarray): audio signal with the range from -1 to 1
        mu (int): quantized level

    Return:
        (ndarray): quantized audio signal with the range from 0 to mu - 1
    """
    mu = mu - 1
    fx = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / np.log(1 + mu) # log1p(x) = log_e(1+x)
    return torch.floor((fx + 1) / 2 * mu + 0.5).long()


def decode_mu_law(y, mu=256):
    """FUNCTION TO PERFORM MU-LAW DECODING

    Args:
        x (ndarray): quantized audio signal with the range from 0 to mu - 1
        mu (int): quantized level

    Return:
        (ndarray): audio signal with the range from -1 to 1
    """
    #fx = 2 * y / (mu - 1.) - 1.
    mu = mu - 1
    fx = y / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return x


def decode_mu_law_torch(y, mu=256):
    """FUNCTION TO PERFORM MU-LAW DECODING

    Args:
        x (ndarray): quantized audio signal with the range from 0 to mu - 1
        mu (int): quantized level

    Return:
        (ndarray): audio signal with the range from -1 to 1
    """
    #fx = 2 * y / (mu - 1.) - 1.
    mu = mu - 1
    fx = y / mu * 2 - 1
    x = torch.sign(fx) / mu * ((1 + mu) ** torch.abs(fx) - 1)
    return x


class ConvTranspose2d(nn.ConvTranspose2d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(ConvTranspose2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.constant_(self.weight, 1.0)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpSampling(nn.Module):
    """UPSAMPLING LAYER WITH DECONVOLUTION

    Arg:
        upsampling_factor (int): upsampling factor
    """

    def __init__(self, upsampling_factor, bias=True):
        super(UpSampling, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.bias = bias
        self.conv = ConvTranspose2d(1, 1,
                                       kernel_size=(1, self.upsampling_factor),
                                       stride=(1, self.upsampling_factor),
                                       bias=self.bias)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T')
                        where T' = T * upsampling_factor
        """
        return self.conv(x.unsqueeze(1)).squeeze(1)


class SkewedConv1d(nn.Module):
    """1D SKEWED CONVOLUTION"""

    def __init__(self, in_dim=39, kernel_size=7, right_size=1, nonlinear=False, pad_first=False):
        super(SkewedConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.right_size = right_size
        self.rec_field = self.kernel_size
        self.left_size = self.kernel_size - 1 - self.right_size
        self.pad_first = pad_first
        if self.right_size < self.left_size:
            self.padding = self.left_size
            self.skew_left = True
            self.padding_1 = self.padding-self.right_size
        else:
            self.padding = self.right_size
            self.skew_left = False
            self.padding_1 = self.padding-self.left_size
        self.out_dim = self.in_dim*self.rec_field
        if nonlinear:
            if not self.pad_first:
                module_list = [nn.Conv1d(self.in_dim, self.in_dim*self.rec_field, self.kernel_size, padding=self.padding),\
                                nn.PReLU(out_chn)]
            else:
                module_list = [nn.Conv1d(self.in_dim, self.in_dim*self.rec_field, self.kernel_size), nn.PReLU(out_chn)]
            self.conv = nn.Sequential(*module_list)
        else:
            if not self.pad_first:
                self.conv = nn.Conv1d(self.in_dim, self.in_dim*self.rec_field, self.kernel_size, padding=self.padding)
            else:
                self.conv = nn.Conv1d(self.in_dim, self.in_dim*self.rec_field, self.kernel_size)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """

        if not self.pad_first:
            if self.padding_1 > 0:
                if self.skew_left:
                    return self.conv(x)[:,:,:-self.padding_1]
                else:
                    return self.conv(x)[:,:,self.padding_1:]
            else:
                return self.conv(x)
        else:
            return self.conv(x)


class TwoSidedDilConv1d(nn.Module):
    """1D TWO-SIDED DILATED CONVOLUTION"""

    def __init__(self, in_dim=39, kernel_size=3, layers=2, nonlinear=False, pad_first=False):
        super(TwoSidedDilConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.layers = layers
        self.rec_field = self.kernel_size**self.layers
        self.padding = int((self.rec_field-1)/2)
        self.pad_first = pad_first
        module_list = []
        self.out_dim = self.in_dim*(self.kernel_size**self.layers)
        if nonlinear:
            for i in range(self.layers):
                if i > 0:
                    in_chn = self.in_dim*(self.kernel_size**(i))
                    out_chn = self.in_dim*(self.kernel_size**(i+1))
                    module_list += [nn.Conv1d(in_chn, out_chn, self.kernel_size, dilation=self.kernel_size**i), \
                                    nn.PReLU(out_chn)]
                else:
                    out_chn = self.in_dim*(self.kernel_size**(i+1))
                    if not self.pad_first:
                        module_list += [nn.Conv1d(self.in_dim, out_chn, self.kernel_size, padding=self.padding),\
                                        nn.PReLU(out_chn)]
                    else:
                        module_list += [nn.Conv1d(self.in_dim, out_chn, self.kernel_size), nn.PReLU(out_chn)]
        else:
            for i in range(self.layers):
                if i > 0:
                    module_list += [nn.Conv1d(self.in_dim*(self.kernel_size**(i)), \
                                    self.in_dim*(self.kernel_size**(i+1)), self.kernel_size, \
                                        dilation=self.kernel_size**i)]
                else:
                    if not self.pad_first:
                        module_list += [nn.Conv1d(self.in_dim, self.in_dim*(self.kernel_size**(i+1)), \
                                        self.kernel_size, padding=self.padding)]
                    else:
                        module_list += [nn.Conv1d(self.in_dim, self.in_dim*(self.kernel_size**(i+1)), self.kernel_size)]
        self.conv = nn.Sequential(*module_list)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """

        return self.conv(x)


class CausalDilConv1d(nn.Module):
    """1D Causal DILATED CONVOLUTION"""

    def __init__(self, in_dim=11, kernel_size=2, layers=2, nonlinear=False, pad_first=False):
        super(CausalDilConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.layers = layers
        self.padding_list = [self.kernel_size**(i+1)-self.kernel_size**(i) for i in range(self.layers)]
        self.padding = sum(self.padding_list)
        self.rec_field = self.padding + 1
        self.pad_first = pad_first
        module_list = []
        if nonlinear:
            for i in range(self.layers):
                if i > 0:
                    in_chn = self.in_dim*(sum(self.padding_list[:i])+1)
                    out_chn = self.in_dim*(sum(self.padding_list[:i+1])+1)
                    module_list += [nn.Conv1d(in_chn, out_chn, self.kernel_size, dilation=self.kernel_size**i), \
                                    nn.PReLU(out_chn)]
                else:
                    out_chn = self.in_dim*(sum(self.padding_list[:i+1])+1)
                    if not self.pad_first:
                        module_list += [nn.Conv1d(self.in_dim, out_chn, self.kernel_size, padding=self.padding), \
                                        nn.PReLU(out_chn)]
                    else:
                        module_list += [nn.Conv1d(self.in_dim, out_chn, self.kernel_size), nn.PReLU(out_chn)]
        else:
            for i in range(self.layers):
                if i > 0:
                    module_list += [nn.Conv1d(self.in_dim*(sum(self.padding_list[:i])+1), \
                                    self.in_dim*(sum(self.padding_list[:i+1])+1), self.kernel_size, \
                                        dilation=self.kernel_size**i)]
                else:
                    if not self.pad_first:
                        module_list += [nn.Conv1d(self.in_dim, self.in_dim*(sum(self.padding_list[:i+1])+1), \
                                        self.kernel_size, padding=self.padding)]
                    else:
                        module_list += [nn.Conv1d(self.in_dim, self.in_dim*(sum(self.padding_list[:i+1])+1), \
                                        self.kernel_size)]
        self.conv = nn.Sequential(*module_list)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """

        if not self.pad_first:
            return self.conv(x)[:,:,:-self.padding]
        else:
            return self.conv(x)


class DualFC_CF(nn.Module):
    """Compact Dual Fully Connected layers based on LPCNet"""

    def __init__(self, in_dim=16, out_dim=32, lpc=6, bias=True, n_bands=10, mid_out=16):
        super(DualFC_CF, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lpc = lpc
        self.n_bands = n_bands
        self.lpc2 = self.lpc*2 #signs and scales
        self.bias = bias
        self.lpc4 = self.lpc2*2
        self.lpc2bands = self.lpc2*self.n_bands
        self.lpc4bands = self.lpc4*self.n_bands
        self.mid_out = mid_out

        self.mid_out_bands = self.mid_out*self.n_bands
        self.mid_out_bands2 = self.mid_out_bands*2
        self.conv = nn.Conv1d(self.in_dim, self.mid_out_bands2+self.lpc4bands, 1, bias=self.bias)
        #self.fact = EmbeddingZero(1, self.mid_out_bands2+self.lpc4)
        self.fact = EmbeddingZero(1, self.mid_out_bands2+self.lpc4bands)
        self.out = nn.Conv1d(self.mid_out, self.out_dim, 1, bias=self.bias)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C_in x T)

        Return:
            (Variable): float tensor variable with the shape (B x T x C_out)
        """

        # out = fact_1 o tanh(conv_1 * x) + fact_2 o tanh(conv_2 * x)
        if self.n_bands > 1:
            if self.lpc > 0:
                conv = self.conv(x).transpose(1,2) # B x T x n_bands*(K*4+mid_dim*2)
                fact_weight = 0.5*torch.exp(self.fact.weight[0]) # K*4+256*2
                B = x.shape[0]
                T = x.shape[2]
                # B x T x n_bands x K*2 --> B x T x n_bands x K
                #return torch.sum((torch.tanh(conv[:,:,:self.lpc2bands]).reshape(B,T,self.n_bands,-1)*fact_weight[:self.lpc2]).reshape(B,T,self.n_bands,2,-1), 3), \
                #        torch.sum((torch.exp(conv[:,:,self.lpc2bands:self.lpc4bands]).reshape(B,T,self.n_bands,-1)*fact_weight[self.lpc2:self.lpc4]).reshape(B,T,self.n_bands,2,-1), 3), \
                #            F.tanhshrink(self.out(torch.sum((F.relu(conv[:,:,self.lpc4bands:])\
                #                *fact_weight[self.lpc4:]).reshape(B,T,self.n_bands,2,-1), 3).reshape(B,T*self.n_bands,-1).transpose(1,2))).transpose(1,2).reshape(B,T,self.n_bands,-1)
                return torch.sum((torch.tanh(conv[:,:,:self.lpc2bands])*fact_weight[:self.lpc2bands]).reshape(B,T,self.n_bands,2,-1), 3), \
                        torch.sum((torch.exp(conv[:,:,self.lpc2bands:self.lpc4bands])*fact_weight[self.lpc2bands:self.lpc4bands]).reshape(B,T,self.n_bands,2,-1), 3), \
                            F.tanhshrink(self.out(torch.sum((F.relu(conv[:,:,self.lpc4bands:])\
                                *fact_weight[self.lpc4bands:]).reshape(B,T,self.n_bands,2,-1), 3).reshape(B,T*self.n_bands,-1).transpose(1,2))).transpose(1,2).reshape(B,T,self.n_bands,-1)
                # B x T x n_bands x mid*2 --> B x (T x n_bands) x mid --> B x mid x (T x n_bands) --> B x T x n_bands x 256
            else:
                # B x T x n_bands x mid*2 --> B x (T x n_bands) x mid --> B x mid x (T x n_bands) --> B x T x n_bands x 256
                B = x.shape[0]
                T = x.shape[2]
                return F.tanhshrink(self.out(torch.sum((F.relu(self.conv(x).transpose(1,2))\
                            *(0.5*torch.exp(self.fact.weight[0]))).reshape(B,T,self.n_bands,2,-1), 3).reshape(B,T*self.n_bands,-1).transpose(1,2))).transpose(1,2).reshape(B,T,self.n_bands,-1)
        else:
            if self.lpc > 0:
                conv = self.conv(x).transpose(1,2)
                fact_weight = 0.5*torch.exp(self.fact.weight[0])
                return torch.sum((torch.tanh(conv[:,:,:self.lpc2])*fact_weight[:self.lpc2]).reshape(x.shape[0],x.shape[2],2,-1), 2), \
                        torch.sum((torch.exp(conv[:,:,self.lpc2:self.lpc4])*fact_weight[self.lpc2:self.lpc4]).reshape(x.shape[0],x.shape[2],2,-1), 2), \
                            F.tanhshrink(self.out(torch.sum((F.relu(conv[:,:,self.lpc4:])*fact_weight[self.lpc4:]).reshape(x.shape[0],x.shape[2],2,-1), 2).transpose(1,2))).transpose(1,2)
            else:
                return F.tanhshrink(self.out(torch.sum((F.relu(self.conv(x).transpose(1,2))*(0.5*torch.exp(self.fact.weight[0]))).reshape(x.shape[0],x.shape[2],2,-1), 2).transpose(1,2))).transpose(1,2)


class DualFC(nn.Module):
    """Compact Dual Fully Connected layers based on LPCNet"""

    def __init__(self, in_dim=16, out_dim=256, lpc=12, bias=True, n_bands=1, mid_out=None):
        super(DualFC, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lpc = lpc
        self.n_bands = n_bands
        self.lpc_out_dim = self.lpc+self.out_dim
        self.lpc_out_dim_lpc = self.lpc_out_dim+self.lpc
        self.lpc2 = self.lpc*2
        self.out_dim2 = self.out_dim*2
        self.bias = bias
        self.lpc4 = self.lpc2*2
        self.lpc2bands = self.lpc2*self.n_bands
        self.lpc4bands = self.lpc4*self.n_bands
        self.mid_out = mid_out

        if self.mid_out is None:
            self.conv = nn.Conv1d(self.in_dim, self.out_dim2*self.n_bands+self.lpc4bands, 1, bias=self.bias)
            self.fact = EmbeddingZero(1, self.out_dim2+self.lpc4)
        else:
            self.mid_out_bands = self.mid_out*self.n_bands
            self.mid_out_bands2 = self.mid_out_bands*2
            self.conv = nn.Conv1d(self.in_dim, self.mid_out_bands2+self.lpc4bands, 1, bias=self.bias)
            #self.fact = EmbeddingZero(1, self.mid_out_bands2+self.lpc4)
            self.fact = EmbeddingZero(1, self.mid_out_bands2+self.lpc4bands)
            self.out = nn.Conv1d(self.mid_out, self.out_dim, 1, bias=self.bias)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C_in x T)

        Return:
            (Variable): float tensor variable with the shape (B x T x C_out)
        """

        # out = fact_1 o tanh(conv_1 * x) + fact_2 o tanh(conv_2 * x)
        if self.n_bands > 1:
            if self.mid_out is None:
                if self.lpc > 0:
                    conv = self.conv(x).transpose(1,2) # B x T x n_bands*(K*4+256*2)
                    fact_weight = 0.5*torch.exp(self.fact.weight[0]) # K*4+256*2
                    B = x.shape[0]
                    T = x.shape[2]
                    # B x T x n_bands x K*2 --> B x T x n_bands x K
                    return torch.sum((torch.tanh(conv[:,:,:self.lpc2bands]).reshape(B,T,self.n_bands,-1)*fact_weight[:self.lpc2]).reshape(B,T,self.n_bands,2,-1), 3), \
                            torch.sum((torch.exp(conv[:,:,self.lpc2bands:self.lpc4bands]).reshape(B,T,self.n_bands,-1)*fact_weight[self.lpc2:self.lpc4]).reshape(B,T,self.n_bands,2,-1), 3), \
                                torch.sum((F.tanhshrink(conv[:,:,self.lpc4bands:]).reshape(B,T,self.n_bands,-1)*fact_weight[self.lpc4:]).reshape(B,T,self.n_bands,2,-1), 3)
                    # B x T x n_bands x 256*2 --> B x T x n_bands x 256
                else:
                    # B x T x n_bands x 256*2 --> B x T x n_bands x 256
                    B = x.shape[0]
                    T = x.shape[2]
                    return torch.sum((F.tanhshrink(self.conv(x).transpose(1,2)).reshape(B,T,self.n_bands,-1)*(0.5*torch.exp(self.fact.weight[0]))).reshape(B,T,self.n_bands,2,-1), 3)
            else:
                if self.lpc > 0:
                    conv = self.conv(x).transpose(1,2) # B x T x n_bands*(K*4+mid_dim*2)
                    fact_weight = 0.5*torch.exp(self.fact.weight[0]) # K*4+256*2
                    B = x.shape[0]
                    T = x.shape[2]
                    # B x T x n_bands x K*2 --> B x T x n_bands x K
                    #return torch.sum((torch.tanh(conv[:,:,:self.lpc2bands]).reshape(B,T,self.n_bands,-1)*fact_weight[:self.lpc2]).reshape(B,T,self.n_bands,2,-1), 3), \
                    #        torch.sum((torch.exp(conv[:,:,self.lpc2bands:self.lpc4bands]).reshape(B,T,self.n_bands,-1)*fact_weight[self.lpc2:self.lpc4]).reshape(B,T,self.n_bands,2,-1), 3), \
                    #            F.tanhshrink(self.out(torch.sum((F.relu(conv[:,:,self.lpc4bands:])\
                    #                *fact_weight[self.lpc4:]).reshape(B,T,self.n_bands,2,-1), 3).reshape(B,T*self.n_bands,-1).transpose(1,2))).transpose(1,2).reshape(B,T,self.n_bands,-1)
                    return torch.sum((torch.tanh(conv[:,:,:self.lpc2bands])*fact_weight[:self.lpc2bands]).reshape(B,T,self.n_bands,2,-1), 3), \
                            torch.sum((torch.exp(conv[:,:,self.lpc2bands:self.lpc4bands])*fact_weight[self.lpc2bands:self.lpc4bands]).reshape(B,T,self.n_bands,2,-1), 3), \
                                F.tanhshrink(self.out(torch.sum((F.relu(conv[:,:,self.lpc4bands:])\
                                    *fact_weight[self.lpc4bands:]).reshape(B,T,self.n_bands,2,-1), 3).reshape(B,T*self.n_bands,-1).transpose(1,2))).transpose(1,2).reshape(B,T,self.n_bands,-1)
                    # B x T x n_bands x mid*2 --> B x (T x n_bands) x mid --> B x mid x (T x n_bands) --> B x T x n_bands x 256
                else:
                    # B x T x n_bands x mid*2 --> B x (T x n_bands) x mid --> B x mid x (T x n_bands) --> B x T x n_bands x 256
                    B = x.shape[0]
                    T = x.shape[2]
                    return F.tanhshrink(self.out(torch.sum((F.relu(self.conv(x).transpose(1,2))\
                                *(0.5*torch.exp(self.fact.weight[0]))).reshape(B,T,self.n_bands,2,-1), 3).reshape(B,T*self.n_bands,-1).transpose(1,2))).transpose(1,2).reshape(B,T,self.n_bands,-1)
        else:
            if self.mid_out is None:
                if self.lpc > 0:
                    conv = self.conv(x).transpose(1,2)
                    fact_weight = 0.5*torch.exp(self.fact.weight[0])
                    return torch.sum((torch.tanh(conv[:,:,:self.lpc2])*fact_weight[:self.lpc2]).reshape(x.shape[0],x.shape[2],2,-1), 2), \
                            torch.sum((torch.exp(conv[:,:,self.lpc2:self.lpc4])*fact_weight[self.lpc2:self.lpc4]).reshape(x.shape[0],x.shape[2],2,-1), 2), \
                    #return signs, scales, logits
                else:
                    return torch.sum((F.tanhshrink(self.conv(x).transpose(1,2))*(0.5*torch.exp(self.fact.weight[0]))).reshape(x.shape[0],x.shape[2],2,-1), 2)
            else:
                if self.lpc > 0:
                    conv = self.conv(x).transpose(1,2)
                    fact_weight = 0.5*torch.exp(self.fact.weight[0])
                    return torch.sum((torch.tanh(conv[:,:,:self.lpc2])*fact_weight[:self.lpc2]).reshape(x.shape[0],x.shape[2],2,-1), 2), \
                            torch.sum((torch.exp(conv[:,:,self.lpc2:self.lpc4])*fact_weight[self.lpc2:self.lpc4]).reshape(x.shape[0],x.shape[2],2,-1), 2), \
                            F.tanhshrink(self.out(torch.sum((F.relu(conv[:,:,self.lpc4:])*fact_weight[self.lpc4:]).reshape(x.shape[0],x.shape[2],2,-1), 2).transpose(1,2))).transpose(1,2)
                else:
                    return F.tanhshrink(self.out(torch.sum((F.relu(self.conv(x).transpose(1,2))*(0.5*torch.exp(self.fact.weight[0]))).reshape(x.shape[0],x.shape[2],2,-1), 2).transpose(1,2))).transpose(1,2)


class OutputConv1d(nn.Module):
    """Output Convolution 1d"""

    def __init__(self, in_dim=1024, lin_dim=320, out_dim=256, lpc=6, n_bands=10, compact=True):
        super(OutputConv1d, self).__init__()
        self.in_dim = in_dim
        self.lin_dim = lin_dim
        self.n_bands = n_bands
        self.lpc = lpc
        self.lpc2 = self.lpc*2
        self.out_dim = out_dim
        self.compact = compact
        self.conv = nn.Conv1d(self.in_dim, self.lin_dim, 1)
        if self.compact:
            if self.n_bands > 1:
                self.mid_dim = self.lin_dim // self.n_bands
                self.mid_dim_bands = self.mid_dim * self.n_bands
                if self.lpc > 0:
                    self.out_mid = nn.Conv1d(self.lin_dim, self.mid_dim_bands, 1)
                    self.out = nn.Conv1d(self.mid_dim, self.out_dim+self.lpc2, 1)
                else:
                    self.out_mid = nn.Conv1d(self.lin_dim, self.mid_dim_bands, 1)
                    self.out = nn.Conv1d(self.mid_dim, self.out_dim, 1)
            else:
                if self.lpc > 0:
                    self.out_lpc = nn.Conv1d(self.lin_dim, self.lpc2, 1)
                    self.out = nn.Conv1d(self.lin_dim, self.out_dim, 1)
                else:
                    self.out = nn.Conv1d(self.lin_dim, self.out_dim, 1)
        else:
            self.out = nn.Conv1d(self.lin_dim, self.out_dim+self.lpc2, 1)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x T x C)
        """

        if self.compact:
            if self.n_bands > 1:
                if self.lpc > 0:
                    B = x.shape[0]
                    T = x.shape[2]
                    # B x (n_bands x C_mid) x T --> B x T x (n_bands x C_mid) --> B x (T x n_bands) x C_mid --> B x C_mid x (T x n_bands) --> B x (lpc*2+256) x T --> BxTx(lpc*2+256) --> BxTxn_bandsxlpc2+256
                    out = self.out(F.relu(self.out_mid(F.relu(self.conv(x)))).transpose(1,2).reshape(B, T*self.n_bands, -1).transpose(1,2)).transpose(1,2).reshape(B, T, self.n_bands, -1)
                    return torch.tanh(out[:,:,:,:self.lpc]), torch.exp(F.relu(out[:,:,:,self.lpc:self.lpc2])), F.tanhshrink(out[:,:,:,self.lpc2:])
                else:
                    B = x.shape[0]
                    T = x.shape[2]
                    return  F.tanhshrink(self.out(F.relu(self.out_mid(F.relu(self.conv(x)))).transpose(1,2).reshape(B, T*self.n_bands, -1).transpose(1,2)).transpose(1,2).reshape(B, T, self.n_bands, -1))
                    # B x T x n_bands x 256
            else:
                if self.lpc > 0:
                    out = F.relu(self.conv(x))
                    lpc = self.out_lpc(out).transpose(1,2)
                    return torch.tanh(lpc[:,:,:self.lpc]), torch.exp(F.relu(lpc[:,:,self.lpc:])), F.tanhshrink(self.out(out).transpose(1,2))
                else:
                    return F.tanhshrink(self.out(F.relu(self.conv(x))).transpose(1,2))
        else:
            if self.lpc > 0:
                out = self.out(F.relu(self.conv(x))).transpose(1,2)
                return torch.tanh(out[:,:,:self.lpc]), torch.exp(F.relu(out[:,:,self.lpc:self.lpc2])), F.tanhshrink(out[:,:,self.lpc2:])
            else:
                return F.tanhshrink(self.out(F.relu(self.conv(x))).transpose(1,2))


class EmbeddingZero(nn.Embedding):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(EmbeddingZero, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.constant_(self.weight, 0)


class EmbeddingOne(nn.Embedding):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(EmbeddingOne, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.constant_(self.weight, 1)


class EmbeddingHalf(nn.Embedding):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(EmbeddingHalf, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.constant_(self.weight, 0.5)


def nn_search(encoding, centroids):
    T = encoding.shape[0]
    K = centroids.shape[0]
    dist2 = torch.sum((encoding.unsqueeze(1).repeat(1,K,1)-centroids.unsqueeze(0).repeat(T,1,1)).abs(),2) # TxK
    ctr_ids = torch.argmin(dist2, dim=-1)

    return ctr_ids


def nn_search_batch(encoding, centroids):
    B = encoding.shape[0]
    T = encoding.shape[1]
    K = centroids.shape[0]
    dist2 = torch.sum((encoding.unsqueeze(2).repeat(1,1,K,1)-\
                    centroids.unsqueeze(0).unsqueeze(0).repeat(B,T,1,1)).abs(),3) # B x T x K
    ctr_ids = torch.argmin(dist2, dim=-1) # B x T

    return ctr_ids


def cross_entropy_with_logits(logits, probs):
    logsumexp = torch.log(torch.sum(torch.exp(logits), -1, keepdim=True)) # B x T x K --> B x T x 1

    return torch.sum(-probs * (logits - logsumexp), -1) # B x T x K --> B x T


def kl_categorical_categorical_logits(p, logits_p, logits_q):
    """ sum_{k=1}^K q_k * (ln q_k - ln p_k) """

    return -cross_entropy_with_logits(logits_p, p) + cross_entropy_with_logits(logits_q, p) # B x T x K --> B x T


def sampling_laplace_wave(loc, scale):
    eps = torch.empty_like(loc).uniform_(torch.finfo(loc.dtype).eps-1,1)

    return loc - scale * eps.sign() * torch.log1p(-eps.abs()) # scale

 
def sampling_normal(mu, var):
    eps = torch.randn(mu.shape).cuda()

    return mu + torch.sqrt(var) * eps # var


def kl_normal(mu_q, var_q):
    """ 1/2 [µ_i^2 + σ^2_i − 1 - ln(σ^2_i) ] """

    var_q = torch.clamp(var_q, min=1e-9)

    return torch.mean(torch.sum(0.5*(torch.pow(mu_q, 2) + var_q - 1 - torch.log(var_q)), -1)) # B x T x C --> B x T --> 1


def kl_normal_normal(mu_q, var_q, p):
    """ 1/2*σ^2_j [(µ_i − µ_j)^2 + σ^2_i − σ^2_j] + ln σ_j/σ_i """

    var_q = torch.clamp(var_q, min=1e-9)

    mu_p = p[:mu_q.shape[-1]]
    var_p = p[mu_q.shape[-1]:]
    var_p = torch.clamp(var_p, min=1e-9)

    return torch.mean(torch.sum(0.5*(torch.pow(mu_q-mu_p, 2)/var_p + var_q/var_p - 1 + torch.log(var_p/var_q)), -1)) # B x T x C --> B x T --> 1


def neg_entropy_laplace(log_b):
    #-ln(2be) = -((ln(2)+1) + ln(b))
    return -(1.69314718055994530941723212145818 + log_b)


def kl_laplace(param):
    """ - ln(λ_i) + |θ_i| + λ_i * exp(−|θ_i|/λ_i) − 1 """

    k = param.shape[-1]//2
    if len(param.shape) > 2:
        mu_q = param[:,:,:k]
        scale_q = torch.exp(param[:,:,k:])
    else:
        mu_q = param[:,:k]
        scale_q = torch.exp(param[:,k:])

    scale_q = torch.clamp(scale_q, min=1e-12)

    mu_q_abs = torch.abs(mu_q)

    return -torch.log(scale_q) + mu_q_abs + scale_q*torch.exp(-mu_q_abs/scale_q) - 1 # B x T x C / T x C


def kl_laplace_param(mu_q, sigma_q):
    """ - ln(λ_i) + |θ_i| + λ_i * exp(−|θ_i|/λ_i) − 1 """

    scale_q = torch.clamp(sigma_q.exp(), min=1e-12)
    mu_q_abs = torch.abs(mu_q)

    return torch.mean(torch.sum(-torch.log(scale_q) + mu_q_abs + scale_q*torch.exp(-mu_q_abs/scale_q) - 1, -1), -1) # B / 1


def sampling_laplace(param, log_scale=None):
    if log_scale is not None:
        mu = param
        scale = torch.exp(log_scale)
    else:
        k = param.shape[-1]//2
        mu = param[:,:,:k]
        scale = torch.exp(param[:,:,k:])
    eps = torch.empty_like(mu).uniform_(torch.finfo(mu.dtype).eps-1,1)

    return mu - scale * eps.sign() * torch.log1p(-eps.abs()) # scale
 

def kl_laplace_laplace_param(mu_q, sigma_q, mu_p, sigma_p):
    """ ln(λ_j/λ_i) + |θ_i-θ_j|/λ_j + λ_i/λ_j * exp(−|θ_i-θ_j|/λ_i) − 1 """

    scale_q = torch.clamp(sigma_q.exp(), min=1e-12)
    scale_p = torch.clamp(sigma_p.exp(), min=1e-12)

    mu_abs = torch.abs(mu_q-mu_p)

    return torch.mean(torch.sum(torch.log(scale_p/scale_q) + mu_abs/scale_p + (scale_q/scale_p)*torch.exp(-mu_abs/scale_q) - 1, -1), -1) # B / 1


def kl_laplace_laplace(q, p):
    """ ln(λ_j/λ_i) + |θ_i-θ_j|/λ_j + λ_i/λ_j * exp(−|θ_i-θ_j|/λ_i) − 1 """

    D = q.shape[-1] // 2
    if len(q.shape) > 2:
        scale_q = torch.clamp(torch.exp(q[:,:,D:]), min=1e-12)
        scale_p = torch.clamp(torch.exp(p[:,:,D:]), min=1e-12)

        mu_abs = torch.abs(q[:,:,:D]-p[:,:,:D])

        return torch.mean(torch.sum(torch.log(scale_p/scale_q) + mu_abs/scale_p + (scale_q/scale_p)*torch.exp(-mu_abs/scale_q) - 1, -1), -1) # B x T x C --> B x T --> B
    else:
        scale_q = torch.clamp(torch.exp(q[:,D:]), min=1e-12)
        scale_p = torch.clamp(torch.exp(p[:,D:]), min=1e-12)

        mu_abs = torch.abs(q[:,:D]-p[:,:D])

        return torch.mean(torch.sum(torch.log(scale_p/scale_q) + mu_abs/scale_p + (scale_q/scale_p)*torch.exp(-mu_abs/scale_q) - 1, -1)) # T x C --> T --> 1


class GRU_VAE_ENCODER_(nn.Module):
    def __init__(self, in_dim=50, lat_dim=50, hidden_layers=1, hidden_units=1024, kernel_size=7,
            dilation_size=1, do_prob=0, use_weight_norm=True, causal_conv=False, right_size=0,
                pad_first=True, scale_out_flag=False, n_spk=None, cont=True):
        super(GRU_VAE_ENCODER_, self).__init__()
        self.in_dim = in_dim
        self.lat_dim = lat_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.right_size = right_size
        self.pad_first = pad_first
        self.use_weight_norm = use_weight_norm
        self.cont = cont
        if self.cont:
            self.out_dim = self.lat_dim*2
        else:
            self.out_dim = self.lat_dim
        self.n_spk = n_spk
        if self.n_spk is not None:
            self.out_dim += self.n_spk
        self.scale_out_flag = scale_out_flag

        # Normalization layer
        self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)

        # Conv. layers
        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = self.conv.padding
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = 0
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                        right_size=self.right_size, pad_first=self.pad_first)
            self.pad_left = self.conv.left_size
            self.pad_right = self.conv.right_size
        self.gru_in_dim = self.in_dim*self.conv.rec_field
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if self.do_prob > 0 and self.hidden_layers > 1:
            self.gru = nn.GRU(self.gru_in_dim, self.hidden_units, self.hidden_layers,
                                dropout=self.do_prob, batch_first=True)
        else:
            self.gru = nn.GRU(self.gru_in_dim, self.hidden_units, self.hidden_layers,
                                batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        self.out = nn.Conv1d(self.hidden_units, self.out_dim, 1)
        if self.scale_out_flag:
            self.scale_out = nn.Conv1d(self.lat_dim, self.lat_dim, 1)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
        else:
            self.apply(initialize)

    def forward(self, x, h=None, do=False, sampling=True, outpad_right=0):
        x_in = self.conv(self.scale_in(x.transpose(1,2))).transpose(1,2)
        # Input s layers
        if self.do_prob > 0 and do:
            s = self.conv_drop(x_in) # B x C x T --> B x T x C
        else:
            s = x_in # B x C x T --> B x T x C
        if outpad_right > 0:
            # GRU s layers
            if h is None:
                out, h = self.gru(s[:,:-outpad_right]) # B x T x C
            else:
                out, h = self.gru(s[:,:-outpad_right], h) # B x T x C
            out_, _ = self.gru(s[:,-outpad_right:], h) # B x T x C
            s = torch.cat((out, out_), 1)
        else:
            # GRU s layers
            if h is None:
                s, h = self.gru(s) # B x T x C
            else:
                s, h = self.gru(s, h) # B x T x C
        # Output s layers
        if self.do_prob > 0 and do:
            s = self.out(self.gru_drop(s).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
        else:
            s = self.out(s.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C

        if self.n_spk is not None: #with speaker posterior
            if self.cont: #continuous latent
                spk_logits = F.selu(out[:,:,:self.n_spk])
                if self.scale_out_flag:
                    mus = self.scale_out(F.tanhshrink(s[:,:,self.n_spk:-self.lat_dim]).transpose(1,2)).transpose(1,2)
                else:
                    mus = F.tanhshrink(s[:,:,self.n_spk:-self.lat_dim])
                log_scales = F.logsigmoid(s[:,:,-self.lat_dim:])
                if sampling:
                    if do:
                        return spk_logits, torch.cat((mus, torch.clamp(log_scales, min=CLIP_1E16)), 2), \
                                sampling_laplace(mus, log_scales), h.detach()
                    else:
                        return spk_logits, torch.cat((mus, log_scales), 2), \
                                sampling_laplace(mus, log_scales), h.detach()
                else:
                    return spk_logits, torch.cat((mus, log_scales), 2), mus, h.detach()
            else: #discrete latent
                if self.scale_out_flag:
                    return F.selu(out[:,:,:self.n_spk]), \
                        self.scale_out(F.tanhshrink(s[:,:,-self.lat_dim:]).transpose(1,2)).transpose(1,2), \
                            h.detach()
                else:
                    return F.selu(out[:,:,:self.n_spk]), F.tanhshrink(s[:,:,-self.lat_dim:]), h.detach()
        else: #without speaker posterior
            if self.cont: #continuous latent
                if self.scale_out_flag:
                    mus = self.scale_out(F.tanhshrink(s[:,:,:self.lat_dim]).transpose(1,2)).transpose(1,2)
                else:
                    mus = F.tanhshrink(s[:,:,:self.lat_dim])
                log_scales = F.logsigmoid(s[:,:,self.lat_dim:])
                if sampling:
                    if do:
                        return torch.cat((mus, torch.clamp(log_scales, min=CLIP_1E16)), 2), \
                                sampling_laplace(mus, log_scales), h.detach()
                    else:
                        return torch.cat((mus, log_scales), 2), \
                                sampling_laplace(mus, log_scales), h.detach()
                else:
                    return torch.cat((mus, log_scales), 2), mus, h.detach()
            else: #discrete latent
                if self.scale_out_flag:
                    return self.scale_out(F.tanhshrink(s).transpose(1,2)).transpose(1,2), h.detach()
                else:
                    return F.tanhshrink(s), h.detach()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:
                return

        self.apply(_remove_weight_norm)


class GRU_SPEC_DECODER_(nn.Module):
    def __init__(self, feat_dim=50, out_dim=50, hidden_layers=1, hidden_units=1024, causal_conv=False,
            kernel_size=7, dilation_size=1, do_prob=0, n_spk=14, use_weight_norm=True,
                excit_dim=None, pad_first=True, right_size=None, pdf=False, scale_in_flag=False):
        super(GRU_SPEC_DECODER_, self).__init__()
        self.n_spk = n_spk
        self.feat_dim = feat_dim
        if self.n_spk is not None:
            self.in_dim = self.n_spk+self.feat_dim
        else:
            self.in_dim = self.feat_dim
        self.spec_dim = out_dim
        self.excit_dim = excit_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.use_weight_norm = use_weight_norm
        self.pad_first = pad_first
        self.right_size = right_size
        self.pdf = pdf
        if self.pdf:
            self.out_dim = self.spec_dim*2
        else:
            self.out_dim = self.spec_dim
        self.scale_in_flag = scale_in_flag

        if self.excit_dim is not None:
            if self.scale_in_flag:
                self.scale_in = nn.Conv1d(self.feat_dim+self.excit_dim, self.feat_dim+self.excit_dim, 1)
            else:
                self.scale_in = nn.Conv1d(self.excit_dim, self.excit_dim, 1)
            self.in_dim += self.excit_dim
        elif self.scale_in_flag:
            self.scale_in = nn.Conv1d(self.feat_dim, self.feat_dim, 1)

        # Conv. layers
        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = self.conv.padding
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = 0
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                        right_size=self.right_size, pad_first=self.pad_first)
            self.pad_left = self.conv.left_size
            self.pad_right = self.conv.right_size

        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if self.do_prob > 0 and self.hidden_layers > 1:
            self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers,
                                dropout=self.do_prob, batch_first=True)
        else:
            self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers,
                                batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        self.out = nn.Conv1d(self.hidden_units, self.out_dim, 1)

        # De-normalization layers
        self.scale_out = nn.Conv1d(self.spec_dim, self.spec_dim, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
        else:
            self.apply(initialize)

    def forward(self, z, y=None, h=None, do=False, e=None, outpad_right=0, sampling=True):
        if y is not None:
            if len(y.shape) == 2:
                y = F.one_hot(y, num_classes=self.n_spk).float()
            if e is not None:
                if self.scale_in_flag:
                    z = torch.cat((y, self.scale_in((torch.cat(e, z), 2).transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
                else:
                    z = torch.cat((y, self.scale_in(e.transpose(1,2)).transpose(1,2), z), 2) # B x T_frm x C
            else:
                if self.scale_in_flag:
                    z = torch.cat((y, self.scale_in(z.transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
                else:
                    z = torch.cat((y, z), 2) # B x T_frm x C
        else:
            if e is not None:
                if self.scale_in_flag:
                    z = self.scale_in((torch.cat(e, z), 2).transpose(1,2)).transpose(1,2) # B x T_frm x C
                else:
                    z = torch.cat((self.scale_in(e.transpose(1,2)).transpose(1,2), z), 2) # B x T_frm x C
            elif self.scale_in_flag:
                    z = self.scale_in(z.transpose(1,2)).transpose(1,2) # B x T_frm x C
        # Input e layers
        if self.do_prob > 0 and do:
            e = self.conv_drop(self.conv(z.transpose(1,2)).transpose(1,2)) # B x C x T --> B x T x C
        else:
            e = self.conv(z.transpose(1,2)).transpose(1,2) # B x C x T --> B x T x C
        if outpad_right > 0:
            # GRU e layers
            if h is None:
                out, h = self.gru(e[:,:-outpad_right]) # B x T x C
            else:
                out, h = self.gru(e[:,:-outpad_right], h) # B x T x C
            out_, _ = self.gru(e[:,-outpad_right:], h) # B x T x C
            e = torch.cat((out, out_), 1)
        else:
            # GRU e layers
            if h is None:
                e, h = self.gru(e) # B x T x C
            else:
                e, h = self.gru(e, h) # B x T x C
        # Output e layers
        if self.do_prob > 0 and do:
            e = self.out(self.gru_drop(e).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
        else:
            e = self.out(e.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C

        if self.pdf:
            if self.scale_out:
                mus = self.scale_out(F.tanhshrink(e[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
            else:
                mus = F.tanhshrink(e[:,:,:self.spec_dim])
            log_scales = F.logsigmoid(e[:,:,self.spec_dim:])
            if sampling:
                if do:
                    return torch.cat((mus, torch.clamp(log_scales, min=CLIP_1E16)), 2), \
                            sampling_laplace(mus, log_scales), h.detach()
                else:
                    return torch.cat((mus, log_scales), 2), \
                            sampling_laplace(mus, log_scales), h.detach()
            else:
                return torch.cat((mus, log_scales), 2), mus, h.detach()
        else:
            return self.scale_out(F.tanhshrink(e).transpose(1,2)).transpose(1,2), h.detach()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:
                return

        self.apply(_remove_weight_norm)


class GRU_POST_NET_(nn.Module):
    def __init__(self, spec_dim=80, excit_dim=6, hidden_layers=1, hidden_units=1024, causal_conv=True,
            kernel_size=7, dilation_size=1, do_prob=0, n_spk=14, use_weight_norm=True, 
                pad_first=True, right_size=None, res=False, laplace=False, ar=False):
        super(GRU_POST_NET_, self).__init__()
        self.n_spk = n_spk
        self.spec_dim = spec_dim
        self.excit_dim = excit_dim
        if self.excit_dim is not None:
            self.feat_dim = self.spec_dim+self.excit_dim
        else:
            self.feat_dim = self.spec_dim
        self.ar = ar
        if self.n_spk is not None:
            self.in_dim = self.feat_dim+self.n_spk
        else:
            self.in_dim = self.feat_dim
        self.laplace = laplace
        if not self.laplace:
            self.out_dim = self.spec_dim
        else:
            self.out_dim = self.spec_dim*2
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.use_weight_norm = use_weight_norm
        self.pad_first = pad_first
        self.right_size = right_size
        if self.laplace:
            self.res = True
        else:
            self.res = res

        self.scale_in = nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        if self.ar:
            self.scale_in_spec = nn.Conv1d(self.spec_dim, self.spec_dim, 1)

        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, nonlinear=False, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = self.conv.padding
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, nonlinear=False, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = 0
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                        right_size=self.right_size, nonlinear=False, pad_first=self.pad_first)
            self.pad_left = self.conv.left_size
            self.pad_right = self.conv.right_size

        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if not self.ar:
            if self.do_prob > 0 and self.hidden_layers > 1:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=False, batch_first=True)
            else:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers, \
                                    bidirectional=False, batch_first=True)
        else:
            if self.do_prob > 0 and self.hidden_layers > 1:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field+self.spec_dim, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=False, batch_first=True)
            else:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field+self.spec_dim, self.hidden_units, self.hidden_layers, \
                                    bidirectional=False, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        self.out = nn.Conv1d(self.hidden_units, self.out_dim, 1)

        # De-normalization layers
        self.scale_out = nn.Conv1d(self.spec_dim, self.spec_dim, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
        #    #torch.nn.utils.remove_weight_norm(self.scale_out)
        else:
            self.apply(initialize)

    def forward(self, x, y=None, e=None, h=None, do=False, outpad_right=0, x_prev=None, x_prev_1=None):
        if y is not None:
            if len(y.shape) == 2:
                if e is not None:
                    z = torch.cat((F.one_hot(y, num_classes=self.n_spk).float(), self.scale_in(torch.cat((e, x), 2).transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
                else:
                    z = torch.cat((F.one_hot(y, num_classes=self.n_spk).float(), self.scale_in(x.transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
            else:
                if e is not None:
                    z = torch.cat((y, self.scale_in(torch.cat((e, x), 2).transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
                else:
                    z = torch.cat((y, self.scale_in(x.transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
        else:
            if e is not None:
                z = self.scale_in(torch.cat((e, x), 2).transpose(1,2)).transpose(1,2) # B x T_frm x C
            else:
                z = self.scale_in(x.transpose(1,2)).transpose(1,2) # B x T_frm x C
        # Input e layers
        if self.do_prob > 0 and do:
            e = self.conv_drop(self.conv(z.transpose(1,2)).transpose(1,2)) # B x C x T --> B x T x C
        else:
            e = self.conv(z.transpose(1,2)).transpose(1,2) # B x C x T --> B x T x C
        if outpad_right > 0:
            # GRU e layers
            if h is None:
                out, h = self.gru(e[:,:-outpad_right]) # B x T x C
            else:
                out, h = self.gru(e[:,:-outpad_right], h) # B x T x C
            out_, _ = self.gru(e[:,-outpad_right:], h) # B x T x C
            e = torch.cat((out, out_), 1)
        else:
            # GRU e layers
            if h is None:
                e, h = self.gru(e) # B x T x C
            else:
                e, h = self.gru(e, h) # B x T x C
        # Output e layers
        if self.do_prob > 0 and do:
            e = self.out(self.gru_drop(e).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
        else:
            e = self.out(e.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C

        if self.laplace:
            if self.pad_right == 0:
                x_ = x[:,self.pad_left:]
            else:
                x_ = x[:,self.pad_left:-self.pad_right]
            mus_e = self.scale_out(F.tanhshrink(e[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
            mus = x_+mus_e
            log_scales = F.logsigmoid(e[:,:,self.spec_dim:])
            e = sampling_laplace(mus_e, log_scales)
            if do:
                return torch.cat((mus, torch.clamp(log_scales, min=CLIP_1E16)), 2), x_+e, h.detach()
            else:
                return torch.cat((mus, log_scales), 2), x_+e, h.detach()
        else:
            if self.res:
                if self.pad_right == 0:
                    return x[:,self.pad_left:]+self.scale_out(e.transpose(1,2)).transpose(1,2), h.detach()
                else:
                    return x[:,self.pad_left:-self.pad_right]+self.scale_out(e.transpose(1,2)).transpose(1,2), h.detach()
            else:
                return self.scale_out(e.transpose(1,2)).transpose(1,2), h.detach()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                #logging.debug(f"Weight norm is applied to {m}.")
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_VAE_ENCODER(nn.Module):
    def __init__(self, in_dim=50, n_spk=14, lat_dim=50, hidden_layers=1, hidden_units=1024, kernel_size=7,
            dilation_size=1, do_prob=0, bi=False, nonlinear_conv=False, n_quantize=None, excit_dim=None,
                use_weight_norm=True, causal_conv=False, cont=True, ar=False, right_size=0, pad_first=True):
        super(GRU_VAE_ENCODER, self).__init__()
        self.in_dim = in_dim
        self.n_spk = n_spk
        self.lat_dim = lat_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.bi = bool(bi)
        self.nonlinear_conv = nonlinear_conv
        self.causal_conv = causal_conv
        self.cont = cont
        self.right_size = right_size
        self.pad_first = pad_first
        self.use_weight_norm = use_weight_norm
        self.n_quantize = n_quantize
        self.excit_dim = excit_dim
        if self.cont:
            self.out_dim = self.n_spk+self.lat_dim*2
        else:
            self.out_dim = self.n_spk+self.lat_dim
        self.ar = ar
        if self.bi:
            self.hidden_units_out = 2*self.hidden_units
        else:
            self.hidden_units_out = self.hidden_units

        # Normalization layer
        if self.n_quantize is None:
            self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)
        elif self.n_quantize is not None and self.excit_dim is not None:
            self.scale_in = nn.Conv1d(self.excit_dim, self.excit_dim, 1)

        # Conv. layers
        if self.n_quantize is not None:
            if self.excit_dim is not None:
                self.spec_dim = self.in_dim - self.excit_dim
            else:
                self.spec_dim = self.in_dim
            self.spec_emb_dim = self.n_quantize // 4
            self.embed_spec = nn.Embedding(self.n_quantize, self.spec_emb_dim)
            self.spec_in_dim = self.spec_emb_dim * self.spec_dim
            if self.kernel_size > 1:
                if self.right_size <= 0:
                    if not self.causal_conv:
                        self.conv = TwoSidedDilConv1d(in_dim=self.in_dim-self.spec_dim+self.spec_in_dim, kernel_size=self.kernel_size, \
                                                    layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                        self.pad_left = self.conv.padding
                        self.pad_right = self.conv.padding
                    else:
                        self.conv = CausalDilConv1d(in_dim=self.in_dim-self.spec_dim+self.spec_in_dim, kernel_size=self.kernel_size, \
                                                    layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                        self.pad_left = self.conv.padding
                        self.pad_right = 0
                else:
                    self.conv = SkewedConv1d(in_dim=self.in_dim-self.spec_dim+self.spec_in_dim, kernel_size=self.kernel_size, \
                                                right_size=self.right_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                    self.pad_left = self.conv.left_size
                    self.pad_right = self.conv.right_size
                self.s_dim = 320
                conv_in = [nn.Conv1d(self.conv.out_dim, self.s_dim, 1), nn.ReLU()]
                self.conv_in = nn.Sequential(*conv_in)
                self.gru_in_dim = self.s_dim
            else:
                self.s_dim = 320
                conv_in = [nn.Conv1d(self.in_dim-self.spec_dim+self.spec_in_dim, self.s_dim, 1), nn.ReLU()]
                self.conv_in = nn.Sequential(*conv_in)
                self.gru_in_dim = self.s_dim
                self.pad_left = 0
                self.pad_right = 0
        else:
            if self.right_size <= 0:
                if not self.causal_conv:
                    self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                                layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                    self.pad_left = self.conv.padding
                    self.pad_right = self.conv.padding
                else:
                    self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                                layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                    self.pad_left = self.conv.padding
                    self.pad_right = 0
            else:
                self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            right_size=self.right_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.left_size
                self.pad_right = self.conv.right_size
            self.gru_in_dim = self.in_dim*self.conv.rec_field
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if self.do_prob > 0 and self.hidden_layers > 1:
            if self.ar:
                self.gru = nn.GRU(self.gru_in_dim+self.out_dim, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=self.bi, batch_first=True)
            else:
                self.gru = nn.GRU(self.gru_in_dim, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=self.bi, batch_first=True)
        else:
            if self.ar:
                self.gru = nn.GRU(self.gru_in_dim+self.out_dim, self.hidden_units, self.hidden_layers, \
                                    bidirectional=self.bi, batch_first=True)
            else:
                self.gru = nn.GRU(self.gru_in_dim, self.hidden_units, self.hidden_layers, \
                                    bidirectional=self.bi, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        self.out = nn.Conv1d(self.hidden_units_out, self.out_dim, 1)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
        else:
            self.apply(initialize)

    def forward(self, x, yz_in=None, h=None, do=False, sampling=True, outpad_right=0):
        if self.n_quantize is None:
            x_in = self.conv(self.scale_in(x.transpose(1,2))).transpose(1,2)
        else:
            if self.excit_dim is not None:
                e_in = self.scale_in(x[:,:,:self.excit_dim].transpose(1,2))
                x_in = torch.cat((e_in, self.embed_spec(x[:,:,self.excit_dim:].long()).reshape(x.shape[0], x.shape[1], -1).transpose(1,2)), 1)
            else:
                x_in = self.embed_spec(x.long()).reshape(x.shape[0], x.shape[1], -1).transpose(1,2)
            if self.kernel_size > 1:
                x_in = self.conv_in(self.conv(x_in)).transpose(1,2)
            else:
                x_in = self.conv_in(x_in).transpose(1,2)
        if not self.ar:
            # Input s layers
            if self.do_prob > 0 and do:
                s = self.conv_drop(x_in) # B x C x T --> B x T x C
            else:
                s = x_in # B x C x T --> B x T x C
            if outpad_right > 0:
                # GRU s layers
                if h is None:
                    out, h = self.gru(s[:,:-outpad_right]) # B x T x C
                else:
                    out, h = self.gru(s[:,:-outpad_right], h) # B x T x C
                out_, _ = self.gru(s[:,-outpad_right:], h) # B x T x C
                s = torch.cat((out, out_), 1)
            else:
                # GRU s layers
                if h is None:
                    s, h = self.gru(s) # B x T x C
                else:
                    s, h = self.gru(s, h) # B x T x C
            # Output s layers
            if self.do_prob > 0 and do:
                s = self.out(self.gru_drop(s).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
            else:
                s = self.out(s.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C

            if self.cont:
                qy_logits = F.selu(s[:,:,:self.n_spk])
                qz_alpha = torch.cat((s[:,:,self.n_spk:self.n_spk+self.lat_dim], F.logsigmoid(s[:,:,self.n_spk+self.lat_dim:])), 2)

                if sampling:
                    return qy_logits, qz_alpha, h.detach()
                else:
                    return qy_logits, qz_alpha, qz_alpha[:,:,:self.lat_dim], h.detach()
            else:
                return F.selu(s[:,:,:self.n_spk]), s[:,:,self.n_spk:], h.detach()
        else:
            # Input layers
            if self.do_prob > 0 and do:
                x_conv = self.conv_drop(x_in) # B x C x T --> B x T x C
            else:
                x_conv = x_in # B x C x T --> B x T x C

            T = x_conv.shape[1]
            T_last = T-outpad_right

            # GRU layers
            if h is None:
                out, h = self.gru(torch.cat((x_conv[:,:1], yz_in), 2)) # B x T x C
            else:
                out, h = self.gru(torch.cat((x_conv[:,:1], yz_in), 2), h) # B x T x C
            if self.do_prob > 0 and do:
                out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
            else:
                out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
            qy_logit = F.selu(out[:,:,:self.n_spk])
            if self.cont:
                qz_alpha = torch.cat((out[:,:,self.n_spk:self.n_spk+self.lat_dim], F.logsigmoid(out[:,:,self.n_spk+self.lat_dim:])), 2)
            else:
                qz_alpha = out[:,:,self.n_spk:]
            yz_in = torch.cat((qy_logit, qz_alpha), 2)
            qy_logits = qy_logit
            qz_alphas = qz_alpha
            if self.cont:
                if self.do_prob > 0 and do:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((x_conv[:,t:t+1], yz_in), 2), h) # B x T x C
                        out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        qy_logit = F.selu(out[:,:,:self.n_spk])
                        qz_alpha = torch.cat((out[:,:,self.n_spk:self.n_spk+self.lat_dim], F.logsigmoid(out[:,:,self.n_spk+self.lat_dim:])), 2)
                        yz_in = torch.cat((qy_logit, qz_alpha), 2)
                        qy_logits = torch.cat((qy_logits, qy_logit), 1)
                        qz_alphas = torch.cat((qz_alphas, qz_alpha), 1)
                    if T_last < T:
                        h_ = h
                        yz_in_ = yz_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((x_conv[:,t:t+1], yz_in_), 2), h_) # B x T x C
                            out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            qy_logit = F.selu(out[:,:,:self.n_spk])
                            qz_alpha = torch.cat((out[:,:,self.n_spk:self.n_spk+self.lat_dim], F.logsigmoid(out[:,:,self.n_spk+self.lat_dim:])), 2)
                            yz_in_ = torch.cat((qy_logit, qz_alpha), 2)
                            qy_logits = torch.cat((qy_logits, qy_logit), 1)
                            qz_alphas = torch.cat((qz_alphas, qz_alpha), 1)
                else:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((x_conv[:,t:t+1], yz_in), 2), h) # B x T x C
                        out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        qy_logit = F.selu(out[:,:,:self.n_spk])
                        qz_alpha = torch.cat((out[:,:,self.n_spk:self.n_spk+self.lat_dim], F.logsigmoid(out[:,:,self.n_spk+self.lat_dim:])), 2)
                        yz_in = torch.cat((qy_logit, qz_alpha), 2)
                        qy_logits = torch.cat((qy_logits, qy_logit), 1)
                        qz_alphas = torch.cat((qz_alphas, qz_alpha), 1)
                    if T_last < T:
                        h_ = h
                        yz_in_ = yz_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((x_conv[:,t:t+1], yz_in_), 2), h_) # B x T x C
                            out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            qy_logit = F.selu(out[:,:,:self.n_spk])
                            qz_alpha = torch.cat((out[:,:,self.n_spk:self.n_spk+self.lat_dim], F.logsigmoid(out[:,:,self.n_spk+self.lat_dim:])), 2)
                            yz_in_ = torch.cat((qy_logit, qz_alpha), 2)
                            qy_logits = torch.cat((qy_logits, qy_logit), 1)
                            qz_alphas = torch.cat((qz_alphas, qz_alpha), 1)
                if sampling:
                    return qy_logits, qz_alphas, h.detach(), yz_in.detach()
                else:
                    return qy_logits, qz_alphas, qz_alphas[:,:,:self.lat_dim], h.detach(), yz_in.detach()
            else:
                if self.do_prob > 0 and do:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((x_conv[:,t:t+1], yz_in), 2), h) # B x T x C
                        out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        qy_logit = F.selu(out[:,:,:self.n_spk])
                        qz_alpha = out[:,:,self.n_spk:]
                        yz_in = torch.cat((qy_logit, qz_alpha), 2)
                        qy_logits = torch.cat((qy_logits, qy_logit), 1)
                        qz_alphas = torch.cat((qz_alphas, qz_alpha), 1)
                    if T_last < T:
                        h_ = h
                        yz_in_ = yz_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((x_conv[:,t:t+1], yz_in_), 2), h_) # B x T x C
                            out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            qy_logit = F.selu(out[:,:,:self.n_spk])
                            qz_alpha = out[:,:,self.n_spk:]
                            yz_in_ = torch.cat((qy_logit, qz_alpha), 2)
                            qy_logits = torch.cat((qy_logits, qy_logit), 1)
                            qz_alphas = torch.cat((qz_alphas, qz_alpha), 1)
                else:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((x_conv[:,t:t+1], yz_in), 2), h) # B x T x C
                        out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        qy_logit = F.selu(out[:,:,:self.n_spk])
                        qz_alpha = out[:,:,self.n_spk:]
                        yz_in = torch.cat((qy_logit, qz_alpha), 2)
                        qy_logits = torch.cat((qy_logits, qy_logit), 1)
                        qz_alphas = torch.cat((qz_alphas, qz_alpha), 1)
                    if T_last < T:
                        h_ = h
                        yz_in_ = yz_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((x_conv[:,t:t+1], yz_in_), 2), h_) # B x T x C
                            out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            qy_logit = F.selu(out[:,:,:self.n_spk])
                            qz_alpha = out[:,:,self.n_spk:]
                            yz_in_ = torch.cat((qy_logit, qz_alpha), 2)
                            qy_logits = torch.cat((qy_logits, qy_logit), 1)
                            qz_alphas = torch.cat((qz_alphas, qz_alpha), 1)
                return qy_logits, qz_alphas, h.detach(), yz_in.detach()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                #logging.debug(f"Weight norm is applied to {m}.")
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_LAT_FEAT_CLASSIFIER(nn.Module):
    def __init__(self, lat_dim=None, feat_dim=50, n_spk=14, hidden_layers=1, hidden_units=32,
            use_weight_norm=True, adversarial=False, feat_aux_dim=None, do_prob=0):
        super(GRU_LAT_FEAT_CLASSIFIER, self).__init__()
        self.lat_dim = lat_dim
        self.feat_aux_dim = feat_aux_dim
        self.feat_dim = feat_dim
        self.n_spk = n_spk
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.use_weight_norm = use_weight_norm
        self.adversarial = adversarial
        self.do_prob = do_prob

        # Conv. layers
        if self.lat_dim is not None:
            conv_lat = [nn.Conv1d(self.lat_dim, self.hidden_units, 1), nn.ReLU()]
            self.conv_lat = nn.Sequential(*conv_lat)
        if self.feat_aux_dim is not None:
            conv_feat_aux = [nn.Conv1d(self.feat_aux_dim, self.hidden_units, 1), nn.ReLU()]
            self.conv_feat_aux = nn.Sequential(*conv_feat_aux)
        conv_feat = [nn.Conv1d(self.feat_dim, self.hidden_units, 1), nn.ReLU()]
        self.conv_feat = nn.Sequential(*conv_feat)

        # GRU layer(s)
        self.gru = nn.GRU(self.hidden_units, self.hidden_units, self.hidden_layers, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        if self.adversarial:
            self.out = nn.Conv1d(self.hidden_units, self.n_spk+1, 1)
        else:
            self.out = nn.Conv1d(self.hidden_units, self.n_spk, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
        else:
            self.apply(initialize)

    def forward(self, lat=None, feat=None, feat_aux=None, h=None, detach_adv=True, do=False):
        # Input layers
        if lat is not None:
            c = self.conv_lat(lat.transpose(1,2)).transpose(1,2)
            if self.adversarial and detach_adv:
                c_detach = self.conv_lat(lat.detach().transpose(1,2)).transpose(1,2)
        elif feat_aux is not None:
            c = self.conv_feat_aux(feat_aux.transpose(1,2)).transpose(1,2)
            if self.adversarial and detach_adv:
                c_detach = self.conv_feat_aux(feat_aux.detach().transpose(1,2)).transpose(1,2)
        else:
            c = self.conv_feat(feat.transpose(1,2)).transpose(1,2)
            if self.adversarial and detach_adv:
                c_detach = self.conv_feat(feat.detach().transpose(1,2)).transpose(1,2)
        if self.adversarial and detach_adv:
            if h is not None:
                out_detach, _ = self.gru(c_detach, h) # B x T x C
            else:
                out_detach, _ = self.gru(c_detach) # B x T x C
        # GRU layers
        if h is not None:
            out, h = self.gru(c, h) # B x T x C
        else:
            out, h = self.gru(c) # B x T x C
        if do and self.do_prob > 0:
            out = self.gru_drop(out)
        # Output layers
        if self.adversarial:
            out = F.selu(self.out(out.transpose(1,2))).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
            if detach_adv:
                return F.selu(out[:,:,:-1]), torch.sigmoid(out[:,:,-1]), torch.sigmoid(self.out(out_detach.transpose(1,2)).transpose(1,2)[:,:,-1]), h.detach()
            else:
                return F.selu(out[:,:,:-1]), torch.sigmoid(out[:,:,-1]), h.detach()
        else:
            return F.selu(self.out(out.transpose(1,2))).transpose(1,2), h.detach() # B x T x C -> B x C x T -> B x T x C

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                #logging.debug(f"Weight norm is applied to {m}.")
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class SPKID_TRANSFORM_LAYER(nn.Module):
    def __init__(self, n_spk=14, spkidtr_dim=2, use_weight_norm=True):
        super(SPKID_TRANSFORM_LAYER, self).__init__()

        self.n_spk = n_spk
        self.spkidtr_dim = spkidtr_dim
        self.use_weight_norm = use_weight_norm

        self.conv = nn.Conv1d(self.n_spk, self.spkidtr_dim, 1)
        deconv = [nn.Conv1d(self.spkidtr_dim, self.n_spk, 1), nn.ReLU()]
        #deconv = [nn.Conv1d(self.spkidtr_dim, self.n_spk, 1), nn.Sigmoid()]
        #deconv = [nn.Conv1d(self.spkidtr_dim, self.n_spk, 1), nn.Tanh()]
        self.deconv = nn.Sequential(*deconv)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
        #    #torch.nn.utils.remove_weight_norm(self.scale_out)
        else:
            self.apply(initialize)

    def forward(self, x):
        # in: B x T
        # out: B x T x C
        return self.deconv(self.conv(F.one_hot(x, num_classes=self.n_spk).float().transpose(1,2))).transpose(1,2)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                #logging.debug(f"Weight norm is applied to {m}.")
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_POST_NET(nn.Module):
    def __init__(self, spec_dim=80, excit_dim=6, hidden_layers=1, hidden_units=1024, causal_conv=True,
            kernel_size=7, dilation_size=1, do_prob=0, n_spk=14, use_weight_norm=True, 
                pad_first=True, right_size=None, res=False, laplace=False, ar=False):
        super(GRU_POST_NET, self).__init__()
        self.n_spk = n_spk
        self.spec_dim = spec_dim
        self.excit_dim = excit_dim
        if self.excit_dim is not None:
            self.feat_dim = self.spec_dim+self.excit_dim
        else:
            self.feat_dim = self.spec_dim
        self.ar = ar
        self.in_dim = self.feat_dim+self.n_spk
        self.laplace = laplace
        if not self.laplace:
            self.out_dim = self.spec_dim
        else:
            self.out_dim = self.spec_dim*2
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.use_weight_norm = use_weight_norm
        self.pad_first = pad_first
        self.right_size = right_size
        if self.laplace:
            self.res = True
        else:
            self.res = res

        self.scale_in = nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        if self.ar:
            self.scale_in_spec = nn.Conv1d(self.spec_dim, self.spec_dim, 1)

        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, nonlinear=False, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = self.conv.padding
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, nonlinear=False, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = 0
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                        right_size=self.right_size, nonlinear=False, pad_first=self.pad_first)
            self.pad_left = self.conv.left_size
            self.pad_right = self.conv.right_size

        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if not self.ar:
            if self.do_prob > 0 and self.hidden_layers > 1:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=False, batch_first=True)
            else:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers, \
                                    bidirectional=False, batch_first=True)
        else:
            if self.do_prob > 0 and self.hidden_layers > 1:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field+self.spec_dim, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=False, batch_first=True)
            else:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field+self.spec_dim, self.hidden_units, self.hidden_layers, \
                                    bidirectional=False, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        self.out = nn.Conv1d(self.hidden_units, self.out_dim, 1)

        # De-normalization layers
        self.scale_out = nn.Conv1d(self.spec_dim, self.spec_dim, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
        #    #torch.nn.utils.remove_weight_norm(self.scale_out)
        else:
            self.apply(initialize)

    def forward(self, y, x, e=None, h=None, do=False, outpad_right=0, x_prev=None, x_prev_1=None):
        if len(y.shape) == 2:
            if e is not None:
                z = torch.cat((F.one_hot(y, num_classes=self.n_spk).float(), self.scale_in(torch.cat((x, e), 2).transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
            else:
                z = torch.cat((F.one_hot(y, num_classes=self.n_spk).float(), self.scale_in(x.transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
        else:
            if e is not None:
                z = torch.cat((y, self.scale_in(torch.cat((x, e), 2).transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
            else:
                z = torch.cat((y, self.scale_in(x.transpose(1,2)).transpose(1,2)), 2) # B x T_frm x C
        # Input e layers
        if self.do_prob > 0 and do:
            e = self.conv_drop(self.conv(z.transpose(1,2)).transpose(1,2)) # B x C x T --> B x T x C
        else:
            e = self.conv(z.transpose(1,2)).transpose(1,2) # B x C x T --> B x T x C
        if not self.ar:
            if outpad_right > 0:
                # GRU e layers
                if h is None:
                    out, h = self.gru(e[:,:-outpad_right]) # B x T x C
                else:
                    out, h = self.gru(e[:,:-outpad_right], h) # B x T x C
                out_, _ = self.gru(e[:,-outpad_right:], h) # B x T x C
                e = torch.cat((out, out_), 1)
            else:
                # GRU e layers
                if h is None:
                    e, h = self.gru(e) # B x T x C
                else:
                    e, h = self.gru(e, h) # B x T x C
        else:
            if x_prev is not None:
                if self.pad_right == 0:
                    x_prev = self.scale_in_spec(x_prev[:,self.pad_left:].transpose(1,2)).transpose(1,2)
                else:
                    x_prev = self.scale_in_spec(x_prev[:,self.pad_left:-self.pad_right].transpose(1,2)).transpose(1,2)
                if outpad_right > 0:
                    # GRU e layers
                    if h is None:
                        out, h = self.gru(torch.cat((e, x_prev), 2)[:,:-outpad_right]) # B x T x C
                    else:
                        out, h = self.gru(torch.cat((e, x_prev), 2)[:,:-outpad_right], h) # B x T x C
                    out_, _ = self.gru(torch.cat((e, x_prev), 2)[:,-outpad_right:], h) # B x T x C
                    e = torch.cat((out, out_), 1)
                else:
                    # GRU e layers
                    if h is None:
                        e, h = self.gru(torch.cat((e, x_prev), 2)) # B x T x C
                    else:
                        e, h = self.gru(torch.cat((e, x_prev), 2), h) # B x T x C
            else:
                if x_prev_1 is not None:
                    x_prev = x_prev_1
                else:
                    x_prev = torch.zeros_like(x[:,:1])
                if self.pad_right == 0:
                    x_ = x[:,self.pad_left:]
                else:
                    x_ = x[:,self.pad_left:-self.pad_right]
                if h is None:
                    out, h = self.gru(torch.cat((e[:,:1], x_prev), 2)) # B x T x C
                else:
                    out, h = self.gru(torch.cat((e[:,:1], x_prev), 2), h) # B x T x C
                if self.do_prob > 0 and do:
                    out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                else:
                    out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                mus_e_ = self.scale_out(F.tanhshrink(out[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
                x_t = x_[:,:1]
                mus = mus_ = x_t+mus_e_
                log_scales = log_scales_ = F.logsigmoid(out[:,:,self.spec_dim:])
                x_e = x_e_ = x_t+sampling_laplace(mus_e_, log_scales_)
                x_prev = self.scale_in_spec(x_e_.transpose(1,2)).transpose(1,2)
                if self.do_prob > 0 and do:
                    if outpad_right > 0:
                        T = e.shape[1]-outpad_right
                        for t in range(1,T):
                            t_1 = t+1
                            out, h = self.gru(torch.cat((e[:,t:t_1], x_prev), 2), h) # B x T x C
                            out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            mus_e_ = self.scale_out(F.tanhshrink(out[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
                            x_t = x_[:,t:t_1]
                            mus_ = x_t+mus_e_
                            log_scales_ = F.logsigmoid(out[:,:,self.spec_dim:])
                            x_e_ = x_t+sampling_laplace(mus_e_, log_scales_)
                            x_prev = self.scale_in_spec(x_e_.transpose(1,2)).transpose(1,2)
                            mus = torch.cat((mus, mus_), 1)
                            log_scales = torch.cat((log_scales, log_scales_), 1)
                            x_e = torch.cat((x_e, x_e_), 1)
                        T = e.shape[1]
                        t_ = t
                        h_ = h
                        x_prev_ = x_prev
                        for t in range(t_,T):
                            t_1 = t+1
                            out, h_ = self.gru(torch.cat((e[:,t:t_1], x_prev_), 2), h_) # B x T x C
                            out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            mus_e_ = self.scale_out(F.tanhshrink(out[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
                            x_t = x_[:,t:t_1]
                            mus_ = x_t+mus_e_
                            mus_ = x_t+mus_e_
                            log_scales_ = F.logsigmoid(out[:,:,self.spec_dim:])
                            x_e_ = x_t+sampling_laplace(mus_e_, log_scales_)
                            x_prev_ = self.scale_in_spec(x_e_.transpose(1,2)).transpose(1,2)
                            mus = torch.cat((mus, mus_), 1)
                            log_scales = torch.cat((log_scales, log_scales_), 1)
                            x_e = torch.cat((x_e, x_e_), 1)
                    else:
                        T = e.shape[1]
                        for t in range(1,T):
                            t_1 = t+1
                            out, h = self.gru(torch.cat((e[:,t:t_1], x_prev), 2), h) # B x T x C
                            out = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            mus_e_ = self.scale_out(F.tanhshrink(out[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
                            x_t = x_[:,t:t_1]
                            mus_ = x_t+mus_e_
                            log_scales_ = F.logsigmoid(out[:,:,self.spec_dim:])
                            x_e_ = x_t+sampling_laplace(mus_e_, log_scales_)
                            x_prev = self.scale_in_spec(x_e_.transpose(1,2)).transpose(1,2)
                            mus = torch.cat((mus, mus_), 1)
                            log_scales = torch.cat((log_scales, log_scales_), 1)
                            x_e = torch.cat((x_e, x_e_), 1)
                else:
                    if outpad_right > 0:
                        T = e.shape[1]-outpad_right
                        for t in range(1,T):
                            t_1 = t+1
                            out, h = self.gru(torch.cat((e[:,t:t_1], x_prev), 2), h) # B x T x C
                            out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            mus_e_ = self.scale_out(F.tanhshrink(out[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
                            x_t = x_[:,t:t_1]
                            mus_ = x_t+mus_e_
                            log_scales_ = F.logsigmoid(out[:,:,self.spec_dim:])
                            x_e_ = x_t+sampling_laplace(mus_e_, log_scales_)
                            x_prev = self.scale_in_spec(x_e_.transpose(1,2)).transpose(1,2)
                            mus = torch.cat((mus, mus_), 1)
                            log_scales = torch.cat((log_scales, log_scales_), 1)
                            x_e = torch.cat((x_e, x_e_), 1)
                        T = e.shape[1]
                        t_ = t
                        h_ = h
                        x_prev_ = x_prev
                        for t in range(t_,T):
                            t_1 = t+1
                            out, h_ = self.gru(torch.cat((e[:,t:t_1], x_prev_), 2), h_) # B x T x C
                            out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            mus_e_ = self.scale_out(F.tanhshrink(out[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
                            x_t = x_[:,t:t_1]
                            mus_ = x_t+mus_e_
                            log_scales_ = F.logsigmoid(out[:,:,self.spec_dim:])
                            x_e_ = x_t+sampling_laplace(mus_e_, log_scales_)
                            x_prev_ = self.scale_in_spec(x_e_.transpose(1,2)).transpose(1,2)
                            mus = torch.cat((mus, mus_), 1)
                            log_scales = torch.cat((log_scales, log_scales_), 1)
                            x_e = torch.cat((x_e, x_e_), 1)
                    else:
                        T = e.shape[1]
                        for t in range(1,T):
                            t_1 = t+1
                            out, h = self.gru(torch.cat((e[:,t:t_1], x_prev), 2), h) # B x T x C
                            out = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            mus_e_ = self.scale_out(F.tanhshrink(out[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
                            x_t = x_[:,t:t_1]
                            mus_ = x_t+mus_e_
                            log_scales_ = F.logsigmoid(out[:,:,self.spec_dim:])
                            x_e_ = x_t+sampling_laplace(mus_e_, log_scales_)
                            x_prev = self.scale_in_spec(x_e_.transpose(1,2)).transpose(1,2)
                            mus = torch.cat((mus, mus_), 1)
                            log_scales = torch.cat((log_scales, log_scales_), 1)
                            x_e = torch.cat((x_e, x_e_), 1)
                if do:
                    return mus, torch.clamp(log_scales, min=CLIP_1E16), x_e, x_prev.detach(), h.detach()
                else:
                    return mus, log_scales, x_e, x_prev.detach(), h.detach()
        # Output e layers
        if self.do_prob > 0 and do:
            e = self.out(self.gru_drop(e).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
        else:
            e = self.out(e.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C

        if self.laplace:
            if self.pad_right == 0:
                x_ = x[:,self.pad_left:]
            else:
                x_ = x[:,self.pad_left:-self.pad_right]
            mus_e = self.scale_out(F.tanhshrink(e[:,:,:self.spec_dim]).transpose(1,2)).transpose(1,2)
            mus = x_+mus_e
            log_scales = F.logsigmoid(e[:,:,self.spec_dim:])
            e = sampling_laplace(mus_e, log_scales)
            if do:
                return mus, torch.clamp(log_scales, min=CLIP_1E16), x_+e, h.detach()
            else:
                return mus, log_scales, x_+e, h.detach()
        else:
            if self.res:
                if self.pad_right == 0:
                    return x[:,self.pad_left:]+self.scale_out(e.transpose(1,2)).transpose(1,2), h.detach()
                else:
                    return x[:,self.pad_left:-self.pad_right]+self.scale_out(e.transpose(1,2)).transpose(1,2), h.detach()
            else:
                return self.scale_out(e.transpose(1,2)).transpose(1,2), h.detach()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                #logging.debug(f"Weight norm is applied to {m}.")
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_SPEC_DECODER(nn.Module):
    def __init__(self, feat_dim=50, out_dim=50, hidden_layers=1, hidden_units=1024, causal_conv=False,
            kernel_size=7, dilation_size=1, do_prob=0, n_spk=14, bi=False, nonlinear_conv=False,
                use_weight_norm=True, ar=False, cap_dim=None, excit_dim=None, spkidtr_dim=0,
                    pad_first=True, diff=False, n_quantize=None, n_quantize_reg=None, right_size=None):
        super(GRU_SPEC_DECODER, self).__init__()
        self.n_spk = n_spk
        self.feat_dim = feat_dim
        self.spkidtr_dim = spkidtr_dim
        self.in_dim = self.n_spk+self.feat_dim
        self.cap_dim = cap_dim
        self.n_quantize = n_quantize
        if n_quantize_reg is not None:
            self.n_quantize_reg = n_quantize_reg // 2
        else:
            self.n_quantize_reg = None
        self.spec_dim = out_dim
        if self.cap_dim is not None:
            if self.n_quantize is not None:
                self.spec_dim_out = self.n_quantize // 4
                self.out_dim = self.spec_dim_out*self.spec_dim + 1 + self.cap_dim
            else:
                self.out_dim = self.spec_dim+1+self.cap_dim
            self.uvcap_dim = self.cap_dim+1
        else:
            if self.n_quantize is not None:
                self.spec_dim_out = self.n_quantize // 4
                self.out_dim = self.spec_dim_out*self.spec_dim
            else:
                self.out_dim = self.spec_dim
            self.uvcap_dim = 0
        self.excit_dim = excit_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.bi = bool(bi)
        self.nonlinear_conv = nonlinear_conv
        self.ar = ar
        self.use_weight_norm = use_weight_norm
        self.diff = diff
        self.pad_first = pad_first
        self.right_size = right_size
        if self.bi:
            self.hidden_units_out = 2*self.hidden_units
        else:
            self.hidden_units_out = self.hidden_units

        if self.excit_dim is not None:
            self.scale_in = nn.Conv1d(self.excit_dim, self.excit_dim, 1)
            self.in_dim += self.excit_dim

        # Conv. layers
        if self.spkidtr_dim > 0:
            self.spkidtr_conv = nn.Conv1d(self.n_spk, self.spkidtr_dim, 1)
            spkidtr_deconv = [nn.Conv1d(self.spkidtr_dim, self.n_spk, 1), nn.ReLU()]
            self.spkidtr_deconv = nn.Sequential(*spkidtr_deconv)

        #if not self.causal_conv:
        #    self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
        #                                layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
        #    self.pad_left = self.conv.padding
        #    self.pad_right = self.conv.padding
        #else:
        #    self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
        #                                layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
        #    self.pad_left = self.conv.padding
        #    self.pad_right = 0
        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = self.conv.padding
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = 0
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                        right_size=self.right_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
            self.pad_left = self.conv.left_size
            self.pad_right = self.conv.right_size

        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if self.do_prob > 0 and self.hidden_layers > 1:
            if self.ar:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field+self.out_dim, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=self.bi, batch_first=True)
            else:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=self.bi, batch_first=True)
        else:
            if self.ar:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field+self.out_dim, self.hidden_units, self.hidden_layers, \
                                    bidirectional=self.bi, batch_first=True)
            else:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers, \
                                    bidirectional=self.bi, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        if self.n_quantize is not None:
            self.out = nn.Conv1d(self.hidden_units_out, self.spec_dim_out*self.spec_dim, 1)
            self.out_logits = nn.Conv1d(self.spec_dim_out, self.n_quantize, 1)
        else:
            self.out = nn.Conv1d(self.hidden_units_out, self.out_dim, 1)

        # De-normalization layers
        if self.cap_dim is not None:
            self.scale_out_cap = nn.Conv1d(self.cap_dim, self.cap_dim, 1)
        if self.n_quantize is None and self.n_quantize_reg is None:
            self.scale_out = nn.Conv1d(self.out_dim-self.uvcap_dim, self.out_dim-self.uvcap_dim, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
        #    #torch.nn.utils.remove_weight_norm(self.scale_out)
        else:
            self.apply(initialize)

    def forward(self, y, z, x_in=None, h=None, x_prev=None, do=False, e=None, outpad_right=0):
        if len(y.shape) == 2:
            if self.spkidtr_dim > 0:
                if e is not None:
                    z = torch.cat((self.spkidtr_deconv(self.spkidtr_conv(F.one_hot(y, num_classes=self.n_spk).float().transpose(1,2))).transpose(1,2), self.scale_in(e.transpose(1,2)).transpose(1,2), z), 2)
                else:
                    z = torch.cat((self.spkidtr_deconv(self.spkidtr_conv(F.one_hot(y, num_classes=self.n_spk).float().transpose(1,2))).transpose(1,2), z), 2) # B x T_frm x C
            else:
                if e is not None:
                    #logging.info(y.shape)
                    #logging.info(e.shape)
                    #logging.info(z.shape)
                    z = torch.cat((F.one_hot(y, num_classes=self.n_spk).float(), self.scale_in(e.transpose(1,2)).transpose(1,2), z), 2) # B x T_frm x C
                else:
                    z = torch.cat((F.one_hot(y, num_classes=self.n_spk).float(), z), 2) # B x T_frm x C
        else:
            if self.spkidtr_dim > 0:
                if e is not None:
                    z = torch.cat((self.spkidtr_deconv(y.transpose(1,2)).transpose(1,2), self.scale_in(e.transpose(1,2)).transpose(1,2), z), 2) # B x T_frm x C
                else:
                    z = torch.cat((self.spkidtr_deconv(y.transpose(1,2)).transpose(1,2), z), 2) # B x T_frm x C
            else:
                if e is not None:
                    z = torch.cat((y, self.scale_in(e.transpose(1,2)).transpose(1,2), z), 2) # B x T_frm x C
                else:
                    z = torch.cat((y, z), 2) # B x T_frm x C
        if not self.ar:
            # Input e layers
            if self.do_prob > 0 and do:
                e = self.conv_drop(self.conv(z.transpose(1,2)).transpose(1,2)) # B x C x T --> B x T x C
            else:
                e = self.conv(z.transpose(1,2)).transpose(1,2) # B x C x T --> B x T x C
            if outpad_right > 0:
                # GRU e layers
                if h is None:
                    out, h = self.gru(e[:,:-outpad_right]) # B x T x C
                else:
                    out, h = self.gru(e[:,:-outpad_right], h) # B x T x C
                out_, _ = self.gru(e[:,-outpad_right:], h) # B x T x C
                e = torch.cat((out, out_), 1)
            else:
                # GRU e layers
                if h is None:
                    e, h = self.gru(e) # B x T x C
                else:
                    e, h = self.gru(e, h) # B x T x C
            # Output e layers
            if self.n_quantize is not None:
                if self.do_prob > 0 and do:
                    e = self.out(self.gru_drop(e).transpose(1,2))
                else:
                    e = self.out(e.transpose(1,2))
                e = F.tanhshrink(self.out_logits(e.reshape(e.shape[0], self.spec_dim_out, -1)).reshape(e.shape[0], self.n_quantize, -1, e.shape[2]).permute(0,3,2,1))
            else:
                if self.do_prob > 0 and do:
                    e = self.out(self.gru_drop(e).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                else:
                    e = self.out(e.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                #if self.n_quantize_reg is not None:
                #    #e = torch.round(torch.clamp(e, min=0, max=self.n_quantize_reg*2))
                #    e = torch.tanh(e) * self.n_quantize_reg + self.n_quantize_reg
                #    #e = ((((torch.round(torch.clamp(e, min=0, max=self.n_quantize_reg*2)) - self.n_quantize_reg) / self.n_quantize_reg)+1)*diff_mcep_bound+2*min_mcep_bound)/2

            if self.cap_dim is not None:
                if not self.diff:
                    return torch.cat((torch.sigmoid(e[:,:,:1]), self.scale_out_cap(e[:,:,1:self.uvcap_dim].transpose(1,2)).transpose(1,2), \
                                    self.scale_out(e[:,:,self.uvcap_dim:].transpose(1,2)).transpose(1,2)), 2), h.detach()
                else:
                    e = torch.cat((x_prev, e[:,:-1]), 1) + e
                    return torch.cat((torch.sigmoid(e[:,:,:1]), self.scale_out_cap(e[:,:,1:self.uvcap_dim].transpose(1,2)).transpose(1,2), \
                                    self.scale_out(e[:,:,self.uvcap_dim:].transpose(1,2)).transpose(1,2)), 2), h.detach(), e[:,-1:].detach()
            else:
                if not self.diff:
                    if self.n_quantize is not None:
                        return e, h.detach()
                    else:
                        return self.scale_out(e.transpose(1,2)).transpose(1,2), h.detach()
                else:
                    e = torch.cat((x_prev, e[:,:-1]), 1) + e
                    return self.scale_out(e.transpose(1,2)).transpose(1,2), h.detach(), e[:,-1:].detach()
        else:
            # Input layers
            if self.do_prob > 0 and do:
                z_conv = self.conv_drop(self.conv(z.transpose(1,2)).transpose(1,2)) # B x C x T --> B x T x C
            else:
                z_conv = self.conv(z.transpose(1,2)).transpose(1,2) # B x C x T --> B x T x C
    
            T = z_conv.shape[1]
            T_last = T-outpad_right

            # GRU layers
            if h is None:
                out, h = self.gru(torch.cat((z_conv[:,:1], x_in), 2)) # B x T x C
            else:
                out, h = self.gru(torch.cat((z_conv[:,:1], x_in), 2), h) # B x T x C
            if self.do_prob > 0 and do:
                if not self.diff:
                    x_in = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                else:
                    x_in = x_in + self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
            else:
                if not self.diff:
                    x_in = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                else:
                    x_in = x_in + self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
            spec = x_in
            if self.do_prob > 0 and do:
                if not self.diff:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((z_conv[:,t:t+1], x_in), 2), h) # B x T x C
                        x_in = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        spec = torch.cat((spec, x_in), 1)
                    if T_last < T:
                        h_ = h
                        x_in_ = x_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((z_conv[:,t:t+1], x_in_), 2), h_) # B x T x C
                            x_in_ = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            spec = torch.cat((spec, x_in_), 1)
                else:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((z_conv[:,t:t+1], x_in), 2), h) # B x T x C
                        x_in = x_in + self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        spec = torch.cat((spec, x_in), 1)
                    if T_last < T:
                        h_ = h
                        x_in_ = x_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((z_conv[:,t:t+1], x_in_), 2), h_) # B x T x C
                            x_in_ = x_in_ + self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            spec = torch.cat((spec, x_in_), 1)
            else:
                if not self.diff:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((z_conv[:,t:t+1], x_in), 2), h) # B x T x C
                        x_in = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        spec = torch.cat((spec, x_in), 1)
                    if T_last < T:
                        h_ = h
                        x_in_ = x_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((z_conv[:,t:t+1], x_in_), 2), h_) # B x T x C
                            x_in_ = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            spec = torch.cat((spec, x_in_), 1)
                else:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((z_conv[:,t:t+1], x_in), 2), h) # B x T x C
                        x_in = x_in + self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        spec = torch.cat((spec, x_in), 1)
                    if T_last < T:
                        h_ = h
                        x_in_ = x_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((z_conv[:,t:t+1], x_in_), 2), h_) # B x T x C
                            x_in_ = x_in_ + self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            spec = torch.cat((spec, x_in_), 1)
    
            if self.cap_dim is not None:
                return torch.cat((torch.sigmoid(spec[:,:,:1]), self.scale_out_cap(spec[:,:,1:self.uvcap_dim].transpose(1,2)).transpose(1,2), \
                                self.scale_out(spec[:,:,self.uvcap_dim:].transpose(1,2)).transpose(1,2)), 2), h.detach()
            else:
                return self.scale_out(spec.transpose(1,2)).transpose(1,2), h.detach(), x_in.detach()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                #logging.debug(f"Weight norm is applied to {m}.")
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_EXCIT_DECODER(nn.Module):
    def __init__(self, feat_dim=50, hidden_layers=1, hidden_units=1024, causal_conv=False,
            kernel_size=7, dilation_size=1, do_prob=0, n_spk=14, bi=False, nonlinear_conv=False,
                use_weight_norm=True, ar=False, cap_dim=None, spkidtr_dim=0, right_size=0,
                    pad_first=True, diff=False):
        super(GRU_EXCIT_DECODER, self).__init__()
        self.n_spk = n_spk
        self.feat_dim = feat_dim
        self.spkidtr_dim = spkidtr_dim
        self.in_dim = self.n_spk+self.feat_dim
        self.cap_dim = cap_dim
        if self.cap_dim is not None:
            self.out_dim = 2+1+self.cap_dim
        else:
            self.out_dim = 2
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.bi = bool(bi)
        self.nonlinear_conv = nonlinear_conv
        self.ar = ar
        self.use_weight_norm = use_weight_norm
        self.diff = diff
        self.pad_first = pad_first
        self.right_size = right_size
        if self.bi:
            self.hidden_units_out = 2*self.hidden_units
        else:
            self.hidden_units_out = self.hidden_units

        # Conv. layers
        if self.spkidtr_dim > 0:
            self.spkidtr_conv = nn.Conv1d(self.n_spk, self.spkidtr_dim, 1)
            spkidtr_deconv = [nn.Conv1d(self.spkidtr_dim, self.n_spk, 1), nn.ReLU()]
            self.spkidtr_deconv = nn.Sequential(*spkidtr_deconv)

        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = self.conv.padding
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = 0
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                        right_size=self.right_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
            self.pad_left = self.conv.left_size
            self.pad_right = self.conv.right_size
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if self.do_prob > 0 and self.hidden_layers > 1:
            if self.ar:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field+self.out_dim, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=self.bi, batch_first=True)
            else:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers, \
                                    dropout=self.do_prob, bidirectional=self.bi, batch_first=True)
        else:
            if self.ar:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field+self.out_dim, self.hidden_units, self.hidden_layers, \
                                    bidirectional=self.bi, batch_first=True)
            else:
                self.gru = nn.GRU(self.in_dim*self.conv.rec_field, self.hidden_units, self.hidden_layers, \
                                    bidirectional=self.bi, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        self.out = nn.Conv1d(self.hidden_units_out, self.out_dim, 1)

        # De-normalization layers
        self.scale_out = nn.Conv1d(1, 1, 1)
        if self.cap_dim is not None:
            self.scale_out_cap = nn.Conv1d(self.cap_dim, self.cap_dim, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
            #torch.nn.utils.remove_weight_norm(self.scale_in)
            #torch.nn.utils.remove_weight_norm(self.scale_out)
        #    #if self.cap_dim is not None:
        #    #    torch.nn.utils.remove_weight_norm(self.scale_out_cap)
        else:
            self.apply(initialize)

    def forward(self, y, z, e_in=None, h=None, do=False, outpad_right=0):
        if len(y.shape) == 2:
            if self.spkidtr_dim > 0:
                z = torch.cat((self.spkidtr_deconv(self.spkidtr_conv(F.one_hot(y, num_classes=self.n_spk).float().transpose(1,2))).transpose(1,2), z), 2) # B x T_frm x C
            else:
                #logging.info(y.shape)
                #logging.info(z.shape)
                z = torch.cat((F.one_hot(y, num_classes=self.n_spk).float(), z), 2) # B x T_frm x C
        else:
            if self.spkidtr_dim > 0:
                z = torch.cat((self.spkidtr_deconv(y.transpose(1,2)).transpose(1,2), z), 2) # B x T_frm x C
            else:
                z = torch.cat((y, z), 2) # B x T_frm x C
        if not self.ar:
            # Input e layers
            if self.do_prob > 0 and do:
                e = self.conv_drop(self.conv(z.transpose(1,2)).transpose(1,2)) # B x C x T --> B x T x C
            else:
                e = self.conv(z.transpose(1,2)).transpose(1,2) # B x C x T --> B x T x C
            if outpad_right > 0:
                # GRU e layers
                if h is None:
                    out, h = self.gru(e[:,:-outpad_right]) # B x T x C
                else:
                    out, h = self.gru(e[:,:-outpad_right], h) # B x T x C
                out_, _ = self.gru(e[:,-outpad_right:], h) # B x T x C
                e = torch.cat((out, out_), 1)
            else:
                # GRU e layers
                if h is None:
                    e, h = self.gru(e) # B x T x C
                else:
                    e, h = self.gru(e, h) # B x T x C
            # Output e layers
            if self.do_prob > 0 and do:
                e = self.out(self.gru_drop(e).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
            else:
                e = self.out(e.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C

            if self.cap_dim is not None:
                return torch.cat((torch.sigmoid(e[:,:,:1]), self.scale_out(e[:,:,1:2].transpose(1,2)).transpose(1,2), \
                                torch.sigmoid(e[:,:,2:3]), self.scale_out_cap(e[:,:,3:].transpose(1,2)).transpose(1,2)), 2), h.detach()
                #return torch.cat((torch.sigmoid(e[:,:,:1]), torch.clamp(self.scale_out(e[:,:,1:2].transpose(1,2)).transpose(1,2), max=8), \
                #                torch.sigmoid(e[:,:,2:3]), torch.clamp(self.scale_out_cap(e[:,:,3:].transpose(1,2)).transpose(1,2), max=8)), 2), h.detach()
                #return torch.cat((torch.sigmoid(e[:,:,:1]), torch.clamp(F.relu(e[:,:,1:2]).transpose(1,2),max=6.8).transpose(1,2), \
                #                torch.sigmoid(e[:,:,2:3]), torch.clamp(F.relu(e[:,:,3:]).transpose(1,2),max=4).transpose(1,2)), 2), h.detach()
            else:
                #return torch.cat((torch.sigmoid(e[:,:,:1]), self.scale_out(e[:,:,1:].transpose(1,2)).transpose(1,2)), 2), \
                #        h.detach()
                return torch.cat((torch.sigmoid(e[:,:,:1]), torch.clamp(self.scale_out(e[:,:,1:].transpose(1,2)).transpose(1,2), max=8)), 2), \
                        h.detach()
                #return torch.cat((torch.sigmoid(torch.clamp(e[:,:,:1], min=-27.631021)), torch.clamp(self.scale_out(e[:,:,1:].transpose(1,2)).transpose(1,2), max=8)), 2), \
                #        h.detach()
        else:
            # Input layers
            if self.do_prob > 0 and do:
                z_conv = self.conv_drop(self.conv(z.transpose(1,2)).transpose(1,2)) # B x C x T --> B x T x C
            else:
                z_conv = self.conv(z.transpose(1,2)).transpose(1,2) # B x C x T --> B x T x C
    
            T = z_conv.shape[1]
            T_last = T-outpad_right

            # GRU layers
            if h is None:
                out, h = self.gru(torch.cat((z_conv[:,:1], e_in), 2)) # B x T x C
            else:
                out, h = self.gru(torch.cat((z_conv[:,:1], e_in), 2), h) # B x T x C
            if self.do_prob > 0 and do:
                if not self.diff:
                    e_in = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                else:
                    e_in_ = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                    e_in = torch.cat((e_in_[:,:,:3], e_in[:,:,3:]+e_in_[:,:,3:]), 2)
            else:
                if not self.diff:
                    e_in = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                else:
                    e_in_ = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                    e_in = torch.cat((e_in_[:,:,:3], e_in[:,:,3:]+e_in_[:,:,3:]), 2)
            excit = e_in
            if self.do_prob > 0 and do:
                if not self.diff:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((z_conv[:,t:t+1], e_in), 2), h) # B x T x C
                        e_in = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        excit = torch.cat((excit, e_in), 1)
                    if T_last < T:
                        h_ = h
                        e_in_ = e_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((z_conv[:,t:t+1], e_in_), 2), h_) # B x T x C
                            e_in_ = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            excit = torch.cat((excit, e_in_), 1)
                else:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((z_conv[:,t:t+1], e_in), 2), h) # B x T x C
                        e_in_ = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        e_in = torch.cat((e_in_[:,:,:3], e_in[:,:,3:]+e_in_[:,:,3:]), 2)
                        excit = torch.cat((excit, e_in), 1)
                    if T_last < T:
                        h_ = h
                        e_in_ = e_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((z_conv[:,t:t+1], e_in_), 2), h_) # B x T x C
                            e_in__ = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            e_in_ = torch.cat((e_in_[:,:,:3], e_in_[:,:,3:]+e_in__[:,:,3:]), 2)
                            excit = torch.cat((excit, e_in_), 1)
            else:
                if not self.diff:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((z_conv[:,t:t+1], e_in), 2), h) # B x T x C
                        e_in = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        excit = torch.cat((excit, e_in), 1)
                    if T_last < T:
                        h_ = h
                        e_in_ = e_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((z_conv[:,t:t+1], e_in_), 2), h_) # B x T x C
                            e_in_ = self.out(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            excit = torch.cat((excit, e_in_), 1)
                else:
                    for t in range(1,T_last):
                        out, h = self.gru(torch.cat((z_conv[:,t:t+1], e_in), 2), h) # B x T x C
                        e_in_ = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                        e_in = torch.cat((e_in_[:,:,:3], e_in[:,:,3:]+e_in_[:,:,3:]), 2)
                        excit = torch.cat((excit, e_in), 1)
                    if T_last < T:
                        h_ = h
                        e_in_ = e_in
                        for t in range(T_last,T):
                            out, h_ = self.gru(torch.cat((z_conv[:,t:t+1], e_in_), 2), h_) # B x T x C
                            e_in__ = self.out(self.gru_drop(out).transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                            e_in_ = torch.cat((e_in_[:,:,:3], e_in_[:,:,3:]+e_in__[:,:,3:]), 2)
                            excit = torch.cat((excit, e_in_), 1)

            if self.cap_dim is not None:
                return torch.cat((torch.sigmoid(excit[:,:,:1]), torch.clamp(self.scale_out(excit[:,:,1:2].transpose(1,2)).transpose(1,2), max=8), \
                                torch.sigmoid(excit[:,:,2:3]), torch.clamp(self.scale_out_cap(excit[:,:,3:].transpose(1,2)).transpose(1,2), max=8)), 2), h.detach(), e_in.detach()
            else:
                return torch.cat((torch.sigmoid(excit[:,:,:1]), torch.clamp(self.scale_out(excit[:,:,1:].transpose(1,2)).transpose(1,2), max=8)), 2), h.detach(), e_in.detach()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                #logging.debug(f"Weight norm is applied to {m}.")
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF(nn.Module):
    def __init__(self, feat_dim=52, upsampling_factor=120, hidden_units=384, hidden_units_2=24, n_quantize=65536, lpc=6,
            kernel_size=7, dilation_size=1, do_prob=0, causal_conv=False, use_weight_norm=True, nonlinear_conv=False,
                right_size=0, n_bands=2, n_spk=None, lat_in=0, excit_dim=0, pad_first=False, quantize_spec=None, remove_weight_norm_scale=True):
        super(GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND_CF, self).__init__()
        self.feat_dim = feat_dim
        self.in_dim = self.feat_dim
        self.n_quantize = n_quantize
        self.cf_dim = int(np.sqrt(self.n_quantize))
        if self.cf_dim > 64:
            self.cf_dim_in = 64
        else:
            self.cf_dim_in = self.cf_dim
        self.out_dim = self.n_quantize
        self.n_bands = n_bands
        self.upsampling_factor = upsampling_factor // self.n_bands
        self.hidden_units = hidden_units
        self.hidden_units_2 = hidden_units_2
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.nonlinear_conv = nonlinear_conv
        self.s_dim = 320
        self.wav_dim = self.s_dim // self.n_bands
        self.wav_dim = self.cf_dim_in
        self.wav_dim_bands = self.wav_dim * self.n_bands
        self.use_weight_norm = use_weight_norm
        self.lpc = lpc
        self.right_size = right_size
        self.n_spk = n_spk
        self.lat_in = lat_in
        self.excit_dim = excit_dim
        self.pad_first = pad_first
        self.quantize_spec = quantize_spec
        self.remove_weight_norm_scale = remove_weight_norm_scale

        # Conv. layers
        if self.quantize_spec is not None:
            if self.excit_dim > 0:
                self.scale_in = nn.Conv1d(self.excit_dim, self.excit_dim, 1)
            if self.right_size <= 0:
                if not self.causal_conv:
                    self.conv = TwoSidedDilConv1d(in_dim=self.lat_in+self.excit_dim, kernel_size=self.kernel_size,
                                                layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                    self.pad_left = self.conv.padding
                    self.pad_right = self.conv.padding
                else:
                    self.conv = CausalDilConv1d(in_dim=self.lat_in+self.excit_dim, kernel_size=self.kernel_size,
                                                layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                    self.pad_left = self.conv.padding
                    self.pad_right = 0
            else:
                self.conv = SkewedConv1d(in_dim=self.lat_in+self.excit_dim, kernel_size=self.kernel_size,
                                            right_size=self.right_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.left_size
                self.pad_right = self.conv.right_size
            if self.excit_dim is not None:
                self.spec_dim = self.in_dim - self.excit_dim
            else:
                self.spec_dim = self.in_dim
            self.spec_emb_dim = self.quantize_spec // 4
            self.embed_spec = nn.Embedding(self.quantize_spec, self.spec_emb_dim)
            self.spec_in_dim = self.spec_emb_dim * self.spec_dim
            conv_s_c = [nn.Conv1d((self.lat_in+self.excit_dim)*self.conv.rec_field+self.spec_in_dim, self.s_dim, 1), nn.ReLU()]
            self.conv_s_c = nn.Sequential(*conv_s_c)
        else:
            self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)
            if self.right_size <= 0:
                if not self.causal_conv:
                    self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                                layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                    self.pad_left = self.conv.padding
                    self.pad_right = self.conv.padding
                else:
                    self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                                layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                    self.pad_left = self.conv.padding
                    self.pad_right = 0
            else:
                self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            right_size=self.right_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.left_size
                self.pad_right = self.conv.right_size
            conv_s_c = [nn.Conv1d(self.in_dim*self.conv.rec_field, self.s_dim, 1), nn.ReLU()]
            self.conv_s_c = nn.Sequential(*conv_s_c)
        if self.do_prob > 0:
            self.drop = nn.Dropout(p=self.do_prob)

        self.embed_c_wav = nn.Embedding(self.cf_dim, self.wav_dim)
        self.embed_f_wav = nn.Embedding(self.cf_dim, self.wav_dim)

        # GRU layer(s) coarse
        self.gru = nn.GRU(self.s_dim+self.wav_dim_bands*2, self.hidden_units, 1, batch_first=True)
        self.gru_2 = nn.GRU(self.s_dim+self.hidden_units, self.hidden_units_2, 1, batch_first=True)

        # Output layers coarse
        self.out = DualFC_CF(self.hidden_units_2, self.cf_dim, self.lpc, n_bands=self.n_bands, mid_out=self.cf_dim_in//2)
        #self.out = DualFC_CF(self.hidden_units_2, self.cf_dim, self.lpc, n_bands=self.n_bands, mid_out=self.cf_dim_in//4)

        # GRU layer(s) fine
        self.gru_f = nn.GRU(self.s_dim+self.wav_dim_bands+self.hidden_units_2, self.hidden_units_2, 1, batch_first=True)

        # Output layers fine
        self.out_f = DualFC_CF(self.hidden_units_2, self.cf_dim, self.lpc, n_bands=self.n_bands, mid_out=self.cf_dim_in//2)
        #self.out_f = DualFC_CF(self.hidden_units_2, self.cf_dim, self.lpc, n_bands=self.n_bands, mid_out=self.cf_dim_in//4)

        # Prev logits if using data-driven lpc
        if self.lpc > 0:
            self.logits = nn.Embedding(self.cf_dim, self.cf_dim)
            logits_param = torch.empty(self.cf_dim, self.cf_dim).fill_(0)
            for i in range(self.cf_dim):
                logits_param[i,i] = 1
            self.logits.weight = torch.nn.Parameter(logits_param)

        if self.n_spk is not None:
            conv_spk_in = [nn.Conv1d(self.cf_dim*self.n_bands*2, self.hidden_units_2, 1), nn.ReLU()]
            self.conv_spk_in = nn.Sequential(*conv_spk_in)
            self.gru_spk = nn.GRU(self.hidden_units_2, self.hidden_units_2, 1, batch_first=True)
            self.conv_spk = nn.Conv1d(self.hidden_units_2, self.n_spk*self.n_bands, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
            if self.quantize_spec is None and self.remove_weight_norm_scale:
                torch.nn.utils.remove_weight_norm(self.scale_in)
        else:
            self.apply(initialize)

    def forward(self, c, x_c_prev, x_f_prev, x_c, h=None, h_2=None, h_f=None, h_spk=None, do=False, x_c_lpc=None, x_f_lpc=None, in_spec=None, in_lat=None):
        # Input
        if self.quantize_spec is not None:
            if self.excit_dim > 0:
                x_in = self.conv(torch.cat((self.scale_in(c.transpose(1,2)), in_lat.transpose(1,2)), 1))
            else:
                x_in = in_lat.transpose(1,2)
            x_in = torch.cat((x_in, self.embed_spec(in_spec.long()).reshape(in_spec.shape[0], in_spec.shape[1], -1).transpose(1,2)), 1)
            if self.do_prob > 0 and do:
                conv = self.drop(torch.repeat_interleave(self.conv_s_c(x_in).transpose(1,2),self.upsampling_factor,dim=1))
            else:
                conv = torch.repeat_interleave(self.conv_s_c(x_in).transpose(1,2),self.upsampling_factor,dim=1)
        else:
            if self.do_prob > 0 and do:
                conv = self.drop(torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2),self.upsampling_factor,dim=1))
            else:
                conv = torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2),self.upsampling_factor,dim=1)

        # GRU1
        if h is not None:
            out, h = self.gru(torch.cat((conv, self.embed_c_wav(x_c_prev).reshape(x_c_prev.shape[0], x_c_prev.shape[1], -1),
                        self.embed_f_wav(x_f_prev).reshape(x_f_prev.shape[0], x_f_prev.shape[1], -1)), 2), h) # B x T x C -> B x C x T -> B x T x C
        else:
            out, h = self.gru(torch.cat((conv, self.embed_c_wav(x_c_prev).reshape(x_c_prev.shape[0], x_c_prev.shape[1], -1),
                        self.embed_f_wav(x_f_prev).reshape(x_f_prev.shape[0], x_f_prev.shape[1], -1)), 2))

        # GRU2
        if h_2 is not None:
            out, h_2 = self.gru_2(torch.cat((conv, out), 2), h_2) # B x T x C -> B x C x T -> B x T x C
        else:
            out, h_2 = self.gru_2(torch.cat((conv, out), 2))

        # GRU_fine
        if h_f is not None:
            out_f, h_f = self.gru_f(torch.cat((conv, self.embed_c_wav(x_c).reshape(x_c.shape[0], x_c.shape[1], -1), out), 2), h_f)
        else:
            out_f, h_f = self.gru_f(torch.cat((conv, self.embed_c_wav(x_c).reshape(x_c.shape[0], x_c.shape[1], -1), out), 2))

        # output
        if self.lpc > 0:
            signs_c, scales_c, logits_c = self.out(out.transpose(1,2))
            signs_f, scales_f, logits_f = self.out_f(out_f.transpose(1,2))
            # B x T x x n_bands x K, B x T x n_bands x K and B x T x n_bands x 256
            #logging.info(torch.mean(torch.mean(torch.mean(signs_c, 2), 1), 0))
            #logging.info(torch.mean(torch.mean(torch.mean(scales_c, 2), 1), 0))
            #logging.info(torch.mean(torch.mean(torch.mean(signs_f, 2), 1), 0))
            #logging.info(torch.mean(torch.mean(torch.mean(scales_f, 2), 1), 0))
            # x_lpc B x T_lpc x n_bands --> B x T x n_bands x K --> B x T x n_bands x K x 256
            # unfold put new dimension on the last
            if self.n_spk is not None:
                logits_c = logits_c + torch.sum((signs_c*scales_c).flip(-1).unsqueeze(-1)*self.logits(x_c_lpc.unfold(1, self.lpc, 1)), 3)
                logits_f = logits_f + torch.sum((signs_f*scales_f).flip(-1).unsqueeze(-1)*self.logits(x_f_lpc.unfold(1, self.lpc, 1)), 3)
                B = logits_c.shape[0]
                T = logits_c.shape[1]
                if h_spk is not None:
                    out, h_spk = self.gru_spk(self.conv_spk_in(torch.cat((logits_c, logits_f), 2).reshape(B, T, -1).transpose(1,2)).transpose(1,2), h_spk)
                else:
                    out, h_spk = self.gru_spk(self.conv_spk_in(torch.cat((logits_c, logits_f), 2).reshape(B, T, -1).transpose(1,2)).transpose(1,2))
                return logits_c, logits_f, F.selu(self.conv_spk(out.transpose(1,2)).transpose(1,2).reshape(B, T, self.n_bands, -1)), h.detach(), h_2.detach(), h_f.detach(), h_spk.detach()
            else:
                #return logits_c + torch.sum((signs_c*scales_c).flip(-1).unsqueeze(-1)*self.logits(x_c_lpc.unfold(1, self.lpc, 1)), 3), \
                #    logits_f + torch.sum((signs_f*scales_f).flip(-1).unsqueeze(-1)*self.logits(x_f_lpc.unfold(1, self.lpc, 1)), 3), h.detach(), h_2.detach(), h_f.detach()
                return torch.clamp(logits_c + torch.sum((signs_c*scales_c).flip(-1).unsqueeze(-1)*self.logits(x_c_lpc.unfold(1, self.lpc, 1)), 3), -32, 32), \
                    torch.clamp(logits_f + torch.sum((signs_f*scales_f).flip(-1).unsqueeze(-1)*self.logits(x_f_lpc.unfold(1, self.lpc, 1)), 3), -32, 32), h.detach(), h_2.detach(), h_f.detach()
            # B x T x n_bands x 256
        else:
            logits_c = self.out(out.transpose(1,2))
            logits_f = self.out_f(out_f.transpose(1,2))
            if self.n_spk is not None:
                B = logits_c.shape[0]
                T = logits_f.shape[1]
                if h_spk is not None:
                    out, h_spk = self.gru_spk(self.conv_spk_in(torch.cat((logits_c, logits_f), 2).reshape(B, T, -1).transpose(1,2)).transpose(1,2), h_spk)
                else:
                    out, h_spk = self.gru_spk(self.conv_spk_in(torch.cat((logits_c, logits_f), 2).reshape(B, T, -1).transpose(1,2)).transpose(1,2))
                return logits_c, logits_f, F.selu(self.conv_spk(out.transpose(1,2)).transpose(1,2).reshape(B, T, self.n_bands, -1)), h.detach(), h_2.detach(), h_f.detach(), h_spk.detach()
            else:
                #return logits_c, logits_f, h.detach(), h_2.detach(), h_f.detach()
                return torch.clamp(logits_c, -32, 32), torch.clamp(logits_f, -32, 32), h.detach(), h_2.detach(), h_f.detach()

    def generate(self, c, in_spec=None, in_lat=None, intervals=4000):
        start = time.time()
        time_sample = []
        intervals /= self.n_bands

        upsampling_factor = self.upsampling_factor

        c_pad = (self.n_quantize // 2) // self.cf_dim
        f_pad = (self.n_quantize // 2) % self.cf_dim

        B = c.shape[0]
        c = F.pad(c.transpose(1,2), (-self.pad_left,self.pad_right), "replicate").transpose(1,2)
        if self.quantize_spec is not None:
            if self.excit_dim > 0:
                x_in = self.conv(torch.cat((self.scale_in(c.transpose(1,2)), in_lat.transpose(1,2)), 1))
            else:
                x_in = in_lat.transpose(1,2)
            x_in = torch.cat((x_in, self.embed_spec(in_spec.long()).reshape(in_spec.shape[0], in_spec.shape[1], -1).transpose(1,2)), 1)
            c = self.conv_s_c(x_in).transpose(1,2)
        else:
            c = self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2)
        if self.lpc > 0:
            x_c_lpc = torch.empty(B,1,self.n_bands,self.lpc).cuda().fill_(c_pad).long() # B x 1 x n_bands x K
            x_f_lpc = torch.empty(B,1,self.n_bands,self.lpc).cuda().fill_(f_pad).long() # B x 1 x n_bands x K
        T = c.shape[1]*upsampling_factor

        c_f = c[:,:1]
        out, h = self.gru(torch.cat((c_f,self.embed_c_wav(torch.empty(B,1,self.n_bands).cuda().fill_(c_pad).long()).reshape(B,1,-1),
                                        self.embed_f_wav(torch.empty(B,1,self.n_bands).cuda().fill_(f_pad).long()).reshape(B,1,-1)),2))
        out, h_2 = self.gru_2(torch.cat((c_f,out), 2))
        #eps = torch.finfo(out.dtype).eps
        #eps_1 = 1-eps
        #logging.info(f"eps: {eps}\neps_1: {eps_1}")
        if self.lpc > 0:
            # coarse part
            signs_c, scales_c, logits_c = self.out(out.transpose(1,2)) # B x 1 x n_bands x K or 32
            #dist = OneHotCategorical(F.softmax(logits_c + torch.sum((signs_c*scales_c).unsqueeze(-1)*self.logits(x_c_lpc), 3), dim=-1))
            dist = OneHotCategorical(F.softmax(torch.clamp(logits_c + torch.sum((signs_c*scales_c).unsqueeze(-1)*self.logits(x_c_lpc), 3), min=-32, max=32), dim=-1))
            # B x 1 x n_bands x 256, B x 1 x n_bands x K x 256 --> B x 1 x n_bands x 2 x 256
            x_c_out = x_c_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
            #u = torch.empty_like(logits_c)
            #logits_c += torch.sum((signs_c*scales_c).unsqueeze(-1)*self.logits(x_c_lpc), 3) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
            #logits_c = torch.clamp(logits_c + torch.sum((signs_c*scales_c).unsqueeze(-1)*self.logits(x_c_lpc), 3), -32, 32) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
            #x_c_out = x_c_wav = logits_c.argmax(dim=-1) # B x 1 x n_bands
            x_c_lpc[:,:,:,1:] = x_c_lpc[:,:,:,:-1]
            x_c_lpc[:,:,:,0] = x_c_wav
            # fine part
            embed_x_c_wav = self.embed_c_wav(x_c_wav).reshape(B,1,-1)
            out, h_f = self.gru_f(torch.cat((c_f, embed_x_c_wav, out), 2))
            signs_f, scales_f, logits_f = self.out_f(out.transpose(1,2)) # B x 1 x n_bands x K or 32
            #dist = OneHotCategorical(F.softmax(logits_f + torch.sum((signs_f*scales_f).unsqueeze(-1)*self.logits(x_f_lpc), 3), dim=-1))
            dist = OneHotCategorical(F.softmax(torch.clamp(logits_f + torch.sum((signs_f*scales_f).unsqueeze(-1)*self.logits(x_f_lpc), 3), min=-32, max=32), dim=-1))
            x_f_out = x_f_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
            #logits_f += torch.sum((signs_f*scales_f).unsqueeze(-1)*self.logits(x_f_lpc), 3) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
            #logits_f = torch.clamp(logits_f + torch.sum((signs_f*scales_f).unsqueeze(-1)*self.logits(x_f_lpc), 3), -32, 32) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
            #x_f_out = x_f_wav = logits_f.argmax(dim=-1) # B x 1 x n_bands
            x_f_lpc[:,:,:,1:] = x_f_lpc[:,:,:,:-1]
            x_f_lpc[:,:,:,0] = x_f_wav
        else:
            # coarse part
            #dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2)), dim=-1))
            dist = OneHotCategorical(F.softmax(torch.clamp(self.out(out.transpose(1,2)), min=-32, max=32), dim=-1))
            x_c_out = x_c_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
            #logits = self.out(out.transpose(1,2))
            #u = torch.empty_like(logits)
            #x_c_out = x_c_wav = (logits - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))).argmax(dim=-1) # B x 1 x n_bands
            # fine part
            embed_x_c_wav = self.embed_c_wav(x_c_wav).reshape(B,1,-1)
            out, h_f = self.gru_f(torch.cat((c_f, embed_x_c_wav, out), 2))
            #dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2)), dim=-1))
            dist = OneHotCategorical(F.softmax(torch.clamp(self.out(out.transpose(1,2)), min=-32, max=32), dim=-1))
            x_f_out = x_f_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
            #logits = self.out(out.transpose(1,2))
            #x_f_out = x_f_wav = (logits - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))).argmax(dim=-1) # B x 1 x n_bands

        time_sample.append(time.time()-start)
        if self.lpc > 0:
            for t in range(1,T):
                start_sample = time.time()

                if t % upsampling_factor  == 0:
                    idx_t_f = t//upsampling_factor
                    c_f = c[:,idx_t_f:idx_t_f+1]

                out, h = self.gru(torch.cat((c_f, embed_x_c_wav, self.embed_f_wav(x_f_wav).reshape(B,1,-1)),2), h)
                out, h_2 = self.gru_2(torch.cat((c_f,out), 2), h_2)

                # coarse part
                signs_c, scales_c, logits_c = self.out(out.transpose(1,2)) # B x 1 x n_bands x K or 32
                #dist = OneHotCategorical(F.softmax(logits_c + torch.sum((signs_c*scales_c).unsqueeze(-1)*self.logits(x_c_lpc), 3), dim=-1))
                dist = OneHotCategorical(F.softmax(torch.clamp(logits_c + torch.sum((signs_c*scales_c).unsqueeze(-1)*self.logits(x_c_lpc), 3), min=-32, max=32), dim=-1))
                x_c_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands x 2
                #logits_c += torch.sum((signs_c*scales_c).unsqueeze(-1)*self.logits(x_c_lpc), 3) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
                #logits_c = torch.clamp(logits_c + torch.sum((signs_c*scales_c).unsqueeze(-1)*self.logits(x_c_lpc), 3), -32, 32) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
                #x_c_wav = logits_c.argmax(dim=-1) # B x 1 x n_bands
                x_c_out = torch.cat((x_c_out, x_c_wav), 1) # B x t+1 x n_bands
                x_c_lpc[:,:,:,1:] = x_c_lpc[:,:,:,:-1]
                x_c_lpc[:,:,:,0] = x_c_wav

                # fine part
                embed_x_c_wav = self.embed_c_wav(x_c_wav).reshape(B,1,-1)
                out, h_f = self.gru_f(torch.cat((c_f, embed_x_c_wav, out), 2), h_f)
                signs_f, scales_f, logits_f = self.out_f(out.transpose(1,2)) # B x 1 x n_bands x K or 32
                #dist = OneHotCategorical(F.softmax(logits_f + torch.sum((signs_f*scales_f).unsqueeze(-1)*self.logits(x_f_lpc), 3), dim=-1))
                dist = OneHotCategorical(F.softmax(torch.clamp(logits_f + torch.sum((signs_f*scales_f).unsqueeze(-1)*self.logits(x_f_lpc), 3), min=-32, max=32), dim=-1))
                x_f_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
                #logits_f += torch.sum((signs_f*scales_f).unsqueeze(-1)*self.logits(x_f_lpc), 3) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
                #logits_f = torch.clamp(logits_f + torch.sum((signs_f*scales_f).unsqueeze(-1)*self.logits(x_f_lpc), 3), -32, 32) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
                #x_f_wav = logits_f.argmax(dim=-1) # B x 1 x n_bands
                x_f_out = torch.cat((x_f_out, x_f_wav), 1) # B x t+1 x n_bands
                x_f_lpc[:,:,:,1:] = x_f_lpc[:,:,:,:-1]
                x_f_lpc[:,:,:,0] = x_f_wav

                time_sample.append(time.time()-start_sample)
                if (t + 1) % intervals == 0:
                    logging.info("%d/%d estimated time = %.6f sec (%.6f sec / sample)" % (
                        (t + 1), T,
                        ((T - t - 1) / intervals) * (time.time() - start),
                        (time.time() - start) / intervals))
                    start = time.time()
        else:
            for t in range(1,T):
                start_sample = time.time()

                if t % upsampling_factor  == 0:
                    idx_t_f = t//upsampling_factor
                    c_f = c[:,idx_t_f:idx_t_f+1]

                out, h = self.gru(torch.cat((c_f, embed_x_c_wav, self.embed_f_wav(x_f_wav).reshape(B,1,-1)),2), h)
                out, h_2 = self.gru_2(torch.cat((c_f,out),2), h_2)

                # coarse part
                #dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2)), dim=-1))
                dist = OneHotCategorical(F.softmax(torch.clamp(self.out(out.transpose(1,2)), min=-32, max=32), dim=-1))
                x_c_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
                #logits = self.out(out.transpose(1,2))
                #x_c_wav = (logits - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))).argmax(dim=-1) # B x 1 x n_bands
                x_c_out = torch.cat((x_c_out, x_c_wav), 1) # B x t+1 x n_bands

                # fine part
                embed_x_c_wav = self.embed_c_wav(x_c_wav).reshape(B,1,-1)
                out, h_f = self.gru_f(torch.cat((c_f, embed_x_c_wav, out), 2), h_f)
                #dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2)), dim=-1))
                dist = OneHotCategorical(F.softmax(torch.clamp(self.out(out.transpose(1,2)), min=-32, max=32), dim=-1))
                x_f_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
                #logits = self.out(out.transpose(1,2))
                #x_f_wav = (logits - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))).argmax(dim=-1) # B x 1 x n_bands
                x_f_out = torch.cat((x_f_out, x_f_wav), 1) # B x t+1 x n_bands

                time_sample.append(time.time()-start_sample)
                if (t + 1) % intervals == 0:
                    logging.info("%d/%d estimated time = %.6f sec (%.6f sec / sample)" % (
                        (t + 1), T,
                        ((T - t - 1) / intervals) * (time.time() - start),
                        (time.time() - start) / intervals))
                    start = time.time()

        time_sample = np.array(time_sample)
        logging.info("average time / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % \
                        (np.mean(time_sample), len(time_sample), 1.0/(1000*np.mean(time_sample))))
        logging.info("average throughput / sample = %.6f sec (%ld samples * %ld) [%.3f kHz/s]" % \
                        (np.sum(time_sample)/(len(time_sample)*c.shape[0]), len(time_sample), c.shape[0], \
                            len(time_sample)*c.shape[0]/(1000*np.sum(time_sample))))

        if self.n_quantize == 65536:
            return ((x_c_out*self.cf_dim+x_f_out).transpose(1,2).float() - 32768.0) / 32768.0 # B x T x n_bands --> B x n_bands x T
        else:
            return decode_mu_law_torch((x_c_out*self.cf_dim+x_f_out).transpose(1,2).float(), mu=self.n_quantize) # B x T x n_bands --> B x n_bands x T

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) \
                or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.utils.weight_norm(m)
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d) \
                    or isinstance(m, torch.nn.ConvTranspose2d):
                    torch.nn.utils.remove_weight_norm(m)
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND(nn.Module):
    def __init__(self, feat_dim=52, upsampling_factor=120, hidden_units=384, hidden_units_2=24, n_quantize=256, lpc=6,
            kernel_size=7, dilation_size=1, do_prob=0, causal_conv=False, use_weight_norm=True, nonlinear_conv=False,
                right_size=0, n_bands=2, n_spk=None, pad_first=False):
        super(GRU_WAVE_DECODER_DUALGRU_COMPACT_MBAND, self).__init__()
        self.feat_dim = feat_dim
        self.in_dim = self.feat_dim
        self.n_quantize = n_quantize
        self.out_dim = self.n_quantize
        self.n_bands = n_bands
        self.upsampling_factor = upsampling_factor // self.n_bands
        self.hidden_units = hidden_units
        self.hidden_units_2 = hidden_units_2
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.nonlinear_conv = nonlinear_conv
        self.s_dim = 320
        #self.wav_dim = self.s_dim // self.n_bands
        self.wav_dim = 64
        self.wav_dim_bands = self.wav_dim * self.n_bands
        self.use_weight_norm = use_weight_norm
        self.lpc = lpc
        self.right_size = right_size
        self.n_spk = n_spk
        self.pad_first = pad_first

        self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)

        # Conv. layers
        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = self.conv.padding
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
                self.pad_left = self.conv.padding
                self.pad_right = 0
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size,
                                        right_size=self.right_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
            self.pad_left = self.conv.left_size
            self.pad_right = self.conv.right_size
        conv_s_c = [nn.Conv1d(self.in_dim*self.conv.rec_field, self.s_dim, 1), nn.ReLU()]
        self.conv_s_c = nn.Sequential(*conv_s_c)
        if self.do_prob > 0:
            self.drop = nn.Dropout(p=self.do_prob)
        self.embed_wav = nn.Embedding(self.n_quantize, self.wav_dim)

        # GRU layer(s)
        self.gru = nn.GRU(self.s_dim+self.wav_dim_bands, self.hidden_units, 1, batch_first=True)
        self.gru_2 = nn.GRU(self.s_dim+self.hidden_units, self.hidden_units_2, 1, batch_first=True)

        # Output layers
        self.out = DualFC(self.hidden_units_2, self.n_quantize, self.lpc, n_bands=self.n_bands, mid_out=32)

        # Prev logits if using data-driven lpc
        if self.lpc > 0:
            self.logits = nn.Embedding(self.n_quantize, self.n_quantize)
            logits_param = torch.empty(self.n_quantize, self.n_quantize).fill_(0)
            for i in range(self.n_quantize):
                logits_param[i,i] = 1
            self.logits.weight = torch.nn.Parameter(logits_param)

        if self.n_spk is not None:
            self.gru_spk = nn.GRU(self.n_quantize*self.n_bands, self.hidden_units_2, 1, batch_first=True)
            self.conv_spk = nn.Conv1d(self.hidden_units_2, self.n_spk*self.n_bands, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
            torch.nn.utils.remove_weight_norm(self.scale_in)
        else:
            self.apply(initialize)

    def forward(self, c, x_prev, h=None, h_2=None, h_spk=None, do=False, x_lpc=None):
        # Input
        if self.do_prob > 0 and do:
            conv = self.drop(torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2),self.upsampling_factor,dim=1))
        else:
            conv = torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2),self.upsampling_factor,dim=1)

        # GRU1
        if h is not None:
            out, h = self.gru(torch.cat((conv, self.embed_wav(x_prev).reshape(x_prev.shape[0], x_prev.shape[1], -1)),2), h) # B x T x C -> B x C x T -> B x T x C
        else:
            out, h = self.gru(torch.cat((conv, self.embed_wav(x_prev).reshape(x_prev.shape[0], x_prev.shape[1], -1)),2))

        # GRU2
        if h_2 is not None:
            out, h_2 = self.gru_2(torch.cat((conv, out),2), h_2) # B x T x C -> B x C x T -> B x T x C
        else:
            out, h_2 = self.gru_2(torch.cat((conv, out),2))

        # output
        if self.lpc > 0:
            signs, scales, logits = self.out(out.transpose(1,2)) # B x T x x n_bands x K, B x T x n_bands x K and B x T x n_bands x 256
            #logging.info(torch.mean(torch.mean(torch.mean(signs, 2), 1), 0))
            #logging.info(torch.mean(torch.mean(torch.mean(scales, 2), 1), 0))
            # x_lpc B x T_lpc x n_bands --> B x T x n_bands x K --> B x T x n_bands x K x 256
            # unfold put new dimension on the last
            if self.n_spk is not None:
                logits = logits + torch.sum((signs*scales).flip(-1).unsqueeze(-1)*self.logits(x_lpc.unfold(1, self.lpc, 1)), 3)
                B = logits.shape[0]
                T = logits.shape[1]
                if h_spk is not None:
                    #out, h_spk = self.gru_spk(logits.reshape(B, T*self.n_bands, -1), h_spk)
                    out, h_spk = self.gru_spk(logits.reshape(B, T, -1), h_spk)
                else:
                    #out, h_spk = self.gru_spk(logits.reshape(B, T*self.n_bands, -1))
                    out, h_spk = self.gru_spk(logits.reshape(B, T, -1))
                return logits, F.selu(self.conv_spk(out.transpose(1,2)).transpose(1,2).reshape(B, T, self.n_bands, -1)), h.detach(), h_2.detach(), h_spk.detach()
            else:
                return torch.clamp(logits + torch.sum((signs*scales).flip(-1).unsqueeze(-1)*self.logits(x_lpc.unfold(1, self.lpc, 1)), 3), -32, 32), h.detach(), h_2.detach()
            # B x T x n_bands x 256
        else:
            if self.n_spk is not None:
                logits = self.out(out.transpose(1,2))
                B = logits.shape[0]
                T = logits.shape[1]
                if h_spk is not None:
                    #out, h_spk = self.gru_spk(logits.reshape(B, T*self.n_bands, -1), h_spk)
                    out, h_spk = self.gru_spk(logits.reshape(B, T, -1), h_spk)
                else:
                    #out, h_spk = self.gru_spk(logits.reshape(B, T*self.n_bands, -1))
                    out, h_spk = self.gru_spk(logits.reshape(B, T, -1))
                return logits, F.selu(self.conv_spk(out.transpose(1,2)).transpose(1,2).reshape(B, T, self.n_bands, -1)), h.detach(), h_2.detach(), h_spk.detach()
            else:
                #return self.out(out.transpose(1,2)), h.detach(), h_2.detach()
                return torch.clamp(self.out(out.transpose(1,2)), -32, 32), h.detach(), h_2.detach()

    def generate(self, c, intervals=4000):
        start = time.time()
        time_sample = []
        intervals /= self.n_bands

        upsampling_factor = self.upsampling_factor

        B = c.shape[0]
        c = F.pad(c.transpose(1,2), (-self.pad_left,self.pad_right), "replicate").transpose(1,2)
        c = self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2)
        if self.lpc > 0:
            x_lpc = torch.empty(B,1,self.n_bands,self.lpc).cuda().fill_(self.n_quantize // 2).long() # B x 1 x n_bands x K
        T = c.shape[1]*upsampling_factor

        c_f = c[:,:1]
        out, h = self.gru(torch.cat((c_f,self.embed_wav(torch.empty(B,1,self.n_bands).cuda().fill_(self.n_quantize//2).long()).reshape(B,1,-1)),2))
        out, h_2 = self.gru_2(torch.cat((c_f,out),2))
        #eps = torch.finfo(out.dtype).eps
        #eps_1 = 1-eps
        #logging.info(f"eps: {eps}\neps_1: {eps_1}")
        if self.lpc > 0:
            signs, scales, logits = self.out(out.transpose(1,2)) # B x T x C -> B x C x T -> B x T x C
            #pred_logits = torch.sum(signs.unsqueeze(-1)*self.logits(x_lpc)*scales.unsqueeze(-1), 2)
            #dist = OneHotCategorical(F.softmax(logits + torch.sum((signs*scales).unsqueeze(-1)*self.logits(x_lpc), 3), dim=-1)) # B x 1 x n_bands x 256, B x 1 x n_bands x K x 256 --> B x 1 x n_bands x 256
            dist = OneHotCategorical(F.softmax(torch.clamp(logits + torch.sum((signs*scales).unsqueeze(-1)*self.logits(x_lpc), 3), min=-32, max=32), dim=-1)) # B x 1 x n_bands x 256, B x 1 x n_bands x K x 256 --> B x 1 x n_bands x 256
            x_out = x_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
            #u = torch.empty_like(logits)
            #logits += torch.sum((signs*scales).unsqueeze(-1)*self.logits(x_lpc), 3) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
            #x_out = x_wav = logits.argmax(dim=-1) # B x 1 x n_bands
            x_lpc[:,:,:,1:] = x_lpc[:,:,:,:-1]
            x_lpc[:,:,:,0] = x_wav
            #dist = OneHotCategorical(F.softmax(logits, dim=-1))
            #x_out_res = dist.sample().argmax(dim=-1)
            ##dist = OneHotCategorical(F.softmax(pred_logits, dim=-1))
            #dist = OneHotCategorical(F.softmax(-pred_logits, dim=-1))
            #x_out_pred = dist.sample().argmax(dim=-1)
        else:
            #dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2)), dim=-1))
            dist = OneHotCategorical(F.softmax(torch.clamp(self.out(out.transpose(1,2)), min=-32, max=32), dim=-1))
            x_out = x_wav = dist.sample().argmax(dim=-1)
            #logits = self.out(out.transpose(1,2))
            #u = torch.empty_like(logits)
            #x_out = x_wav = (logits - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))).argmax(dim=-1) # B x 1 x n_bands

        time_sample.append(time.time()-start)
        if self.lpc > 0:
            for t in range(1,T):
                start_sample = time.time()

                if t % upsampling_factor  == 0:
                    idx_t_f = t//upsampling_factor
                    c_f = c[:,idx_t_f:idx_t_f+1]

                out, h = self.gru(torch.cat((c_f, self.embed_wav(x_wav).reshape(B,1,-1)),2), h)
                out, h_2 = self.gru_2(torch.cat((c_f,out),2), h_2)

                signs, scales, logits = self.out(out.transpose(1,2)) # B x T x C -> B x C x T -> B x T x C
                #pred_logits = torch.sum(signs.unsqueeze(-1)*self.logits(x_lpc)*scales.unsqueeze(-1), 2)
                #dist = OneHotCategorical(F.softmax(logits + torch.sum((signs*scales).unsqueeze(-1)*self.logits(x_lpc), 3), dim=-1)) # B x 1 x n_bands x 256, B x 1 x n_bands x K x 256 --> B x 1 x n_bands x 256
                dist = OneHotCategorical(F.softmax(torch.clamp(logits + torch.sum((signs*scales).unsqueeze(-1)*self.logits(x_lpc), 3), min=-32, max=32), dim=-1)) # B x 1 x n_bands x 256, B x 1 x n_bands x K x 256 --> B x 1 x n_bands x 256
                x_wav = dist.sample().argmax(dim=-1) # B x 1 x n_bands
                #logits += torch.sum((signs*scales).unsqueeze(-1)*self.logits(x_lpc), 3) - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))
                #x_wav = logits.argmax(dim=-1) # B x 1 x n_bands
                x_out = torch.cat((x_out, x_wav), 1) # B x t+1 x n_bands
                x_lpc[:,:,:,1:] = x_lpc[:,:,:,:-1]
                x_lpc[:,:,:,0] = x_wav
                #dist = OneHotCategorical(F.softmax(logits, dim=-1))
                #x_out_res = torch.cat((x_out_res, dist.sample().argmax(dim=-1)), 1)
                ##dist = OneHotCategorical(F.softmax(pred_logits, dim=-1))
                #dist = OneHotCategorical(F.softmax(-pred_logits, dim=-1))
                #x_out_pred = torch.cat((x_out_pred, dist.sample().argmax(dim=-1)), 1)

                time_sample.append(time.time()-start_sample)
                if (t + 1) % intervals == 0:
                    logging.info("%d/%d estimated time = %.6f sec (%.6f sec / sample)" % (
                        (t + 1), T,
                        ((T - t - 1) / intervals) * (time.time() - start),
                        (time.time() - start) / intervals))
                    start = time.time()
        else:
            for t in range(1,T):
                start_sample = time.time()

                if t % upsampling_factor  == 0:
                    idx_t_f = t//upsampling_factor
                    c_f = c[:,idx_t_f:idx_t_f+1]

                out, h = self.gru(torch.cat((c_f, self.embed_wav(x_wav).reshape(B,1,-1)),2), h)
                out, h_2 = self.gru_2(torch.cat((c_f,out),2), h_2)

                #dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2)), dim=-1))
                dist = OneHotCategorical(F.softmax(torch.clamp(self.out(out.transpose(1,2)), min=-32, max=32), dim=-1))
                x_wav = dist.sample().argmax(dim=-1)
                #logits = self.out(out.transpose(1,2))
                #x_wav = (logits - torch.log(-torch.log(torch.clamp(u.uniform_(), eps, eps_1)))).argmax(dim=-1) # B x 1 x n_bands
                x_out = torch.cat((x_out, x_wav), 1)

                time_sample.append(time.time()-start_sample)
                if (t + 1) % intervals == 0:
                    logging.info("%d/%d estimated time = %.6f sec (%.6f sec / sample)" % (
                        (t + 1), T,
                        ((T - t - 1) / intervals) * (time.time() - start),
                        (time.time() - start) / intervals))
                    start = time.time()

        time_sample = np.array(time_sample)
        logging.info("average time / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % \
                        (np.mean(time_sample), len(time_sample), 1.0/(1000*np.mean(time_sample))))
        logging.info("average throughput / sample = %.6f sec (%ld samples * %ld) [%.3f kHz/s]" % \
                        (np.sum(time_sample)/(len(time_sample)*c.shape[0]), len(time_sample), c.shape[0], \
                            len(time_sample)*c.shape[0]/(1000*np.sum(time_sample))))

        return decode_mu_law_torch(x_out.transpose(1,2).float(), mu=self.n_quantize) # B x T x n_bands --> B x n_bands x T
        #if self.lpc > 0:
        #    return decode_mu_law(x_out.cpu().data.numpy()), decode_mu_law(x_out_res.cpu().data.numpy()), \
        #            decode_mu_law(x_out_pred.cpu().data.numpy())
        #else:
        #    return decode_mu_law(x_out.cpu().data.numpy())


    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) \
                or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.utils.weight_norm(m)
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d) \
                    or isinstance(m, torch.nn.ConvTranspose2d):
                    torch.nn.utils.remove_weight_norm(m)
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class ModulationSpectrumLoss(nn.Module):
    def __init__(self, fftsize=256):
        super(ModulationSpectrumLoss, self).__init__()
        self.fftsize = fftsize

    def forward(self, x, y):
        """ x : B x T x C / T x C
            y : B x T x C / T x C
            return : B, B / 1, 1 [frobenius-norm, L1-loss] """
        if len(x.shape) > 2: # B x T x C
            padded_x = F.pad(x, (0, 0, 0, self.fftsize-x.shape[1]), "constant", 0)
            padded_y = F.pad(y, (0, 0, 0, self.fftsize-y.shape[1]), "constant", 0)
            #csp_x = torch.clamp(torch.rfft(padded_x, 3, onesided=False), -1e12, 1e12)
            #csp_y = torch.clamp(torch.rfft(padded_y, 3, onesided=False), -1e12, 1e12)
            #csp_x = torch.rfft(padded_x, 3, onesided=False)
            #logging.info(csp_x.shape)
            #logging.info(csp_x[0,0,0])
            #logging.info(csp_x[0,0,0,0]**2+csp_x[0,0,0,1]**2)
            #csp_y = torch.rfft(padded_y, 3, onesided=False)
            csp_x = torch.fft.fftn(padded_x)
            #logging.info(csp_x.shape)
            #logging.info(csp_x[0,0,0])
            #logging.info(torch.abs(csp_x[0,0,0])**2)
            csp_y = torch.fft.fftn(padded_y)
            #logging.info("ms")
            #logging.info(torch.isinf(csp_x.mean()))
            #logging.info(torch.isinf(csp_y.mean()))
            #magsp_x = (csp_x[:,:,:,0]**2 + csp_x[:,:,:,1]**2).sqrt()
            #magsp_x = torch.clamp(csp_x[:,:,:,0]**2 + csp_x[:,:,:,1]**2, min=1e-7).sqrt()
            magsp_x = torch.clamp(torch.abs(csp_x)**2, min=1e-7).sqrt()
            #logging.info(csp_x.max())
            #logging.info(csp_x.min())
            #logging.info(torch.isinf((csp_x[:,:,:,0]**2).mean()))
            #logging.info(torch.isinf((csp_x[:,:,:,1]**2).mean()))
            #logging.info(torch.isinf(magsp_x.mean()))
            #magsp_y = (csp_y[:,:,:,0]**2 + csp_y[:,:,:,1]**2).sqrt()
            #magsp_y = torch.clamp(csp_y[:,:,:,0]**2 + csp_y[:,:,:,1]**2, min=1e-7).sqrt()
            magsp_y = torch.clamp(torch.abs(csp_y)**2, min=1e-7).sqrt()
            #logging.info(torch.isinf(magsp_y.mean()))
            norm = torch.norm(magsp_y - magsp_x, p="fro", dim=(1,2)) / torch.norm(magsp_y, p="fro", dim=(1,2))
            #err = F.l1_loss(torch.log10(magsp_y), torch.log10(magsp_x), reduction='none').sum(-1).mean(-1)
            err = F.l1_loss(torch.log10(magsp_y), torch.log10(magsp_x), reduction='none').mean(-1).mean(-1)
            #logging.info(csp_x.shape)
            #logging.info(csp_y.shape)
            #logging.info(magsp_x.shape)
            #logging.info(magsp_y.shape)
        else: # T x C
            padded_x = F.pad(x, (0, self.fftsize-x.shape[1]), "constant", 0)
            padded_y = F.pad(y, (0, self.fftsize-y.shape[1]), "constant", 0)
            #csp_x = torch.clamp(torch.rfft(padded_x, 2, onesided=False), -1e12, 1e12)
            #csp_y = torch.clamp(torch.rfft(padded_y, 2, onesided=False), -1e12, 1e12)
            #csp_x = torch.rfft(padded_x, 2, onesided=False)
            #csp_y = torch.rfft(padded_y, 2, onesided=False)
            csp_x = torch.fft.fftn(padded_x)
            csp_y = torch.fft.fftn(padded_y)
            #magsp_x = (csp_x[:,:,0]**2 + csp_x[:,:,1]**2).sqrt()
            #magsp_x = torch.clamp(csp_x[:,:,0]**2 + csp_x[:,:,1]**2, min=1e-7).sqrt()
            magsp_x = torch.clamp(torch.abs(csp_x)**2, min=1e-7).sqrt()
            #magsp_y = (csp_y[:,:,0]**2 + csp_y[:,:,1]**2).sqrt()
            #magsp_y = torch.clamp(csp_y[:,:,0]**2 + csp_y[:,:,1]**2, min=1e-7).sqrt()
            magsp_y = torch.clamp(torch.abs(csp_y)**2, min=1e-7).sqrt()
            norm = torch.norm(magsp_y - magsp_x, p="fro") / torch.norm(magsp_y, p="fro")
            #err = F.l1_loss(torch.log10(magsp_y), torch.log10(magsp_x), reduction='none').sum(-1).mean()
            err = F.l1_loss(torch.log10(magsp_y), torch.log10(magsp_x), reduction='none').mean()
            #logging.info(csp_x.shape)
            #logging.info(csp_y.shape)
            #logging.info(magsp_x.shape)
            #logging.info(magsp_y.shape)
        return norm, err


class LaplaceLoss(nn.Module):
    def __init__(self):
        super(LaplaceLoss, self).__init__()
        self.c = 0.69314718055994530941723212145818 # ln(2)

    def forward(self, mu, log_b, target):
        if len(mu.shape) > 2: # B x T x C
            return torch.mean(torch.sum(self.c + log_b + torch.abs(target-mu)/log_b.exp(), -1), -1) # B x 1
        else: # T x C
            return torch.mean(torch.sum(self.c + log_b + torch.abs(target-mu)/log_b.exp(), -1)) # 1
        #else: # T
        #    return torch.mean(self.c + log_b + torch.abs(target-mu)/b)


def laplace_logits(mu, b, disc, log_b):
    return -0.69314718055994530941723212145818 - log_b - torch.abs(disc-mu)/b # log_like (Laplace)


class LaplaceLogits(nn.Module):
    def __init__(self):
        super(LaplaceLogits, self).__init__()
        self.c = 0.69314718055994530941723212145818 # ln(2)

    def forward(self, mu, b, disc, log_b):
        return -self.c - log_b - torch.abs(disc-mu)/b # log_like (Laplace)


class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION"""

    def __init__(self, in_channels, out_channels, kernel_size, dil_fact=0, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dil_fact = dil_fact
        self.dilation = self.kernel_size**self.dil_fact
        self.padding = self.kernel_size**(self.dil_fact+1) - self.dilation
        self.bias = bias
        self.conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, \
                                dilation=self.dilation, bias=self.bias)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """
        return self.conv(x)[:,:,:x.shape[2]]


class DSWNV(nn.Module):
    """SHALLOW WAVENET VOCODER WITH SOFTMAX OUTPUT"""

    def __init__(self, n_quantize=256, n_aux=54, hid_chn=192, skip_chn=256, aux_kernel_size=3, nonlinear_conv=False,
                aux_dilation_size=2, dilation_depth=3, dilation_repeat=3, kernel_size=6, right_size=0, pad_first=True,
                upsampling_factor=110, audio_in_flag=False, wav_conv_flag=False, do_prob=0, use_weight_norm=True):
        super(DSWNV, self).__init__()
        self.n_aux = n_aux
        self.n_quantize = n_quantize
        self.upsampling_factor = upsampling_factor
        self.in_audio_dim = self.n_quantize
        self.n_hidch = hid_chn
        self.n_skipch = skip_chn
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.aux_kernel_size = aux_kernel_size
        self.aux_dilation_size = aux_dilation_size
        self.do_prob = do_prob
        self.audio_in_flag = audio_in_flag
        self.wav_conv_flag = wav_conv_flag
        self.use_weight_norm = use_weight_norm
        self.right_size = right_size
        self.nonlinear_conv = nonlinear_conv
        self.s_dim = 320
        self.pad_first = pad_first

        # Input Layers
        self.scale_in = nn.Conv1d(self.n_aux, self.n_aux, 1)
        if self.right_size <= 0:
            self.conv = CausalDilConv1d(in_dim=self.n_aux, kernel_size=aux_kernel_size,
                                        layers=aux_dilation_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
            self.pad_left = self.conv.padding
            self.pad_right = 0
        else:
            self.conv = SkewedConv1d(in_dim=self.n_aux, kernel_size=aux_kernel_size,
                                        right_size=self.right_size, nonlinear=self.nonlinear_conv, pad_first=self.pad_first)
            self.pad_left = self.conv.left_size
            self.pad_right = self.conv.right_size
        conv_s_c = [nn.Conv1d(self.n_aux*self.conv.rec_field, self.s_dim, 1), nn.ReLU()]
        self.conv_s_c = nn.Sequential(*conv_s_c)

        self.in_aux_dim = self.s_dim
        self.upsampling = UpSampling(self.upsampling_factor)
        if self.do_prob > 0:
            self.aux_drop = nn.Dropout(p=self.do_prob)
        if not self.audio_in_flag:
            self.in_tot_dim = self.in_aux_dim
        else:
            self.in_tot_dim = self.in_aux_dim+self.in_audio_dim
        if self.wav_conv_flag:
            self.wav_conv = nn.Conv1d(self.in_audio_dim, self.n_hidch, 1)
            self.causal = CausalConv1d(self.n_hidch, self.n_hidch, self.kernel_size, dil_fact=0)
        else:
            self.causal = CausalConv1d(self.in_audio_dim, self.n_hidch, self.kernel_size, dil_fact=0)

        # Dilated Convolutional Recurrent Neural Network (DCRNN)
        self.padding = []
        self.dil_facts = [i for i in range(self.dilation_depth)] * self.dilation_repeat
        logging.info(self.dil_facts)
        self.in_x = nn.ModuleList()
        self.dil_h = nn.ModuleList()
        self.out_skip = nn.ModuleList()
        for i, d in enumerate(self.dil_facts):
            self.in_x += [nn.Conv1d(self.in_tot_dim, self.n_hidch*2, 1)]
            self.dil_h += [CausalConv1d(self.n_hidch, self.n_hidch*2, self.kernel_size, dil_fact=d)]
            self.padding.append(self.dil_h[i].padding)
            self.out_skip += [nn.Conv1d(self.n_hidch, self.n_skipch, 1)]
        logging.info(self.padding)
        self.receptive_field = sum(self.padding) + self.kernel_size-1
        logging.info(self.receptive_field)
        if self.do_prob > 0:
            self.dcrnn_drop = nn.Dropout(p=self.do_prob)

        # Output Layers
        self.out_1 = nn.Conv1d(self.n_skipch, self.n_quantize, 1)
        self.out_2 = nn.Conv1d(self.n_quantize, self.n_quantize, 1)

        ## apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
            torch.nn.utils.remove_weight_norm(self.scale_in)
        else:
            self.apply(initialize)

    def forward(self, aux, audio, first=False, do=False):
        audio = F.one_hot(audio, num_classes=self.n_quantize).float().transpose(1,2)
        # Input	Features
        x = self.upsampling(self.conv_s_c(self.conv(self.scale_in(aux.transpose(1,2)))))
        if first:
            x = F.pad(x, (self.receptive_field, 0), "replicate")
        if self.do_prob > 0 and do:
            x = self.aux_drop(x)
        if self.audio_in_flag:
            x = torch.cat((x,audio),1) # B x C x T
        # Initial Hidden Units
        if not self.wav_conv_flag:
            h = F.softsign(self.causal(audio)) # B x C x T
        else:
            h = F.softsign(self.causal(self.wav_conv(audio))) # B x C x T
        # DCRNN blocks
        sum_out, h = self._dcrnn_forward(x, h, self.in_x[0], self.dil_h[0], self.out_skip[0])
        if self.do_prob > 0 and do:
            for l in range(1,len(self.dil_facts)):
                if (l+1)%self.dilation_depth == 0:
                    out, h = self._dcrnn_forward_drop(x, h, self.in_x[l], self.dil_h[l], self.out_skip[l])
                else:
                    out, h = self._dcrnn_forward(x, h, self.in_x[l], self.dil_h[l], self.out_skip[l])
                sum_out += out
        else:
            for l in range(1,len(self.dil_facts)):
                out, h = self._dcrnn_forward(x, h, self.in_x[l], self.dil_h[l], self.out_skip[l])
                sum_out += out
        # Output
        return self.out_2(F.relu(self.out_1(F.relu(sum_out)))).transpose(1,2)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) \
                or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.utils.weight_norm(m)
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d) \
                    or isinstance(m, torch.nn.ConvTranspose2d):
                    torch.nn.utils.remove_weight_norm(m)
                    logging.info(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def _dcrnn_forward_drop(self, x, h, in_x, dil_h, out_skip):
        x_h_ = in_x(x)*dil_h(h)
        z = torch.sigmoid(x_h_[:,:self.n_hidch,:])
        h = (1-z)*torch.tanh(x_h_[:,self.n_hidch:,:]) + z*h
        return out_skip(h), self.dcrnn_drop(h)

    def _dcrnn_forward(self, x, h, in_x, dil_h, out_skip):
        x_h_ = in_x(x)*dil_h(h)
        z = torch.sigmoid(x_h_[:,:self.n_hidch,:])
        h = (1-z)*torch.tanh(x_h_[:,self.n_hidch:,:]) + z*h
        return out_skip(h), h

    def _generate_dcrnn_forward(self, x, h, in_x, dil_h, out_skip):
        x_h_ = in_x(x)*dil_h(h)[:,:,-1:]
        z = torch.sigmoid(x_h_[:,:self.n_hidch,:])
        h = (1-z)*torch.tanh(x_h_[:,self.n_hidch:,:]) + z*h[:,:,-1:]
        return out_skip(h), h

    def batch_fast_generate(self, audio, aux, n_samples_list, intervals=4410):
        with torch.no_grad():
            # set max length
            max_samples = max(n_samples_list)
    
            # upsampling
            aux = F.pad(aux.transpose(1,2), (self.pad_left,self.pad_right), "replicate").transpose(1,2)
            x = self.upsampling(self.conv_s_c(self.conv(self.scale_in(aux.transpose(1,2))))) # B x C x T
    
            logging.info(x.shape)
            # padding if the length less than
            n_pad = self.receptive_field
            if n_pad > 0:
                audio = F.pad(audio, (n_pad, 0), "constant", self.n_quantize // 2)
                x = F.pad(x, (n_pad, 0), "replicate")

            logging.info(x.shape)
            #audio = OneHot(audio).transpose(1,2)
            audio = F.one_hot(audio, num_classes=self.n_quantize).float().transpose(1,2)
            #audio = OneHot(audio)
            if not self.audio_in_flag:
                x_ = x[:, :, :audio.size(2)]
            else:
                x_ = torch.cat((x[:, :, :audio.size(2)],audio),1)
            if self.wav_conv_flag:
                audio = self.wav_conv(audio) # B x C x T
            output = F.softsign(self.causal(audio)) # B x C x T
            output_buffer = []
            buffer_size = []
            for l in range(len(self.dil_facts)):
                _, output = self._dcrnn_forward(
                    x_, output, self.in_x[l], self.dil_h[l],
                    self.out_skip[l])
                if l < len(self.dil_facts)-1:
                    buffer_size.append(self.padding[l+1])
                else:
                    buffer_size.append(self.kernel_size - 1)
                output_buffer.append(output[:, :, -buffer_size[l] - 1: -1])
    
            # generate
            samples = audio.data  # B x T
            time_sample = []
            start = time.time()
            out_idx = self.kernel_size*2-1
            for i in range(max_samples):
                start_sample = time.time()
                samples_size = samples.size(-1)
                if not self.audio_in_flag:
                    x_ = x[:, :, (samples_size-1):samples_size]
                else:
                    x_ = torch.cat((x[:, :, (samples_size-1):samples_size],samples[:,:,-1:]),1)
                output = F.softsign(self.causal(samples[:,:,-out_idx:])[:,:,-self.kernel_size:]) # B x C x T
                output_buffer_next = []
                skip_connections = []
                for l in range(len(self.dil_facts)):
                    #start_ = time.time()
                    skip, output = self._generate_dcrnn_forward(
                        x_, output, self.in_x[l], self.dil_h[l],
                        self.out_skip[l])
                    output = torch.cat((output_buffer[l], output), 2)
                    output_buffer_next.append(output[:, :, -buffer_size[l]:])
                    skip_connections.append(skip)
    
                # update buffer
                output_buffer = output_buffer_next
    
                # get predicted sample
                output = self.out_2(F.relu(self.out_1(F.relu(sum(skip_connections))))).transpose(1,2)[:,-1]

                posterior = F.softmax(output, dim=-1)
                dist = torch.distributions.OneHotCategorical(posterior)
                sample = dist.sample().data  # B
                if i > 0:
                    out_samples = torch.cat((out_samples, torch.argmax(sample, dim=--1).unsqueeze(1)), 1)
                else:
                    out_samples = torch.argmax(sample, dim=--1).unsqueeze(1)

                if self.wav_conv_flag:
                    samples = torch.cat((samples, self.wav_conv(sample.unsqueeze(2))), 2)
                else:
                    samples = torch.cat((samples, sample.unsqueeze(2)), 2)
    
                # show progress
                time_sample.append(time.time()-start_sample)
                #if intervals is not None and (i + 1) % intervals == 0:
                if (i + 1) % intervals == 0:
                    logging.info("%d/%d estimated time = %.6f sec (%.6f sec / sample)" % (
                        (i + 1), max_samples,
                        (max_samples - i - 1) * ((time.time() - start) / intervals),
                        (time.time() - start) / intervals))
                    start = time.time()
                    #break
            logging.info("average time / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % (
                        np.mean(np.array(time_sample)), len(time_sample),
                        1.0/(1000*np.mean(np.array(time_sample)))))
            logging.info("average throughput / sample = %.6f sec (%ld samples * %ld) [%.3f kHz/s]" % (
                        sum(time_sample)/(len(time_sample)*len(n_samples_list)), len(time_sample),
                        len(n_samples_list), len(time_sample)*len(n_samples_list)/(1000*sum(time_sample))))
            samples = out_samples
    
            # devide into each waveform
            samples = samples[:, -max_samples:].cpu().numpy()
            samples_list = np.split(samples, samples.shape[0], axis=0)
            samples_list = [s[0, :n_s] for s, n_s in zip(samples_list, n_samples_list)]
    
            return samples_list
