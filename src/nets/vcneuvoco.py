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
from torch import nn

from torch.distributions.one_hot_categorical import OneHotCategorical

import numpy as np

CLIP_1E12 = -14.162084148244246758816564788835


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
    mu = mu - 1
    #fx = (y - 0.5) / mu * 2 - 1
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
    mu = mu - 1
    #fx = (y - 0.5) / mu * 2 - 1
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


class OutputConv1d(nn.Module):
    """Output Convolution 1d"""

    def __init__(self, in_dim=1024, lin_dim=256, out_dim=64, nonlinear=False):
        super(OutputConv1d, self).__init__()
        self.in_dim = in_dim
        self.lin_dim = lin_dim
        self.out_dim = out_dim
        module_list = []
        if nonlinear:
            module_list += [nn.ReLU(), nn.Conv1d(self.in_dim, self.lin_dim, 1), \
                            nn.Conv1d(self.lin_dim, self.out_dim, 1)]
        else:
            module_list += [nn.Conv1d(self.in_dim, self.lin_dim, 1), \
                            nn.Conv1d(self.lin_dim, self.out_dim, 1)]
        self.conv = nn.Sequential(*module_list)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """

        return self.conv(x)


class EmbeddingHalf(nn.Embedding):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(EmbeddingHalf, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.constant_(self.weight, 0.5)


class DualFC(nn.Module):
    """Compact Dual Fully Connected layers based on LPCNet"""

    def __init__(self, in_dim=16, out_dim=256, lpc=12, bias=True):
        super(DualFC, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lpc = lpc
        self.lpc_out_dim = self.lpc+self.out_dim
        self.lpc_out_dim_lpc = self.lpc_out_dim+self.lpc
        self.lpc2 = self.lpc*2
        self.out_dim2 = self.out_dim*2
        self.bias = bias

        self.conv = nn.Conv1d(self.in_dim, self.out_dim2+self.lpc2, 1, bias=self.bias)
        self.fact = EmbeddingHalf(1, self.out_dim2+self.lpc2)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C_in x T)

        Return:
            (Variable): float tensor variable with the shape (B x T x C_out)
        """

        # out = fact_1 o ((conv_1 * x)-tanh(conv_1 * x)) + fact_2 o ((conv_2 * x)-tanh(conv_2 * x))
        if self.lpc > 0:
            conv = self.conv(x).transpose(1,2)
            fact_weight = self.fact.weight[0]
            lpc = torch.sum((conv[:,:,:self.lpc2]*fact_weight[:self.lpc2]).reshape(x.shape[0],x.shape[2],2,-1), 2)
            logits = torch.sum((F.tanhshrink(conv[:,:,self.lpc2:])*fact_weight[self.lpc2:]).reshape(x.shape[0],x.shape[2],2,-1), 2)
            return lpc, logits
        else:
            return torch.sum((F.tanhshrink(self.conv(x).transpose(1,2))*self.fact.weight[0]).reshape(x.shape[0],x.shape[2],2,-1), 2)


class DualFCMult(nn.Module):
    """Compact Dual Fully Connected layers based on LPCNet with multiple samples output"""

    def __init__(self, in_dim=16, out_dim=256, seg=5, lpc=4, bias=True):
        super(DualFCMult, self).__init__()
        self.in_dim = in_dim
        self.seg = seg
        self.mid_dim = 160 // self.seg
        self.mid_dim_seg = self.mid_dim * self.seg
        self.mid_dim_seg_mid_dim = self.mid_dim_seg+self.mid_dim
        self.out_dim = out_dim
        self.lpc = lpc
        self.lpc_mid_dim_seg = self.lpc+self.mid_dim_seg
        self.lpc_mid_dim_seg_lpc = self.lpc_mid_dim_seg+self.lpc
        self.lpc_mid_dim_seg_lpc_mid_dim = self.lpc_mid_dim_seg_lpc+self.mid_dim
        self.lpc_mid_dim = self.lpc+self.mid_dim
        self.lpc_out_dim = self.lpc+self.out_dim
        self.lpc_out_dim_lpc = self.lpc_out_dim+self.lpc
        self.lpc2 = self.lpc*2
        self.mid_dim2 = self.mid_dim*2
        self.out_dim2 = self.out_dim*2
        self.mid_dim_seg2 = self.mid_dim_seg*2
        self.bias = bias

        self.conv = nn.Conv1d(self.in_dim, self.mid_dim_seg2+self.lpc2, 1, bias=self.bias)
        self.out = nn.Conv1d(self.mid_dim2, self.out_dim2 ,1)
        self.fact = EmbeddingHalf(1, self.out_dim2+self.lpc2)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C_in x T)

        Return:
            (Variable): float tensor variable with the shape (B x T x C_out)
        """

        # out = fact_1 o tanh(conv_1 * x) + fact_2 o tanh(conv_2 * x)
        conv = self.conv(x).transpose(1,2)
        B = conv.shape[0]
        T_seg = conv.shape[1]
        if self.lpc > 0:
            fact_weight = self.fact.weight[0]
            # B x T_seg x K*2 --> B x T_seg x K
            lpc = torch.sum((conv[:,:,:self.lpc2]*fact_weight[:self.lpc2]).reshape(B,T_seg,2,-1), 2)
            # B x T_seg x mid_dim_seg*2 --> B x T x mid_dim*2 --> B x T x out_dim*2 --> B x T x out_dim
            logits = torch.sum((F.tanhshrink(self.out(conv[:,:,self.lpc2:].reshape(B,-1,self.mid_dim*2).transpose(1,2)).transpose(1,2))
                        *fact_weight[self.lpc2:]).reshape(B,T_seg*self.seg,2,-1), 2)
            return lpc, logits
        else:
            # B x T_seg x mid_dim_seg*2 --> B x T x mid_dim*2 --> B x T x out_dim*2 --> B x T x out_dim
            return torch.sum((F.tanhshrink(self.out(self.conv(x).transpose(1,2).reshape(x.shape[0],-1,self.mid_dim*2).transpose(1,2)).transpose(1,2))
                    *self.fact.weight[0]).reshape(x.shape[0],x.shape[2]*self.seg,2,-1), 2)


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


def sampling_laplace(param):
    k = param.shape[-1]//2
    mu = param[:,:,:k]
    scale = torch.exp(param[:,:,k:])
    eps = torch.empty_like(mu).uniform_(torch.finfo(mu.dtype).eps-1,1)

    return mu - scale * eps.sign() * torch.log1p(-eps.abs()) # scale
 

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


class GRU_VAE_ENCODER(nn.Module):
    def __init__(self, in_dim=50, n_spk=14, lat_dim=50, hidden_layers=1, hidden_units=1024, kernel_size=7, \
            dilation_size=1, do_prob=0, bi=False, nonlinear_conv=False, onehot_lat=False, disc_cont=False, \
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
        self.disc_cont = disc_cont
        self.onehot_lat = onehot_lat
        self.right_size = right_size
        self.pad_first = pad_first
        if self.onehot_lat:
            self.cont = False
        if self.disc_cont:
            self.cont = True
        self.use_weight_norm = use_weight_norm
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
        self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)

        # Conv. layers
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

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
        else:
            self.apply(initialize)

    def forward(self, x, yz_in=None, h=None, do=False, sampling=True, outpad_right=0):
        x = self.scale_in(x.transpose(1,2))
        if not self.ar:
            # Input s layers
            if self.do_prob > 0 and do:
                s = self.conv_drop(self.conv(x).transpose(1,2)) # B x C x T --> B x T x C
            else:
                s = self.conv(x).transpose(1,2) # B x C x T --> B x T x C
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
                if self.onehot_lat:
                    if do:
                        logits = torch.clamp(s[:,:,self.n_spk:], max=27.631021)
                    else:
                        logits = s[:,:,self.n_spk:]
                    return F.selu(s[:,:,:self.n_spk]), F.softmax(logits, dim=-1), logits, h.detach()
                else:
                    return F.selu(s[:,:,:self.n_spk]), s[:,:,self.n_spk:], h.detach()
        else:
            # Input layers
            if self.do_prob > 0 and do:
                x_conv = self.conv_drop(self.conv(x).transpose(1,2)) # B x C x T --> B x T x C
            else:
                x_conv = self.conv(x).transpose(1,2) # B x C x T --> B x T x C

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
                if sampling or self.disc_cont:
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
                if self.onehot_lat:
                    if do:
                        qz_alphas = torch.clamp(qz_alphas, max=27.631021)
                    return qy_logits, F.softmax(qz_alphas, dim=-1), qz_alphas, h.detach(), yz_in.detach()
                else:
                    return qy_logits, qz_alphas, h.detach(), yz_in.detach()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                #logging.debug(f"Weight norm is applied to {m}.")
                logging.info(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_gru_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_gru_weight_norm(m):
            if isinstance(m, torch.nn.GRU):
                logging.info(m)
                list_name = []
                for name, param in m.named_parameters():
                    list_name.append(name)
                logging.info(list_name)
                for name in list_name:
                    if 'weight' in name:
                        logging.info(name)
                        torch.nn.utils.weight_norm(m, name=name)

                        #logging.debug(f"Weight norm is applied to {m} {name}.")
                        logging.info(f"Weight norm is applied to {m} {name}.")
                for name, param in m.named_parameters():
                    logging.info(name)

        self.apply(_apply_gru_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d) \
                    or isinstance(m, torch.nn.ConvTranspose2d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
                elif isinstance(m, torch.nn.GRU):
                    list_name = []
                    for name, param in m.named_parameters():
                        list_name.append(name)
                    logging.info(list_name)
                    for name in list_name:
                        if 'weight' in name and '_g' in name:
                            torch.nn.utils.remove_weight_norm(m, name=name.replace('_g',''))
                            #logging.debug(f"Weight norm is removed from {m} {name}.")
                            logging.info(f"Weight norm is removed from {m} {name}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_SPEC_DECODER(nn.Module):
    def __init__(self, feat_dim=50, out_dim=50, hidden_layers=1, hidden_units=1024, causal_conv=False,
            kernel_size=7, dilation_size=1, do_prob=0, n_spk=14, bi=False, nonlinear_conv=False, ctr_size=None,
                onehot_lat_dim=None, use_weight_norm=True, ar=False, cap_dim=None, excit_dim=None, spkidtr_dim=0,
                    pad_first=True, diff=False):
        super(GRU_SPEC_DECODER, self).__init__()
        self.n_spk = n_spk
        self.feat_dim = feat_dim
        self.spkidtr_dim = spkidtr_dim
        self.in_dim = self.n_spk+self.feat_dim
        self.cap_dim = cap_dim
        if self.cap_dim is not None:
            self.out_dim = out_dim+1+self.cap_dim
            self.uvcap_dim = self.cap_dim+1
        else:
            self.out_dim = out_dim
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
        self.ctr_size = ctr_size
        self.onehot_lat_dim = onehot_lat_dim
        self.use_weight_norm = use_weight_norm
        self.diff = diff
        self.pad_first = pad_first
        if self.bi:
            self.hidden_units_out = 2*self.hidden_units
        else:
            self.hidden_units_out = self.hidden_units

        if self.excit_dim is not None:
            self.scale_in = nn.Conv1d(self.excit_dim, self.excit_dim, 1)
            self.in_dim += self.excit_dim

        # Conv. layers
        if self.onehot_lat_dim is not None:
            self.onehot_conv = nn.Conv1d(self.onehot_lat_dim, self.feat_dim, 1)
        if self.ctr_size is not None:
            self.ctr_conv = nn.Conv1d(self.ctr_size, self.feat_dim, 1)

        if self.spkidtr_dim > 0:
            self.spkidtr_conv = nn.Conv1d(self.n_spk, self.spkidtr_dim, 1)
            spkidtr_deconv = [nn.Conv1d(self.spkidtr_dim, self.n_spk, 1), nn.ReLU()]
            self.spkidtr_deconv = nn.Sequential(*spkidtr_deconv)

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
        if self.cap_dim is not None:
            self.scale_out_cap = nn.Conv1d(self.cap_dim, self.cap_dim, 1)
        self.scale_out = nn.Conv1d(self.out_dim-self.uvcap_dim, self.out_dim-self.uvcap_dim, 1)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
        #    #torch.nn.utils.remove_weight_norm(self.scale_out)
        else:
            self.apply(initialize)

    def forward(self, y, z, x_in=None, h=None, x_prev=None, do=False, e=None, outpad_right=0):
        if self.onehot_lat_dim is not None:
            z = self.onehot_conv(z.transpose(1,2)).transpose(1,2)
        if self.ctr_size is not None:
            z = self.ctr_conv(z.transpose(1,2)).transpose(1,2)
        if len(y.shape) == 2:
            if self.spkidtr_dim > 0:
                if e is not None:
                    z = torch.cat((self.spkidtr_deconv(self.spkidtr_conv(F.one_hot(y, num_classes=self.n_spk).float().transpose(1,2))).transpose(1,2), self.scale_in(e.transpose(1,2)).transpose(1,2), z), 2)
                else:
                    z = torch.cat((self.spkidtr_deconv(self.spkidtr_conv(F.one_hot(y, num_classes=self.n_spk).float().transpose(1,2))).transpose(1,2), z), 2) # B x T_frm x C
            else:
                if e is not None:
                    z = torch.cat((F.one_hot(y, num_classes=self.n_spk).float(), self.scale_in(e.transpose(1,2)).transpose(1,2), z), 2) # B x T_frm x C
                else:
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
                if not self.diff:
                    return torch.cat((torch.sigmoid(e[:,:,:1]), self.scale_out_cap(e[:,:,1:self.uvcap_dim].transpose(1,2)).transpose(1,2), \
                                    self.scale_out(e[:,:,self.uvcap_dim:].transpose(1,2)).transpose(1,2)), 2), h.detach()
                else:
                    e = torch.cat((x_prev, e[:,:-1]), 1) + e
                    return torch.cat((torch.sigmoid(e[:,:,:1]), self.scale_out_cap(e[:,:,1:self.uvcap_dim].transpose(1,2)).transpose(1,2), \
                                    self.scale_out(e[:,:,self.uvcap_dim:].transpose(1,2)).transpose(1,2)), 2), h.detach(), e[:,-1:].detach()
            else:
                if not self.diff:
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

    def apply_gru_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_gru_weight_norm(m):
            if isinstance(m, torch.nn.GRU):
                logging.info(m)
                list_name = []
                for name, param in m.named_parameters():
                    list_name.append(name)
                logging.info(list_name)
                for name in list_name:
                    if 'weight' in name:
                        logging.info(name)
                        torch.nn.utils.weight_norm(m, name=name)

                        #logging.debug(f"Weight norm is applied to {m} {name}.")
                        logging.info(f"Weight norm is applied to {m} {name}.")
                for name, param in m.named_parameters():
                    logging.info(name)

        self.apply(_apply_gru_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d) \
                    or isinstance(m, torch.nn.ConvTranspose2d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
                elif isinstance(m, torch.nn.GRU):
                    list_name = []
                    for name, param in m.named_parameters():
                        list_name.append(name)
                    logging.info(list_name)
                    for name in list_name:
                        if 'weight' in name and '_g' in name:
                            torch.nn.utils.remove_weight_norm(m, name=name.replace('_g',''))
                            #logging.debug(f"Weight norm is removed from {m} {name}.")
                            logging.info(f"Weight norm is removed from {m} {name}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_EXCIT_DECODER(nn.Module):
    def __init__(self, feat_dim=50, hidden_layers=1, hidden_units=1024, causal_conv=False, ctr_size=None, \
            kernel_size=7, dilation_size=1, do_prob=0, n_spk=14, bi=False, nonlinear_conv=False, \
                onehot_lat_dim=None, use_weight_norm=True, ar=False, cap_dim=None, spkidtr_dim=0, \
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
        self.onehot_lat_dim = onehot_lat_dim
        self.ar = ar
        self.ctr_size = ctr_size
        self.use_weight_norm = use_weight_norm
        self.diff = diff
        self.pad_first = pad_first
        if self.bi:
            self.hidden_units_out = 2*self.hidden_units
        else:
            self.hidden_units_out = self.hidden_units

        #self.scale_in = nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        # Conv. layers
        if self.onehot_lat_dim is not None:
            self.onehot_conv = nn.Conv1d(self.onehot_lat_dim, self.feat_dim, 1)
        if self.ctr_size is not None:
            self.ctr_conv = nn.Conv1d(self.ctr_size, self.feat_dim, 1)

        if self.spkidtr_dim > 0:
            self.spkidtr_conv = nn.Conv1d(self.n_spk, self.spkidtr_dim, 1)
            spkidtr_deconv = [nn.Conv1d(self.spkidtr_dim, self.n_spk, 1), nn.ReLU()]
            self.spkidtr_deconv = nn.Sequential(*spkidtr_deconv)

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
        if self.onehot_lat_dim is not None:
            z = self.onehot_conv(z.transpose(1,2)).transpose(1,2)
        if self.ctr_size is not None:
            z = self.ctr_conv(z.transpose(1,2)).transpose(1,2)
        if len(y.shape) == 2:
            if self.spkidtr_dim > 0:
                z = torch.cat((self.spkidtr_deconv(self.spkidtr_conv(F.one_hot(y, num_classes=self.n_spk).float().transpose(1,2))).transpose(1,2), z), 2) # B x T_frm x C
            else:
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
                #return torch.cat((torch.sigmoid(e[:,:,:1]), self.scale_out(e[:,:,1:2].transpose(1,2)).transpose(1,2), \
                #                torch.sigmoid(e[:,:,2:3]), self.scale_out_cap(e[:,:,3:].transpose(1,2)).transpose(1,2)), 2), h.detach()
                return torch.cat((torch.sigmoid(e[:,:,:1]), torch.clamp(self.scale_out(e[:,:,1:2].transpose(1,2)).transpose(1,2), max=8), \
                                torch.sigmoid(e[:,:,2:3]), torch.clamp(self.scale_out_cap(e[:,:,3:].transpose(1,2)).transpose(1,2), max=8)), 2), h.detach()
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

    def apply_gru_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_gru_weight_norm(m):
            if isinstance(m, torch.nn.GRU):
                logging.info(m)
                list_name = []
                for name, param in m.named_parameters():
                    list_name.append(name)
                logging.info(list_name)
                for name in list_name:
                    if 'weight' in name:
                        logging.info(name)
                        torch.nn.utils.weight_norm(m, name=name)

                        #logging.debug(f"Weight norm is applied to {m} {name}.")
                        logging.info(f"Weight norm is applied to {m} {name}.")
                for name, param in m.named_parameters():
                    logging.info(name)

        self.apply(_apply_gru_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                if isinstance(m, torch.nn.Conv1d) \
                    or isinstance(m, torch.nn.ConvTranspose2d):
                    torch.nn.utils.remove_weight_norm(m)
                    #logging.debug(f"Weight norm is removed from {m}.")
                    logging.info(f"Weight norm is removed from {m}.")
                elif isinstance(m, torch.nn.GRU):
                    list_name = []
                    for name, param in m.named_parameters():
                        list_name.append(name)
                    logging.info(list_name)
                    for name in list_name:
                        if 'weight' in name and '_g' in name:
                            torch.nn.utils.remove_weight_norm(m, name=name.replace('_g',''))
                            #logging.debug(f"Weight norm is removed from {m} {name}.")
                            logging.info(f"Weight norm is removed from {m} {name}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class GRU_WAVE_DECODER_DUALGRU_COMPACT(nn.Module):
    def __init__(self, feat_dim=52, upsampling_factor=120, hidden_units=384, hidden_units_2=16, n_quantize=256, lpc=12,
            kernel_size=7, dilation_size=1, do_prob=0, causal_conv=False, use_weight_norm=True, nonlinear_conv=False,
                right_size=0):
        super(GRU_WAVE_DECODER_DUALGRU_COMPACT, self).__init__()
        self.feat_dim = feat_dim
        self.in_dim = self.feat_dim
        self.n_quantize = n_quantize
        self.out_dim = self.n_quantize
        self.upsampling_factor = upsampling_factor
        self.hidden_units = hidden_units
        self.hidden_units_2 = hidden_units_2
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.nonlinear_conv = nonlinear_conv
        self.s_dim = 256
        self.use_weight_norm = use_weight_norm
        self.lpc = lpc
        self.right_size = right_size

        self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)

        # Conv. layers
        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv)
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv)
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                        right_size=self.right_size, nonlinear=self.nonlinear_conv)
        self.conv_s_c = nn.Conv1d(self.in_dim*self.conv.rec_field, self.s_dim, 1)
        self.embed_wav = nn.Embedding(self.n_quantize, self.s_dim)
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        self.gru = nn.GRU(self.s_dim*2, self.hidden_units, 1, batch_first=True) #concat for small gru
        self.gru_2 = nn.GRU(self.s_dim+self.hidden_units, self.hidden_units_2, 1, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        self.out = DualFC(self.hidden_units_2, self.n_quantize, self.lpc) # for 2nd small gru

        # Prev logits if using data-driven lpc
        if self.lpc > 0:
            self.logits = nn.Embedding(self.n_quantize, self.n_quantize)
            logits_param = torch.empty(self.n_quantize, self.n_quantize).fill_(-9)
            for i in range(self.n_quantize):
                logits_param[i,i] = 9
            self.logits.weight = torch.nn.Parameter(logits_param)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
            torch.nn.utils.remove_weight_norm(self.scale_in)
        else:
            self.apply(initialize)

    def forward(self, c, x_prev, h=None, h_2=None, do=False, x_lpc=None):
        # Input
        if self.do_prob > 0 and do:
            conv = self.conv_drop(torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2),self.upsampling_factor,dim=1))
        else:
            conv = torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2),self.upsampling_factor,dim=1)

        # GRU1
        if h is not None:
            out, h = self.gru(torch.cat((conv,self.embed_wav(x_prev)),2), h)
        else:
            out, h = self.gru(torch.cat((conv,self.embed_wav(x_prev)),2))

        # GRU2
        if self.do_prob > 0 and do:
            if h_2 is not None:
                out, h_2 = self.gru_2(torch.cat((conv,self.gru_drop(out)),2), h_2) # B x T x C -> B x C x T -> B x T x C
            else:
                out, h_2 = self.gru_2(torch.cat((conv,self.gru_drop(out)),2)) # B x T x C -> B x C x T -> B x T x C
        else:
            if h_2 is not None:
                out, h_2 = self.gru_2(torch.cat((conv,out),2), h_2) # B x T x C -> B x C x T -> B x T x C
            else:
                out, h_2 = self.gru_2(torch.cat((conv,out),2)) # B x T x C -> B x C x T -> B x T x C

        # output
        if self.lpc > 0:
            lpc, logits = self.out(out.transpose(1,2)) # B x T x K and B x T x 256
            # x_lpc B x T_lpc --> B x T x K --> B x T x K x 256
            return logits + torch.sum(lpc.flip(-1).unsqueeze(-1)*self.logits(x_lpc.unfold(1, self.lpc, 1)), 2), h.detach(), h_2.detach()
        else:
            return self.out(out.transpose(1,2)), h.detach(), h_2.detach()

    def generate(self, c, intervals=4000):
        start = time.time()
        time_sample = []

        c = self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2)
        if self.lpc > 0:
            x_lpc = torch.empty(c.shape[0],1,self.lpc).cuda().fill_(self.n_quantize // 2).long()
        T = c.shape[1]*self.upsampling_factor

        c_f = c[:,:1]
        out, h = self.gru(torch.cat((c_f,self.embed_wav(torch.zeros(c.shape[0],1).cuda().fill_(self.n_quantize//2).long())),2))
        out, h_2 = self.gru_2(torch.cat((c_f,out),2))
        if self.lpc > 0:
            out = self.out(out.transpose(1,2), clip=False) # B x T x C -> B x C x T -> B x T x C
            dist = OneHotCategorical(F.softmax(out[:,:,self.lpc:] + torch.sum(out[:,:,:self.lpc].unsqueeze(-1)*self.logits(x_lpc), 2), dim=-1))
            x_wav = dist.sample().argmax(dim=-1)
            x_out = x_wav
            x_lpc[:,:,1:] = x_lpc[:,:,:-1]
            x_lpc[:,:,0] = x_wav
        else:
            dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2), clip=False), dim=-1))
            x_wav = dist.sample().argmax(dim=-1)
            x_out = x_wav

        time_sample.append(time.time()-start)
        if self.lpc > 0:
            for t in range(1,T):
                start_sample = time.time()

                if t % self.upsampling_factor  == 0:
                    idx_t_f = t//self.upsampling_factor
                    c_f = c[:,idx_t_f:idx_t_f+1]
                out, h = self.gru(torch.cat((c_f, self.embed_wav(x_wav)),2), h)
                out, h_2 = self.gru_2(torch.cat((c_f,out),2), h_2)

                out = self.out(out.transpose(1,2), clip=False) # B x T x C -> B x C x T -> B x T x C
                dist = OneHotCategorical(F.softmax(out[:,:,self.lpc:] + torch.sum(out[:,:,:self.lpc].unsqueeze(-1)*self.logits(x_lpc), 2), dim=-1))
                x_wav = dist.sample().argmax(dim=-1)
                x_out = torch.cat((x_out, x_wav), 1)
                x_lpc[:,:,1:] = x_lpc[:,:,:-1]
                x_lpc[:,:,0] = x_wav

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

                if t % self.upsampling_factor  == 0:
                    idx_t_f = t//self.upsampling_factor
                    c_f = c[:,idx_t_f:idx_t_f+1]
                out, h = self.gru(torch.cat((c_f, self.embed_wav(x_wav)),2), h)
                out, h_2 = self.gru_2(torch.cat((c_f,out),2), h_2)

                dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2), clip=False), dim=-1))
                x_wav = dist.sample().argmax(dim=-1)
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

        return decode_mu_law(x_out.cpu().data.numpy())

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


class GRU_WAVE_DECODER_DUALGRU_COMPACT_LPCSEG(nn.Module):
    def __init__(self, feat_dim=52, upsampling_factor=120, hidden_units=384, hidden_units_2=32, n_quantize=256, lpc=4,
            kernel_size=7, dilation_size=1, do_prob=0, causal_conv=False, use_weight_norm=True, nonlinear_conv=False,
                right_size=1, seg=2):
        super(GRU_WAVE_DECODER_DUALGRU_COMPACT_LPCSEG, self).__init__()
        self.feat_dim = feat_dim
        self.in_dim = self.feat_dim
        self.n_quantize = n_quantize
        self.out_dim = self.n_quantize
        self.upsampling_factor = upsampling_factor
        self.hidden_units = hidden_units
        self.hidden_units_2 = hidden_units_2
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.causal_conv = causal_conv
        self.nonlinear_conv = nonlinear_conv
        self.use_weight_norm = use_weight_norm
        self.right_size = right_size
        self.lpc = lpc
        self.s_dim = 320
        self.seg = seg
        self.seg_1 = self.seg-1
        self.seg_offset = self.seg+self.seg_1 # include seg-1 of t-seg samples for 1 shift segment grouping generation in training
        self.lpc = lpc
        self.lpc_offset = self.lpc+self.seg_1 # t-(n_lpc + seg-1) until T_batch-1 is the index range of current t LPC
        self.wav_dim = self.s_dim // self.seg
        self.wav_dim_seg = self.wav_dim * self.seg
        self.cond_dim = self.s_dim // self.seg
        self.cond_dim_seg = self.cond_dim * self.seg
        assert(self.wav_dim_seg == self.cond_dim_seg)

        self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)

        # Conv. layers
        if self.right_size <= 0:
            if not self.causal_conv:
                self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv)
            else:
                self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size, nonlinear=self.nonlinear_conv)
        else:
            self.conv = SkewedConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                        right_size=self.right_size, nonlinear=self.nonlinear_conv)
        conv_s_c = [nn.Conv1d(self.in_dim*self.conv.rec_field, self.cond_dim, 1), nn.ReLU()]
        self.conv_s_c = nn.Sequential(*conv_s_c)
        self.embed_wav = nn.Embedding(self.n_quantize, self.wav_dim)
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        self.gru = nn.GRU(self.s_dim*2, self.hidden_units, 1, batch_first=True) #concat for small gru
        self.gru_2 = nn.GRU(self.s_dim+self.hidden_units, self.hidden_units_2, 1, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)

        # Output layers
        self.out = DualFCMult(self.hidden_units_2, self.n_quantize, self.seg, self.lpc) # for 2nd small gru

        # Prev logits if using data-driven lpc
        if self.lpc > 0:
            self.logits = nn.Embedding(self.n_quantize, self.n_quantize)
            logits_param = torch.empty(self.n_quantize, self.n_quantize).fill_(-9)
            for i in range(self.n_quantize):
                logits_param[i,i] = 9
            self.logits.weight = torch.nn.Parameter(logits_param)

        # apply weight norm
        if self.use_weight_norm:
            self.apply_weight_norm()
            torch.nn.utils.remove_weight_norm(self.scale_in)
        else:
            self.apply(initialize)

    def forward(self, c, x_prev, h=None, h_2=None, do=False, x_lpc=None, first=False, shift1=True):
        # input
        B = c.shape[0]
        if shift1:
            if not first:
                # upsample, seg-1 from prev. frame,conv_vec, B x T x C
                c = torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))),self.upsampling_factor,dim=-1)[:,:,self.upsampling_factor-self.seg_1:].transpose(1,2)
            else:
                # upsample, pad left seg-1,conv_vec, B x T x C
                c = F.pad(torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))),self.upsampling_factor,dim=-1), (self.seg_1, 0), "replicate").transpose(1,2)
            x_prev = self.embed_wav(x_prev) # B x T --> B x T x C

            cs = c[:,:-self.seg_1].unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.cond_dim_seg) # concat cond_vec at each seg.
            x_prevs = x_prev[:,:-self.seg_1].unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.wav_dim_seg) # concat wav_vec at each seg.
            if self.seg == 2:
                cs = torch.cat((cs, c[:,1:].unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.cond_dim_seg)), 0)
                x_prevs = torch.cat((x_prevs, x_prev[:,1:].unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.wav_dim_seg)), 0) # concat wav_vec at each seg.
            else:
                for i in range(1,self.seg):
                    if i < self.seg_1:
                        cs = torch.cat((cs, c[:,i:-self.seg_1+i].unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.cond_dim_seg)), 0)
                        x_prevs = torch.cat((x_prevs, x_prev[:,i:-self.seg_1+i].unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.wav_dim_seg)), 0) # concat wav_vec at each seg.
                    else:
                        cs = torch.cat((cs, c[:,i:].unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.cond_dim_seg)), 0)
                        x_prevs = torch.cat((x_prevs, x_prev[:,i:].unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.wav_dim_seg)), 0) # concat wav_vec at each seg.
        else:
            cs = torch.repeat_interleave(self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2),self.upsampling_factor,dim=1).unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.cond_dim_seg)
            x_prevs = self.embed_wav(x_prev).unfold(1, self.seg, self.seg).permute(0,1,3,2).reshape(B,-1,self.wav_dim_seg)
        if self.do_prob > 0 and do:
            cs = self.conv_drop(cs)
        # B_seg x T_seg x C_seg

        # GRU1
        if h is not None:
            out, h = self.gru(torch.cat((cs, x_prevs), 2), h) # B x T x C
        else:
            out, h = self.gru(torch.cat((cs, x_prevs), 2)) # B x T x C

        # GRU2
        if self.do_prob > 0 and do:
            if h_2 is not None:
                out, h_2 = self.gru_2(torch.cat((cs, self.gru_drop(out)), 2), h_2) # B x T x C -> B x C x T -> B x T x C
            else:
                out, h_2 = self.gru_2(torch.cat((cs, self.gru_drop(out)), 2)) # B x T x C -> B x C x T -> B x T x C
        else:
            if h_2 is not None:
                out, h_2 = self.gru_2(torch.cat((cs, out), 2), h_2) # B x T x C -> B x C x T -> B x T x C
            else:
                out, h_2 = self.gru_2(torch.cat((cs, out), 2)) # B x T x C -> B x C x T -> B x T x C

        # output
        if self.lpc > 0:
            lpc, logits = self.out(out.transpose(1,2)) # B_seg x T_seg x K and B_seg x T x 256
            # B_seg x T_seg x K --> B_seg x T_seg x 1 x K x 1 * (B_seg x T_seg x seg x K --> B_seg x T_seg x seg x K x 256) --> B_seg x T x 256
            return logits + torch.sum(lpc.flip(-1).unsqueeze(-1).unsqueeze(2)*self.logits(x_lpc), 3).reshape(logits.shape[0],-1,self.n_quantize), h.detach(), h_2.detach()
        else:
            return self.out(out.transpose(1,2)), h.detach(), h_2.detach()

    def generate(self, c, intervals=4000):
        start = time.time()
        time_sample = []

        c = self.conv_s_c(self.conv(self.scale_in(c.transpose(1,2)))).transpose(1,2)
        if self.lpc > 0:
            x_lpc = torch.empty(c.shape[0],1,self.lpc).cuda().fill_(self.n_quantize // 2).long()
        T = c.shape[1]*self.upsampling_factor

        c_f = c[:,:1]
        out, h = self.gru(torch.cat((c_f,self.embed_wav(torch.zeros(c.shape[0],1).cuda().fill_(self.n_quantize//2).long())),2))
        out, h_2 = self.gru_2(torch.cat((c_f,out),2))
        if self.lpc > 0:
            out = self.out(out.transpose(1,2), clip=False) # B x T x C -> B x C x T -> B x T x C
            dist = OneHotCategorical(F.softmax(out[:,:,self.lpc:] + torch.sum(out[:,:,:self.lpc].unsqueeze(-1)*self.logits(x_lpc), 2), dim=-1))
            x_wav = dist.sample().argmax(dim=-1)
            x_out = x_wav
            x_lpc[:,:,1:] = x_lpc[:,:,:-1]
            x_lpc[:,:,0] = x_wav
        else:
            dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2), clip=False), dim=-1))
            x_wav = dist.sample().argmax(dim=-1)
            x_out = x_wav

        time_sample.append(time.time()-start)
        if self.lpc > 0:
            for t in range(1,T):
                start_sample = time.time()

                if t % self.upsampling_factor  == 0:
                    idx_t_f = t//self.upsampling_factor
                    c_f = c[:,idx_t_f:idx_t_f+1]
                out, h = self.gru(torch.cat((c_f, self.embed_wav(x_wav)),2), h)
                out, h_2 = self.gru_2(torch.cat((c_f,out),2), h_2)

                out = self.out(out.transpose(1,2), clip=False) # B x T x C -> B x C x T -> B x T x C
                dist = OneHotCategorical(F.softmax(out[:,:,self.lpc:] + torch.sum(out[:,:,:self.lpc].unsqueeze(-1)*self.logits(x_lpc), 2), dim=-1))
                x_wav = dist.sample().argmax(dim=-1)
                x_out = torch.cat((x_out, x_wav), 1)
                x_lpc[:,:,1:] = x_lpc[:,:,:-1]
                x_lpc[:,:,0] = x_wav

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

                if t % self.upsampling_factor  == 0:
                    idx_t_f = t//self.upsampling_factor
                    c_f = c[:,idx_t_f:idx_t_f+1]
                out, h = self.gru(torch.cat((c_f, self.embed_wav(x_wav)),2), h)
                out, h_2 = self.gru_2(torch.cat((c_f,out),2), h_2)

                dist = OneHotCategorical(F.softmax(self.out(out.transpose(1,2), clip=False), dim=-1))
                x_wav = dist.sample().argmax(dim=-1)
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

        return decode_mu_law(x_out.cpu().data.numpy())

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


class LaplaceLoss(nn.Module):
    def __init__(self):
        super(LaplaceLoss, self).__init__()
        self.c = 0.69314718055994530941723212145818 # ln(2)

    def forward(self, mu, b, target, log_b):
        return torch.mean(self.c + log_b + torch.abs(target-mu)/b, -1)


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
    def __init__(self, n_quantize=256, n_aux=54, hid_chn=192, skip_chn=256, aux_kernel_size=3, \
                aux_dilation_size=2, dilation_depth=3, dilation_repeat=3, kernel_size=6, \
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

        # Input Layers
        self.scale_in = nn.Conv1d(self.n_aux, self.n_aux, 1)
        self.conv_aux = TwoSidedDilConv1d(in_dim=self.n_aux, kernel_size=aux_kernel_size, \
                                            layers=aux_dilation_size)
        self.in_aux_dim = self.n_aux*self.conv_aux.rec_field
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
        x = self.upsampling(self.conv_aux(self.scale_in(aux.transpose(1,2))))
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
            x = self.upsampling(self.conv_aux(self.scale_in(aux.transpose(1,2)))) # B x C x T
    
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
            logging.info("average time / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % (\
                        np.mean(np.array(time_sample)), len(time_sample), \
                        1.0/(1000*np.mean(np.array(time_sample)))))
            logging.info("average throughput / sample = %.6f sec (%ld samples * %ld) [%.3f kHz/s]" % (\
                        sum(time_sample)/(len(time_sample)*len(n_samples_list)), len(time_sample), \
                        len(n_samples_list), len(time_sample)*len(n_samples_list)/(1000*sum(time_sample))))
            samples = out_samples
    
            # devide into each waveform
            samples = samples[:, -max_samples:].cpu().numpy()
            samples_list = np.split(samples, samples.shape[0], axis=0)
            samples_list = [s[0, :n_s] for s, n_s in zip(samples_list, n_samples_list)]
    
            return samples_list
