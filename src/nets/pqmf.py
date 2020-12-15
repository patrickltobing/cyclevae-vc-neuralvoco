# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)
# Modified by Patrick Lumban Tobing on August 2020

"""Pseudo QMF modules."""

import numpy as np
import torch
import torch.nn.functional as F

from scipy.signal import kaiser


def design_prototype_filter(taps=64, cutoff_ratio=0.15, beta=10.06126):
    """Design prototype filter for PQMF.

    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.

    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.

    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).

    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427

    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid='ignore'):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) \
            / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    """PQMF module.

    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122

    """

    def __init__(self, subbands=4):
        """Initilize PQMF module.

        Args:
            subbands (int): The number of subbands.

        """
        super(PQMF, self).__init__()

        self.subbands = subbands
        # Kaiser parameters calculation
        #self.err = 1e-5 # passband ripple
        #self.err = 1e-10 # passband ripple
        self.err = 1e-20 # passband ripple
        self.A = -20*np.log10(self.err)  # attenuation in stopband [dB]
        self.taps = int((self.A-8)/(2.285*(0.8/self.subbands)*np.pi)) # (0.8/subbands * pi) is the width of band-transition
        if self.taps % 2 != 0:
            self.taps += 1
        self.cutoff_ratio = round(0.6/self.subbands, 4)
        self.beta = round(0.1102*(self.A-8.7), 5)
        #print(f'{subbands} {err} {A} {taps} {cutoff_ratio} {beta}')

        # define filter coefficient
        h_proto = design_prototype_filter(self.taps, self.cutoff_ratio, self.beta)
        # n_bands x (taps+1)
        h_analysis = np.zeros((self.subbands, len(h_proto)))
        h_synthesis = np.zeros((self.subbands, len(h_proto)))
        for k in range(self.subbands):
            h_analysis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * self.subbands)) *
                (np.arange(self.taps + 1) - ((self.taps - 1) / 2)) +
                (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * self.subbands)) *
                (np.arange(self.taps + 1) - ((self.taps - 1) / 2)) -
                (-1) ** k * np.pi / 4)

        # convert to tensor
        # out x in x kernel --> weight shape of Conv1d pytorch
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1) # n_bands x 1 x (taps+1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0) # 1 x n_bands x (taps+1)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        ## filter for downsampling & upsampling
        # down/up-sampling filter is used in the multiband domain, hence out=in=n_bands
        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0 #only the 1st kernel, i.e., zero to the other right samples
        self.register_buffer("updown_filter", updown_filter)

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(self.taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.

        Args:
            x (Tensor): Input tensor (B, 1, T).

        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).

        """
        # B x 1 x T --> B x n_bands x T
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        # B x n_bands x T --> B x n_bands x (T//n_bands) [discard the 2nd-nth indices every n index]
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.

        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).

        Returns:
            Tensor: Output tensor (B, 1, T).

        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        # B x n_bands x (T//n_bands) --> B x n_bands x T 
        # [zeroing the 2nd-nth indices every n index, and multiply by n_bands at each 1st index]
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        # B x n_bands x T --> B x 1 x T
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)
