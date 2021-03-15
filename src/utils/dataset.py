#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import os
import logging
from utils import read_hdf5, check_hdf5, write_hdf5, shape_hdf5
from torch.utils.data import Dataset
import soundfile as sf


def padding(x, flen, value=0):
    """Pad values to end by flen"""
    diff = flen - x.shape[0]
    x_len = len(x.shape)
    if diff > 0:
        if x_len > 1:
            if value is not None: #pad value
                x = np.r_[x, np.ones((diff,) + x.shape[1:]) * value]
            else: #pad replicate
                x = np.r_[x, np.ones((diff,) + x.shape[1:]) * x[-1:]]
        else:
            if value is not None: #pad value
                x = np.r_[x, np.ones(diff) * value]
            else: #pad replicate
                #x = np.r_[x, np.zeros(diff)]
                x = np.r_[x, np.ones(diff) * x[-1:]]
    return x


def validate_length(x, y, upsampling_factor=0):
    """FUNCTION TO VALIDATE LENGTH

    Args:
        x (ndarray): numpy.ndarray with x.shape[0] = len_x
        y (ndarray): numpy.ndarray with y.shape[0] = len_y
        upsampling_factor (int): upsampling factor

    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x
    """
    if upsampling_factor == 0:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        mod_sample = x.shape[0] % upsampling_factor
        if mod_sample > 0:
            x = x[:-mod_sample]
        if x.shape[0] > y.shape[0] * upsampling_factor:
            x = x[:-(x.shape[0]-y.shape[0]*upsampling_factor)]
        elif x.shape[0] < y.shape[0] * upsampling_factor:
            y = y[:-((y.shape[0]*upsampling_factor-x.shape[0])//upsampling_factor)]
        assert len(x) == len(y) * upsampling_factor

    return x, y


class FeatureDatasetNeuVoco(Dataset):
    """Dataset for neural vocoder
    """

    def __init__(self, wav_list, feat_list, pad_wav_transform, pad_feat_transform, upsampling_factor,
                    string_path, pad_wav_f_transform=None, wav_transform=None, wav_transform_in=None, spcidx=False, string_path_ft=None,
                        wav_transform_out=None, with_excit=False, codeap_dim=None, n_bands=1, spk_list=None, cf_dim=None, magsp_flag=False,
                            pad_left=0, pad_right=0, wlat_flag=False, wspk_flag=False, worg_flag=False, worgx_flag=False, worgx_band_flag=False,
                                wrec_flag=True, wf0_flag=False, worgx_rec_flag=None, pad_wav_org_transform=None):
        self.wav_list = wav_list
        self.feat_list = feat_list
        self.pad_wav_transform = pad_wav_transform
        self.pad_feat_transform = pad_feat_transform
        self.upsampling_factor = upsampling_factor
        #self.string_path_org = '/feat_mceplf0cap'
        self.string_path_org = string_path
        #self.string_path_org = '/feat_org_lf0'
        self.wlat_flag = wlat_flag
        self.wspk_flag = wspk_flag
        self.wf0_flag = wf0_flag
        self.pad_wav_org_transform = pad_wav_org_transform
        self.magsp_flag = magsp_flag
        if string_path_ft is not None:
            self.string_path = string_path_ft
        else:
            self.string_path = string_path
        if self.wlat_flag or "rec" in self.string_path:
            self.string_path_lat = self.string_path+"_lat"
            if self.wspk_flag:
                self.string_path_spk = self.string_path+"_spk"
            if self.wf0_flag:
                self.string_path_f0 = self.string_path+"_f0"
            self.wlat_flag = True
        self.wrec_flag = wrec_flag
        self.worg_flag = worg_flag
        self.worgx_flag = worgx_flag
        self.worgx_band_flag = worgx_band_flag
        self.worgx_rec_flag = worgx_rec_flag
        self.pad_wav_f_transform = pad_wav_f_transform
        self.wav_transform = wav_transform
        self.wav_transform_in = wav_transform_in
        self.wav_transform_out = wav_transform_out
        self.with_excit = with_excit
        self.codeap_dim = codeap_dim
        if self.codeap_dim is not None:
            self.excit_dim = 2+1+self.codeap_dim
        else:
            self.excit_dim = 2
        self.n_bands = n_bands
        self.upsampling_factor_bands = self.upsampling_factor // self.n_bands
        self.spk_list = spk_list
        self.cf_dim = cf_dim
        self.spcidx = spcidx
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wavfile = self.wav_list[idx]
        featfile = self.feat_list[idx]
        x_org = None
        x_org_band = None
        
        if (self.spcidx and not check_hdf5(featfile, '/spcidx_range')) or (self.wlat_flag and self.worg_flag):
            file_org = os.path.join(os.path.dirname(os.path.dirname(featfile)), os.path.basename(os.path.dirname(featfile)).split("-")[0], os.path.basename(featfile))
        if self.spcidx:
            if not check_hdf5(featfile, '/spcidx_range'):
                spcidx = read_hdf5(file_org, '/spcidx_range')[0]
            else:
                spcidx = read_hdf5(featfile, '/spcidx_range')[0]
            if check_hdf5(featfile, self.string_path_org):
                frm_len = len(read_hdf5(featfile, '/f0_range'))
            else:
                frm_len = shape_hdf5(featfile, self.string_path)[0]

        if self.n_bands > 1:
            wavfile_pqmf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wavfile)))+"_pqmf_"+str(self.n_bands), \
                os.path.basename(os.path.dirname(os.path.dirname(wavfile))), os.path.basename(os.path.dirname(wavfile)))
            #wavfile_pqmf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wavfile)))+"_pqmf_fsb_"+str(self.n_bands), \
            if self.worgx_flag:
                x_org, _ = sf.read(wavfile, dtype=np.float32)
            elif self.worgx_rec_flag:
                wavfile_org = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wavfile)))+"_pqmf_"+str(self.n_bands)+"_rec", \
                    os.path.basename(os.path.dirname(os.path.dirname(wavfile))), os.path.basename(os.path.dirname(wavfile)), os.path.basename(wavfile))
                #wavfile_org = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wavfile)))+"_pqmf_fsb_"+str(self.n_bands)+"_rec", \
                x_org, _ = sf.read(wavfile_org, dtype=np.float32)
            for i in range(self.n_bands):
                if self.n_bands >= 10:
                    if i < self.n_bands - 1:
                        wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-0"+str(i+1)+".wav"))
                    else:
                        wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-"+str(i+1)+".wav"))
                else:
                    wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-"+str(i+1)+".wav"))
                x_pqmf, _ = sf.read(wavfile_pqmf, dtype=np.float32)
                if i > 0:
                    x_pqmf, _ = validate_length(x_pqmf, h, self.upsampling_factor_bands)
                    x = np.c_[x, np.expand_dims(x_pqmf,-1)]
                else:
                    if not self.with_excit:
                        if self.wrec_flag:
                            if check_hdf5(featfile, self.string_path):
                                h = read_hdf5(featfile, self.string_path)
                            else:
                                h = read_hdf5(featfile, self.string_path_org)
                        if self.wlat_flag:
                            if not self.wrec_flag:
                                h = read_hdf5(featfile, self.string_path_lat)
                            else:
                                h_lat = read_hdf5(featfile, self.string_path_lat)
                            if self.wspk_flag:
                                h_spk = read_hdf5(featfile, self.string_path_spk)
                            if self.wf0_flag:
                                h_f0 = read_hdf5(featfile, self.string_path_f0)
                            if self.worg_flag:
                                h_org = read_hdf5(file_org, self.string_path_org)
                                h_magsp_org = read_hdf5(file_org, '/magsp')
                    else:
                        h = np.c_[read_hdf5(featfile, self.string_path_org)[:,:self.excit_dim], read_hdf5(featfile, self.string_path)]
                    x_pqmf, h = validate_length(x_pqmf, h, self.upsampling_factor_bands)
                    if self.worgx_flag or self.worgx_rec_flag:
                        x_org, _ = validate_length(x_org, h, self.upsampling_factor)
                    if self.magsp_flag:
                        h_magsp = read_hdf5(featfile, '/magsp')
                        _, h_magsp = validate_length(x_pqmf, h_magsp, self.upsampling_factor_bands)
                    if self.wlat_flag:
                        if self.wrec_flag:
                            _, h_lat = validate_length(h, h_lat)
                        if self.wspk_flag:
                            _, h_spk = validate_length(h, h_spk)
                        if self.wf0_flag:
                            _, h_f0 = validate_length(h, h_f0)
                        if self.worg_flag:
                            _, h_org = validate_length(h, h_org)
                            _, h_magsp_org = validate_length(h_org, h_magsp_org)
                    x = np.expand_dims(x_pqmf,-1)

            if self.wav_transform_in is not None:
                x_t = self.wav_transform_in(x) # cont -> disc in/trg n_bands
            if self.wav_transform is not None:
                if self.wav_transform_out is not None:
                    x = self.wav_transform_out(self.wav_transform(x)) # cont -> disc -> cont trg n_bands
                    x_f, _ = sf.read(wavfile, dtype=np.float32)
                    x_f, _ = validate_length(x_f, h, self.upsampling_factor)
                    x_f = self.wav_transform_out(self.wav_transform(x_f)) # cont -> disc -> cont trg full
                    slen_f = x_f.shape[0]
                else:
                    if self.worgx_band_flag:
                        x_org_band = x
                    x = self.wav_transform(x) # cont -> disc in/trg
                    #if self.worgx_band_flag:
                    #    assert((x!=x_org_band).all())
            
            assert(x.shape[0]==h.shape[0]*self.upsampling_factor_bands)
            if self.worgx_flag or self.worgx_rec_flag:
                assert(x_org.shape[0]==h.shape[0]*self.upsampling_factor)
            if self.magsp_flag:
                assert(h.shape[0]==h_magsp.shape[0])
            if self.wlat_flag:
                if self.wrec_flag:
                    assert(h.shape[0]==h_lat.shape[0])
                if self.wspk_flag:
                    assert(h.shape[0]==h_spk.shape[0])
                if self.wf0_flag:
                    assert(h.shape[0]==h_f0.shape[0])
                if self.worg_flag:
                    assert(h.shape[0]==h_magsp_org.shape[0])
                    assert(h.shape[0]==h_org.shape[0])

            frm_len = h.shape[0]
            if self.spcidx:
                f_ss = spcidx[0]-self.pad_left
                idx_end = -1
                spcidx_end = spcidx[idx_end]
                while spcidx_end >= frm_len:
                    idx_end -= 1
                    spcidx_end = spcidx[idx_end]
                f_es = spcidx_end+self.pad_right
                if f_ss < 0:
                    f_ss = 0
                if f_es >= frm_len:
                    f_es = frm_len-1
                f_es += 1
                spcidx_s_e = [f_ss, f_es]
                spcidx_s_e_smpl = [f_ss*self.upsampling_factor_bands, f_es*self.upsampling_factor_bands]
                if self.worgx_flag or self.worgx_rec_flag:
                    spcidx_s_e_smpl_org = [f_ss*self.upsampling_factor, f_es*self.upsampling_factor]

                x = x[spcidx_s_e_smpl[0]:spcidx_s_e_smpl[-1]]
                h = h[spcidx_s_e[0]:spcidx_s_e[-1]]
                assert(x.shape[0]==h.shape[0]*self.upsampling_factor_bands)
                if self.worgx_flag or self.worgx_rec_flag:
                    x_org = x_org[spcidx_s_e_smpl_org[0]:spcidx_s_e_smpl_org[-1]]
                    assert(x_org.shape[0]==h.shape[0]*self.upsampling_factor)
                if self.worgx_band_flag:
                    x_org_band = x_org_band[spcidx_s_e_smpl[0]:spcidx_s_e_smpl[-1]]
                    assert(x_org_band.shape[0]==x.shape[0])
                if self.magsp_flag:
                    h_magsp = h_magsp[spcidx_s_e[0]:spcidx_s_e[-1]]
                    assert(h.shape[0]==h_magsp.shape[0])
                if self.wlat_flag:
                    if self.wrec_flag:
                        h_lat = h_lat[spcidx_s_e[0]:spcidx_s_e[-1]]
                        assert(h.shape[0]==h_lat.shape[0])
                    if self.wspk_flag:
                        h_spk = h_spk[spcidx_s_e[0]:spcidx_s_e[-1]]
                        assert(h.shape[0]==h_spk.shape[0])
                    if self.wf0_flag:
                        h_f0 = h_f0[spcidx_s_e[0]:spcidx_s_e[-1]]
                        assert(h.shape[0]==h_f0.shape[0])
                    if self.worg_flag:
                        h_org = h_org[spcidx_s_e[0]:spcidx_s_e[-1]]
                        h_magsp_org = h_magsp_org[spcidx_s_e[0]:spcidx_s_e[-1]]
                        assert(h.shape[0]==h_magsp_org.shape[0])
                        assert(h.shape[0]==h_org.shape[0])
            slen = x.shape[0]
            flen = h.shape[0]

            h = torch.FloatTensor(self.pad_feat_transform(h))
            if self.magsp_flag:
                h_magsp = torch.FloatTensor(self.pad_feat_transform(h_magsp))
            if self.wlat_flag:
                if self.wrec_flag:
                    h_lat = torch.FloatTensor(self.pad_feat_transform(h_lat))
                if self.wspk_flag:
                    h_spk = torch.FloatTensor(self.pad_feat_transform(h_spk))
                if self.wf0_flag:
                    h_f0 = torch.FloatTensor(self.pad_feat_transform(h_f0))
                if self.worg_flag:
                    h_org = torch.FloatTensor(self.pad_feat_transform(h_org))
                    h_magsp_org = torch.FloatTensor(self.pad_feat_transform(h_magsp_org))
            if self.wav_transform is not None and self.wav_transform_out is None:
                x = torch.LongTensor(self.pad_wav_transform(x)) # disc in/trg
                if self.worgx_flag or self.worgx_rec_flag:
                    x_org = torch.FloatTensor(self.pad_wav_org_transform(x_org)) # disc in/trg
                if self.worgx_band_flag:
                    x_org_band = torch.FloatTensor(self.pad_wav_transform(x_org_band)) # disc in/trg
            else:
                x = torch.FloatTensor(self.pad_wav_transform(x)) # cont trg_n_bands
                if self.wav_transform is not None and self.wav_transform_out is not None:
                    x_f = torch.FloatTensor(self.pad_wav_f_transform(x_f)) # cont trg_full

            if self.spk_list is not None:
                #featfile_spk = os.path.basename(os.path.dirname(featfile)).split("-")[0]
                #spk_idx = (torch.ones(h.shape[0])*self.spk_list.index(featfile_spk)).long()
                idx_spk = self.spk_list.index(os.path.basename(os.path.dirname(featfile)))
                spk_code = torch.LongTensor(self.pad_feat_transform(np.ones(flen)*idx_spk))

            if self.wav_transform_in is not None: # laplace-disc
                x_t = torch.LongTensor(self.pad_wav_transform(x_t)) # disc in/trg_n_bands
                if self.spk_list is not None:
                    if self.cf_dim is not None and self.wav_transform is not None and self.wav_transform_out is None:
                        return {'x_t_c': x_t // self.cf_dim, 'x_t_f': x_t % self.cf_dim, 'x': x, 'x': x_f, 'feat': h, \
                                    'slen': slen, 'slen_f': slen_f, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                    else:
                        return {'x_t': x_t, 'x': x, 'x_f': x_f, 'feat': h, 'slen': slen, 'slen_f': slen_f, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                else:
                    if self.cf_dim is not None and self.wav_transform is not None and self.wav_transform_out is None:
                        return {'x_t_c': x_t // self.cf_dim, 'x_t_f': x_t % self.cf_dim, 'x': x, 'x_f': x_f, 'feat': h, \
                                    'slen': slen, 'slen_f': slen_f, 'flen': flen, 'featfile': featfile}
                    else:
                        return {'x_t': x_t, 'x': x, 'x_f': x_f, 'feat': h, 'slen': slen, 'slen_f': slen_f, 'flen': flen, 'featfile': featfile}
            else: # disc
                if self.spk_list is not None:
                    if self.cf_dim is not None and self.wav_transform is not None and self.wav_transform_out is None:
                        if self.worgx_flag or self.worgx_rec_flag:
                            if self.worgx_band_flag:
                                if not self.magsp_flag:
                                    return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                                else:
                                    return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'feat_magsp': h_magsp, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                            else:
                                return {'x_org': x_org, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                        else:
                            return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                    else:
                        return {'x': x, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                else:
                    if self.cf_dim is not None and self.wav_transform is not None and self.wav_transform_out is None:
                        if self.wlat_flag:
                            if self.worg_flag:
                                if self.wspk_flag:
                                    if not self.wrec_flag:
                                        if not self.wf0_flag:
                                            return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, \
                                                'feat': h, 'feat_org': h_org, 'feat_magsp_org': h_magsp_org, 'spk': h_spk, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                        else:
                                            return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, \
                                                'feat': h, 'feat_org': h_org, 'feat_magsp_org': h_magsp_org, 'f0': h_f0, 'spk': h_spk, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                    else:
                                        if not self.wf0_flag:
                                            return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'lat': h_lat,
                                                'feat': h, 'feat_org': h_org, 'feat_magsp_org': h_magsp_org, 'spk': h_spk, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                        else:
                                            return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'lat': h_lat,
                                                'feat': h, 'feat_org': h_org, 'feat_magsp_org': h_magsp_org, 'f0': h_f0, 'spk': h_spk, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                else:
                                    return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'feat_org': h_org, 'feat_magsp_org': h_magsp_org, 'lat': h_lat, 'slen': slen, 'flen': flen, 'featfile': featfile}
                            else:
                                if self.wspk_flag:
                                    return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'lat': h_lat, 'spk': h_spk, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                else:
                                    return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'lat': h_lat, 'slen': slen, 'flen': flen, 'featfile': featfile}
                        else:
                            if self.worgx_flag or self.worgx_rec_flag:
                                if self.worgx_band_flag:
                                    if not self.magsp_flag:
                                        return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                    else:
                                        return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'feat_magsp': h_magsp, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                else:
                                    return {'x_org': x_org, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile}
                            else:
                                return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile}
                    else:
                        if self.wlat_flag:
                            if self.worg_flag:
                                if self.wspk_flag:
                                    return {'x': x, 'feat': h, 'feat_org': h_org, 'feat_magsp_org': h_magsp_org, 'lat': h_lat, 'spk': h_spk, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                else:
                                    return {'x': x, 'feat': h, 'feat_org': h_org, 'feat_magsp_org': h_magsp_org, 'lat': h_lat, 'slen': slen, 'flen': flen, 'featfile': featfile}
                            else:
                                if self.wspk_flag:
                                    return {'x': x, 'feat': h, 'lat': h_lat, 'spk': h_spk, 'slen': slen, 'flen': flen, 'featfile': featfile}
                                else:
                                    return {'x': x, 'feat': h, 'lat': h_lat, 'slen': slen, 'flen': flen, 'featfile': featfile}
                        else:
                            return {'x': x, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile}
        else:
            x, _ = sf.read(wavfile, dtype=np.float32)
            if not self.with_excit:
                if check_hdf5(featfile, self.string_path):
                    h = read_hdf5(featfile, self.string_path)
                else:
                    h = read_hdf5(featfile, self.string_path_org)
            else:
                h = np.c_[read_hdf5(featfile, self.string_path_org)[:,:self.excit_dim], read_hdf5(featfile, self.string_path)]

            x, h = validate_length(x, h, self.upsampling_factor)

            if self.wav_transform_in is not None:
                x_t = self.wav_transform_in(x)
            if self.wav_transform is not None:
                if self.wav_transform_out is not None:
                    x = self.wav_transform_out(self.wav_transform(x))
                else:
                    x = self.wav_transform(x)
            
            slen = x.shape[0]
            flen = h.shape[0]

            h = torch.FloatTensor(self.pad_feat_transform(h))
            if self.wav_transform is not None and self.wav_transform_out is None:
                x = torch.LongTensor(self.pad_wav_transform(x))
            else:
                x = torch.FloatTensor(self.pad_wav_transform(x))

            if self.spk_list is not None:
                featfile_spk = os.path.basename(os.path.dirname(featfile))
                spk_code = (torch.ones(h.shape[0])*self.spk_list.index(featfile_spk)).long()

            if self.wav_transform_in is not None:
                x_t = torch.LongTensor(self.pad_wav_transform(x_t))
                if self.spk_list is not None:
                    if self.cf_dim is not None and self.wav_transform is not None and self.wav_transform_out is None:
                        return {'x_t_c': x_t // self.cf_dim, 'x_t_f': x_t % self.cf_dim, 'x': x, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                    else:
                        return {'x_t': x_t, 'x': x, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                else:
                    if self.cf_dim is not None and self.wav_transform is not None and self.wav_transform_out is None:
                        return {'x_t_c': x_t // self.cf_dim, 'x_t_f': x_t % self.cf_dim, 'x': x, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile}
                    else:
                        return {'x_t': x_t, 'x': x, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile}
            else:
                if self.spk_list is not None:
                    if self.cf_dim is not None and self.wav_transform is not None and self.wav_transform_out is None:
                        return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                    else:
                        return {'x': x, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile, 'c': spk_code}
                else:
                    if self.cf_dim is not None and self.wav_transform is not None and self.wav_transform_out is None:
                        return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile}
                    else:
                        return {'x': x, 'feat': h, 'slen': slen, 'flen': flen, 'featfile': featfile}


def proc_random_spkcv_statcvexcit(src_idx, spk_list, n_cv, n_frm, n_spk, stat_spk_list, mean_path, scale_path, excit_flag=True):
    if excit_flag:
        mean_trg_list = [None]*n_cv
        std_trg_list = [None]*n_cv
        trg_code_list = [None]*n_cv
        pair_spk_list = [None]*n_cv
        for i in range(n_cv):
            pair_idx = np.random.randint(0,n_spk)
            while pair_idx == src_idx:
                pair_idx = np.random.randint(0,n_spk)
            trg_code_list[i] = np.ones(n_frm, dtype=np.int64)*pair_idx
            mean_trg_list[i] = read_hdf5(stat_spk_list[pair_idx], mean_path)[1:2]
            std_trg_list[i] = read_hdf5(stat_spk_list[pair_idx], scale_path)[1:2]
            pair_spk_list[i] = spk_list[pair_idx]

        return mean_trg_list, std_trg_list, trg_code_list, pair_spk_list
    else:
        trg_code_list = [None]*n_cv
        pair_spk_list = [None]*n_cv
        for i in range(n_cv):
            pair_idx = np.random.randint(0,n_spk)
            while pair_idx == src_idx:
                pair_idx = np.random.randint(0,n_spk)
            trg_code_list[i] = np.ones(n_frm, dtype=np.int64)*pair_idx
            pair_spk_list[i] = spk_list[pair_idx]

        return trg_code_list, pair_spk_list


class FeatureDatasetCycMceplf0WavVAE(Dataset):
    """Dataset for cyclic mceplf0-waveform VAE-based VC
    """

    def __init__(self, feat_list, pad_feat_transform, spk_list, stat_spk_list, n_cyc, string_path, excit_dim=None, cap_exc_dim=None,
            upsampling_factor=None, wav_list=None, pad_wav_transform=None, wav_transform=None, spcidx=True, uvcap_flag=True,
                n_bands=1, cf_dim=None, pad_left=0, pad_right=0, magsp=False, worgx_flag=False, pad_wav_org_transform=None):
        self.wav_list = wav_list
        self.feat_list = feat_list
        self.pad_wav_transform = pad_wav_transform
        self.pad_feat_transform = pad_feat_transform
        self.wav_transform = wav_transform
        self.upsampling_factor = upsampling_factor
        self.stat_spk_list = stat_spk_list
        self.spk_list = spk_list
        self.n_cyc = np.max((n_cyc, 2))
        self.n_cv = int(self.n_cyc/2 + self.n_cyc%2)
        self.n_spk = len(self.spk_list)
        self.string_path = string_path
        self.excit_dim = excit_dim
        self.cap_exc_dim = cap_exc_dim #to exclude cap
        self.spcidx = spcidx
        self.magsp = magsp
        self.worgx_flag = worgx_flag
        self.pad_wav_org_transform = pad_wav_org_transform
        if 'mel' in self.string_path:
            self.mean_path = "/mean_feat_mceplf0cap"
            self.scale_path = "/scale_feat_mceplf0cap"
            self.uvcap = False
            self.mel = True
        else:
            self.mean_path = "/mean_"+self.string_path.replace("/","")
            self.scale_path = "/scale_"+self.string_path.replace("/","")
            self.uvcap_flag = uvcap_flag
            if self.string_path == '/feat_org_lf0' and self.uvcap_flag:
                self.uvcap = True
            else:
                self.uvcap = False
            self.mel = False
        self.n_bands = n_bands
        self.cf_dim = cf_dim
        if self.upsampling_factor is not None:
            self.upsampling_factor_bands = self.upsampling_factor // self.n_bands
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        featfile = self.feat_list[idx]
        if self.mel:
            if self.excit_dim is not None:
                feat = np.c_[read_hdf5(featfile, '/feat_mceplf0cap')[:,:self.excit_dim], read_hdf5(featfile, self.string_path)]
            else:
                feat = read_hdf5(featfile, self.string_path)
            if self.magsp:
                feat_magsp = read_hdf5(featfile, '/magsp')
        else:
            if self.cap_exc_dim is None:
                feat = read_hdf5(featfile, self.string_path)
            else:
                feat = np.c_[read_hdf5(featfile, '/feat_mceplf0cap')[:,:2], read_hdf5(featfile, '/feat_mceplf0cap')[:,self.cap_exc_dim:]]
        frm_len = len(read_hdf5(featfile, '/f0_range'))
        featfile_spk = os.path.basename(os.path.dirname(featfile))
        src_idx = self.spk_list.index(featfile_spk)

        if self.wav_list is not None:
            wavfile = self.wav_list[idx]            
            if self.n_bands > 1:
                wavfile_pqmf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wavfile)))+"_pqmf_"+str(self.n_bands), \
                    os.path.basename(os.path.dirname(os.path.dirname(wavfile))), os.path.basename(os.path.dirname(wavfile)))
                if self.worgx_flag:
                    x_org, _ = sf.read(wavfile, dtype=np.float32)
                for i in range(self.n_bands):
                    if self.n_bands >= 10:
                        if i < self.n_bands - 1:
                            wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-0"+str(i+1)+".wav"))
                        else:
                            wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-"+str(i+1)+".wav"))
                    else:
                        wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-"+str(i+1)+".wav"))
                    x_pqmf, _ = sf.read(wavfile_pqmf, dtype=np.float32)
                    if i > 0:
                        x_pqmf, _ = validate_length(x_pqmf, feat, self.upsampling_factor_bands)
                        x = np.c_[x, np.expand_dims(x_pqmf,-1)]
                    else:
                        x_pqmf, feat = validate_length(x_pqmf, feat, self.upsampling_factor_bands)
                        x = np.expand_dims(x_pqmf,-1)
                        if self.worgx_flag:
                            x_org, _ = validate_length(x_org, feat, self.upsampling_factor)
                if self.worgx_flag:
                    assert(x_org.shape[0]==feat.shape[0]*self.upsampling_factor)
                    x_org_band = x
                x = self.wav_transform(x)
                assert(x.shape[0]==feat.shape[0]*self.upsampling_factor_bands)
                frm_len = feat.shape[0]
                if self.spcidx:
                    spcidx = read_hdf5(featfile, '/spcidx_range')[0]
                    f_ss = spcidx[0]-self.pad_left
                    idx_end = -1
                    spcidx_end = spcidx[idx_end]
                    while spcidx_end >= frm_len:
                        idx_end -= 1
                        spcidx_end = spcidx[idx_end]
                    f_es = spcidx_end+self.pad_right
                    if f_ss < 0:
                        f_ss = 0
                    if f_es >= frm_len:
                        f_es = frm_len-1
                    f_es += 1
                    spcidx_s_e = [f_ss, f_es]
                    spcidx_s_e_smpl = [f_ss*self.upsampling_factor_bands, f_es*self.upsampling_factor_bands]
                    x = x[spcidx_s_e_smpl[0]:spcidx_s_e_smpl[-1]]
                    feat = feat[spcidx_s_e[0]:spcidx_s_e[-1]]
                    assert(x.shape[0]==feat.shape[0]*self.upsampling_factor_bands)
                    if self.worgx_flag:
                        spcidx_s_e_smpl_org = [f_ss*self.upsampling_factor, f_es*self.upsampling_factor]
                        x_org = x_org[spcidx_s_e_smpl_org[0]:spcidx_s_e_smpl_org[-1]]
                        assert(x_org.shape[0]==feat.shape[0]*self.upsampling_factor)
                        x_org_band = x_org_band[spcidx_s_e_smpl[0]:spcidx_s_e_smpl[-1]]
                        assert(x_org_band.shape[0]==x.shape[0])
                    if self.magsp and self.mel:
                        feat_magsp = feat_magsp[spcidx_s_e[0]:spcidx_s_e[-1]]
                        assert(x.shape[0]==feat_magsp.shape[0]*self.upsampling_factor_bands)
            else:
                x, _ = sf.read(wavfile, dtype=np.float32)
                x, feat = validate_length(x, feat, self.upsampling_factor)
                assert(x.shape[0]==feat.shape[0]*self.upsampling_factor)
                frm_len = feat.shape[0]
                if self.spcidx:
                    spcidx = read_hdf5(featfile, '/spcidx_range')[0]
                    f_ss = spcidx[0]-self.pad_left
                    idx_end = -1
                    spcidx_end = spcidx[idx_end]
                    while spcidx_end >= frm_len:
                        idx_end -= 1
                        spcidx_end = spcidx[idx_end]
                    f_es = spcidx_end+self.pad_right
                    if f_ss < 0:
                        f_ss = 0
                    if f_es >= frm_len:
                        f_es = frm_len-1
                    f_es += 1
                    spcidx_s_e = [f_ss, f_es]
                    spcidx_s_e_smpl = [f_ss*self.upsampling_factor, f_es*self.upsampling_factor]
                    x = x[spcidx_s_e_smpl[0]:spcidx_s_e_smpl[-1]]
                    feat = feat[spcidx_s_e[0]:spcidx_s_e[-1]]
                    assert(x.shape[0]==feat.shape[0]*self.upsampling_factor)
                    if self.magsp and self.mel:
                        feat_magsp = feat_magsp[spcidx_s_e[0]:spcidx_s_e[-1]]
                        assert(x.shape[0]==feat_magsp.shape[0]*self.upsampling_factor)
                x = self.wav_transform(x)
            slen = x.shape[0]
        elif self.spcidx:
            spcidx = read_hdf5(featfile, '/spcidx_range')[0]
            f_ss = spcidx[0]-self.pad_left
            spcidx_end = spcidx[-1]
            f_es = spcidx_end+self.pad_right
            if f_ss < 0:
                f_ss = 0
            if f_es >= frm_len:
                f_es = frm_len-1
            f_es += 1
            spcidx_s_e = [f_ss, f_es]
            feat = feat[spcidx_s_e[0]:spcidx_s_e[-1]]
            if self.magsp and self.mel:
                feat_magsp = feat_magsp[spcidx_s_e[0]:spcidx_s_e[-1]]
        flen = feat.shape[0]

        if not self.mel or (self.mel and self.excit_dim is not None):
            mean_trg_list, std_trg_list, trg_code_list, pair_spk_list = \
                proc_random_spkcv_statcvexcit(src_idx, self.spk_list, self.n_cv, flen, self.n_spk, \
                    self.stat_spk_list, self.mean_path, self.scale_path)
            mean_src = read_hdf5(self.stat_spk_list[src_idx], self.mean_path)[1:2]
            std_src = read_hdf5(self.stat_spk_list[src_idx], self.scale_path)[1:2]

            cv_src_list = [None]*self.n_cv
            if self.excit_dim is not None:
                if self.cap_exc_dim is None:
                    for i in range(self.n_cv):
                        cv_src_list[i] = torch.FloatTensor(self.pad_feat_transform(np.c_[feat[:,:1], \
                                            (std_trg_list[i]/std_src)*(feat[:,1:2]-mean_src)+mean_trg_list[i], feat[:,2:self.excit_dim]]))
                else:
                    for i in range(self.n_cv):
                        cv_src_list[i] = torch.FloatTensor(self.pad_feat_transform(np.c_[feat[:,:1], \
                                            (std_trg_list[i]/std_src)*(feat[:,1:2]-mean_src)+mean_trg_list[i]]))
            else:
                for i in range(self.n_cv):
                    cv_src_list[i] = torch.FloatTensor(self.pad_feat_transform(np.c_[feat[:,:1], \
                                        (std_trg_list[i]/std_src)*(feat[:,1:2]-mean_src)+mean_trg_list[i]]))
        else:
            trg_code_list, pair_spk_list = \
                proc_random_spkcv_statcvexcit(src_idx, self.spk_list, self.n_cv, flen, self.n_spk, \
                    self.stat_spk_list, self.mean_path, self.scale_path, False)

        for i in range(self.n_cv):
            trg_code_list[i] = torch.LongTensor(self.pad_feat_transform(trg_code_list[i]))

        if self.uvcap:
            if self.spcidx:
                if self.wav_list is not None:
                    uvcap = read_hdf5(featfile, '/feat_mceplf0cap')[:spcidx_end+1,2:3]
                else:
                    uvcap = read_hdf5(featfile, '/feat_mceplf0cap')[spcidx[0]:spcidx_end+1,2:3]
            else:
                uvcap = read_hdf5(featfile, '/feat_mceplf0cap')[:,2:3]
            feat = torch.FloatTensor(self.pad_feat_transform(np.c_[feat,uvcap]))
        else:
            feat = torch.FloatTensor(self.pad_feat_transform(feat))
        if self.magsp and self.mel:
            feat_magsp = torch.FloatTensor(self.pad_feat_transform(feat_magsp))
        src_codes = torch.LongTensor(self.pad_feat_transform(np.ones(flen, dtype=np.int64)*src_idx))

        if not self.mel or (self.mel and self.excit_dim is not None):
            if not self.magsp or not self.mel:
                if self.wav_list is None:
                    return {'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                            'pair_spk_list': pair_spk_list, 'feat_cv_list': cv_src_list, 'featfile_spk': featfile_spk, \
                                'featfile': featfile, 'feat': feat}
                else:
                    x = torch.LongTensor(self.pad_wav_transform(x))
                    if self.cf_dim is not None:
                        return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen': slen, 'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                                'pair_spk_list': pair_spk_list, 'feat_cv_list': cv_src_list, 'featfile_spk': featfile_spk, \
                                    'featfile': featfile, 'feat': feat}
                    else:
                        return {'x': x, 'slen': slen, 'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                                'pair_spk_list': pair_spk_list, 'feat_cv_list': cv_src_list, 'featfile_spk': featfile_spk, \
                                    'featfile': featfile, 'feat': feat}
            else:
                if self.wav_list is None:
                    return {'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                            'pair_spk_list': pair_spk_list, 'feat_cv_list': cv_src_list, 'featfile_spk': featfile_spk, \
                                'featfile': featfile, 'feat': feat, 'feat_magsp': feat_magsp}
                else:
                    x = torch.LongTensor(self.pad_wav_transform(x))
                    if self.worgx_flag:
                        x_org = torch.FloatTensor(self.pad_wav_org_transform(x_org)) # disc in/trg
                        x_org_band = torch.FloatTensor(self.pad_wav_transform(x_org_band)) # disc in/trg
                    else:
                        x_org = None
                        x_org_band = None
                    if self.cf_dim is not None:
                        return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen': slen, 'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                                'pair_spk_list': pair_spk_list, 'feat_cv_list': cv_src_list, 'featfile_spk': featfile_spk, \
                                    'featfile': featfile, 'feat': feat, 'feat_magsp': feat_magsp}
                    else:
                        return {'x_org': x_org, 'x_org_band': x_org_band, 'x': x, 'slen': slen, 'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                                'pair_spk_list': pair_spk_list, 'feat_cv_list': cv_src_list, 'featfile_spk': featfile_spk, \
                                    'featfile': featfile, 'feat': feat, 'feat_magsp': feat_magsp}
        else:
            if not self.magsp:
                if self.wav_list is None:
                    return {'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                            'pair_spk_list': pair_spk_list, 'featfile_spk': featfile_spk, \
                                'featfile': featfile, 'feat': feat}
                else:
                    x = torch.LongTensor(self.pad_wav_transform(x))
                    if self.cf_dim is not None:
                        return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen': slen, 'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                                'pair_spk_list': pair_spk_list, 'featfile_spk': featfile_spk, \
                                    'featfile': featfile, 'feat': feat}
                    else:
                        return {'x': x, 'slen': slen, 'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                                'pair_spk_list': pair_spk_list, 'featfile_spk': featfile_spk, \
                                    'featfile': featfile, 'feat': feat}
            else:
                if self.wav_list is None:
                    return {'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                            'pair_spk_list': pair_spk_list, 'featfile_spk': featfile_spk, \
                                'featfile': featfile, 'feat': feat, 'feat_magsp': feat_magsp}
                else:
                    x = torch.LongTensor(self.pad_wav_transform(x))
                    if self.worgx_flag:
                        x_org = torch.FloatTensor(self.pad_wav_org_transform(x_org)) # disc in/trg
                        x_org_band = torch.FloatTensor(self.pad_wav_transform(x_org_band)) # disc in/trg
                    else:
                        x_org = None
                        x_org_band = None
                    x = torch.LongTensor(self.pad_wav_transform(x))
                    if self.cf_dim is not None:
                        return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen': slen, 'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                                'pair_spk_list': pair_spk_list, 'featfile_spk': featfile_spk, \
                                    'featfile': featfile, 'feat': feat, 'feat_magsp': feat_magsp}
                    else:
                        return {'x_org': x_org, 'x_org_band': x_org_band, 'x': x, 'slen': slen, 'flen': flen, 'src_codes': src_codes, 'src_trg_codes_list': trg_code_list, \
                                'pair_spk_list': pair_spk_list, 'featfile_spk': featfile_spk, \
                                    'featfile': featfile, 'feat': feat, 'feat_magsp': feat_magsp}


class FeatureDatasetEvalCycMceplf0WavVAE(Dataset):
    """Dataset for evaluation cyclic mceplf0-waveform VAE-based VC
    """

    def __init__(self, file_list, pad_transform, spk_list, stat_spk_list, string_path, excit_dim=None, cap_exc_dim=None,
            upsampling_factor=None, wav_list=None, pad_wav_transform=None, wav_transform=None, spcidx=True, worgx_flag=False,
                uvcap_flag=True, n_bands=1, cf_dim=None, pad_left=0, pad_right=0, magsp=False, pad_wav_org_transform=None, n_spk_data=None):
        self.wav_list = wav_list
        self.file_list = file_list
        self.pad_transform = pad_transform
        self.wav_transform = wav_transform
        self.pad_wav_transform = pad_wav_transform
        self.upsampling_factor = upsampling_factor
        self.spk_list = spk_list
        self.stat_spk_list = stat_spk_list
        self.n_spk = len(self.spk_list)
        if n_spk_data is not None:
            self.n_spk_data = n_spk_data
        else:
            self.n_spk_data = self.n_spk
        self.wav_list_src = []
        self.file_list_src = []
        self.file_list_src_trg = []
        self.list_src_trg_flag = []
        self.excit_dim = excit_dim
        self.cap_exc_dim = cap_exc_dim
        self.string_path = string_path
        self.spcidx = spcidx
        eval_exist = False
        self.uvcap_flag = uvcap_flag
        self.magsp = magsp
        self.n_bands = n_bands
        self.worgx_flag = worgx_flag
        self.pad_wav_org_transform = pad_wav_org_transform
        if self.upsampling_factor is not None:
            self.upsampling_factor_bands = self.upsampling_factor // self.n_bands
        self.cf_dim = cf_dim
        self.pad_left = 0
        self.pad_right = 0
        if 'mel' in self.string_path:
            self.mean_path = "/mean_feat_mceplf0cap"
            self.scale_path = "/scale_feat_mceplf0cap"
            self.uvcap = False
            self.mel = True
        else:
            self.mean_path = "/mean_"+self.string_path.replace("/","")
            self.scale_path = "/scale_"+self.string_path.replace("/","")
            self.uvcap_flag = uvcap_flag
            if self.string_path == '/feat_org_lf0' and self.uvcap_flag:
                self.uvcap = True
            else:
                self.uvcap = False
            self.mel = False
        for i in range(self.n_spk_data):
            if '.' not in spk_list[i] and spk_list[i].find('p') != 0 and len(self.file_list[i]) > 0:
                eval_exist = True
                break
        if eval_exist:
            # deterministically select a conv. pair for each validation utterance,
            # and deal with existence of pair data
            n_pair = self.n_spk_data // 2 
            #n_src = n_pair + self.n_spk % 2
            for spk_src_idx_iter in range(self.n_spk_data):
                #if '.' not in spk_list[spk_src_idx] and spk_list[spk_src_idx].find('p') != 0: 
                #spk_src = self.spk_list[spk_src_idx]
                spk_src_n_utt = len(self.file_list[spk_src_idx_iter])
                if spk_src_n_utt > 0:
                    spk_src = os.path.basename(os.path.dirname(self.file_list[spk_src_idx_iter][0]))
                    spk_trg_idx_iter_start = spk_src_idx_iter + n_pair
                    if spk_trg_idx_iter_start >= self.n_spk_data:
                        spk_trg_idx_iter_start -= self.n_spk_data
                    #while '.' in spk_list[spk_trg_idx_start] or spk_list[spk_trg_idx_start].find('p') == 0:
                    #    spk_trg_idx_start += 1
                    flag = False
                    for spk_trg_idx_iter in range(spk_trg_idx_iter_start,self.n_spk_data):
                        #if '.' not in spk_list[spk_trg_idx] and spk_list[spk_trg_idx].find('p') != 0:
                        if spk_trg_idx_iter != spk_src_idx_iter:
                            #spk_trg = self.spk_list[spk_trg_idx]
                            if len(self.file_list[spk_trg_idx_iter]) > 0:
                                spk_trg = os.path.basename(os.path.dirname(self.file_list[spk_trg_idx_iter][0]))
                                for i in range(spk_src_n_utt):
                                    file_src = self.file_list[spk_src_idx_iter][i]
                                    if self.wav_list is not None:
                                        wav_src = self.wav_list[spk_src_idx_iter][i]
                                    file_trg = os.path.dirname(os.path.dirname(file_src))+"/"+spk_trg+"/"+\
                                                    os.path.basename(file_src)
                                    if (file_trg in self.file_list) or os.path.exists(file_trg):
                                        self.file_list_src.append(file_src)
                                        if self.wav_list is not None:
                                            self.wav_list_src.append(wav_src)
                                        self.file_list_src_trg.append(file_trg)
                                        flag = True
                                        self.list_src_trg_flag.append(flag)
                                    elif flag:
                                        self.file_list_src.append(file_src)
                                        if self.wav_list is not None:
                                            self.wav_list_src.append(wav_src)
                                        self.file_list_src_trg.append(file_trg)
                                        self.list_src_trg_flag.append(False)
                            if flag:
                                break
                    if not flag:
                        for spk_trg_idx_iter in range(spk_trg_idx_iter_start):
                            #if '.' not in spk_list[spk_trg_idx] and spk_list[spk_trg_idx].find('p') != 0:
                            if spk_trg_idx_iter != spk_src_idx_iter:
                                #spk_trg = self.spk_list[spk_trg_idx]
                                if len(self.file_list[spk_trg_idx_iter]) > 0:
                                    spk_trg = os.path.basename(os.path.dirname(self.file_list[spk_trg_idx_iter][0]))
                                    for i in range(spk_src_n_utt):
                                        file_src = self.file_list[spk_src_idx_iter][i]
                                        if self.wav_list is not None:
                                            wav_src = self.wav_list[spk_src_idx_iter][i]
                                        file_trg = os.path.dirname(os.path.dirname(file_src))+"/"+spk_trg+\
                                                                    "/"+os.path.basename(file_src)
                                        if (file_trg in self.file_list) or os.path.exists(file_trg):
                                            self.file_list_src.append(file_src)
                                            if self.wav_list is not None:
                                                self.wav_list_src.append(wav_src)
                                            self.file_list_src_trg.append(file_trg)
                                            flag = True
                                            self.list_src_trg_flag.append(flag)
                                        elif flag:
                                            self.file_list_src.append(file_src)
                                            if self.wav_list is not None:
                                                self.wav_list_src.append(wav_src)
                                            self.file_list_src_trg.append(file_trg)
                                            self.list_src_trg_flag.append(False)
                                if flag:
                                    break
                        if not flag:
                            spk_trg = self.spk_list[spk_trg_idx_iter_start]
                            for i in range(spk_src_n_utt):
                                file_src = self.file_list[spk_src_idx_iter][i]
                                if self.wav_list is not None:
                                    wav_src = self.wav_list[spk_src_idx_iter][i]
                                file_trg = os.path.dirname(os.path.dirname(file_src))+"/"+spk_trg+\
                                                            "/"+os.path.basename(file_src)
                                self.file_list_src.append(file_src)
                                if self.wav_list is not None:
                                    self.wav_list_src.append(wav_src)
                                self.file_list_src_trg.append(file_trg)
                                #self.file_list_src_trg.append(file_src)
                                self.list_src_trg_flag.append(False)

    def __len__(self):
        return len(self.file_list_src)

    def __getitem__(self, idx):
        featfile_src = self.file_list_src[idx]
        featfile_src_trg = self.file_list_src_trg[idx]
        file_src_trg_flag = self.list_src_trg_flag[idx]

        if self.mel:
            if self.excit_dim is not None:
                h_src = np.c_[read_hdf5(featfile_src, '/feat_mceplf0cap')[:,:self.excit_dim], read_hdf5(featfile_src, self.string_path)]
            else:
                h_src = read_hdf5(featfile_src, self.string_path)
            if self.magsp:
                h_src_magsp = read_hdf5(featfile_src, '/magsp')
        else:
            if self.cap_exc_dim is None:
                h_src = read_hdf5(featfile_src, self.string_path)
            else:
                h_src = np.c_[read_hdf5(featfile_src, '/feat_mceplf0cap')[:,:2], read_hdf5(featfile_src, '/feat_mceplf0cap')[:,self.cap_exc_dim:]]
        spk_src = os.path.basename(os.path.dirname(featfile_src))
        spk_trg = os.path.basename(os.path.dirname(featfile_src_trg))
        idx_src = self.spk_list.index(spk_src)
        idx_trg = self.spk_list.index(spk_trg)

        spcidx_src = read_hdf5(featfile_src, '/spcidx_range')[0]
        frm_len = len(read_hdf5(featfile_src, '/f0_range'))
        if self.wav_list is not None:
            wavfile = self.wav_list_src[idx]            
            if self.n_bands > 1:
                wavfile_pqmf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wavfile)))+"_pqmf_"+str(self.n_bands), \
                    os.path.basename(os.path.dirname(os.path.dirname(wavfile))), os.path.basename(os.path.dirname(wavfile)))
                if self.worgx_flag:
                    x_org, _ = sf.read(wavfile, dtype=np.float32)
                for i in range(self.n_bands):
                    if self.n_bands >= 10:
                        if i < self.n_bands - 1:
                            wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-0"+str(i+1)+".wav"))
                        else:
                            wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-"+str(i+1)+".wav"))
                    else:
                        wavfile_pqmf = os.path.join(wavfile_pqmf_dir, os.path.basename(wavfile).replace(".wav", "_B-"+str(i+1)+".wav"))
                    x_pqmf, _ = sf.read(wavfile_pqmf, dtype=np.float32)
                    if i > 0:
                        x_pqmf, _ = validate_length(x_pqmf, h_src, self.upsampling_factor_bands)
                        x = np.c_[x, np.expand_dims(x_pqmf,-1)]
                    else:
                        x_pqmf, h_src = validate_length(x_pqmf, h_src, self.upsampling_factor_bands)
                        x = np.expand_dims(x_pqmf,-1)
                        if self.worgx_flag:
                            x_org, _ = validate_length(x_org, h_src, self.upsampling_factor)
                if self.worgx_flag:
                    assert(x_org.shape[0]==h_src.shape[0]*self.upsampling_factor)
                    x_org_band = x
                x = self.wav_transform(x)
                assert(x.shape[0]==h_src.shape[0]*self.upsampling_factor_bands)
                frm_len = h_src.shape[0]
                if self.spcidx:
                    h_src_full = h_src
                    flen_src_full = h_src_full.shape[0]
                    f_ss = spcidx_src[0]-self.pad_left
                    idx_end = -1
                    spcidx_src_end = spcidx_src[idx_end]
                    while spcidx_src_end >= frm_len:
                        idx_end -= 1
                        spcidx_src_end = spcidx_src[idx_end]
                    spcidx_src = spcidx_src[:idx_end]
                    f_es = spcidx_src_end+self.pad_right
                    if f_ss < 0:
                        f_ss = 0
                    if f_es >= frm_len:
                        f_es = frm_len-1
                    f_es += 1
                    spcidx_s_e = [f_ss, f_es]
                    spcidx_s_e_smpl = [f_ss*self.upsampling_factor_bands, f_es*self.upsampling_factor_bands]
                    x = x[spcidx_s_e_smpl[0]:spcidx_s_e_smpl[-1]]
                    h_src = h_src[spcidx_s_e[0]:spcidx_s_e[-1]]
                    assert(x.shape[0]==h_src.shape[0]*self.upsampling_factor_bands)
                    if self.worgx_flag:
                        spcidx_s_e_smpl_org = [f_ss*self.upsampling_factor, f_es*self.upsampling_factor]
                        x_org = x_org[spcidx_s_e_smpl_org[0]:spcidx_s_e_smpl_org[-1]]
                        assert(x_org.shape[0]==h_src.shape[0]*self.upsampling_factor)
                        x_org_band = x_org_band[spcidx_s_e_smpl[0]:spcidx_s_e_smpl[-1]]
                        assert(x_org_band.shape[0]==x.shape[0])
                    if self.magsp and self.mel:
                        h_src_magsp = h_src_magsp[spcidx_s_e[0]:spcidx_s_e[-1]]
                        assert(x.shape[0]==h_src_magsp.shape[0]*self.upsampling_factor_bands)
            else:
                x, _ = sf.read(wavfile, dtype=np.float32)
                x, h_src = validate_length(x, h_src, self.upsampling_factor)
                assert(x.shape[0]==h_src.shape[0]*self.upsampling_factor)
                frm_len = h_src.shape[0]
                if self.spcidx:
                    h_src_full = h_src
                    flen_src_full = h_src_full.shape[0]
                    f_ss = spcidx_src[0]-self.pad_left
                    idx_end = -1
                    spcidx_src_end = spcidx_src[idx_end]
                    while spcidx_src_end >= frm_len:
                        idx_end -= 1
                        spcidx_src_end = spcidx_src[idx_end]
                    spcidx_src = spcidx_src[:idx_end]
                    f_es = spcidx_src_end+self.pad_right
                    if f_ss < 0:
                        f_ss = 0
                    if f_es >= frm_len:
                        f_es = frm_len-1
                    f_es += 1
                    spcidx_s_e = [f_ss, f_es]
                    spcidx_s_e_smpl = [f_ss*self.upsampling_factor, f_es*self.upsampling_factor]
                    x = x[spcidx_s_e_smpl[0]:spcidx_s_e_smpl[-1]]
                    h_src = h_src[spcidx_s_e[0]:spcidx_s_e[-1]]
                    assert(x.shape[0]==h_src.shape[0]*self.upsampling_factor)
                    if self.magsp and self.mel:
                        h_src_magsp = h_src_magsp[spcidx_s_e[0]:spcidx_s_e[-1]]
                        assert(x.shape[0]==h_src_magsp.shape[0]*self.upsampling_factor)
                x = self.wav_transform(x)
            slen = x.shape[0]
        elif self.spcidx:
            h_src_full = h_src
            flen_src_full = h_src_full.shape[0]
            f_ss = spcidx_src[0]-self.pad_left
            idx_end = -1
            spcidx_src_end = spcidx_src[idx_end]
            while spcidx_src_end >= frm_len:
                idx_end -= 1
                spcidx_src_end = spcidx_src[idx_end]
            spcidx_src = spcidx_src[:idx_end]
            f_es = spcidx_src_end+self.pad_right
            if f_ss < 0:
                f_ss = 0
            if f_es >= frm_len:
                f_es = frm_len-1
            f_es += 1
            spcidx_s_e = [f_ss, f_es]
            h_src = h_src[spcidx_s_e[0]:spcidx_s_e[-1]]
            if self.magsp and self.mel:
                h_src_magsp = h_src_magsp[spcidx_s_e[0]:spcidx_s_e[-1]]
        flen = h_src.shape[0]

        if not self.mel or (self.mel and self.excit_dim is not None):
            mean_src = read_hdf5(self.stat_spk_list[idx_src], self.mean_path)[1:2]
            std_src = read_hdf5(self.stat_spk_list[idx_src], self.scale_path)[1:2]
            mean_trg = read_hdf5(self.stat_spk_list[idx_trg], self.mean_path)[1:2]
            std_trg = read_hdf5(self.stat_spk_list[idx_trg], self.scale_path)[1:2]

        flen_src = h_src.shape[0]
        flen_spc_src = spcidx_src.shape[0]
        src_code = np.ones(flen_src)*idx_src
        src_trg_code = np.ones(flen_src)*idx_trg
        if self.spcidx:
            src_code_full = np.ones(flen_src_full)*idx_src
            src_trg_code_full = np.ones(flen_src_full)*idx_trg
        if not self.mel or (self.mel and self.excit_dim is not None):
            if self.excit_dim is not None:
                if self.cap_exc_dim is None:
                    cv_src = np.c_[h_src[:,:1], (std_trg/std_src)*(h_src[:,1:2]-mean_src)+mean_trg, h_src[:,2:self.excit_dim]]
                    cv_src_full = np.c_[h_src_full[:,:1], (std_trg/std_src)*(h_src_full[:,1:2]-mean_src)+mean_trg, h_src_full[:,2:self.excit_dim]]
                else:
                    cv_src = np.c_[h_src[:,:1], (std_trg/std_src)*(h_src[:,1:2]-mean_src)+mean_trg]
                    cv_src_full = np.c_[h_src_full[:,:1], (std_trg/std_src)*(h_src_full[:,1:2]-mean_src)+mean_trg]
            else:
                cv_src = np.c_[h_src[:,:1], (std_trg/std_src)*(h_src[:,1:2]-mean_src)+mean_trg]
                cv_src_full = np.c_[h_src_full[:,:1], (std_trg/std_src)*(h_src_full[:,1:2]-mean_src)+mean_trg]

        if file_src_trg_flag:
            if self.mel:
                if self.excit_dim is not None:
                    h_src_trg = np.c_[read_hdf5(featfile_src_trg, '/feat_mceplf0cap')[:,:self.excit_dim], read_hdf5(featfile_src_trg, self.string_path)]
                else:
                    h_src_trg = read_hdf5(featfile_src_trg, self.string_path)
            else:
                if self.cap_exc_dim is None:
                    h_src_trg = read_hdf5(featfile_src_trg, self.string_path)
                else:
                    h_src_trg = np.c_[read_hdf5(featfile_src_trg, '/feat_mceplf0cap')[:,:2], read_hdf5(featfile_src_trg, '/feat_mceplf0cap')[:,self.cap_exc_dim:]]
            spcidx_src_trg = read_hdf5(featfile_src_trg, "/spcidx_range")[0]
            flen_src_trg = h_src_trg.shape[0]
            flen_spc_src_trg = spcidx_src_trg.shape[0]
            if self.uvcap:
                uvcap_trg = read_hdf5(featfile_src_trg, '/feat_mceplf0cap')[:,2:3]
                h_src_trg = torch.FloatTensor(self.pad_transform(np.c_[h_src_trg,uvcap_trg]))
            else:
                h_src_trg = torch.FloatTensor(self.pad_transform(h_src_trg))
            spcidx_src_trg = torch.LongTensor(self.pad_transform(spcidx_src_trg))

        if self.uvcap:
            if self.spcidx:
                uvcap_full = read_hdf5(featfile, '/feat_mceplf0cap')
                if self.wav_list is not None:
                    uvcap = uvcap_full[:spcidx_src_end+1,2:3]
                else:
                    uvcap = uvcap_full[spcidx_src[0]:spcidx_src_end+1,2:3]
                h_src_full = torch.FloatTensor(self.pad_transform(np.c_[h_src_full,uvcap_full]))
            else:
                uvcap = read_hdf5(featfile_src, '/feat_mceplf0cap')[:,2:3]
            h_src = torch.FloatTensor(self.pad_transform(np.c_[h_src,uvcap]))
        else:
            h_src = torch.FloatTensor(self.pad_transform(h_src))
            if self.spcidx:
                h_src_full = torch.FloatTensor(self.pad_transform(h_src_full))
        if self.magsp and self.mel:
            h_src_magsp = torch.FloatTensor(self.pad_transform(h_src_magsp))
        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))
        src_code = torch.LongTensor(self.pad_transform(src_code))
        src_trg_code = torch.LongTensor(self.pad_transform(src_trg_code))
        if self.spcidx:
            src_code_full = torch.LongTensor(self.pad_transform(src_code_full))
            src_trg_code_full = torch.LongTensor(self.pad_transform(src_trg_code_full))
        if not self.mel or (self.mel and self.excit_dim is not None):
            cv_src = torch.FloatTensor(self.pad_transform(cv_src))
            cv_src_full = torch.FloatTensor(self.pad_transform(cv_src_full))

        if not file_src_trg_flag:
            flen_src_trg = flen_src
            flen_spc_src_trg = flen_spc_src
            h_src_trg = h_src
            spcidx_src_trg = spcidx_src

        if self.spcidx:
            if not self.mel or (self.mel and self.excit_dim is not None):
                if not self.magsp or not self.mel:
                    if self.wav_list is None:
                        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                                'cv_src': cv_src, 'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                                'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                                'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg, \
                                'h_src_full': h_src_full, 'cv_src_full': cv_src_full, 'flen_src_full': flen_src_full, \
                                'src_code_full': src_code_full, 'src_trg_code_full': src_trg_code_full}
                    else:
                        x = torch.LongTensor(self.pad_wav_transform(x))
                        return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen_src': slen, 'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                                'cv_src': cv_src, 'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                                'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                                'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg, \
                                'h_src_full': h_src_full, 'cv_src_full': cv_src_full, 'flen_src_full': flen_src_full, \
                                'src_code_full': src_code_full, 'src_trg_code_full': src_trg_code_full}
                else:
                    if self.wav_list is None:
                        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                                'cv_src': cv_src, 'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                                'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                                'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg, \
                                'h_src_full': h_src_full, 'cv_src_full': cv_src_full, 'flen_src_full': flen_src_full, \
                                'src_code_full': src_code_full, 'src_trg_code_full': src_trg_code_full, \
                                'h_src_magsp': h_src_magsp}
                    else:
                        x = torch.LongTensor(self.pad_wav_transform(x))
                        if self.worgx_flag:
                            x_org = torch.FloatTensor(self.pad_wav_org_transform(x_org)) # disc in/trg
                            x_org_band = torch.FloatTensor(self.pad_wav_transform(x_org_band)) # disc in/trg
                        else:
                            x_org = None
                            x_org_band = None
                        return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen_src': slen, 'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                                'cv_src': cv_src, 'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                                'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                                'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg, \
                                'h_src_full': h_src_full, 'cv_src_full': cv_src_full, 'flen_src_full': flen_src_full, \
                                'src_code_full': src_code_full, 'src_trg_code_full': src_trg_code_full, \
                                'h_src_magsp': h_src_magsp}
            else:
                if not self.magsp:
                    if self.wav_list is None:
                        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                                'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                                'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg, \
                                'h_src_full': h_src_full, 'flen_src_full': flen_src_full, \
                                'src_code_full': src_code_full, 'src_trg_code_full': src_trg_code_full}
                    else:
                        x = torch.LongTensor(self.pad_wav_transform(x))
                        return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen_src': slen, 'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                                'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                                'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg, \
                                'h_src_full': h_src_full, 'flen_src_full': flen_src_full, \
                                'src_code_full': src_code_full, 'src_trg_code_full': src_trg_code_full}
                else:
                    if self.wav_list is None:
                        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                                'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                                'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg, \
                                'h_src_full': h_src_full, 'flen_src_full': flen_src_full, \
                                'src_code_full': src_code_full, 'src_trg_code_full': src_trg_code_full, \
                                'h_src_magsp': h_src_magsp}
                    else:
                        x = torch.LongTensor(self.pad_wav_transform(x))
                        if self.worgx_flag:
                            x_org = torch.FloatTensor(self.pad_wav_org_transform(x_org)) # disc in/trg
                            x_org_band = torch.FloatTensor(self.pad_wav_transform(x_org_band)) # disc in/trg
                        else:
                            x_org = None
                            x_org_band = None
                        return {'x_org': x_org, 'x_org_band': x_org_band, 'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen_src': slen, 'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                                'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                                'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg, \
                                'h_src_full': h_src_full, 'flen_src_full': flen_src_full, \
                                'src_code_full': src_code_full, 'src_trg_code_full': src_trg_code_full, \
                                'h_src_magsp': h_src_magsp}
        else:
            if self.wav_list is None:
                return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                        'cv_src': cv_src, 'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                        'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                        'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg}
            else:
                x = torch.LongTensor(self.pad_wav_transform(x))
                return {'x_c': x // self.cf_dim, 'x_f': x % self.cf_dim, 'slen_src': slen, 'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                        'cv_src': cv_src, 'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'featfile': featfile_src, \
                        'file_src_trg_flag': file_src_trg_flag, 'spk_trg': spk_trg, 'spcidx_src': spcidx_src, \
                        'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src': flen_spc_src, 'flen_spc_src_trg': flen_spc_src_trg}


class FeatureDatasetVAE(Dataset):
    """Dataset for VAE
    """

    def __init__(self, feat_list, pad_feat_transform, string_path, magsp=False, spk_list=None):
        self.feat_list = feat_list
        self.pad_feat_transform = pad_feat_transform
        self.string_path = string_path
        self.magsp = magsp
        self.spk_list = spk_list
        if "mel" in self.string_path:
            self.mel = True
        else:
            self.mel = False

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        featfile = self.feat_list[idx]
        feat = read_hdf5(featfile, self.string_path)
        #if self.excit_dim is not None:
        #    feat = np.c_[read_hdf5(featfile, '/feat_mceplf0cap')[:,:self.excit_dim], read_hdf5(featfile, self.string_path)]
        frm_len = len(read_hdf5(featfile, '/f0_range'))

        spcidx = read_hdf5(featfile, '/spcidx_range')[0]
        f_ss = spcidx[0]
        f_es = spcidx[-1]
        if f_ss < 0:
            f_ss = 0
        if f_es > frm_len:
            f_es = frm_len
        spcidx_s_e = [f_ss, f_es]
        feat = feat[spcidx_s_e[0]:spcidx_s_e[-1]]
        flen = feat.shape[0]
        if self.spk_list is not None:
            idx_spk = self.spk_list.index(os.path.basename(os.path.dirname(featfile)))
            spk_code = torch.LongTensor(self.pad_feat_transform(np.ones(flen)*idx_spk))

        #mean_trg_list, std_trg_list, trg_code_list, pair_spk_list = \
        #    proc_random_spkcv_statcvexcit(src_idx, self.spk_list, self.n_cv, flen, self.n_spk, \
        #        self.stat_spk_list, self.mean_path, self.scale_path)

        feat = torch.FloatTensor(self.pad_feat_transform(feat))
        if self.magsp:
            if self.mel:
                feat_magsp = read_hdf5(featfile, '/magsp')[spcidx_s_e[0]:spcidx_s_e[-1]]
            else:
                feat_magsp = read_hdf5(featfile, '/worldsp')[spcidx_s_e[0]:spcidx_s_e[-1]]
            feat_magsp = torch.FloatTensor(self.pad_feat_transform(feat_magsp))
            if self.spk_list is not None:
                return {'flen': flen, 'featfile': featfile, 'feat': feat, 'feat_magsp': feat_magsp, 'sc': spk_code}
            else:
                return {'flen': flen, 'featfile': featfile, 'feat': feat, 'feat_magsp': feat_magsp}
        else:
            if self.spk_list is not None:
                return {'flen': flen, 'featfile': featfile, 'feat': feat, 'sc': spk_code}
            else:
                return {'flen': flen, 'featfile': featfile, 'feat': feat}
