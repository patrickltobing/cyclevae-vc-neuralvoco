#!/bin/bash
##################################################################################################
#   SCRIPT FOR NON-PARALLEL VOICE CONVERSION based on CycleVAE/CycleVQVAE and WaveRNN/WaveNet    #
##################################################################################################

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# USER SETTINGS {{{
#######################################
#           STAGE SETTING             #
#######################################
# {{{
# 0: data preparation step
# init: feature extraction with initial speaker config
# 1: feature extraction step
# 2: statistics calculation step
# 3: apply noise shaping [pre-emphasis] and multiband processing [neural vocoder waveform train. data]
# 4: vc training step
# post: vc post-net training step
# 5: reconstruction decoding step [for possible neural vocoder training data augmentation]
# 6: conversion decoding step [converted waveform with Griffin-Lim]
# 7: wavernn training step
# 8: copy-synthesis decoding step with wavernn [original waveform with neural vocoder]
# 9: restore noise shaping step [de-emphasis] from copy-synthesis
# a: vc decoding step with wavernn [converted waveform with neural vocoder]
# b: restore noise shaping step [de-emphasis] from vc
# }}}
#stage=0
#stage=init
#stage=01
#stage=open
#stage=012
#stage=0123
#stage=12
#stage=1
#stage=2
#stage=123
#stage=23
#stage=3
#stage=4
#stage=post
#stage=5
#stage=56
#stage=6
#stage=7
#stage=89
#stage=8
#stage=9
#stage=ab
#stage=6ab
#stage=56ab
#stage=
#stage=ab
#stage=a
#stage=b

##number of parallel jobs in feature extraction / noise-shaping proc. / statistics calculation
n_jobs=1
#n_jobs=10
#n_jobs=25
n_jobs=30
#n_jobs=35
n_jobs=40
#n_jobs=45
#n_jobs=50
n_jobs=60

#######################################
#          TRAINING SETTING           #
#######################################

#spks_atr_todalab=(mizokuchi morikawa okada otake taga takada uchino yamada)
#spks_open=(taga)
#spks_open=(p276)
#spks_open=(morikawa mizokuchi okada takada otake taga)
#spks_open=(morikawa mizokuchi okada takada otake taga p276)
#spks_open=(morikawa mizokuchi okada takada otake)
spks=(SEF1 TEF1)
spks=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
data_name=vcc2020
#spks=(VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2)
#data_name=vcc18
#spks=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 p237 p245 p251 p252 p259 p274 p304 p311 p326 p345 p360 p363 TEF1 TEM1 TEF2 TEM2 \
#    TFF1 TGF1 TMF1 p231 p238 p248 p253 p264 p265 p266 p276 p305 p308 p318 p335)
#data_name=vcc2020vctk

## fs: sampling rate
fs=`awk '{if ($1 == "fs:") print $2}' conf/config.yml`
## shiftms: frame shift in ms
shiftms=`awk '{if ($1 == "shiftms:") print $2}' conf/config.yml`
## upsampling_factor: upsampling factor for neural vocoder
upsampling_factor=`echo "${shiftms} * ${fs} / 1000" | bc`

# uv-f0 and log-f0 occupied the first two dimensions,
# then uv-codeap, log-negative-codeap and mel-ceps
## [uv-f0,log-f0,uv-codeap,log-negative-codeap,mel-ceps]
## fftl: length of FFT window analysis
## WORLD F0_floor for cheaptrick: 3.0 * fs / (fft_size - 3.0)
## [https://github.com/mmorise/World/blob/master/src/cheaptrick.cpp] line 197
## mcep_alpha: frequency warping parameter for mel-cepstrum
if [ $fs -eq 22050 ]; then
    wav_org_dir=wav_22.05kHz
    data_name=${data_name}_22.05kHz
    mcep_alpha=0.455 #22.05k ## frequency warping based on pysptk.util.mcepalpha
    fftl=2048
    if [ $shiftms -eq 5 ]; then
        shiftms=4.9886621315192743764172335600907 #22.05k rounding 110/22050 5ms shift
    elif [ $shiftms -eq 10 ]; then
        shiftms=9.9773242630385487528344671201814 #22.05k rounding 220/22050 10ms shift
    fi
    full_excit_dim=5
elif [ $fs -eq 24000 ]; then
    wav_org_dir=wav_24kHz
    data_name=${data_name}_24kHz
    mcep_alpha=0.466 #24k
    fftl=2048
    full_excit_dim=6
elif [ $fs -eq 48000 ]; then
    wav_org_dir=wav_48kHz
    data_name=${data_name}_48kHz
    mcep_alpha=0.554 #48k
    fftl=4096
    full_excit_dim=8
elif [ $fs -eq 44100 ]; then
    wav_org_dir=wav_44.1kHz
    data_name=${data_name}_44.1kHz
    mcep_alpha=0.544 #44.1k
    fftl=4096
    if [ $shiftms -eq 5 ]; then
        shiftms=4.9886621315192743764172335600907 #44.1k rounding 220/44100 5ms shift
    elif [ $shiftms -eq 10 ]; then
        shiftms=9.9773242630385487528344671201814 #44.1k rounding 440/44100 10ms shift
    fi
    full_excit_dim=8
elif [ $fs -eq 16000 ]; then
    wav_org_dir=wav_16kHz
    data_name=${data_name}_16kHz
    mcep_alpha=0.41000000000000003 #16k
    fftl=1024
    full_excit_dim=4
elif [ $fs -eq 8000 ]; then
    wav_org_dir=wav_8kHz
    data_name=${data_name}_8kHz
    mcep_alpha=0.312 #8k
    fftl=1024
    full_excit_dim=4
else
    echo "sampling rate not available"
    exit 1
fi
## from WORLD: number of code-aperiodicities = min(15000,fs/2-3000)/3000
## [https://github.com/mmorise/World/blob/master/src/codec.cpp] line 212

## mcep_dim: number of mel-cepstrum dimension
mcep_dim=`awk '{if ($1 == "mcep_dim:") print $2}' conf/config.yml`
## powmcep_dim: 0th power + mcep_dim
powmcep_dim=`expr ${mcep_dim} + 1`
## winms: window length analysis for mel-spectrogram extraction
winms=`awk '{if ($1 == "winms:") print $2}' conf/config.yml`
## mel_dim: number of mel-spectrogram dimension
mel_dim=`awk '{if ($1 == "mel_dim:") print $2}' conf/config.yml`
## highpass_cutoff: cutoff frequency for low-cut filter to remove DC-component in recording
highpass_cutoff=`awk '{if ($1 == "highpass_cutoff:") print $2}' conf/config.yml`
## alpha: coefficient for pre-emphasis
alpha=`awk '{if ($1 == "alpha:") print $2}' conf/config.yml`
## n_bands: number of bands for multiband modeling
n_bands=`awk '{if ($1 == "n_bands:") print $2}' conf/config.yml`

trn=tr_${data_name}
dev=dv_${data_name}
tst=ts_${data_name}

GPU_device=0
#GPU_device=1
GPU_device=2
#GPU_device=3
GPU_device=4
#GPU_device=5
#GPU_device=6
#GPU_device=7
#GPU_device=8
#GPU_device=9

## Please see the conf/config.yml for explanation of the rest of variables
mdl_name=`awk '{if ($1 == "mdl_name:") print $2}' conf/config.yml`
#mdl_name_post=none #set none to not use post-net
mdl_name_post=`awk '{if ($1 == "mdl_name_post:") print $2}' conf/config.yml`
epoch_count=`awk '{if ($1 == "epoch_count:") print $2}' conf/config.yml`
epoch_count_wave=`awk '{if ($1 == "epoch_count_wave:") print $2}' conf/config.yml`
n_half_cyc=`awk '{if ($1 == "n_half_cyc:") print $2}' conf/config.yml`

use_mcep=`awk '{if ($1 == "use_mcep:") print $2}' conf/config.yml`
with_excit=`awk '{if ($1 == "with_excit:") print $2}' conf/config.yml`

if [ $use_mcep == "true" ]; then
    string_path="/feat_mceplf0cap"
else
    string_path="/log_1pmelmagsp"
fi

lr=`awk '{if ($1 == "lr:") print $2}' conf/config.yml`

### settings for spec/excit network
batch_size=`awk '{if ($1 == "batch_size:") print $2}' conf/config.yml`
batch_size_utt=`awk '{if ($1 == "batch_size_utt:") print $2}' conf/config.yml`
batch_size_utt_eval=`awk '{if ($1 == "batch_size_utt_eval:") print $2}' conf/config.yml`
lat_dim=`awk '{if ($1 == "lat_dim:") print $2}' conf/config.yml`
lat_dim_e=`awk '{if ($1 == "lat_dim_e:") print $2}' conf/config.yml`
ctr_size=`awk '{if ($1 == "ctr_size:") print $2}' conf/config.yml`
hidden_units_enc=`awk '{if ($1 == "hidden_units_enc:") print $2}' conf/config.yml`
hidden_layers_enc=`awk '{if ($1 == "hidden_layers_enc:") print $2}' conf/config.yml`
hidden_units_dec=`awk '{if ($1 == "hidden_units_dec:") print $2}' conf/config.yml`
hidden_layers_dec=`awk '{if ($1 == "hidden_layers_dec:") print $2}' conf/config.yml`
hidden_units_lf0=`awk '{if ($1 == "hidden_units_lf0:") print $2}' conf/config.yml`
hidden_layers_lf0=`awk '{if ($1 == "hidden_layers_lf0:") print $2}' conf/config.yml`
hidden_units_post=`awk '{if ($1 == "hidden_units_post:") print $2}' conf/config.yml`
hidden_layers_post=`awk '{if ($1 == "hidden_layers_post:") print $2}' conf/config.yml`
kernel_size_enc=`awk '{if ($1 == "kernel_size_enc:") print $2}' conf/config.yml`
dilation_size_enc=`awk '{if ($1 == "dilation_size_enc:") print $2}' conf/config.yml`
kernel_size_dec=`awk '{if ($1 == "kernel_size_dec:") print $2}' conf/config.yml`
dilation_size_dec=`awk '{if ($1 == "dilation_size_dec:") print $2}' conf/config.yml`
kernel_size_lf0=`awk '{if ($1 == "kernel_size_lf0:") print $2}' conf/config.yml`
dilation_size_lf0=`awk '{if ($1 == "dilation_size_lf0:") print $2}' conf/config.yml`
kernel_size_post=`awk '{if ($1 == "kernel_size_post:") print $2}' conf/config.yml`
dilation_size_post=`awk '{if ($1 == "dilation_size_post:") print $2}' conf/config.yml`
causal_conv_enc=`awk '{if ($1 == "causal_conv_enc:") print $2}' conf/config.yml`
causal_conv_dec=`awk '{if ($1 == "causal_conv_dec:") print $2}' conf/config.yml`
causal_conv_lf0=`awk '{if ($1 == "causal_conv_lf0:") print $2}' conf/config.yml`
causal_conv_post=`awk '{if ($1 == "causal_conv_post:") print $2}' conf/config.yml`
do_prob=`awk '{if ($1 == "do_prob:") print $2}' conf/config.yml`
n_workers=`awk '{if ($1 == "n_workers:") print $2}' conf/config.yml`
pad_len=`awk '{if ($1 == "pad_len:") print $2}' conf/config.yml`
spkidtr_dim=`awk '{if ($1 == "spkidtr_dim:") print $2}' conf/config.yml`
right_size_enc=`awk '{if ($1 == "right_size_enc:") print $2}' conf/config.yml`
right_size_dec=`awk '{if ($1 == "right_size_dec:") print $2}' conf/config.yml`
right_size_lf0=`awk '{if ($1 == "right_size_lf0:") print $2}' conf/config.yml`
right_size_post=`awk '{if ($1 == "right_size_post:") print $2}' conf/config.yml`

### settings for neural vocoder
mdl_name_wave=`awk '{if ($1 == "mdl_name_wave:") print $2}' conf/config.yml`
hidden_units_wave=`awk '{if ($1 == "hidden_units_wave:") print $2}' conf/config.yml`
hidden_units_wave_2=`awk '{if ($1 == "hidden_units_wave_2:") print $2}' conf/config.yml`
hidden_units_wave_enc=`awk '{if ($1 == "hidden_units_wave_enc:") print $2}' conf/config.yml`
kernel_size_wave=`awk '{if ($1 == "kernel_size_wave:") print $2}' conf/config.yml`
dilation_size_wave=`awk '{if ($1 == "dilation_size_wave:") print $2}' conf/config.yml`
kernel_size=`awk '{if ($1 == "kernel_size:") print $2}' conf/config.yml`
hid_chn=`awk '{if ($1 == "hid_chn:") print $2}' conf/config.yml`
skip_chn=`awk '{if ($1 == "skip_chn:") print $2}' conf/config.yml`
dilation_depth=`awk '{if ($1 == "dilation_depth:") print $2}' conf/config.yml`
dilation_repeat=`awk '{if ($1 == "dilation_repeat:") print $2}' conf/config.yml`
batch_size_wave=`awk '{if ($1 == "batch_size_wave:") print $2}' conf/config.yml`
batch_size_utt_wave=`awk '{if ($1 == "batch_size_utt_wave:") print $2}' conf/config.yml`
batch_size_utt_eval_wave=`awk '{if ($1 == "batch_size_utt_eval_wave:") print $2}' conf/config.yml`
t_start=`awk '{if ($1 == "t_start:") print $2}' conf/config.yml`
t_end=`awk '{if ($1 == "t_end:") print $2}' conf/config.yml`
interval=`awk '{if ($1 == "interval:") print $2}' conf/config.yml`
densities=`awk '{if ($1 == "densities:") print $2}' conf/config.yml`
densities_enc=`awk '{if ($1 == "densities_enc:") print $2}' conf/config.yml`
n_stage=`awk '{if ($1 == "n_stage:") print $2}' conf/config.yml`
lpc=`awk '{if ($1 == "lpc:") print $2}' conf/config.yml`
compact=`awk '{if ($1 == "compact:") print $2}' conf/config.yml`
causal_conv_wave=`awk '{if ($1 == "causal_conv_wave:") print $2}' conf/config.yml`
right_size_wave=`awk '{if ($1 == "right_size_wave:") print $2}' conf/config.yml`


#######################################
#     DECODING/FINE-TUNING SETTING    #
#######################################

#idx_resume_cycvae=1 #for resume cyclevae
idx_resume_cycvae=0 #set <= 0 for not resume

#idx_resume=1 #for resume post-net
idx_resume=0 #set <= 0 for not resume

#idx_resume_wave=1 #for resume wavernn
idx_resume_wave=0 #set <= 0 for not resume

min_idx_cycvae= #for raw cyclevae
#min_idx_cycvae=2

if [ $mdl_name_post == none ]; then
    min_idx= #for cyclevae without post-net
else
    min_idx= #for cyclevae with post-net
    #min_idx=2
fi

min_idx_wave= #for wavernn model
#min_idx_wave=1
#min_idx_wave=23
#min_idx_wave=30

n_interp=0
#n_interp=10 #for speaker interpolation in 2-dim space with spec-excit cyclevae/cyclevqvae
#n_interp=20

gv_coeff=`awk '{if ($1 == "gv_coeff:") print $2}' conf/config.yml`

if [ $mdl_name_post == "cycmelspxlf0capspkvae-post-smpl-laplace" ]; then
    string_path_rec=/feat_rec_${mdl_name_post}-${mdl_name}-${epoch_count}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}-${min_idx}
    string_path_cv=/feat_cv_${mdl_name_post}-${mdl_name}-${epoch_count}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}-${min_idx}
elif [ $mdl_name == "cycmelspxlf0capspkvae-laplace" ]; then
    string_path_rec=/feat_rec_${mdl_name}-${epoch_count}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}
    string_path_cv=/feat_cv_${mdl_name}-${epoch_count}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}
fi


### Set GPU_device_str and n_gpus for VC/neural vocoder decoding with synchronized values
GPU_device_str="0"
#GPU_device_str="0,1,2"
#GPU_device_str="7,8,9"
#GPU_device_str="2,8,9"
GPU_device_str="5,6,7"
#GPU_device_str="0,2,8,9"
#GPU_device_str="9,2,8,0"
#GPU_device_str="8,9"
#GPU_device_str="0,4,7,8,9"
#GPU_device_str="2,0,3,4,1"
GPU_device_str="3,2,0,4,1"

n_gpus=1
#n_gpus=2
n_gpus=3
#n_gpus=4
n_gpus=5
###


### This is for reconstruction generation
#spks_trg_rec=(VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2)
spks_trg_rec=(TEM2)
spks_trg_rec=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
#spks_trg_rec=(TEF2 TGF1)
#spks_trg_rec=(TEM2 TMF1 TFF1 TEF1)
#spks_trg_rec=(TMM1 TGM1 TFM1 TEM1)
#spks_trg_rec=(TEF1)
#spks_trg_rec=(SEF1 TEF1)

#spks_trg_rec=(p237 p245 p251 p252 p259 p274)
#spks_trg_rec=(p266 p276 p304 p311 p326 p345)
#spks_trg_rec=(p360 p363 p305 p308 p231 p238)
#spks_trg_rec=(p248 p253 p264 p265 p318 p335)
###


### This is for VC with griffin-lim or neural vocoder synthesizer
#spks_src_dec=(VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4)
spks_src_dec=(SEM1 SEF2 SEM2 SEF1)
#spks_src_dec=(SEM1)
#spks_src_dec=(SEF2)
#spks_src_dec=(SEM2)
spks_src_dec=(SEF1)

#spks_trg_dec=(VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2)
spks_trg_dec=(TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
#spks_trg_dec=(TEM2)
#spks_trg_dec=(TEF2 TGF1)
#spks_trg_dec=(TEM2 TMF1 TFF1 TEF1)
#spks_trg_dec=(TMM1 TGM1 TFM1 TEM1)
spks_trg_dec=(TEF1)
###


### This is for copy-synthesis using neural vocoder
#spks_dec=(TFF1 TFM1 TMF1 TMM1 TGF1 TGM1 TEF1 TEF2 TEM1 TEM2 p237 p245 p248 p253 p276)
#spks_dec=(p276 TGM1 taga TEM2 okada otake mizokuchi morikawa takada TFF1 TMM1)
#spks_dec=(TFM1)
#spks_dec=(TGM1 TFF1 TMM1)
#spks_dec=(TEM2)
#spks_dec=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
spks_dec=(TEF1)
#spks_dec=(TGM1)
#spks_dec=(VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2)
###


echo $GPU_device $GPU_device_str
#echo $min_idx_wave $idx_resume_wave $epoch_count $batch_size $batch_size_utt


## This is for number of batch sequence when decoding with wavernn
decode_batch_size=1
#decode_batch_size=2
decode_batch_size=3
#decode_batch_size=5
decode_batch_size=9
#decode_batch_size=10
decode_batch_size=13
#decode_batch_size=15
decode_batch_size=17

# parse options
. parse_options.sh

# set params

echo $mdl_name $data_name

# stop when error occured
set -e
# }}}


# STAGE 0 {{{
if [ `echo ${stage} | grep 0` ];then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    mkdir -p data/${trn}
    mkdir -p data/${dev}
    mkdir -p data/${tst}
    [ -e data/${trn}/wav.scp ] && rm data/${trn}/wav.scp
    [ -e data/${dev}/wav.scp ] && rm data/${dev}/wav.scp
    [ -e data/${tst}/wav.scp ] && rm data/${tst}/wav.scp
    if true; then
    #if false; then
    for spk in ${spks[@]};do
        if [ -n "$(echo $spk | sed -n 's/\(p265\)/\1/p')" ]; then
            echo vctk1 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 22 >> data/${tst}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 315 >> data/${trn}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(p\)/\1/p')" ]; then
            if [ -n "$(echo $spk | sed -n 's/\(p253\)/\1/p')" ]; then
                echo vctk2 $spk
                find ${wav_org_dir}/${spk} -name "*.wav" \
                    | sort | head -n 21 >> data/${tst}/wav.scp
            elif [ -n "$(echo $spk | sed -n 's/\(p264\)/\1/p')" ] \
                || [ -n "$(echo $spk | sed -n 's/\(p318\)/\1/p')" ]; then
                echo vctk3 $spk
                find ${wav_org_dir}/${spk} -name "*.wav" \
                    | sort | head -n 22 >> data/${tst}/wav.scp
            elif [ -n "$(echo $spk | sed -n 's/\(p237\)/\1/p')" ] \
                || [ -n "$(echo $spk | sed -n 's/\(p274\)/\1/p')" ] \
                    || [ -n "$(echo $spk | sed -n 's/\(p231\)/\1/p')" ] \
                        || [ -n "$(echo $spk | sed -n 's/\(p276\)/\1/p')" ] \
                            || [ -n "$(echo $spk | sed -n 's/\(p335\)/\1/p')" ]; then
                echo vctk4 $spk
                find ${wav_org_dir}/${spk} -name "*.wav" \
                    | sort | head -n 23 >> data/${tst}/wav.scp
            else
                echo vctk5 $spk
                find ${wav_org_dir}/${spk} -name "*.wav" \
                    | sort | head -n 24 >> data/${tst}/wav.scp
            fi
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 339 | tail -n 315 >> data/${trn}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(VCC\)/\1/p')" ]; then
            echo vcc18_1 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 71 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/test/${spk} -name "*.wav" \
                | sort >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(SE\)/\1/p')" ]; then
            echo vcc20_1 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 60 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 70 | tail -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/test/${spk} -name "*.wav" \
                | sort >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(TE\)/\1/p')" ]; then
            echo vcc20_2 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 10 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 20 | tail -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 70 | tail -n 50 >> data/${trn}/wav.scp
            find ${wav_org_dir}/test/${spk} -name "*.wav" \
                | sort >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(TF\)/\1/p')" ] \
            || [ -n "$(echo $spk | sed -n 's/\(TM\)/\1/p')" ]; then
            echo vcc20_3 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 70 | tail -n 60 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 70 | head -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/test/${spk} -name "*.wav" \
                | sort >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(TG\)/\1/p')" ]; then
            echo vcc20_4 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 70 | head -n 60 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 70 | tail -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/test/${spk} -name "*.wav" \
                | sort >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(S\)/\1/p')" ]; then
            echo vcc18_1 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 71 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 81 | tail -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/test/${spk} -name "*.wav" \
                | sort >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(T\)/\1/p')" ]; then
            echo vcc18_1 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 71 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 81 | tail -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/test/${spk} -name "*.wav" \
                | sort >> data/${tst}/wav.scp
        fi
        set +e
        touch conf/spkr.yml
        tmp=`yq ".${spk}" conf/spkr.yml`
        if [[ -z $tmp ]] || [[ $tmp == "null" ]]; then
            echo $spk: >> conf/spkr.yml
            if [ -f "conf/${spk}.f0" ]; then
                minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
                yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
                yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
                echo "minF0 and maxF0 of ${spk} is initialized from .f0 file"
            else
                yq -yi ".${spk}.minf0=40" conf/spkr.yml
                yq -yi ".${spk}.maxf0=700" conf/spkr.yml
                echo "minF0 and maxF0 of ${spk} is initialized, please run stage init, then change accordingly"
            fi
            if [ -f "conf/${spk}.pow" ]; then
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
                yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                echo "npow of ${spk} is initialized from .pow file"
            else
                yq -yi ".${spk}.npow=-25" conf/spkr.yml
                echo "npow of ${spk} is initialized, please run stage init, then change accordingly"
            fi
        else
            tmp=`yq ".${spk}.minf0" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.f0" ]; then
                    minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                    yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
                    echo "minF0 of ${spk} is initialized .f0 file"
                else
                    yq -yi ".${spk}.minf0=40" conf/spkr.yml
                    echo "minF0 of ${spk} is initialized, please run stage init, then change accordingly"
                fi
            fi
            tmp=`yq ".${spk}.maxf0" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.f0" ]; then
                    maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
                    yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
                    echo "maxF0 of ${spk} is initialized .f0 file"
                else
                    yq -yi ".${spk}.maxf0=700" conf/spkr.yml
                    echo "maxF0 of ${spk} is initialized, please run stage init, then change accordingly"
                fi
            fi
            tmp=`yq ".${spk}.npow" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.pow" ]; then
                    pow=`cat conf/${spk}.pow | awk '{print $1}'`
                    yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                    echo "npow of ${spk} is initialized from .pow file"
                else
                    yq -yi ".${spk}.npow=-25" conf/spkr.yml
                    echo "npow of ${spk} is initialized, please run stage init, then change accordingly"
                fi
            fi
        fi
        set -e
    done
    fi
fi
# }}}


# STAGE open {{{
if [ `echo ${stage} | grep open` ];then
    echo "###########################################################"
    echo "#              OPEN DATA PREPARATION STEP                 #"
    echo "###########################################################"
    mkdir -p data/${tst}
    if true; then
    #if false; then
    for spk in ${spks_open[@]};do
        if printf '%s\0' "${spks_atr_todalab[@]}" | grep -xq --null "$spk";then
            echo todalab $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n +50 >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(p237\)/\1/p')" ] \
                || [ -n "$(echo $spk | sed -n 's/\(p274\)/\1/p')" ] \
                    || [ -n "$(echo $spk | sed -n 's/\(p231\)/\1/p')" ] \
                        || [ -n "$(echo $spk | sed -n 's/\(p276\)/\1/p')" ] \
                            || [ -n "$(echo $spk | sed -n 's/\(p335\)/\1/p')" ]; then
                echo vctk4 $spk
                find ${wav_org_dir}/${spk} -name "*.wav" \
                    | sort | head -n 23 >> data/${tst}/wav.scp
        fi
        set +e
        tmp=`yq ".${spk}" conf/spkr.yml`
        if [[ -z $tmp ]] || [[ $tmp == "null" ]]; then
            echo $spk: >> conf/spkr.yml
            if [ -f "conf/${spk}.f0" ]; then
                minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
                yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
                yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
                echo "minF0 and maxF0 of ${spk} is initialized from .f0 file"
            else
                yq -yi ".${spk}.minf0=40" conf/spkr.yml
                yq -yi ".${spk}.maxf0=700" conf/spkr.yml
                echo "minF0 and maxF0 of ${spk} is initialized, please run stage init, then change accordingly"
            fi
            if [ -f "conf/${spk}.pow" ]; then
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
                yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                echo "npow of ${spk} is initialized from .pow file"
            else
                yq -yi ".${spk}.npow=-25" conf/spkr.yml
                echo "npow of ${spk} is initialized, please run stage init, then change accordingly"
            fi
        else
            tmp=`yq ".${spk}.minf0" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.f0" ]; then
                    minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                    yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
                    echo "minF0 of ${spk} is initialized .f0 file"
                else
                    yq -yi ".${spk}.minf0=40" conf/spkr.yml
                    echo "minF0 of ${spk} is initialized, please run stage init, then change accordingly"
                fi
            fi
            tmp=`yq ".${spk}.maxf0" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.f0" ]; then
                    maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
                    yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
                    echo "maxF0 of ${spk} is initialized .f0 file"
                else
                    yq -yi ".${spk}.maxf0=700" conf/spkr.yml
                    echo "maxF0 of ${spk} is initialized, please run stage init, then change accordingly"
                fi
            fi
            tmp=`yq ".${spk}.npow" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.pow" ]; then
                    pow=`cat conf/${spk}.pow | awk '{print $1}'`
                    yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                    echo "npow of ${spk} is initialized from .pow file"
                else
                    yq -yi ".${spk}.npow=-25" conf/spkr.yml
                    echo "npow of ${spk} is initialized, please run stage init, then change accordingly"
                fi
            fi
        fi
        set -e
    done
    fi
fi
# }}}


# STAGE init {{{
if [ `echo ${stage} | grep "init"` ];then
    echo "###########################################################"
    echo "#               INIT FEATURE EXTRACTION STEP              #"
    echo "###########################################################"
    if true; then
    #if false; then
        # extract feat and wav_anasyn src_speaker
        nj=0
        for set in ${trn} ${dev};do
            echo $set
            expdir=exp/feature_extract_init/${set}
            mkdir -p $expdir
            for spk in ${spks[@]}; do
                echo $spk
                scp=${expdir}/wav_${spk}.scp
                n_wavs=`cat data/${set}/wav.scp | grep "\/${spk}\/" | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav.scp | grep "\/${spk}\/" > ${scp}
                    ${train_cmd} --num-threads ${n_jobs} ${expdir}/feature_extract_${spk}.log \
                        feature_extract.py \
                            --expdir $expdir \
                            --waveforms ${scp} \
                            --hdf5dir hdf5_init/${set}/${spk} \
                            --fs ${fs} \
                            --shiftms ${shiftms} \
                            --mcep_dim ${mcep_dim} \
                            --mcep_alpha ${mcep_alpha} \
                            --fftl ${fftl} \
                            --highpass_cutoff ${highpass_cutoff} \
                            --init true \
                            --n_jobs ${n_jobs}
        
                    # check the number of feature files
                    n_feats=`find hdf5/${set}/${spk} -name "*.h5" | wc -l`
                    echo "${n_feats}/${n_wavs} files are successfully processed."

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            done
        done
    fi
    for set in ${trn} ${dev};do
        echo $set
        find hdf5_init/${set} -name "*.h5" | sort > tmp2
        rm -f data/${set}/feats_init.scp
        for spk in ${spks[@]}; do
            cat tmp2 | grep "\/${spk}\/" >> data/${set}/feats_init.scp
        done
        rm -f tmp2
    done
    echo "###########################################################"
    echo "#              INIT SPEAKER STATISTICS STEP               #"
    echo "###########################################################"
    expdir=exp/init_spk_stat/${trn}
    mkdir -p $expdir
    if true; then
    #if false; then
        rm -f $expdir/spk_stat.log
        for spk in ${spks[@]};do
            echo $spk
            cat data/${trn}/feats_init.scp | grep \/${spk}\/ > data/${trn}/feats_init_all_spk-${spk}.scp
            cat data/${dev}/feats_init.scp | grep \/${spk}\/ >> data/${trn}/feats_init_all_spk-${spk}.scp
            ${train_cmd} exp/init_spk_stat/init_stat_${data_name}_spk-${spk}.log \
                spk_stat.py \
                    --expdir ${expdir} \
                    --feats data/${trn}/feats_init_all_spk-${spk}.scp
        done
        echo "init spk statistics are successfully calculated. please change the initial values accordingly"
    fi
fi
# }}}


# STAGE 1 {{{
if [ `echo ${stage} | grep 1` ];then
    echo "###########################################################"
    echo "#               FEATURE EXTRACTION STEP                   #"
    echo "###########################################################"
    if true; then
    #if false; then
        # extract feat and wav_anasyn
        nj=0
        #for set in ${trn} ${dev};do
        for set in ${trn} ${dev} ${tst};do
        #for set in ${tst};do
            echo $set
            expdir=exp/feature_extract/${set}
            mkdir -p $expdir
            rm -f $expdir/feature_extract.log
            for spk in ${spks[@]}; do
            #for spk in ${spks_open[@]}; do
                echo $spk
                minf0=`yq ".${spk}.minf0" conf/spkr.yml`
                maxf0=`yq ".${spk}.maxf0" conf/spkr.yml`
                pow=`yq ".${spk}.npow" conf/spkr.yml`
                echo $minf0 $maxf0 $pow
                scp=${expdir}/wav_${spk}.scp
                n_wavs=`cat data/${set}/wav.scp | grep "\/${spk}\/" | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav.scp | grep "\/${spk}\/" > ${scp}
                    ${train_cmd} --num-threads ${n_jobs} ${expdir}/feature_extract_${spk}.log \
                        feature_extract.py \
                            --expdir $expdir \
                            --waveforms ${scp} \
                            --wavdir wav_anasyn/${set}/${spk} \
                            --wavgfdir wav_anasyn_gf/${set}/${spk} \
                            --wavfiltdir wav_filtered/${set}/${spk} \
                            --hdf5dir hdf5/${set}/${spk} \
                            --fs ${fs} \
                            --shiftms ${shiftms} \
                            --winms ${winms} \
                            --minf0 ${minf0} \
                            --maxf0 ${maxf0} \
                            --pow ${pow} \
                            --mel_dim ${mel_dim} \
                            --mcep_dim ${mcep_dim} \
                            --mcep_alpha ${mcep_alpha} \
                            --fftl ${fftl} \
                            --highpass_cutoff ${highpass_cutoff} \
                            --n_jobs ${n_jobs}
        
                    # check the number of feature files
                    n_feats=`find hdf5/${set}/${spk} -name "*.h5" | wc -l`
                    echo "${n_feats}/${n_wavs} files are successfully processed."

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            done
        done
    fi
    # make scp for feats
    set +e
    rm -f data/${trn}/feats_all.scp
    #for set in ${trn} ${dev};do
    for set in ${trn} ${dev} ${tst};do
    #for set in ${tst};do
        echo $set
        find hdf5/${set} -name "*.h5" | sort > tmp2
        find wav_filtered/${set} -name "*.wav" | sort > tmp3
        rm -f data/${set}/feats.scp data/${set}/wav_filtered.scp
        for spk in ${spks[@]}; do
            cat tmp2 | grep "\/${spk}\/" >> data/${set}/feats.scp
            cat tmp3 | grep "\/${spk}\/" >> data/${set}/wav_filtered.scp
            echo $set $spk
            if [[ $set != $tst ]]; then
                cat tmp2 | grep "\/${spk}\/" >> data/${trn}/feats_all.scp
            fi
        done
        rm -f tmp2 tmp3
    done
    #for set in ${tst};do
    #    echo $set
    #    find hdf5/${set} -name "*.h5" | sort > tmp2
    #    for spk in ${spks_open[@]}; do
    #        cat tmp2 | grep "\/${spk}\/" >> data/${set}/feats.scp
    #        echo $set $spk
    #    done
    #    rm -f tmp2
    #done
    set -e
fi
# }}}


# STAGE 2 {{{
if [ `echo ${stage} | grep 2` ];then
    echo "###########################################################"
    echo "#            CALCULATE SPEAKER STATISTICS STEP            #"
    echo "###########################################################"
    expdir=exp/calculate_statistics
    rm -f $expdir/calc_stats.log
    if true; then
    #if false; then
        for spk in ${spks[@]};do
            echo $spk
            cat data/${trn}/feats.scp | grep \/${spk}\/ > data/${trn}/feats_spk-${spk}.scp
            cat data/${trn}/feats_spk-${spk}.scp > data/${trn}/feats_all_spk-${spk}.scp
            n_feats_dev=`cat data/${dev}/feats.scp | grep "\/${spk}\/" | wc -l`
            if [ $n_feats_dev -gt 0 ]; then
                cat data/${dev}/feats.scp | grep \/${spk}\/ >> data/${trn}/feats_all_spk-${spk}.scp
                cat data/${dev}/feats.scp | grep \/${spk}\/ > data/${dev}/feats_spk-${spk}.scp
            fi
            n_feats_tst=`cat data/${tst}/feats.scp | grep "\/${spk}\/" | wc -l`
            if [ $n_feats_tst -gt 0 ]; then
                cat data/${tst}/feats.scp | grep \/${spk}\/ > data/${tst}/feats_spk-${spk}.scp
            fi
            ${train_cmd} exp/calculate_statistics/calc_stats_${trn}_spk-${spk}.log \
                calc_stats.py \
                    --expdir ${expdir} \
                    --feats data/${trn}/feats_all_spk-${spk}.scp \
                    --mcep_dim ${powmcep_dim} \
                    --n_jobs ${n_jobs} \
                    --stats data/${trn}/stats_spk-${spk}.h5
        done
        echo "speaker statistics are successfully calculated."
    fi
    echo "###########################################################"
    echo "#             CALCULATE JOINT STATISTICS STEP             #"
    echo "###########################################################"
    if true; then
    #if false; then
        ${train_cmd} exp/calculate_statistics/calc_stats_${trn}.log \
            calc_stats.py \
                --expdir ${expdir} \
                --feats data/${trn}/feats_all.scp \
                --mcep_dim ${powmcep_dim} \
                --n_jobs ${n_jobs} \
                --stats data/${trn}/stats_jnt.h5
        echo "joint statistics are successfully calculated."
    fi
fi
# }}}


# STAGE 3 {{{
if [ `echo ${stage} | grep 3` ];then
    if true; then
    #if false; then
        echo "###########################################################"
        echo "#                   NOISE SHAPING STEP                    #"
        echo "###########################################################"
        nj=0
        expdir=exp/noise_shaping
        mkdir -p ${expdir}
        for set in ${trn} ${dev};do
            echo $set
            mkdir -p ${expdir}/${set}
            for spk in ${spks[@]};do
                echo $spk
                # make scp of each speaker
                scp=${expdir}/${set}/wav_filtered.${set}.${spk}.scp
                n_wavs=`cat data/${set}/wav_filtered.scp | grep "\/${spk}\/" | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav_filtered.scp | grep "\/${spk}\/" > ${scp}
            
                    # apply noise shaping
                    ${train_cmd} --num-threads ${n_jobs} \
                        ${expdir}/${set}/noise_shaping_emph_apply.${set}.${spk}.log \
                        noise_shaping_emph.py \
                            --waveforms ${scp} \
                            --writedir wav_ns/${set}/${spk} \
                            --fs ${fs} \
                            --alpha ${alpha} \
                            --n_jobs ${n_jobs} & 

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            done
            wait
            # check the number of feature files
            n_wavs=`cat data/${set}/wav_filtered.scp | wc -l`
            n_ns=`find wav_ns/${set} -name "*.wav" | wc -l`
            echo "${n_ns}/${n_wavs} files are successfully processed [emph]."

            # make scp files
            find wav_ns/${set} -name "*.wav" | sort > data/${set}/wav_ns.scp
        done
    fi
    if true; then
    #if false; then
        echo "###########################################################"
        echo "#                  PQMF MULTI-BAND STEP                   #"
        echo "###########################################################"
        nj=0
        expdir=exp/pqmf
        mkdir -p ${expdir}
        for set in ${trn} ${dev};do
            echo $set
            mkdir -p ${expdir}/${set}
            for spk in ${spks[@]};do
                echo $spk
                # make scp of each speaker
                scp=${expdir}/${set}/wav_ns.${set}.${spk}.scp
                n_wavs=`cat data/${set}/wav_ns.scp | grep "\/${spk}\/" | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav_ns.scp | grep "\/${spk}\/" > ${scp}
            
                    # apply noise shaping
                    ${train_cmd} --num-threads ${n_jobs} \
                        ${expdir}/${set}/noise_shaping_emph_pqmf_${n_bands}_apply.${set}.${spk}.log \
                        proc_wav_pqmf.py \
                            --waveforms ${scp} \
                            --writedir wav_ns_pqmf_${n_bands}/${set}/${spk} \
                            --writesyndir wav_ns_pqmf_${n_bands}_rec/${set}/${spk} \
                            --fs ${fs} \
                            --n_bands ${n_bands} \
                            --n_jobs ${n_jobs}

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            done
            wait
            # check the number of feature files
            n_wavs=`cat data/${set}/wav_ns.scp | wc -l`
            n_ns=`find wav_ns_pqmf_${n_bands}/${set} -name "*.wav" | wc -l`
            echo "${n_ns}/${n_wavs} files are successfully processed [emph pqmf ${n_bands}-bands]."

            # make scp files
            find wav_ns_pqmf_${n_bands}/${set} -name "*.wav" | sort > data/${set}/wav_ns_pqmf_${n_bands}.scp
        done
    fi
fi
# }}}


stats_list=()
feats_eval_list=()
wavs_eval_list=()
for spk in ${spks[@]};do
    stats_list+=(data/${trn}/stats_spk-${spk}.h5)
    if [ -n "$(echo $spk | sed -n 's/\(p\)/\1/p')" ]; then
        touch data/${dev}/feats_spk-${spk}.scp
        touch data/${dev}/wav_ns_spk-${spk}.scp
    else
        if [ ! -f data/${dev}/wav_ns_spk-${spk}.scp ]; then
            cat data/${dev}/wav_ns.scp | grep "\/${spk}\/" > data/${dev}/wav_ns_spk-${spk}.scp
        fi
    fi
    feats_eval_list+=(data/${dev}/feats_spk-${spk}.scp)
    wavs_eval_list+=(data/${dev}/wav_ns_spk-${spk}.scp)
done

stats_list_list="$(IFS="@"; echo "${stats_list[*]}")"
feats_list_eval_list="$(IFS="@"; echo "${feats_eval_list[*]}")"
wavs_list_eval_list="$(IFS="@"; echo "${wavs_eval_list[*]}")"

spk_list="$(IFS="@"; echo "${spks[*]}")"
echo ${spk_list}

echo $mdl_name
if [ $mdl_name == "cycmelspxlf0capspkvae-laplace" ]; then
    setting=${mdl_name}_${data_name}_lr${lr}_bs${batch_size}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hud${hidden_units_dec}_huf${hidden_units_lf0}_kse${kernel_size_enc}_ksd${kernel_size_dec}_ksf${kernel_size_lf0}_rse${right_size_enc}_rsd${right_size_dec}_rsf${right_size_lf0}_do${do_prob}_ep${epoch_count}_mel${mel_dim}_nhcyc${n_half_cyc}_s${spkidtr_dim}
fi

# STAGE 4 {{
# set variables
expdir=exp/tr_${setting}
if [ `echo ${stage} | grep 4` ];then
    mkdir -p $expdir
    echo "###########################################################"
    echo "#               FEATURE MODELING STEP                     #"
    echo "###########################################################"
    echo $expdir

    if [ $mdl_name == "cycmelspxlf0capspkvae-laplace" ];then
        feats=data/${trn}/feats.scp
        if [ $idx_resume_cycvae -gt 0 ]; then
            ${cuda_cmd} ${expdir}/log/train_resume-${idx_resume_cycvae}.log \
                train_gru-cycle-melsp-x-lf0cap-spk-vae-laplace_noar.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count} \
                    --mel_dim ${mel_dim} \
                    --lat_dim ${lat_dim} \
                    --lat_dim_e ${lat_dim_e} \
                    --stats_list ${stats_list_list} \
                    --spk_list ${spk_list} \
                    --hidden_units_enc ${hidden_units_enc} \
                    --hidden_layers_enc ${hidden_layers_enc} \
                    --hidden_units_dec ${hidden_units_dec} \
                    --hidden_layers_dec ${hidden_layers_dec} \
                    --hidden_units_lf0 ${hidden_units_lf0} \
                    --hidden_layers_lf0 ${hidden_layers_lf0} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --kernel_size_lf0 ${kernel_size_lf0} \
                    --dilation_size_lf0 ${dilation_size_lf0} \
                    --causal_conv_enc ${causal_conv_enc} \
                    --causal_conv_dec ${causal_conv_dec} \
                    --causal_conv_lf0 ${causal_conv_lf0} \
                    --batch_size ${batch_size} \
                    --batch_size_utt ${batch_size_utt} \
                    --batch_size_utt_eval ${batch_size_utt_eval} \
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --right_size_lf0 ${right_size_lf0} \
                    --full_excit_dim ${full_excit_dim} \
                    --resume ${expdir}/checkpoint-${idx_resume_cycvae}.pkl \
                    --GPU_device ${GPU_device}
        else
            ${cuda_cmd} ${expdir}/log/train.log \
                train_gru-cycle-melsp-x-lf0cap-spk-vae-laplace_noar.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count} \
                    --mel_dim ${mel_dim} \
                    --lat_dim ${lat_dim} \
                    --lat_dim_e ${lat_dim_e} \
                    --stats_list ${stats_list_list} \
                    --spk_list ${spk_list} \
                    --hidden_units_enc ${hidden_units_enc} \
                    --hidden_layers_enc ${hidden_layers_enc} \
                    --hidden_units_dec ${hidden_units_dec} \
                    --hidden_layers_dec ${hidden_layers_dec} \
                    --hidden_units_lf0 ${hidden_units_lf0} \
                    --hidden_layers_lf0 ${hidden_layers_lf0} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --kernel_size_lf0 ${kernel_size_lf0} \
                    --dilation_size_lf0 ${dilation_size_lf0} \
                    --causal_conv_enc ${causal_conv_enc} \
                    --causal_conv_dec ${causal_conv_dec} \
                    --causal_conv_lf0 ${causal_conv_lf0} \
                    --batch_size ${batch_size} \
                    --batch_size_utt ${batch_size_utt} \
                    --batch_size_utt_eval ${batch_size_utt_eval} \
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --right_size_lf0 ${right_size_lf0} \
                    --full_excit_dim ${full_excit_dim} \
                    --GPU_device ${GPU_device}
        fi
    fi
fi
# }}}


echo $min_idx_cycvae $setting
echo $mdl_name_post
setting_cycvae=${setting}
if [ $mdl_name_post == "cycmelspxlf0capspkvae-post-smpl-laplace" ]; then
    setting=${mdl_name_post}_${data_name}_lr${lr}_bs${batch_size}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hud${hidden_units_dec}_huf${hidden_units_lf0}_hup${hidden_units_post}_kse${kernel_size_enc}_ksd${kernel_size_dec}_ksf${kernel_size_lf0}_ksp${kernel_size_post}_rse${right_size_enc}_rsd${right_size_dec}_rsf${right_size_lf0}_rsp${right_size_post}_do${do_prob}_ep${epoch_count}_mel${mel_dim}_nhcyc${n_half_cyc}_s${spkidtr_dim}_c${min_idx_cycvae}
fi

# STAGE post {{
# set variables
expdir_cycvae=exp/tr_${setting_cycvae}
expdir=exp/tr_${setting}
if [ `echo ${stage} | grep post` ];then
    mkdir -p $expdir
    echo "###########################################################"
    echo "#               POST NETWORK TRAINING                     #"
    echo "###########################################################"
    echo $expdir

    if [ $mdl_name_post == "cycmelspxlf0capspkvae-post-smpl-laplace" ];then
        feats=data/${trn}/feats.scp
        if [ $idx_resume -gt 0 ]; then
            ${cuda_cmd} ${expdir}/log/train_resume-${idx_resume}.log \
                train_gru-cycle-melsp-x-lf0cap-spk-vae-post-smpl-laplace_noar.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count} \
                    --mel_dim ${mel_dim} \
                    --lat_dim ${lat_dim} \
                    --lat_dim_e ${lat_dim_e} \
                    --stats_list ${stats_list_list} \
                    --spk_list ${spk_list} \
                    --hidden_units_enc ${hidden_units_enc} \
                    --hidden_layers_enc ${hidden_layers_enc} \
                    --hidden_units_dec ${hidden_units_dec} \
                    --hidden_layers_dec ${hidden_layers_dec} \
                    --hidden_units_lf0 ${hidden_units_lf0} \
                    --hidden_layers_lf0 ${hidden_layers_lf0} \
                    --hidden_units_post ${hidden_units_post} \
                    --hidden_layers_post ${hidden_layers_post} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --kernel_size_lf0 ${kernel_size_lf0} \
                    --dilation_size_lf0 ${dilation_size_lf0} \
                    --kernel_size_post ${kernel_size_post} \
                    --dilation_size_post ${dilation_size_post} \
                    --causal_conv_enc ${causal_conv_enc} \
                    --causal_conv_dec ${causal_conv_dec} \
                    --causal_conv_lf0 ${causal_conv_lf0} \
                    --causal_conv_post ${causal_conv_post} \
                    --batch_size ${batch_size} \
                    --batch_size_utt ${batch_size_utt} \
                    --batch_size_utt_eval ${batch_size_utt_eval} \
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --right_size_lf0 ${right_size_lf0} \
                    --right_size_post ${right_size_post} \
                    --full_excit_dim ${full_excit_dim} \
                    --fftl ${fftl} \
                    --fs ${fs} \
                    --gen_model ${expdir_cycvae}/checkpoint-${min_idx_cycvae}.pkl \
                    --resume ${expdir}/checkpoint-${idx_resume}.pkl \
                    --GPU_device ${GPU_device}
        else
            ${cuda_cmd} ${expdir}/log/train.log \
                train_gru-cycle-melsp-x-lf0cap-spk-vae-post-smpl-laplace_noar.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count} \
                    --mel_dim ${mel_dim} \
                    --lat_dim ${lat_dim} \
                    --lat_dim_e ${lat_dim_e} \
                    --stats_list ${stats_list_list} \
                    --spk_list ${spk_list} \
                    --hidden_units_enc ${hidden_units_enc} \
                    --hidden_layers_enc ${hidden_layers_enc} \
                    --hidden_units_dec ${hidden_units_dec} \
                    --hidden_layers_dec ${hidden_layers_dec} \
                    --hidden_units_lf0 ${hidden_units_lf0} \
                    --hidden_layers_lf0 ${hidden_layers_lf0} \
                    --hidden_units_post ${hidden_units_post} \
                    --hidden_layers_post ${hidden_layers_post} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --kernel_size_lf0 ${kernel_size_lf0} \
                    --dilation_size_lf0 ${dilation_size_lf0} \
                    --kernel_size_post ${kernel_size_post} \
                    --dilation_size_post ${dilation_size_post} \
                    --causal_conv_enc ${causal_conv_enc} \
                    --causal_conv_dec ${causal_conv_dec} \
                    --causal_conv_lf0 ${causal_conv_lf0} \
                    --causal_conv_post ${causal_conv_post} \
                    --batch_size ${batch_size} \
                    --batch_size_utt ${batch_size_utt} \
                    --batch_size_utt_eval ${batch_size_utt_eval} \
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --right_size_lf0 ${right_size_lf0} \
                    --right_size_post ${right_size_post} \
                    --full_excit_dim ${full_excit_dim} \
                    --fftl ${fftl} \
                    --fs ${fs} \
                    --gen_model ${expdir_cycvae}/checkpoint-${min_idx_cycvae}.pkl \
                    --GPU_device ${GPU_device}
        fi
    fi
fi
# }}}


if [ $mdl_name_post == none ]; then
    expdir=${expdir_cycvae}
fi

# STAGE 5 {{{
if [ `echo ${stage} | grep 5` ];then
    echo $expdir $n_gpus $GPU_device $GPU_device_str
    config=${expdir}/model.conf
    for spk_trg in ${spks_trg_rec[@]};do
        if true; then
        #if false; then
                echo "########################################################"
                echo "#          DECODING RECONST. FEAT and GV stat          #"
                echo "########################################################"
                echo $spk_trg $min_idx
                outdir=${expdir}/rec-cycrec_${spk_trg}_${min_idx}
                mkdir -p $outdir
                feats_tr=data/${trn}/feats.scp
                feats_dv=data/${dev}/feats.scp
                feats_scp=${outdir}/feats_${spk_trg}.scp
                cat ${feats_tr} | grep "\/${spk_trg}\/" > ${feats_scp}
                n_feats_dev=`cat ${feats_dv} | grep "\/${spk_trg}\/" | wc -l`
                if [ $n_feats_dev -gt 0 ]; then
                    cat ${feats_dv} | grep "\/${spk_trg}\/" >> ${feats_scp}
                fi
                if [ $mdl_name_post == "cycmelspxlf0capspkvae-post-smpl-laplace" ];then
                    model=${expdir}/checkpoint-${min_idx}.pkl
                    ${cuda_cmd} ${expdir}/log/decode_rec-cycrec_${spk_trg}_${min_idx_cycvae}-${min_idx}.log \
                        calc_rec-cycrec-gv_gru-cycle-melspxlf0capspkvae-post-smpl-laplace_noar.py \
                            --feats ${feats_scp} \
                            --spk ${spk_trg} \
                            --outdir ${outdir} \
                            --model ${model} \
                            --config ${config} \
                            --GPU_device_str ${GPU_device_str} \
                            --string_path ${string_path_rec} \
                            --n_gpus ${n_gpus}
                            #--GPU_device ${GPU_device} \
                elif [ $mdl_name == "cycmelspxlf0capspkvae-laplace" ];then
                    model=${expdir}/checkpoint-${min_idx_cycvae}.pkl
                    ${cuda_cmd} ${expdir}/log/decode_rec-cycrec_${spk_trg}_${min_idx_cycvae}.log \
                        calc_rec-cycrec-gv_gru-cycle-melspxlf0capspkvae-laplace_noar.py \
                            --feats ${feats_scp} \
                            --spk ${spk_trg} \
                            --outdir ${outdir} \
                            --model ${model} \
                            --config ${config} \
                            --GPU_device_str ${GPU_device_str} \
                            --string_path ${string_path_rec} \
                            --n_gpus ${n_gpus}
                            #--GPU_device ${GPU_device} \
                fi
        fi
    done
    if true; then
    #if false; then
    waveforms=data/${trn}/wav_ns.scp
    waveforms_eval=data/${dev}/wav_ns.scp
    feats_ft_scp=data/${trn}/feats_ft.scp
    feats_ft_eval_scp=data/${dev}/feats_ft.scp
    rm -f ${feats_ft_scp} ${feats_ft_eval_scp}
    waveforms_ft_scp=data/${trn}/wav_ns_ft.scp
    waveforms_ft_eval_scp=data/${dev}/wav_ns_ft.scp
    rm -f ${waveforms_ft_scp} ${waveforms_ft_eval_scp}
    for spk in ${spks_trg_rec[@]}; do
        echo $spk
        ## org
        #find hdf5/${trn}/${spk} -name "*.h5" | sort >> ${feats_ft_scp}
        #cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ft_scp}
        ## rec/cycrec
        find hdf5/${trn}/${spk}-${spk} -name "*.h5" | sort >> ${feats_ft_scp}
        cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ft_scp}
        find hdf5/${trn}/${spk}-${spk}-${spk} -name "*.h5" | sort >> ${feats_ft_scp}
        cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ft_scp}
        #n_feats=`find hdf5/${dev}/${spk} -name "*.h5" | wc -l `
        #if [ $n_feats -gt 0 ]; then
        #    find hdf5/${dev}/${spk} -name "*.h5" | sort >> ${feats_ft_eval_scp}
        #    cat ${waveforms_eval} | grep "\/${spk}\/" >> ${waveforms_ft_eval_scp}
        #fi
        n_feats=`find hdf5/${dev}/${spk}-${spk} -name "*.h5" | wc -l `
        if [ $n_feats -gt 0 ]; then
            find hdf5/${dev}/${spk}-${spk} -name "*.h5" | sort >> ${feats_ft_eval_scp}
            cat ${waveforms_eval} | grep "\/${spk}\/" >> ${waveforms_ft_eval_scp}
        fi
        n_feats=`find hdf5/${dev}/${spk}-${spk}-${spk} -name "*.h5" | wc -l `
        if [ $n_feats -gt 0 ]; then
            find hdf5/${dev}/${spk}-${spk}-${spk} -name "*.h5" | sort >> ${feats_ft_eval_scp}
            cat ${waveforms_eval} | grep "\/${spk}\/" >> ${waveforms_ft_eval_scp}
        fi
    done
    fi
fi
# }}}


# STAGE 6 {{{
if [ `echo ${stage} | grep 6` ];then
for spkr in ${spks_src_dec[@]};do
#if [ -n "$(echo $spkr | sed -n 's/\(S\)/\1/p')" ]; then
for spk_trg in ${spks_trg_dec[@]};do
if [ $spkr != $spk_trg ]; then
    echo $spkr $spk_trg $min_idx_cycvae $min_idx
    echo $expdir $n_gpus $GPU_device $GPU_device_str

    config=${expdir}/model.conf

    h5outdir=hdf5/${dev}/${spkr}-${spk_trg}
    echo $h5outdir
    if true; then
    #if false; then
    echo "######################################################"
    echo "#                DECODING CONV. FEAT DEV             #"
    echo "######################################################"
    if [ $mdl_name_post == "cycmelspxlf0capspkvae-post-smpl-laplace" ]; then
        model=${expdir}/checkpoint-${min_idx}.pkl
        if [ $spkidtr_dim != "0" ]; then
            outdir=${expdir}/wav_cv_${mdl_name_post}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx_cycvae}-${min_idx}_${spkr}-${spk_trg}-${n_interp}_dev
        else
            outdir=${expdir}/wav_cv_${mdl_name_post}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx_cycvae}-${min_idx}_${spkr}-${spk_trg}_dev
        fi
    elif [ $mdl_name == "cycmelspxlf0capspkvae-laplace" ]; then
        model=${expdir}/checkpoint-${min_idx_cycvae}.pkl
        if [ $spkidtr_dim != "0" ]; then
            outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx_cycvae}_${spkr}-${spk_trg}-${n_interp}_dev
        else
            outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx_cycvae}_${spkr}-${spk_trg}_dev
        fi
    fi
    mkdir -p ${outdir}
    feats_scp=${outdir}/feats.scp
    cat data/${dev}/feats.scp | grep "\/${spkr}\/" > ${feats_scp}
    if [ $mdl_name_post == "cycmelspxlf0capspkvae-post-smpl-laplace" ]; then
        if [ $spkidtr_dim != "0" ]; then
            ${cuda_cmd} ${expdir}/log/decode_dev_${min_idx_cycvae}-${min_idx}_${spkr}-${spk_trg}-${n_interp}.log \
                decode_gru-cycle-melspxlf0capspkvae-post-smpl-laplace_noar.py \
                    --feats ${feats_scp} \
                    --spk_trg ${spk_trg} \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config} \
                    --fs ${fs} \
                    --winms ${winms} \
                    --fftl ${fftl} \
                    --shiftms ${shiftms} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --n_interp ${n_interp} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device} \
        else
            ${cuda_cmd} ${expdir}/log/decode_dev_${min_idx_cycvae}-${min_idx}_${spkr}-${spk_trg}.log \
                decode_gru-cycle-melspxlf0capspkvae-post-smpl-laplace_noar.py \
                    --feats ${feats_scp} \
                    --spk_trg ${spk_trg} \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config} \
                    --fs ${fs} \
                    --winms ${winms} \
                    --fftl ${fftl} \
                    --shiftms ${shiftms} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device} \
        fi
    elif [ $mdl_name == "cycmelspxlf0capspkvae-laplace" ]; then
        if [ $spkidtr_dim != "0" ]; then
            ${cuda_cmd} ${expdir}/log/decode_dev_${min_idx_cycvae}_${spkr}-${spk_trg}-${n_interp}.log \
                decode_gru-cycle-melspxlf0capspkvae-laplace_noar.py \
                    --feats ${feats_scp} \
                    --spk_trg ${spk_trg} \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config} \
                    --fs ${fs} \
                    --winms ${winms} \
                    --fftl ${fftl} \
                    --shiftms ${shiftms} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --n_interp ${n_interp} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device} \
        else
            ${cuda_cmd} ${expdir}/log/decode_dev_${min_idx_cycvae}_${spkr}-${spk_trg}.log \
                decode_gru-cycle-melspxlf0capspkvae-laplace_noar.py \
                    --feats ${feats_scp} \
                    --spk_trg ${spk_trg} \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config} \
                    --fs ${fs} \
                    --winms ${winms} \
                    --fftl ${fftl} \
                    --shiftms ${shiftms} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device} \
        fi
    fi
    fi
    find ${h5outdir} -name "*.h5" | sort > data/${dev}/feats_cv_${spkr}-${spk_trg}.scp

    h5outdir=hdf5/${tst}/${spkr}-${spk_trg}
    echo $h5outdir
    if true; then
    #if false; then
    echo "######################################################"
    echo "#                DECODING CONV. FEAT TST             #"
    echo "######################################################"
    if [ $mdl_name_post == "cycmelspxlf0capspkvae-post-smpl-laplace" ]; then
        model=${expdir}/checkpoint-${min_idx}.pkl
        if [ $spkidtr_dim != "0" ]; then
            outdir=${expdir}/wav_cv_${mdl_name_post}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx_cycvae}-${min_idx}_${spkr}-${spk_trg}-${n_interp}_tst
        else
            outdir=${expdir}/wav_cv_${mdl_name_post}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx_cycvae}-${min_idx}_${spkr}-${spk_trg}_tst
        fi
    elif [ $mdl_name == "cycmelspxlf0capspkvae-laplace" ]; then
        model=${expdir}/checkpoint-${min_idx_cycvae}.pkl
        if [ $spkidtr_dim != "0" ]; then
            outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx_cycvae}_${spkr}-${spk_trg}-${n_interp}_tst
        else
            outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx_cycvae}_${spkr}-${spk_trg}_tst
        fi
    fi
    mkdir -p ${outdir}
    feats_scp=${outdir}/feats.scp
    cat data/${tst}/feats.scp | grep "\/${spkr}\/" > ${feats_scp}
    if [ $mdl_name_post == "cycmelspxlf0capspkvae-post-smpl-laplace" ]; then
        if [ $spkidtr_dim != "0" ]; then
            ${cuda_cmd} ${expdir}/log/decode_tst_${min_idx_cycvae}-${min_idx}_${spkr}-${spk_trg}-${n_interp}.log \
                decode_gru-cycle-melspxlf0capspkvae-post-smpl-laplace_noar.py \
                    --feats ${feats_scp} \
                    --spk_trg ${spk_trg} \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config} \
                    --fs ${fs} \
                    --winms ${winms} \
                    --fftl ${fftl} \
                    --shiftms ${shiftms} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --n_interp ${n_interp} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device} \
        else
            ${cuda_cmd} ${expdir}/log/decode_tst_${min_idx_cycvae}-${min_idx}_${spkr}-${spk_trg}.log \
                decode_gru-cycle-melspxlf0capspkvae-post-smpl-laplace_noar.py \
                    --feats ${feats_scp} \
                    --spk_trg ${spk_trg} \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config} \
                    --fs ${fs} \
                    --winms ${winms} \
                    --fftl ${fftl} \
                    --shiftms ${shiftms} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device} \
        fi
    elif [ $mdl_name == "cycmelspxlf0capspkvae-laplace" ]; then
        if [ $spkidtr_dim != "0" ]; then
            ${cuda_cmd} ${expdir}/log/decode_tst_${min_idx_cycvae}_${spkr}-${spk_trg}-${n_interp}.log \
                decode_gru-cycle-melspxlf0capspkvae-laplace_noar.py \
                    --feats ${feats_scp} \
                    --spk_trg ${spk_trg} \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config} \
                    --fs ${fs} \
                    --winms ${winms} \
                    --fftl ${fftl} \
                    --shiftms ${shiftms} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --n_interp ${n_interp} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device} \
        else
            ${cuda_cmd} ${expdir}/log/decode_tst_${min_idx_cycvae}_${spkr}-${spk_trg}.log \
                decode_gru-cycle-melspxlf0capspkvae-laplace_noar.py \
                    --feats ${feats_scp} \
                    --spk_trg ${spk_trg} \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config} \
                    --fs ${fs} \
                    --winms ${winms} \
                    --fftl ${fftl} \
                    --shiftms ${shiftms} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device} \
        fi
    fi
    fi
    find ${h5outdir} -name "*.h5" | sort > data/${tst}/feats_cv_${spkr}-${spk_trg}.scp
fi
done
#fi
done
fi
# }}}


echo $mdl_name_wave
if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_16bit" ] \
    || [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_9bit" ] \
        || [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf" ]; then
    if [ $use_mcep == "true" ]; then
        setting_wave=${mdl_name_wave}_${data_name}_lr${lr}_bs${batch_size_wave}_bsu${batch_size_utt_wave}_bsue${batch_size_utt_eval_wave}_huw${hidden_units_wave}_hu2w${hidden_units_wave_2}_ksw${kernel_size_wave}_dsw${dilation_size_wave}_do${do_prob}_ep${epoch_count_wave}_mcep${use_mcep}_ts${t_start}_te${t_end}_i${interval}_d${densities}_ns${n_stage}_lpc${lpc}_rs${right_size_wave}_nb${n_bands}
    else
        setting_wave=${mdl_name_wave}_${data_name}_lr${lr}_bs${batch_size_wave}_bsu${batch_size_utt_wave}_bsue${batch_size_utt_eval_wave}_huw${hidden_units_wave}_hu2w${hidden_units_wave_2}_ksw${kernel_size_wave}_dsw${dilation_size_wave}_do${do_prob}_ep${epoch_count_wave}_mcep${use_mcep}_ts${t_start}_te${t_end}_i${interval}_d${densities}_ns${n_stage}_lpc${lpc}_rs${right_size_wave}_nb${n_bands}_exc${with_excit}
    fi
fi

# STAGE 7 {{
# set variables
expdir_wave=exp/tr_${setting_wave}
if [ `echo ${stage} | grep 7` ];then
    mkdir -p $expdir_wave
    if [ $use_mcep == "false" ]; then
        powmcep_dim=$mel_dim
    fi
    echo "###########################################################"
    echo "#               WAVEFORM MODELING STEP                    #"
    echo "###########################################################"
    echo $expdir_wave
   
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf" ];then
        feats=${expdir_wave}/feats_tr.scp
        feats_eval=${expdir_wave}/feats_ev.scp
        waveforms=${expdir_wave}/wavs_tr.scp
        waveforms_eval=${expdir_wave}/wavs_ev.scp
        ### Use these if not using reconst./cyclic reconst. feats
        cat data/${trn}/feats.scp | sort > ${feats}
        cat data/${dev}/feats.scp | sort > ${feats_eval}
        cat data/${trn}/wav_ns.scp | sort > ${waveforms}
        cat data/${dev}/wav_ns.scp | sort > ${waveforms_eval}
        if [ $idx_resume_wave -gt 0 ]; then
            ${cuda_cmd} ${expdir_wave}/log/train_resume-${idx_resume_wave}.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_10bit_cf.py \
                    --waveforms ${waveforms} \
                    --waveforms_eval $waveforms_eval \
                    --feats ${feats} \
                    --feats_eval $feats_eval \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_wave} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --batch_size ${batch_size_wave} \
                    --mcep_dim ${powmcep_dim} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --batch_size_utt ${batch_size_utt_wave} \
                    --batch_size_utt_eval ${batch_size_utt_eval_wave} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities ${densities} \
                    --n_stage ${n_stage} \
                    --lpc ${lpc} \
                    --right_size ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --with_excit ${with_excit} \
                    --string_path ${string_path} \
                    --resume ${expdir_wave}/checkpoint-${idx_resume_wave}.pkl \
                    --GPU_device ${GPU_device}
                    #--string_path_ft ${string_path_rec} \
        else
            ${cuda_cmd} ${expdir_wave}/log/train.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_10bit_cf.py \
                    --waveforms ${waveforms} \
                    --waveforms_eval $waveforms_eval \
                    --feats ${feats} \
                    --feats_eval $feats_eval \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_wave} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --batch_size ${batch_size_wave} \
                    --mcep_dim ${powmcep_dim} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --batch_size_utt ${batch_size_utt_wave} \
                    --batch_size_utt_eval ${batch_size_utt_eval_wave} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities ${densities} \
                    --n_stage ${n_stage} \
                    --lpc ${lpc} \
                    --right_size ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --with_excit ${with_excit} \
                    --string_path ${string_path} \
                    --GPU_device ${GPU_device}
                    #--string_path_ft ${string_path_rec} \
        fi
    elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_9bit" ];then
        feats=${expdir_wave}/feats_tr.scp
        feats_eval=${expdir_wave}/feats_ev.scp
        waveforms=${expdir_wave}/wavs_tr.scp
        waveforms_eval=${expdir_wave}/wavs_ev.scp
        ### Use these if not using reconst./cyclic reconst. feats
        cat data/${trn}/feats.scp | sort > ${feats}
        cat data/${dev}/feats.scp | sort > ${feats_eval}
        cat data/${trn}/wav_ns.scp | sort > ${waveforms}
        cat data/${dev}/wav_ns.scp | sort > ${waveforms_eval}
        if [ $idx_resume_wave -gt 0 ]; then
            ${cuda_cmd} ${expdir_wave}/log/train_resume-${idx_resume_wave}.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_9bit.py \
                    --waveforms ${waveforms} \
                    --waveforms_eval $waveforms_eval \
                    --feats ${feats} \
                    --feats_eval $feats_eval \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_wave} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --batch_size ${batch_size_wave} \
                    --mcep_dim ${powmcep_dim} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --batch_size_utt ${batch_size_utt_wave} \
                    --batch_size_utt_eval ${batch_size_utt_eval_wave} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities ${densities} \
                    --n_stage ${n_stage} \
                    --lpc ${lpc} \
                    --right_size ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --with_excit ${with_excit} \
                    --string_path ${string_path} \
                    --resume ${expdir_wave}/checkpoint-${idx_resume_wave}.pkl \
                    --GPU_device ${GPU_device}
                    #--string_path_ft ${string_path_rec} \
        else
            ${cuda_cmd} ${expdir_wave}/log/train.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_9bit.py \
                    --waveforms ${waveforms} \
                    --waveforms_eval $waveforms_eval \
                    --feats ${feats} \
                    --feats_eval $feats_eval \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_wave} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --batch_size ${batch_size_wave} \
                    --mcep_dim ${powmcep_dim} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --batch_size_utt ${batch_size_utt_wave} \
                    --batch_size_utt_eval ${batch_size_utt_eval_wave} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities ${densities} \
                    --n_stage ${n_stage} \
                    --lpc ${lpc} \
                    --right_size ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --with_excit ${with_excit} \
                    --string_path ${string_path} \
                    --GPU_device ${GPU_device}
                    #--string_path_ft ${string_path_rec} \
        fi
    elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_16bit" ];then
        feats=${expdir_wave}/feats_tr.scp
        feats_eval=${expdir_wave}/feats_ev.scp
        waveforms=${expdir_wave}/wavs_tr.scp
        waveforms_eval=${expdir_wave}/wavs_ev.scp
        ### Use these if not using reconst./cyclic reconst. feats
        cat data/${trn}/feats.scp | sort > ${feats}
        cat data/${dev}/feats.scp | sort > ${feats_eval}
        cat data/${trn}/wav_ns.scp | sort > ${waveforms}
        cat data/${dev}/wav_ns.scp | sort > ${waveforms_eval}
        if [ $idx_resume_wave -gt 0 ]; then
            ${cuda_cmd} ${expdir_wave}/log/train_resume-${idx_resume_wave}.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_16bit.py \
                    --waveforms ${waveforms} \
                    --waveforms_eval $waveforms_eval \
                    --feats ${feats} \
                    --feats_eval $feats_eval \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_wave} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --batch_size ${batch_size_wave} \
                    --mcep_dim ${powmcep_dim} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --batch_size_utt ${batch_size_utt_wave} \
                    --batch_size_utt_eval ${batch_size_utt_eval_wave} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities ${densities} \
                    --n_stage ${n_stage} \
                    --lpc ${lpc} \
                    --right_size ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --with_excit ${with_excit} \
                    --string_path ${string_path} \
                    --resume ${expdir_wave}/checkpoint-${idx_resume_wave}.pkl \
                    --GPU_device ${GPU_device}
                    #--string_path_ft ${string_path_rec} \
        else
            ${cuda_cmd} ${expdir_wave}/log/train.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_16bit.py \
                    --waveforms ${waveforms} \
                    --waveforms_eval $waveforms_eval \
                    --feats ${feats} \
                    --feats_eval $feats_eval \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_wave} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --epoch_count ${epoch_count_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --batch_size ${batch_size_wave} \
                    --mcep_dim ${powmcep_dim} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --batch_size_utt ${batch_size_utt_wave} \
                    --batch_size_utt_eval ${batch_size_utt_eval_wave} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities ${densities} \
                    --n_stage ${n_stage} \
                    --lpc ${lpc} \
                    --right_size ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --with_excit ${with_excit} \
                    --string_path ${string_path} \
                    --GPU_device ${GPU_device}
                    #--string_path_ft ${string_path_rec} \
        fi
    fi
fi
# }}}


# STAGE 8 {{{
if [ `echo ${stage} | grep 8` ] || [ `echo ${stage} | grep 9` ];then
for spk_src in ${spks_dec[@]};do
        if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_16bit" ] \
            || [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_9bit" ] \
                || [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf" ];then
            outdir=${expdir_wave}/${mdl_name_wave}-${data_name}_dev-${hidden_units_wave}-${epoch_count_wave}-${lpc}-${n_bands}-${batch_size_wave}-${min_idx_wave}_${spk_src}
            #outdir=${expdir_wave}/${mdl_name_wave}-${data_name}_tst-${hidden_units_wave}-${epoch_count_wave}-${lpc}-${n_bands}-${batch_size_wave}-${min_idx_wave}_${spk_src}
        fi
if [ `echo ${stage} | grep 8` ];then
        echo $spk_src
        echo $spk_src $min_idx_wave $data_name $mdl_name_wave
        echo $outdir
        echo "###################################################"
        echo "#               DECODING STEP                     #"
        echo "###################################################"
        echo ${setting_wave}
        checkpoint=${expdir_wave}/checkpoint-${min_idx_wave}.pkl
        config=${expdir_wave}/model.conf

        #feats=data/${trn}/feats.scp
        feats=data/${dev}/feats.scp
        #feats=data/${tst}/feats.scp

        feats_scp=${expdir_wave}/feats_${min_idx_wave}_${spk_src}.scp
        cat $feats | grep "\/${spk_src}\/" > ${feats_scp}

        # decode
        if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_9bit" ]; then
            #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_wave}_${spk_src}.log \
            ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_wave}_${spk_src}.log \
                decode_wavernn_dualgru_compact_lpc_mband_9bit.py \
                    --feats ${feats_scp} \
                    --outdir ${outdir} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --GPU_device_str ${GPU_device_str}
        elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_16bit" ]; then
            #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_wave}_${spk_src}.log \
            ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_wave}_${spk_src}.log \
                decode_wavernn_dualgru_compact_lpc_mband_16bit.py \
                    --feats ${feats_scp} \
                    --outdir ${outdir} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --GPU_device_str ${GPU_device_str}
        fi
fi
# }}}


# STAGE 9 {{{
if [ `echo ${stage} | grep 9` ];then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    scp=${expdir_wave}/wav_generated_${min_idx_wave}_${spk_src}.scp
    find ${outdir} -name "*.wav" > ${scp}

    # restore noise shaping
    ${train_cmd} --num-threads ${n_jobs} \
        ${expdir_wave}/${log}/noise_shaping_restore_${min_idx_wave}_${spk_src}.log \
        noise_shaping_emph.py \
            --waveforms ${scp} \
            --writedir ${outdir}_restored \
            --alpha ${alpha} \
            --fs ${fs} \
            --inv true \
            --n_jobs ${n_jobs}
fi
# }}}
done
fi


# STAGE a {{{
if [ `echo ${stage} | grep a` ] || [ `echo ${stage} | grep b` ];then
for spk_src in ${spks_src_dec[@]};do
for spk_trg in ${spks_trg_dec[@]};do
        if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_16bit" ] \
            || [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_9bit" ] \
                || [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf" ];then
            outdir=${expdir_wave}/${mdl_name_post}-${mdl_name}-${mdl_name_wave}-${data_name}_dev-${hidden_units_wave}-${epoch_count}-${epoch_count_wave}-${lpc}-${n_bands}-${batch_size_wave}-${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}
            #outdir=${expdir_wave}/${mdl_name_post}-${mdl_name}-${mdl_name_wave}-${data_name}_tst-${hidden_units_wave}-${epoch_count}-${epoch_count_wave}-${lpc}-${n_bands}-${batch_size_wave}-${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}
        fi
if [ `echo ${stage} | grep a` ];then
        echo $spk_src $spk_trg $min_idx_cycvae $min_idx $min_idx_wave
        echo $data_name $mdl_name $mdl_name_post $mdl_name_wave
        echo $outdir
        echo "###################################################"
        echo "#               DECODING STEP                     #"
        echo "###################################################"
        echo ${setting_wave}
        checkpoint=${expdir_wave}/checkpoint-${min_idx_wave}.pkl
        config=${expdir_wave}/model.conf

        feats=data/${dev}/feats_cv_${spk_src}-${spk_trg}.scp
        #feats=data/${tst}/feats_cv_${spk_src}-${spk_trg}.scp

        feats_scp=${expdir_wave}/feats_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.scp
        cat $feats | grep "\/${spk_src}-${spk_trg}\/" > ${feats_scp}

        # decode
        if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf" ]; then
            #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
            ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
                decode_wavernn_dualgru_compact_lpc_mband_10bit_cf.py \
                    --feats ${feats_scp} \
                    --outdir ${outdir} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --GPU_device_str ${GPU_device_str}
        elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_9bit" ]; then
            #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
            ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
                decode_wavernn_dualgru_compact_lpc_mband_9bit.py \
                    --feats ${feats_scp} \
                    --outdir ${outdir} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --GPU_device_str ${GPU_device_str}
        elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_16bit" ]; then
            #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
            ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
                decode_wavernn_dualgru_compact_lpc_mband_16bit.py \
                    --feats ${feats_scp} \
                    --outdir ${outdir} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --string_path ${string_path_cv} \
                    --GPU_device_str ${GPU_device_str}
        fi
fi
# }}}


# STAGE b {{{
if [ `echo ${stage} | grep b` ];then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    scp=${expdir_wave}/wav_generated_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.scp
    find ${outdir} -name "*.wav" > ${scp}

    # restore noise shaping
    ${train_cmd} --num-threads ${n_jobs} \
        ${expdir_wave}/${log}/noise_shaping_restore_${min_idx_cycvae}-${min_idx}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
        noise_shaping_emph.py \
            --waveforms ${scp} \
            --writedir ${outdir}_restored \
            --alpha ${alpha} \
            --fs ${fs} \
            --inv true \
            --n_jobs ${n_jobs}
fi
# }}}
done
done
fi

