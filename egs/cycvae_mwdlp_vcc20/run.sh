#!/bin/bash
###################################################################################################
#        SCRIPT FOR Low-latency multispeaker VC with Cyclic Variational Autoencoder (CycleVAE)    #
#        and Multiband WaveRNN using Data-driven Linear Prediction (MWDLP)                             #
###################################################################################################

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# USER SETTINGS {{{
#######################################
#           STAGE SETTING             #
#######################################
# {{{
# 0: data preparation step
# init: feature extraction w/ initial speaker conf. for f0 and power histogram calc. to obtain proper speaker conf.
# 1: feature extraction step
# 2: feature statistics calculation step
# 3: apply noise shaping [pre-emphasis] and multiband processing
# 4: vc training step
# 5: mwdlp training step
# 6: fine-tune vc with fixed mwdlp
# 7: fine-tune decoder vc
# 8: copy-synthesis mwdlp using gpu
# 9: restore noise-shaping copy-synthesis mwdlp
# a: decode vc step using gpu
# b: synthesis vc with mwdlp using gpu
# c: restore-noise shaping vc synthesis with mwdlp
# d: decode fine-tune vc using gpu
# e: synthesis fine-tune vc with mwdlp using gpu
# f: restore-noise shaping fine-tune vc synthesis with mwdlp
# g: decode fine-tune vc decoder using gpu
# h: synthesis fine-tune vc decoder with mwdlp using gpu
# j: restore-noise shaping fine-tune vc decoder synthesis with mwdlp
# }}}
#stage=0
#stage=init
#stage=0init
stage=0init123
#stage=init123
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
#stage=4567
#stage=4
#stage=567
#stage=5
#stage=6
#stage=7
#stage=8
#stage=89
#stage=9
#stage=a
#stage=bc
#stage=abc
#stage=b
#stage=c
#stage=d
#stage=ef
#stage=def
#stage=e
#stage=f
#stage=g
#stage=hj
#stage=ghj
#stage=h
#stage=j

##number of parallel jobs in feature extraction / noise-shaping & pqmf proc. / statistics calculation
n_jobs=1
n_jobs=5
#n_jobs=10
#n_jobs=20
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

## fs: sampling rate
fs=`awk '{if ($1 == "fs:") print $2}' conf/config.yml`
## shiftms: frame shift in ms
shiftms=`awk '{if ($1 == "shiftms:") print $2}' conf/config.yml`
## upsampling_factor: upsampling factor for neural vocoder
upsampling_factor=`echo "${shiftms} * ${fs} / 1000" | bc`

if [ $shiftms -eq 10 ]; then
    ## for using skewed input convolution (> 0), or balanced (0) [only for encoder]
    ## lookup frame limited to only 1/2 frame [for allowing low-latency/real-time processing]
    right_size_enc=1
    right_size_wave=1
    batch_size_wave=6
    batch_size=30
elif [ $shiftms -eq 5 ]; then
    right_size_enc=2
    right_size_wave=2
    batch_size_wave=12
    batch_size=60
else
    echo "shift ms not available"
    exit 1
fi

#spks_open=(p237 p245 p276)
spks=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
data_name=vcc20_${shiftms}ms

# uv-f0 and log-f0 occupied the first two dimensions,
# then uv-codeap, log-negative-codeap and mel-ceps
## [uv-f0,log-f0,uv-codeap,log-negative-codeap,mel-ceps]
## fftl: length of FFT window analysis
## WORLD F0_floor for cheaptrick: 3.0 * fs / (fft_size - 3.0)
## [https://github.com/mmorise/World/blob/master/src/cheaptrick.cpp] line 197
## mcep_alpha: frequency warping parameter for mel-cepstrum
## n_bands: number of bands for multiband modeling [a minimum of 4 kHz per band for proper modeling]
if [ $fs -eq 22050 ]; then
    wav_org_dir=wav_22kHz
    data_name=${data_name}_22kHz
    mcep_alpha=0.455 #22.05k ## frequency warping based on pysptk.util.mcepalpha
    fftl=2048
    if [ $shiftms -eq 5 ]; then
        shiftms=4.9886621315192743764172335600907 #22.05k rounding 110/22050 5ms shift
    elif [ $shiftms -eq 10 ]; then
        shiftms=9.9773242630385487528344671201814 #22.05k rounding 220/22050 10ms shift
    fi
    full_excit_dim=5
    n_bands=5
elif [ $fs -eq 24000 ]; then
    wav_org_dir=wav_24kHz
    data_name=${data_name}_24kHz
    mcep_alpha=0.466 #24k
    fftl=2048
    full_excit_dim=6
    n_bands=6
elif [ $fs -eq 48000 ]; then
    wav_org_dir=wav_48kHz
    data_name=${data_name}_48kHz
    mcep_alpha=0.554 #48k
    fftl=4096
    full_excit_dim=8
    n_bands=12
elif [ $fs -eq 44100 ]; then
    wav_org_dir=wav_44kHz
    data_name=${data_name}_44kHz
    mcep_alpha=0.544 #44.1k
    fftl=4096
    if [ $shiftms -eq 5 ]; then
        shiftms=4.9886621315192743764172335600907 #44.1k rounding 220/44100 5ms shift
    elif [ $shiftms -eq 10 ]; then
        shiftms=9.9773242630385487528344671201814 #44.1k rounding 440/44100 10ms shift
    fi
    full_excit_dim=8
    n_bands=10
elif [ $fs -eq 16000 ]; then
    wav_org_dir=wav_16kHz
    data_name=${data_name}_16kHz
    mcep_alpha=0.41000000000000003 #16k
    fftl=1024
    full_excit_dim=4
    n_bands=4
elif [ $fs -eq 8000 ]; then
    wav_org_dir=wav_8kHz
    data_name=${data_name}_8kHz
    mcep_alpha=0.312 #8k
    fftl=1024
    full_excit_dim=4
    n_bands=2
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

trn=tr_${data_name}
dev=dv_${data_name}
tst=ts_${data_name}

GPU_device=0
GPU_device=1
GPU_device=2
#GPU_device=3
#GPU_device=4
GPU_device=5
#GPU_device=6
#GPU_device=7
GPU_device=8
#GPU_device=9

string_path="/log_1pmelmagsp"

lr=`awk '{if ($1 == "lr:") print $2}' conf/config.yml`
do_prob=`awk '{if ($1 == "do_prob:") print $2}' conf/config.yml`
n_workers=`awk '{if ($1 == "n_workers:") print $2}' conf/config.yml`

### settings for VC network
step_count=`awk '{if ($1 == "step_count:") print $2}' conf/config.yml`
mdl_name_vc=`awk '{if ($1 == "mdl_name_vc:") print $2}' conf/config.yml`
mdl_name_ft=`awk '{if ($1 == "mdl_name_ft:") print $2}' conf/config.yml`
mdl_name_sp=`awk '{if ($1 == "mdl_name_sp:") print $2}' conf/config.yml`
n_half_cyc=`awk '{if ($1 == "n_half_cyc:") print $2}' conf/config.yml`
lat_dim=`awk '{if ($1 == "lat_dim:") print $2}' conf/config.yml`
lat_dim_e=`awk '{if ($1 == "lat_dim_e:") print $2}' conf/config.yml`
hidden_units_enc=`awk '{if ($1 == "hidden_units_enc:") print $2}' conf/config.yml`
hidden_layers_enc=`awk '{if ($1 == "hidden_layers_enc:") print $2}' conf/config.yml`
hidden_units_dec=`awk '{if ($1 == "hidden_units_dec:") print $2}' conf/config.yml`
hidden_layers_dec=`awk '{if ($1 == "hidden_layers_dec:") print $2}' conf/config.yml`
hidden_units_lf0=`awk '{if ($1 == "hidden_units_lf0:") print $2}' conf/config.yml`
hidden_layers_lf0=`awk '{if ($1 == "hidden_layers_lf0:") print $2}' conf/config.yml`
kernel_size_enc=`awk '{if ($1 == "kernel_size_enc:") print $2}' conf/config.yml`
dilation_size_enc=`awk '{if ($1 == "dilation_size_enc:") print $2}' conf/config.yml`
kernel_size_dec=`awk '{if ($1 == "kernel_size_dec:") print $2}' conf/config.yml`
dilation_size_dec=`awk '{if ($1 == "dilation_size_dec:") print $2}' conf/config.yml`
kernel_size_lf0=`awk '{if ($1 == "kernel_size_lf0:") print $2}' conf/config.yml`
dilation_size_lf0=`awk '{if ($1 == "dilation_size_lf0:") print $2}' conf/config.yml`
causal_conv_enc=`awk '{if ($1 == "causal_conv_enc:") print $2}' conf/config.yml`
causal_conv_dec=`awk '{if ($1 == "causal_conv_dec:") print $2}' conf/config.yml`
causal_conv_lf0=`awk '{if ($1 == "causal_conv_lf0:") print $2}' conf/config.yml`
spkidtr_dim=`awk '{if ($1 == "spkidtr_dim:") print $2}' conf/config.yml`
n_weight_emb=`awk '{if ($1 == "n_weight_emb:") print $2}' conf/config.yml`
right_size_dec=`awk '{if ($1 == "right_size_dec:") print $2}' conf/config.yml`
right_size_lf0=`awk '{if ($1 == "right_size_lf0:") print $2}' conf/config.yml`
s_conv_flag=`awk '{if ($1 == "s_conv_flag:") print $2}' conf/config.yml`
seg_conv_flag=`awk '{if ($1 == "seg_conv_flag:") print $2}' conf/config.yml`
t_start_cycvae=`awk '{if ($1 == "t_start_cycvae:") print $2}' conf/config.yml`
t_end_cycvae=`awk '{if ($1 == "t_end_cycvae:") print $2}' conf/config.yml`
interval_cycvae=`awk '{if ($1 == "interval_cycvae:") print $2}' conf/config.yml`
densities_cycvae_enc=`awk '{if ($1 == "densities_cycvae_enc:") print $2}' conf/config.yml`
densities_cycvae_dec=`awk '{if ($1 == "densities_cycvae_dec:") print $2}' conf/config.yml`
n_stage_cycvae=`awk '{if ($1 == "n_stage_cycvae:") print $2}' conf/config.yml`

### settings for neural vocoder
step_count_wave=`awk '{if ($1 == "step_count_wave:") print $2}' conf/config.yml`
mdl_name_wave=`awk '{if ($1 == "mdl_name_wave:") print $2}' conf/config.yml`
hidden_units_wave=`awk '{if ($1 == "hidden_units_wave:") print $2}' conf/config.yml`
hidden_units_wave_2=`awk '{if ($1 == "hidden_units_wave_2:") print $2}' conf/config.yml`
kernel_size_wave=`awk '{if ($1 == "kernel_size_wave:") print $2}' conf/config.yml`
dilation_size_wave=`awk '{if ($1 == "dilation_size_wave:") print $2}' conf/config.yml`
kernel_size=`awk '{if ($1 == "kernel_size:") print $2}' conf/config.yml`
hid_chn=`awk '{if ($1 == "hid_chn:") print $2}' conf/config.yml`
skip_chn=`awk '{if ($1 == "skip_chn:") print $2}' conf/config.yml`
dilation_depth=`awk '{if ($1 == "dilation_depth:") print $2}' conf/config.yml`
dilation_repeat=`awk '{if ($1 == "dilation_repeat:") print $2}' conf/config.yml`
t_start=`awk '{if ($1 == "t_start:") print $2}' conf/config.yml`
t_end=`awk '{if ($1 == "t_end:") print $2}' conf/config.yml`
interval=`awk '{if ($1 == "interval:") print $2}' conf/config.yml`
densities=`awk '{if ($1 == "densities:") print $2}' conf/config.yml`
n_stage=`awk '{if ($1 == "n_stage:") print $2}' conf/config.yml`
lpc=`awk '{if ($1 == "lpc:") print $2}' conf/config.yml`
causal_conv_wave=`awk '{if ($1 == "causal_conv_wave:") print $2}' conf/config.yml`
seg_conv_flag_wave=`awk '{if ($1 == "seg_conv_flag_wave:") print $2}' conf/config.yml`
s_dim=`awk '{if ($1 == "s_dim:") print $2}' conf/config.yml`
mid_dim=`awk '{if ($1 == "mid_dim:") print $2}' conf/config.yml`


#######################################
#          DECODING SETTING           #
#######################################

### Set GPU_device_str and n_gpus for GPU decoding with synchronized values
GPU_device_str="0"
GPU_device_str="4,6"
#GPU_device_str="0,1,2"
GPU_device_str="0,5,2,7,6"
GPU_device_str="4,8,2,1,0"
GPU_device_str="4,5,0,7,6"
GPU_device_str="4,5,6,8,9"
GPU_device_str="0,1,2,4,5"
#GPU_device_str="0,1,2,3,7"
#GPU_device_str="9,7,6,5,3"
#GPU_device_str="4,5,6"
#GPU_device_str="4"

n_gpus=1
#n_gpus=3
#n_gpus=5
###

### This is for VC with griffin-lim or neural vocoder synthesizer
spks_src_dec=(SEM1 SEF2 SEM2 SEF1)
spks_src_dec=(SEM1 SEF1)
spks_src_dec=(SEF1)
spks_src_dec=(SEF2)
spks_trg_dec=(TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
spks_trg_dec=(TEF1 TEM2)
spks_trg_dec=(TEF1)
spks_trg_dec=(TEM2)
###

### This is for speakers that will be used in analysis-synthesis
spks_dec=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
spks_dec=(TEM2 SEF2 TFF1 TFM1 TMF1)
spks_dec=(TEM2)
#spks_dec=(SEF2)
###

### This is the maximum number of waveforms to be decoded per speaker
#n_wav_decode=1
#n_wav_decode=5
n_wav_decode=10
#n_wav_decode=50
###

## This is for number of batch sequence when decoding using GPU
#decode_batch_size=1
#decode_batch_size=2
#decode_batch_size=3
#decode_batch_size=5
#decode_batch_size=9
decode_batch_size=10
#decode_batch_size=13
#decode_batch_size=15
#decode_batch_size=17


# parse options
. parse_options.sh

#echo $mdl_name_vc $data_name

echo $GPU_device $GPU_device_str

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
    touch conf/spkr.yml
    if true; then
    #if false; then
    for spk in ${spks[@]};do
        #n_tr=`find ${wav_org_dir}/train/${spk} -name "*.wav" | wc -l`
        #n_dv=`find ${wav_org_dir}/dev/${spk} -name "*.wav" | wc -l`
        #n_ts=`find ${wav_org_dir}/test/${spk} -name "*.wav" | wc -l`
        #echo $spk $n_tr $n_dv $n_ts
        #find ${wav_org_dir}/train/${spk} -name "*.wav" | sort >> data/${trn}/wav.scp
        #find ${wav_org_dir}/dev/${spk} -name "*.wav" | sort >> data/${dev}/wav.scp
        #find ${wav_org_dir}/test/${spk} -name "*.wav" | sort >> data/${tst}/wav.scp
        if [ -n "$(echo $spk | sed -n 's/\(SE\)/\1/p')" ]; then
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
        else
            echo error, ${spk}
            exit
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
                echo "minF0 and maxF0 of ${spk} is initialized, please run stage init to obtain the proper config."
            fi
            if [ -f "conf/${spk}.pow" ]; then
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
                yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                echo "npow of ${spk} is initialized from .pow file"
            else
                yq -yi ".${spk}.npow=-25" conf/spkr.yml
                echo "npow of ${spk} is initialized, please run stage init to obtain the proper config."
            fi
        else
            if [ -f "conf/${spk}.f0" ]; then
                minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
            fi
            if [ -f "conf/${spk}.pow" ]; then
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
            fi
            tmp=`yq ".${spk}.minf0" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.f0" ]; then
                    yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
                    echo "minF0 of ${spk} is initialized from .f0 file"
                else
                    yq -yi ".${spk}.minf0=40" conf/spkr.yml
                    echo "minF0 of ${spk} is initialized, please run stage init to obtain the proper config."
                fi
            elif [[ $tmp -ne $minf0 ]]; then
                yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
                echo "minF0 of ${spk} is changed based on .f0 file"
            fi
            tmp=`yq ".${spk}.maxf0" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.f0" ]; then
                    yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
                    echo "maxF0 of ${spk} is initialized from .f0 file"
                else
                    yq -yi ".${spk}.maxf0=700" conf/spkr.yml
                    echo "maxF0 of ${spk} is initialized, please run stage init to obtain the proper config."
                fi
            elif [[ $tmp -ne $maxf0 ]]; then
                yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
                echo "maxF0 of ${spk} is changed based on .f0 file"
            fi
            tmp=`yq ".${spk}.npow" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                if [ -f "conf/${spk}.pow" ]; then
                    yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                    echo "npow of ${spk} is initialized from .pow file"
                else
                    yq -yi ".${spk}.npow=-25" conf/spkr.yml
                    echo "npow of ${spk} is initialized, please run stage init to get the proper config."
                fi
            elif [[ "$tmp" != "$pow" ]]; then
                yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                echo "npow of ${spk} is changed based on .pow file"
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
        echo open $spk
        #find ${wav_org_dir}_unseen/test/${spk} -name "*.wav" | sort >> data/${tst}/wav.scp
        if printf '%s\0' "${spks_open[@]}" | grep -xq --null "$spk";then
            echo open $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 23 >> data/${tst}/wav.scp
                #| sort | head -n +50 >> data/${tst}/wav.scp
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
                echo "minF0 and maxF0 of ${spk} is initialized, please run stage init with using spks_open list"
            fi
            if [ -f "conf/${spk}.pow" ]; then
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
                yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                echo "npow of ${spk} is initialized from .pow file"
            else
                yq -yi ".${spk}.npow=-25" conf/spkr.yml
                echo "npow of ${spk} is initialized, please run stage init with using spks_open list"
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
                    echo "minF0 of ${spk} is initialized, please run stage init with using spks_open list"
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
                    echo "maxF0 of ${spk} is initialized, please run stage init with using spks_open list"
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
                    echo "npow of ${spk} is initialized, please run stage init with using spks_open list"
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
        #for set in ${trn} ${dev};do
        for set in ${trn};do
            echo $set
            expdir=exp/feature_extract_init/${set}
            mkdir -p $expdir
            for spk in ${spks[@]}; do
            #for spk in ${spks_open[@]}; do
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
                    n_feats=`find hdf5_init/${set}/${spk} -name "*.h5" | wc -l`
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
    set +e
    #for set in ${trn} ${dev};do
    for set in ${trn};do
        echo $set
        find hdf5_init/${set} -name "*.h5" | sort > tmp2
        rm -f data/${set}/feats_init.scp
        for spk in ${spks[@]}; do
        #for spk in ${spks_open[@]}; do
            cat tmp2 | grep "\/${spk}\/" >> data/${set}/feats_init.scp
        done
        rm -f tmp2
    done
    set -e
    echo "###########################################################"
    echo "#              SPEAKER HISTOGRAM CALC. STEP               #"
    echo "###########################################################"
    expdir=exp/init_spk_stat/${trn}
    mkdir -p $expdir
    if true; then
    #if false; then
        rm -f $expdir/spk_stat.log
        for spk in ${spks[@]};do
        #for spk in ${spks_open[@]};do
            echo $spk
            cat data/${trn}/feats_init.scp | grep \/${spk}\/ > data/${trn}/feats_init_spk-${spk}.scp
            ${train_cmd} exp/init_spk_stat/init_stat_${data_name}_spk-${spk}.log \
                spk_stat.py \
                    --expdir ${expdir} \
                    --feats data/${trn}/feats_init_spk-${spk}.scp
        done
        echo "spk histograms are successfully calculated"
    fi
    echo "###########################################################"
    echo "#             F0 RANGE and MIN. POW CALC. STEP            #"
    echo "###########################################################"
    featdir=${expdir}
    expdir=exp/spk_conf/${trn}
    mkdir -p $expdir
    if true; then
    #if false; then
        rm -f $expdir/f0_range.log
        spk_list="$(IFS="@"; echo "${spks[*]}")"
        #spk_list="$(IFS="@"; echo "${spks_open[*]}")"
        echo ${spk_list}
        ${train_cmd} exp/spk_conf/f0_range_${data_name}.log \
            f0_range.py \
                --expdir ${expdir} \
                --featdir ${featdir} \
                --confdir conf/ \
                --spk_list ${spk_list}
        echo "f0 range spk confs. are successfully calculated"
        rm -f $expdir/min_pow.log
        ${train_cmd} exp/spk_conf/min_pow_${data_name}.log \
            min_pow.py \
                --expdir ${expdir} \
                --featdir ${featdir} \
                --confdir conf/ \
                --spk_list ${spk_list}
        echo "min. pow spk confs. are successfully calculated"
    fi
    echo "###########################################################"
    echo "#     PUTTING PROPER F0 RANGE and MIN. POW CONFS. STEP    #"
    echo "###########################################################"
    if true; then
    #if false; then
    set +e
    touch conf/spkr.yml
    for spk in ${spks[@]};do
    #for spk in ${spks_open[@]};do
        echo $spk
        tmp=`yq ".${spk}" conf/spkr.yml`
        if [[ -z $tmp ]] || [[ $tmp == "null" ]]; then
            echo $spk: >> conf/spkr.yml
            minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
            maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
            echo $minf0 $maxf0
            yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
            yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
            echo "minF0 and maxF0 of ${spk} is initialized from .f0 file"
            pow=`cat conf/${spk}.pow | awk '{print $1}'`
            echo $pow
            yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
            echo "npow of ${spk} is initialized from .pow file"
        else
            if [ -f "conf/${spk}.f0" ]; then
                minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
            fi
            echo $minf0 $maxf0
            if [ -f "conf/${spk}.pow" ]; then
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
            fi
            echo $pow
            tmp=`yq ".${spk}.minf0" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
                echo "minF0 of ${spk} is initialized from .f0 file"
            elif [[ $tmp -ne $minf0 ]]; then
                yq -yi ".${spk}.minf0=${minf0}" conf/spkr.yml
                echo "minF0 of ${spk} is changed based on .f0 file"
            fi
            tmp=`yq ".${spk}.maxf0" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
                echo "maxF0 of ${spk} is initialized from .f0 file"
            elif [[ $tmp -ne $maxf0 ]]; then
                yq -yi ".${spk}.maxf0=${maxf0}" conf/spkr.yml
                echo "maxF0 of ${spk} is changed based on .f0 file"
            fi
            tmp=`yq ".${spk}.npow" conf/spkr.yml`
            if [[ $tmp == "null" ]]; then
                yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                echo "npow of ${spk} is initialized from .pow file"
            elif [[ "$tmp" != "$pow" ]]; then
                yq -yi ".${spk}.npow=${pow}" conf/spkr.yml
                echo "npow of ${spk} is changed based on .pow file"
            fi
        fi
    done
    set -e
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
        for set in ${trn} ${dev} ${tst};do
        #for set in ${tst};do
            if [ -f "data/${set}/wav.scp" ]; then
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
            fi
        done
    fi
    # make scp for feats
    set +e
    rm -f data/${trn}/feats_all.scp
    #for set in ${trn} ${dev};do
    for set in ${trn} ${dev} ${tst};do
    #for set in ${tst};do
        if [ -f "data/${set}/wav.scp" ]; then
            echo $set
            find hdf5/${set} -name "*.h5" | sort > tmp2
            find wav_filtered/${set} -name "*.wav" | sort > tmp3
            rm -f data/${set}/feats.scp data/${set}/wav_filtered.scp
            for spk in ${spks[@]}; do
                cat tmp2 | grep "\/${spk}\/" | sort >> data/${set}/feats.scp
                cat tmp3 | grep "\/${spk}\/" | sort >> data/${set}/wav_filtered.scp
                echo $set $spk
            done
            rm -f tmp2 tmp3
        fi
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
            set +e
            cat data/${trn}/feats.scp | grep \/${spk}\/ > data/${trn}/feats_spk-${spk}.scp
            cat data/${dev}/feats.scp | grep \/${spk}\/ > data/${dev}/feats_spk-${spk}.scp
            if [ -f "data/${tst}/wav.scp" ]; then
                cat data/${tst}/feats.scp | grep \/${spk}\/ > data/${tst}/feats_spk-${spk}.scp
            fi
            set -e
            ${train_cmd} exp/calculate_statistics/calc_stats_${trn}_spk-${spk}.log \
                calc_stats.py \
                    --expdir ${expdir} \
                    --feats data/${trn}/feats_spk-${spk}.scp \
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
                --feats data/${trn}/feats.scp \
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
            set +e
            n_wavs=`cat data/${set}/wav_filtered.scp | wc -l`
            n_ns=`find wav_ns/${set} -name "*.wav" | wc -l`
            echo "${n_ns}/${n_wavs} files are successfully processed [emph]."

            # make scp files
            find wav_ns/${set} -name "*.wav" | sort > tmp
            rm -f data/${set}/wav_ns.scp
            for spk in ${spks[@]}; do
                cat tmp | grep "\/${spk}\/" >> data/${set}/wav_ns.scp
                echo $set $spk
            done
            rm -f tmp
            set -e
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
            set +e
            n_wavs=`cat data/${set}/wav_ns.scp | wc -l`
            n_ns=`find wav_ns_pqmf_${n_bands}/${set} -name "*.wav" | wc -l`
            echo "${n_ns}/${n_wavs} files are successfully processed [emph pqmf ${n_bands}-bands]."

            # make scp files
            find wav_ns_pqmf_${n_bands}/${set} -name "*.wav" | sort > tmp
            rm -f data/${set}/wav_ns_pqmf_${n_bands}.scp
            for spk in ${spks[@]}; do
                cat tmp | grep "\/${spk}\/" >> data/${set}/wav_ns_pqmf_${n_bands}.scp
                echo $set $spk
            done
            rm -f tmp
            set -e
        done
    fi
fi
# }}}


if [ -d "exp/feature_extract/tr_${data_name}" ]; then
    tmp=`yq ".${data_name}" conf/spkr.yml`
    if [[ -z $tmp ]] || [[ $tmp == "null" ]]; then
        echo ${data_name}: >> conf/spkr.yml
    fi
    pad_len=`yq ".${data_name}.pad_len" conf/spkr.yml`
    if [[ $pad_len == "null" ]]; then
        max_frame=0
        max_spk=""
        for spk in ${spks[*]}; do
        if [ -f "exp/feature_extract/tr_${data_name}/feature_extract_${spk}.log" ]; then
            echo $spk tr
            max_frame_spk=`awk '{if ($1 == "max_frame:") print $2}' exp/feature_extract/tr_${data_name}/feature_extract_${spk}.log`
            echo $max_frame_spk
            if [[ $max_frame_spk -gt $max_frame ]]; then
                max_frame=$max_frame_spk
                max_spk=$spk
            fi 
            if [ -f "exp/feature_extract/dv_${data_name}/feature_extract_${spk}.log" ]; then
                echo $spk dv
                max_frame_spk=`awk '{if ($1 == "max_frame:") print $2}' exp/feature_extract/dv_${data_name}/feature_extract_${spk}.log`
                echo $max_frame_spk
                if [[ $max_frame_spk -gt $max_frame ]]; then
                    max_frame=$max_frame_spk
                    max_spk=$spk
                fi 
            fi
        else
            echo exp/feature_extract/tr_${data_name}/feature_extract_${spk}.log does not exist, please run stage 1 for feature extraction on speaker ${spk} and frame length checking
            exit
        fi
        done
        echo $max_spk $max_frame
        pad_len=$max_frame
        yq -yi ".${data_name}.pad_len=${pad_len}" conf/spkr.yml
    fi
    echo $pad_len
else
    echo exp/feature_extract/tr_${data_name} does not exist, please run stage 1 for feature extraction and frame length checking
    exit
fi


if [ $mdl_name_vc == "cycmelspxlf0capspkvae-gauss-smpl_sparse_weightemb_v2" ]; then
    setting_vc=${mdl_name_vc}_${data_name}_lr${lr}_bs${batch_size}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hud${hidden_units_dec}_huf${hidden_units_lf0}_do${do_prob}_st${step_count}_mel${mel_dim}_nhcyc${n_half_cyc}_s${spkidtr_dim}_w${n_weight_emb}_ts${t_start_cycvae}_te${t_end_cycvae}_i${interval_cycvae}_de${densities_cycvae_enc}_dd${densities_cycvae_dec}_ns${n_stage_cycvae}_sc${s_conv_flag}_ss${seg_conv_flag}
fi


# STAGE 4 {{
# set variables
expdir_vc=exp/tr_${setting_vc}
if [ `echo ${stage} | grep 4` ];then
    echo $mdl_name_vc
    mkdir -p $expdir_vc
    echo "###########################################################"
    echo "#               FEATURE MODELING STEP                     #"
    echo "###########################################################"

    stats_list=()
    feats_eval_list=()
    for spk in ${spks[@]};do
        stats_list+=(data/${trn}/stats_spk-${spk}.h5)
        feats_eval_list+=(data/${dev}/feats_spk-${spk}.scp)
    done
    
    stats_list_list="$(IFS="@"; echo "${stats_list[*]}")"
    feats_list_eval_list="$(IFS="@"; echo "${feats_eval_list[*]}")"
    
    spk_list="$(IFS="@"; echo "${spks[*]}")"
    echo ${spk_list}

    echo $expdir_vc

    if [ -f "${expdir_vc}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_vc}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_vc} \
                --confdir conf/${data_name}_vc
        idx_resume_cycvae=`cat conf/${data_name}_vc.idx | awk '{print $1}'`
        min_idx_cycvae=`cat conf/${data_name}_vc.idx | awk '{print $2}'`
        echo "${data_name}: idx_resume_cycvae=${idx_resume_cycvae}, min_idx_cycvae=${min_idx_cycvae}"
    else
        idx_resume_cycvae=0
    fi

    n_spk=${#spks[@]}
    n_tr_sum=13800
    n_tr=`expr $n_tr_sum / ${n_spk}`
    if [ `expr $n_tr_sum % ${n_spk}` -gt 0 ]; then
        n_tr=`expr ${n_tr} + 1`
    fi
    echo $n_tr
    if true; then
        feats_sort_list=data/${trn}/feats_sort.scp
        wav_sort_list=data/${trn}/wav_ns_sort.scp
        if [ ! -f ${wav_sort_list} ] || [ ! -f ${feats_sort_list}  ]; then
        #if true; then
            ${cuda_cmd} ${expdir_vc}/log/get_max_frame.log \
                sort_frame_list.py \
                    --feats data/${trn}/feats.scp \
                    --waveforms data/${trn}/wav_ns.scp \
                    --spk_list ${spk_list} \
                    --expdir ${expdir_vc} \
                    --n_jobs ${n_jobs}
        fi
        feats=${expdir_vc}/feats_tr_cut.scp
        if [ ! -f ${feats} ]; then
        #if true; then
            rm -f ${feats}
            n_utt_spks=()
            sum_utt_spks=0
            flag_utt_spk_max=true
            count_utt_spk_gt_max=0
            for spk in ${spks[@]}; do
                n_utt_spk=`cat ${feats_sort_list} | grep "\/${spk}\/" | wc -l`
                if [ $n_utt_spk -gt $n_tr ]; then
                    sum_utt_spks=`expr $sum_utt_spks + $n_tr`
                    flag_utt_spk_max=false
                    count_utt_spk_gt_max=$(( $count_utt_spk_gt_max + 1 ))
                    n_utt_spks+=(${n_tr})
                    echo $spk $n_tr $sum_utt_spks
                else
                    sum_utt_spks=`expr $sum_utt_spks + $n_utt_spk`
                    n_utt_spks+=(${n_utt_spk})
                    echo $spk $n_utt_spk $n_tr $sum_utt_spks
                fi
            done
            if [ $sum_utt_spks -lt $n_tr_sum ] && ! ${flag_utt_spk_max} ; then
                flag=false
                rem_sum=$(( $n_tr_sum - $sum_utt_spks ))
                rem_sum_spk=$(( $rem_sum / $count_utt_spk_gt_max ))
                n_tr=$(( $n_tr + $rem_sum_spk ))
            else
                flag=true
            fi
            while ! $flag; do
                n_utt_spks=()
                sum_utt_spks=0
                flag_utt_spk_max=true
                count_utt_spk_gt_max=0
                for spk in ${spks[@]}; do
                    n_utt_spk=`cat ${feats_sort_list} | grep "\/${spk}\/" | wc -l`
                    if [ $n_utt_spk -gt $n_tr ]; then
                        sum_utt_spks=`expr $sum_utt_spks + $n_tr`
                        flag_utt_spk_max=false
                        count_utt_spk_gt_max=$(( $count_utt_spk_gt_max + 1 ))
                        n_utt_spks+=(${n_tr})
                        echo $spk $n_tr $sum_utt_spks
                    else
                        sum_utt_spks=`expr $sum_utt_spks + $n_utt_spk`
                        n_utt_spks+=(${n_utt_spk})
                        echo $spk $n_utt_spk $n_tr $sum_utt_spks
                    fi
                done
                if [ $sum_utt_spks -lt $n_tr_sum ] && ! ${flag_utt_spk_max} ; then
                    flag=false
                    rem_sum=$(( $n_tr_sum - $sum_utt_spks ))
                    rem_sum_spk=$(( $rem_sum / $count_utt_spk_gt_max ))
                    if [ $rem_sum_spk -eq 0 ]; then
                        rem_sum_spk=1
                    fi
                    n_tr=$(( $n_tr + $rem_sum_spk ))
                else
                    flag=true
                fi
            done
            idx_utt_spk=0
            for spk in ${spks[@]}; do
                n_utt_spk=${n_utt_spks[${idx_utt_spk}]}
                echo tr $spk $n_utt_spk
                cat ${feats_sort_list} | grep "\/${spk}\/" | head -n ${n_utt_spk} | sort >> ${feats}
                idx_utt_spk=$(( $idx_utt_spk + 1 ))
            done
        fi
    fi

    if [ $mdl_name_vc == "cycmelspxlf0capspkvae-gauss-smpl_sparse_weightemb_v2" ];then
        if [ $idx_resume_cycvae -gt 0 ]; then
            echo ""
            echo "vc model is in training, please use less/vim to monitor the training log: ${expdir_vc}/log/train_resume-${idx_resume_cycvae}.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_vc}/log/train_resume-${idx_resume_cycvae}.log \
                train_sparse-gru-cycle-melsp-x-lf0cap-spk-vae-gauss-smpl_weightemb_v2.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_vc} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --step_count ${step_count} \
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
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --n_weight_emb ${n_weight_emb} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --right_size_lf0 ${right_size_lf0} \
                    --s_conv_flag ${s_conv_flag} \
                    --seg_conv_flag ${seg_conv_flag} \
                    --full_excit_dim ${full_excit_dim} \
                    --t_start ${t_start_cycvae} \
                    --t_end ${t_end_cycvae} \
                    --interval ${interval_cycvae} \
                    --densities_enc ${densities_cycvae_enc} \
                    --densities_dec ${densities_cycvae_dec} \
                    --n_stage ${n_stage_cycvae} \
                    --resume ${expdir_vc}/checkpoint-${idx_resume_cycvae}.pkl \
                    --GPU_device ${GPU_device}
        else
            echo ""
            echo "vc model is in training, please use less/vim to monitor the training log: ${expdir_vc}/log/train.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_vc}/log/train.log \
                train_sparse-gru-cycle-melsp-x-lf0cap-spk-vae-gauss-smpl_weightemb_v2.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_vc} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --step_count ${step_count} \
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
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --n_weight_emb ${n_weight_emb} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --right_size_lf0 ${right_size_lf0} \
                    --s_conv_flag ${s_conv_flag} \
                    --seg_conv_flag ${seg_conv_flag} \
                    --t_start ${t_start_cycvae} \
                    --t_end ${t_end_cycvae} \
                    --interval ${interval_cycvae} \
                    --densities_enc ${densities_cycvae_enc} \
                    --densities_dec ${densities_cycvae_dec} \
                    --n_stage ${n_stage_cycvae} \
                    --full_excit_dim ${full_excit_dim} \
                    --GPU_device ${GPU_device}
        fi
        echo ""
        echo "vc training finished, please check the log file, complete mwdlp training, and move to model fine-tuning"
        echo ""
    fi
fi
# }}}


if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb_v2" ]; then
    setting_wave=${mdl_name_wave}_${data_name}_lr${lr}_bs${batch_size_wave}_huw${hidden_units_wave}_hu2w${hidden_units_wave_2}_ksw${kernel_size_wave}_dsw${dilation_size_wave}_do${do_prob}_st${step_count_wave}_mel${mel_dim}_ts${t_start}_te${t_end}_i${interval}_d${densities}_ns${n_stage}_lpc${lpc}_rs${right_size_wave}_nb${n_bands}_s${s_dim}_m${mid_dim}_ss${seg_conv_flag_wave}
fi


# STAGE 5 {{
# set variables
expdir_wave=exp/tr_${setting_wave}
if [ `echo ${stage} | grep 5` ];then
    echo $mdl_name_wave
    mkdir -p $expdir_wave
    powmcep_dim=$mel_dim
    echo "###########################################################"
    echo "#               WAVEFORM MODELING STEP                    #"
    echo "###########################################################"
    echo $expdir_wave
   
    if [ -f "${expdir_wave}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_wave}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_wave} \
                --confdir conf/${data_name}_wave
        idx_resume_wave=`cat conf/${data_name}_wave.idx | awk '{print $1}'`
        min_idx_wave=`cat conf/${data_name}_wave.idx | awk '{print $2}'`
        echo "${data_name}: idx_resume_wave=${idx_resume_wave}, min_idx_wave=${min_idx_wave}"
    else
        idx_resume_wave=0
    fi

    # order of files on feats-wav_ns pair has to be the same
    feats_eval=data/${dev}/feats.scp
    waveforms_eval=data/${dev}/wav_ns.scp
    n_spk=${#spks[@]}
    n_tr_sum=15000
    n_tr=`expr $n_tr_sum / ${n_spk}`
    if [ `expr $n_tr_sum % ${n_spk}` -gt 0 ]; then
        n_tr=`expr ${n_tr} + 1`
    fi
    echo $n_tr
    if true; then
        feats_sort_list=data/${trn}/feats_sort.scp
        wav_sort_list=data/${trn}/wav_ns_sort.scp
        if [ ! -f ${wav_sort_list} ] || [ ! -f ${feats_sort_list}  ]; then
            spk_list="$(IFS="@"; echo "${spks[*]}")"
        #if true; then
            ${cuda_cmd} ${expdir_wave}/log/get_max_frame.log \
                sort_frame_list.py \
                    --feats data/${trn}/feats.scp \
                    --waveforms data/${trn}/wav_ns.scp \
                    --spk_list ${spk_list} \
                    --expdir ${expdir_wave} \
                    --n_jobs ${n_jobs}
        fi
        feats=${expdir_wave}/feats_tr_cut.scp
        waveforms=${expdir_wave}/wavs_tr_cut.scp
        if [ ! -f ${waveforms} ] || [ ! -f ${feats} ]; then
        #if true; then
            rm -f ${feats} ${waveforms}
            n_utt_spks=()
            sum_utt_spks=0
            flag_utt_spk_max=true
            count_utt_spk_gt_max=0
            for spk in ${spks[@]}; do
                n_utt_spk=`cat ${feats_sort_list} | grep "\/${spk}\/" | wc -l`
                if [ $n_utt_spk -gt $n_tr ]; then
                    sum_utt_spks=`expr $sum_utt_spks + $n_tr`
                    flag_utt_spk_max=false
                    count_utt_spk_gt_max=$(( $count_utt_spk_gt_max + 1 ))
                    n_utt_spks+=(${n_tr})
                    echo $spk $n_tr $sum_utt_spks
                else
                    sum_utt_spks=`expr $sum_utt_spks + $n_utt_spk`
                    n_utt_spks+=(${n_utt_spk})
                    echo $spk $n_utt_spk $n_tr $sum_utt_spks
                fi
            done
            if [ $sum_utt_spks -lt $n_tr_sum ] && ! ${flag_utt_spk_max} ; then
                flag=false
                rem_sum=$(( $n_tr_sum - $sum_utt_spks ))
                rem_sum_spk=$(( $rem_sum / $count_utt_spk_gt_max ))
                n_tr=$(( $n_tr + $rem_sum_spk ))
            else
                flag=true
            fi
            while ! $flag; do
                n_utt_spks=()
                sum_utt_spks=0
                flag_utt_spk_max=true
                count_utt_spk_gt_max=0
                for spk in ${spks[@]}; do
                    n_utt_spk=`cat ${feats_sort_list} | grep "\/${spk}\/" | wc -l`
                    if [ $n_utt_spk -gt $n_tr ]; then
                        sum_utt_spks=`expr $sum_utt_spks + $n_tr`
                        flag_utt_spk_max=false
                        count_utt_spk_gt_max=$(( $count_utt_spk_gt_max + 1 ))
                        n_utt_spks+=(${n_tr})
                        echo $spk $n_tr $sum_utt_spks
                    else
                        sum_utt_spks=`expr $sum_utt_spks + $n_utt_spk`
                        n_utt_spks+=(${n_utt_spk})
                        echo $spk $n_utt_spk $n_tr $sum_utt_spks
                    fi
                done
                if [ $sum_utt_spks -lt $n_tr_sum ] && ! ${flag_utt_spk_max} ; then
                    flag=false
                    rem_sum=$(( $n_tr_sum - $sum_utt_spks ))
                    rem_sum_spk=$(( $rem_sum / $count_utt_spk_gt_max ))
                    if [ $rem_sum_spk -eq 0 ]; then
                        rem_sum_spk=1
                    fi
                    n_tr=$(( $n_tr + $rem_sum_spk ))
                else
                    flag=true
                fi
            done
            idx_utt_spk=0
            for spk in ${spks[@]}; do
                n_utt_spk=${n_utt_spks[${idx_utt_spk}]}
                echo tr $spk $n_utt_spk
                cat ${feats_sort_list} | grep "\/${spk}\/" | head -n ${n_utt_spk} | sort >> ${feats}
                cat ${wav_sort_list} | grep "\/${spk}\/" | head -n ${n_utt_spk} | sort >> ${waveforms}
                idx_utt_spk=$(( $idx_utt_spk + 1 ))
            done
        fi
    fi

    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb_v2" ];then
        if [ $idx_resume_wave -gt 0 ]; then
            echo ""
            echo "mwdlp model is in training, please use less/vim to monitor the training log: ${expdir_wave}/log/train_resume-${idx_resume_wave}.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_wave}/log/train_resume-${idx_resume_wave}.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_10bit_cf_smpl_orgx_emb_v2.py \
                    --waveforms ${waveforms} \
                    --waveforms_eval $waveforms_eval \
                    --feats ${feats} \
                    --feats_eval $feats_eval \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_wave} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --step_count ${step_count_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --batch_size ${batch_size_wave} \
                    --mcep_dim ${powmcep_dim} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
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
                    --string_path ${string_path} \
                    --fs ${fs} \
                    --seg_conv_flag_wave ${seg_conv_flag_wave} \
                    --s_dim ${s_dim} \
                    --mid_dim ${mid_dim} \
                    --resume ${expdir_wave}/checkpoint-${idx_resume_wave}.pkl \
                    --GPU_device ${GPU_device}
        else
            echo ""
            echo "mwdlp model is in training, please use less/vim to monitor the training log: ${expdir_wave}/log/train.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_wave}/log/train.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_10bit_cf_smpl_orgx_emb_v2.py \
                    --waveforms ${waveforms} \
                    --waveforms_eval $waveforms_eval \
                    --feats ${feats} \
                    --feats_eval $feats_eval \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_wave} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --step_count ${step_count_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --batch_size ${batch_size_wave} \
                    --mcep_dim ${powmcep_dim} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
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
                    --string_path ${string_path} \
                    --fs ${fs} \
                    --seg_conv_flag_wave ${seg_conv_flag_wave} \
                    --s_dim ${s_dim} \
                    --mid_dim ${mid_dim} \
                    --GPU_device ${GPU_device}
        fi
        echo ""
        echo "mwdlp training finished, please check the log file, and move to vc model fine-tuning"
        echo ""
    fi
fi
# }}}


if [ `echo ${stage} | grep 8` ] || [ `echo ${stage} | grep 9` ]; then
    echo $expdir_wave
    if [ -f "${expdir_wave}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_wave}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_wave} \
                --confdir conf/${data_name}_wave
        min_idx_wave=`cat conf/${data_name}_wave.idx | awk '{print $2}'`
        echo "${data_name}: min_idx_wave=${min_idx_wave}"
    else
        echo "mwdlp checkpoints not found, please run mwdlp training step"
        exit
    fi
elif [ `echo ${stage} | grep 6` ] || [ `echo ${stage} | grep 7` ] \
    || [ `echo ${stage} | grep b` ] || [ `echo ${stage} | grep c` ] \
        || [ `echo ${stage} | grep d` ] || [ `echo ${stage} | grep e` ] || [ `echo ${stage} | grep f` ] \
        || [ `echo ${stage} | grep g` ] || [ `echo ${stage} | grep h` ] || [ `echo ${stage} | grep j` ]; then
    echo $expdir_vc
    if [ -f "${expdir_vc}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_vc}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_vc} \
                --confdir conf/${data_name}_vc
        min_idx_cycvae=`cat conf/${data_name}_vc.idx | awk '{print $2}'`
        echo "${data_name}: min_idx_cycvae=${min_idx_cycvae}"
        string_path_cv=/feat_cv_${mdl_name_vc}-${step_count}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}
    else
        echo "vc checkpoints not found, please run vc training step"
        exit
    fi
    echo $expdir_wave
    if [ -f "${expdir_wave}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_wave}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_wave} \
                --confdir conf/${data_name}_wave
        min_idx_wave=`cat conf/${data_name}_wave.idx | awk '{print $2}'`
        echo "${data_name}: min_idx_wave=${min_idx_wave}"
    else
        echo "mwdlp checkpoints not found, please run mwdlp training step"
        exit
    fi
elif [ `echo ${stage} | grep a` ]; then
    echo $expdir_vc
    if [ -f "${expdir_vc}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_vc}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_vc} \
                --confdir conf/${data_name}_vc
        min_idx_cycvae=`cat conf/${data_name}_vc.idx | awk '{print $2}'`
        echo "${data_name}: min_idx_cycvae=${min_idx_cycvae}"
        string_path_cv=/feat_cv_${mdl_name_vc}-${step_count}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}
    else
        echo "vc checkpoints not found, please run vc training step"
        exit
    fi
fi


if [ $mdl_name_ft == "cycmelspspkvae-gauss-smpl_sparse_weightemb_mwdlp_smpl_v2" ]; then
    setting_ft=${mdl_name_ft}_${data_name}_lr${lr}_bs${batch_size_wave}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hud${hidden_units_dec}_huw${hidden_units_wave}_stc${step_count}_st${step_count_wave}_s${spkidtr_dim}_w${n_weight_emb}_de${densities_cycvae_enc}_dd${densities_cycvae_dec}_nb${n_bands}_sc${s_conv_flag}_ss${seg_conv_flag}_ssw${seg_conv_flag_wave}_${min_idx_cycvae}-${min_idx_wave}
fi


# STAGE 6 {{
# set variables
expdir_ft=exp/tr_${setting_ft}
if [ `echo ${stage} | grep 6` ];then
    echo $mdl_name_ft $mdl_name_vc $mdl_name_wave
    mkdir -p $expdir_ft
    echo "###########################################################"
    echo "#         VC FINE-TUNE WITH WAVEFORM MODELING STEP        #"
    echo "###########################################################"
    echo $expdir_ft
  
    if [ -f "${expdir_ft}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_ft}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_ft} \
                --confdir conf/${data_name}_ft
        idx_resume_ft=`cat conf/${data_name}_ft.idx | awk '{print $1}'`
        min_idx_ft=`cat conf/${data_name}_ft.idx | awk '{print $2}'`
        echo "${data_name}: idx_resume_ft=${idx_resume_ft}, min_idx_ft=${min_idx_ft}"
    else
        idx_resume_ft=0
    fi

    stats_list=()
    for spk in ${spks[@]};do
        stats_list+=(data/${trn}/stats_spk-${spk}.h5)
    done
    stats_list_list="$(IFS="@"; echo "${stats_list[*]}")"
    spk_list="$(IFS="@"; echo "${spks[*]}")"
    echo ${spk_list}

    n_spk=${#spks[@]}
    n_tr_sum=7500
    n_tr=`expr $n_tr_sum / ${n_spk}`
    if [ `expr $n_tr_sum % ${n_spk}` -gt 0 ]; then
        n_tr=`expr ${n_tr} + 1`
    fi
    if [ $n_spk -le 300 ]; then
        n_dv=`expr 300 / ${n_spk}`
    else
        n_dv=1
    fi
    echo $n_tr $n_dv
    if true; then
        feats_sort_list=data/${trn}/feats_sort.scp
        wav_sort_list=data/${trn}/wav_ns_sort.scp
        if [ ! -f ${wav_sort_list} ] || [ ! -f ${feats_sort_list}  ]; then
        #if true; then
            ${cuda_cmd} ${expdir_ft}/log/get_max_frame.log \
                sort_frame_list.py \
                    --feats data/${trn}/feats.scp \
                    --waveforms data/${trn}/wav_ns.scp \
                    --spk_list ${spk_list} \
                    --expdir ${expdir_ft} \
                    --n_jobs ${n_jobs}
        fi
        feats=${expdir_ft}/feats_tr_cut.scp
        waveforms=${expdir_ft}/wavs_tr_cut.scp
        if [ ! -f ${waveforms} ] || [ ! -f ${feats} ]; then
        #if true; then
            rm -f ${feats} ${waveforms}
            n_utt_spks=()
            sum_utt_spks=0
            flag_utt_spk_max=true
            count_utt_spk_gt_max=0
            for spk in ${spks[@]}; do
                n_utt_spk=`cat ${feats_sort_list} | grep "\/${spk}\/" | wc -l`
                if [ $n_utt_spk -gt $n_tr ]; then
                    sum_utt_spks=`expr $sum_utt_spks + $n_tr`
                    flag_utt_spk_max=false
                    count_utt_spk_gt_max=$(( $count_utt_spk_gt_max + 1 ))
                    n_utt_spks+=(${n_tr})
                    echo $spk $n_tr $sum_utt_spks
                else
                    sum_utt_spks=`expr $sum_utt_spks + $n_utt_spk`
                    n_utt_spks+=(${n_utt_spk})
                    echo $spk $n_utt_spk $n_tr $sum_utt_spks
                fi
            done
            if [ $sum_utt_spks -lt $n_tr_sum ] && ! ${flag_utt_spk_max} ; then
                flag=false
                rem_sum=$(( $n_tr_sum - $sum_utt_spks ))
                rem_sum_spk=$(( $rem_sum / $count_utt_spk_gt_max ))
                n_tr=$(( $n_tr + $rem_sum_spk ))
            else
                flag=true
            fi
            while ! $flag; do
                n_utt_spks=()
                sum_utt_spks=0
                flag_utt_spk_max=true
                count_utt_spk_gt_max=0
                for spk in ${spks[@]}; do
                    n_utt_spk=`cat ${feats_sort_list} | grep "\/${spk}\/" | wc -l`
                    if [ $n_utt_spk -gt $n_tr ]; then
                        sum_utt_spks=`expr $sum_utt_spks + $n_tr`
                        flag_utt_spk_max=false
                        count_utt_spk_gt_max=$(( $count_utt_spk_gt_max + 1 ))
                        n_utt_spks+=(${n_tr})
                        echo $spk $n_tr $sum_utt_spks
                    else
                        sum_utt_spks=`expr $sum_utt_spks + $n_utt_spk`
                        n_utt_spks+=(${n_utt_spk})
                        echo $spk $n_utt_spk $n_tr $sum_utt_spks
                    fi
                done
                if [ $sum_utt_spks -lt $n_tr_sum ] && ! ${flag_utt_spk_max} ; then
                    flag=false
                    rem_sum=$(( $n_tr_sum - $sum_utt_spks ))
                    rem_sum_spk=$(( $rem_sum / $count_utt_spk_gt_max ))
                    if [ $rem_sum_spk -eq 0 ]; then
                        rem_sum_spk=1
                    fi
                    n_tr=$(( $n_tr + $rem_sum_spk ))
                else
                    flag=true
                fi
            done
            idx_utt_spk=0
            for spk in ${spks[@]}; do
                n_utt_spk=${n_utt_spks[${idx_utt_spk}]}
                echo tr $spk $n_utt_spk
                cat ${feats_sort_list} | grep "\/${spk}\/" | head -n ${n_utt_spk} | sort >> ${feats}
                cat ${wav_sort_list} | grep "\/${spk}\/" | head -n ${n_utt_spk} | sort >> ${waveforms}
                idx_utt_spk=$(( $idx_utt_spk + 1 ))
            done
        fi
        feats_eval=data/${dev}/feats.scp
        wav_eval=data/${dev}/wav_ns.scp
        feats_eval_list=()
        wavs_eval_list=()
        #for spk in ${spks_trg_rec[@]};do
        for spk in ${spks[@]};do
            if [ ! -f ${expdir_ft}/feats_dv_cut_spk-${spk}.scp ]; then
            #if true; then
                echo dv $spk feat
                cat ${feats_eval} | grep "\/${spk}\/" | head -n ${n_dv} | sort > ${expdir_ft}/feats_dv_cut_spk-${spk}.scp
            fi
            if [ ! -f ${expdir_ft}/wavs_dv_cut_spk-${spk}.scp ]; then
            #if true; then
                echo dv $spk wav
                cat ${wav_eval} | grep "\/${spk}\/" | head -n ${n_dv} | sort > ${expdir_ft}/wavs_dv_cut_spk-${spk}.scp
            fi
            feats_eval_list+=(${expdir_ft}/feats_dv_cut_spk-${spk}.scp)
            wavs_eval_list+=(${expdir_ft}/wavs_dv_cut_spk-${spk}.scp)
        done
        feats_list_eval_list="$(IFS="@"; echo "${feats_eval_list[*]}")"
        wavs_list_eval_list="$(IFS="@"; echo "${wavs_eval_list[*]}")"
    fi

    if [ $mdl_name_ft == "cycmelspspkvae-gauss-smpl_sparse_weightemb_mwdlp_smpl_v2" ];then
        if [ $idx_resume_ft -gt 0 ]; then
            echo ""
            echo "vc fine-tuning is in training, please use less/vim to monitor the training log: ${expdir_ft}/log/train_resume-${idx_resume_ft}.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_ft}/log/train_resume-${idx_resume_ft}.log \
                train_sparse-gru-cycle-melsp-spk-vae-gauss-smpl_weightemb_mwdlp_smpl_v2.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --waveforms ${waveforms} \
                    --waveforms_eval_list $wavs_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_ft} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --step_count ${step_count_wave} \
                    --mel_dim ${mel_dim} \
                    --lat_dim ${lat_dim} \
                    --lat_dim_e ${lat_dim_e} \
                    --stats_list ${stats_list_list} \
                    --spk_list ${spk_list} \
                    --hidden_units_enc ${hidden_units_enc} \
                    --hidden_layers_enc ${hidden_layers_enc} \
                    --hidden_units_dec ${hidden_units_dec} \
                    --hidden_layers_dec ${hidden_layers_dec} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --causal_conv_enc ${causal_conv_enc} \
                    --causal_conv_dec ${causal_conv_dec} \
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --n_weight_emb ${n_weight_emb} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --s_conv_flag ${s_conv_flag} \
                    --seg_conv_flag ${seg_conv_flag} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities_enc ${densities_cycvae_enc} \
                    --densities_dec ${densities_cycvae_dec} \
                    --n_stage ${n_stage} \
                    --fftl ${fftl} \
                    --fs ${fs} \
                    --batch_size ${batch_size_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --lpc ${lpc} \
                    --right_size_wave ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --s_dim ${s_dim} \
                    --mid_dim ${mid_dim} \
                    --gen_model ${expdir_vc}/checkpoint-${min_idx_cycvae}.pkl \
                    --gen_model_waveform ${expdir_wave}/checkpoint-${min_idx_wave}.pkl \
                    --resume ${expdir_ft}/checkpoint-${idx_resume_ft}.pkl \
                    --GPU_device ${GPU_device}
        else
            echo ""
            echo "vc fine-tuning is in training, please use less/vim to monitor the training log: ${expdir_ft}/log/train.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_ft}/log/train.log \
                train_sparse-gru-cycle-melsp-spk-vae-gauss-smpl_weightemb_mwdlp_smpl_v2.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --waveforms ${waveforms} \
                    --waveforms_eval_list $wavs_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_ft} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --step_count ${step_count_wave} \
                    --mel_dim ${mel_dim} \
                    --lat_dim ${lat_dim} \
                    --lat_dim_e ${lat_dim_e} \
                    --stats_list ${stats_list_list} \
                    --spk_list ${spk_list} \
                    --hidden_units_enc ${hidden_units_enc} \
                    --hidden_layers_enc ${hidden_layers_enc} \
                    --hidden_units_dec ${hidden_units_dec} \
                    --hidden_layers_dec ${hidden_layers_dec} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --causal_conv_enc ${causal_conv_enc} \
                    --causal_conv_dec ${causal_conv_dec} \
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --n_weight_emb ${n_weight_emb} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --s_conv_flag ${s_conv_flag} \
                    --seg_conv_flag ${seg_conv_flag} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities_enc ${densities_cycvae_enc} \
                    --densities_dec ${densities_cycvae_dec} \
                    --n_stage ${n_stage} \
                    --fftl ${fftl} \
                    --fs ${fs} \
                    --batch_size ${batch_size_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --lpc ${lpc} \
                    --right_size_wave ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --s_dim ${s_dim} \
                    --mid_dim ${mid_dim} \
                    --gen_model ${expdir_vc}/checkpoint-${min_idx_cycvae}.pkl \
                    --gen_model_waveform ${expdir_wave}/checkpoint-${min_idx_wave}.pkl \
                    --GPU_device ${GPU_device}
        fi
        echo ""
        echo "vc fine-tuning finished, please check the log file, and try to decode/compile real-time demo"
        echo ""
    fi
fi
# }}}


if [ `echo ${stage} | grep 7` ] \
    || [ `echo ${stage} | grep d` ] || [ `echo ${stage} | grep e` ] || [ `echo ${stage} | grep f` ] \
        || [ `echo ${stage} | grep g` ] || [ `echo ${stage} | grep h` ] || [ `echo ${stage} | grep j` ]; then
    echo $expdir_ft
    if [ -f "${expdir_ft}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_ft}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_ft} \
                --confdir conf/${data_name}_ft
        min_idx_ft=`cat conf/${data_name}_ft.idx | awk '{print $2}'`
        echo "${data_name}: min_idx_ft=${min_idx_ft}"
        string_path_ft=/feat_cv_${mdl_name_ft}-${step_count}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}
    else
        echo "fine-tune vc checkpoints not found, please run vc fine-tune training step"
        exit
    fi
fi


#model=${expdir_ft}/checkpoint-${min_idx_ft}.pkl
#config=${expdir_ft}/model.conf
#outdir=${expdir_ft}/spkidtr-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}
#mkdir -p $outdir
#${cuda_cmd} ${expdir_ft}/log/decode_spkidtr_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}.log \
#    decode_spkidtr_map.py \
#        --outdir ${outdir} \
#        --model ${model} \
#        --config ${config}
#exit


if [ $mdl_name_sp == "cycmelspspkvae-ftdec-gauss-smpl_sparse_wemb_mwdlp_smpl_v2" ]; then
    setting_sp=${mdl_name_sp}_${data_name}_lr${lr}_bs${batch_size_wave}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hud${hidden_units_dec}_huw${hidden_units_wave}_stc${step_count}_st${step_count_wave}_s${spkidtr_dim}_w${n_weight_emb}_de${densities_cycvae_enc}_dd${densities_cycvae_dec}_nb${n_bands}_sc${s_conv_flag}_ss${seg_conv_flag}_ssw${seg_conv_flag_wave}_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}
fi


# STAGE 7 {{
# set variables
expdir_sp=exp/tr_${setting_sp}
if [ `echo ${stage} | grep 7` ];then
    echo $mdl_name_sp $mdl_name_ft $mdl_name_vc $mdl_name_wave
    mkdir -p $expdir_sp
    echo "###########################################################"
    echo "#    VC DECODER FINE-TUNE WITH WAVEFORM MODELING STEP     #"
    echo "###########################################################"
    echo $expdir_sp
  
    if [ -f "${expdir_sp}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_sp}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_sp} \
                --confdir conf/${data_name}_sp
        idx_resume_sp=`cat conf/${data_name}_sp.idx | awk '{print $1}'`
        min_idx_sp=`cat conf/${data_name}_sp.idx | awk '{print $2}'`
        echo "${data_name}: idx_resume_sp=${idx_resume_sp}, min_idx_sp=${min_idx_sp}"
    else
        idx_resume_sp=0
    fi

    stats_list=()
    for spk in ${spks[@]};do
        stats_list+=(data/${trn}/stats_spk-${spk}.h5)
    done
    stats_list_list="$(IFS="@"; echo "${stats_list[*]}")"
    spk_list="$(IFS="@"; echo "${spks[*]}")"
    echo ${spk_list}

    n_spk=${#spks[@]}
    n_tr_sum=15000
    n_tr=`expr $n_tr_sum / ${n_spk}`
    if [ `expr $n_tr_sum % ${n_spk}` -gt 0 ]; then
        n_tr=`expr ${n_tr} + 1`
    fi
    if [ $n_spk -le 300 ]; then
        n_dv=`expr 300 / ${n_spk}`
    else
        n_dv=1
    fi
    echo $n_tr $n_dv
    if true; then
        feats_sort_list=data/${trn}/feats_sort.scp
        wav_sort_list=data/${trn}/wav_ns_sort.scp
        if [ ! -f ${wav_sort_list} ] || [ ! -f ${feats_sort_list}  ]; then
        #if true; then
            ${cuda_cmd} ${expdir_sp}/log/get_max_frame.log \
                sort_frame_list.py \
                    --feats data/${trn}/feats.scp \
                    --waveforms data/${trn}/wav_ns.scp \
                    --spk_list ${spk_list} \
                    --expdir ${expdir_sp} \
                    --n_jobs ${n_jobs}
        fi
        feats=${expdir_sp}/feats_tr_cut.scp
        waveforms=${expdir_sp}/wavs_tr_cut.scp
        if [ ! -f ${waveforms} ] || [ ! -f ${feats} ]; then
        #if true; then
            rm -f ${feats} ${waveforms}
            n_utt_spks=()
            sum_utt_spks=0
            flag_utt_spk_max=true
            count_utt_spk_gt_max=0
            for spk in ${spks[@]}; do
                n_utt_spk=`cat ${feats_sort_list} | grep "\/${spk}\/" | wc -l`
                if [ $n_utt_spk -gt $n_tr ]; then
                    sum_utt_spks=`expr $sum_utt_spks + $n_tr`
                    flag_utt_spk_max=false
                    count_utt_spk_gt_max=$(( $count_utt_spk_gt_max + 1 ))
                    n_utt_spks+=(${n_tr})
                    echo $spk $n_tr $sum_utt_spks
                else
                    sum_utt_spks=`expr $sum_utt_spks + $n_utt_spk`
                    n_utt_spks+=(${n_utt_spk})
                    echo $spk $n_utt_spk $n_tr $sum_utt_spks
                fi
            done
            if [ $sum_utt_spks -lt $n_tr_sum ] && ! ${flag_utt_spk_max} ; then
                flag=false
                rem_sum=$(( $n_tr_sum - $sum_utt_spks ))
                rem_sum_spk=$(( $rem_sum / $count_utt_spk_gt_max ))
                n_tr=$(( $n_tr + $rem_sum_spk ))
            else
                flag=true
            fi
            while ! $flag; do
                n_utt_spks=()
                sum_utt_spks=0
                flag_utt_spk_max=true
                count_utt_spk_gt_max=0
                for spk in ${spks[@]}; do
                    n_utt_spk=`cat ${feats_sort_list} | grep "\/${spk}\/" | wc -l`
                    if [ $n_utt_spk -gt $n_tr ]; then
                        sum_utt_spks=`expr $sum_utt_spks + $n_tr`
                        flag_utt_spk_max=false
                        count_utt_spk_gt_max=$(( $count_utt_spk_gt_max + 1 ))
                        n_utt_spks+=(${n_tr})
                        echo $spk $n_tr $sum_utt_spks
                    else
                        sum_utt_spks=`expr $sum_utt_spks + $n_utt_spk`
                        n_utt_spks+=(${n_utt_spk})
                        echo $spk $n_utt_spk $n_tr $sum_utt_spks
                    fi
                done
                if [ $sum_utt_spks -lt $n_tr_sum ] && ! ${flag_utt_spk_max} ; then
                    flag=false
                    rem_sum=$(( $n_tr_sum - $sum_utt_spks ))
                    rem_sum_spk=$(( $rem_sum / $count_utt_spk_gt_max ))
                    if [ $rem_sum_spk -eq 0 ]; then
                        rem_sum_spk=1
                    fi
                    n_tr=$(( $n_tr + $rem_sum_spk ))
                else
                    flag=true
                fi
            done
            idx_utt_spk=0
            for spk in ${spks[@]}; do
                n_utt_spk=${n_utt_spks[${idx_utt_spk}]}
                echo tr $spk $n_utt_spk
                cat ${feats_sort_list} | grep "\/${spk}\/" | head -n ${n_utt_spk} | sort >> ${feats}
                cat ${wav_sort_list} | grep "\/${spk}\/" | head -n ${n_utt_spk} | sort >> ${waveforms}
                idx_utt_spk=$(( $idx_utt_spk + 1 ))
            done
        fi
        feats_eval=data/${dev}/feats.scp
        wav_eval=data/${dev}/wav_ns.scp
        feats_eval_list=()
        wavs_eval_list=()
        #for spk in ${spks_trg_rec[@]};do
        for spk in ${spks[@]};do
            if [ ! -f ${expdir_sp}/feats_dv_cut_spk-${spk}.scp ]; then
            #if true; then
                echo dv $spk feat
                cat ${feats_eval} | grep "\/${spk}\/" | head -n ${n_dv} | sort > ${expdir_sp}/feats_dv_cut_spk-${spk}.scp
            fi
            if [ ! -f ${expdir_sp}/wavs_dv_cut_spk-${spk}.scp ]; then
            #if true; then
                echo dv $spk wav
                cat ${wav_eval} | grep "\/${spk}\/" | head -n ${n_dv} | sort > ${expdir_sp}/wavs_dv_cut_spk-${spk}.scp
            fi
            feats_eval_list+=(${expdir_sp}/feats_dv_cut_spk-${spk}.scp)
            wavs_eval_list+=(${expdir_sp}/wavs_dv_cut_spk-${spk}.scp)
        done
        feats_list_eval_list="$(IFS="@"; echo "${feats_eval_list[*]}")"
        wavs_list_eval_list="$(IFS="@"; echo "${wavs_eval_list[*]}")"
    fi

    if [ $mdl_name_sp == "cycmelspspkvae-ftdec-gauss-smpl_sparse_wemb_mwdlp_smpl_v2" ]; then
        if [ $idx_resume_sp -gt 0 ]; then
            echo ""
            echo "vc fine-tuning is in training, please use less/vim to monitor the training log: ${expdir_sp}/log/train_resume-${idx_resume_sp}.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_sp}/log/train_resume-${idx_resume_sp}.log \
                train_sparse-gru-cycle-melsp-spk-vae-ftdec-gauss-smpl_weightemb_mwdlp_smpl_v2.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --waveforms ${waveforms} \
                    --waveforms_eval_list $wavs_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_sp} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --step_count ${step_count_wave} \
                    --mel_dim ${mel_dim} \
                    --lat_dim ${lat_dim} \
                    --lat_dim_e ${lat_dim_e} \
                    --stats_list ${stats_list_list} \
                    --spk_list ${spk_list} \
                    --hidden_units_enc ${hidden_units_enc} \
                    --hidden_layers_enc ${hidden_layers_enc} \
                    --hidden_units_dec ${hidden_units_dec} \
                    --hidden_layers_dec ${hidden_layers_dec} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --causal_conv_enc ${causal_conv_enc} \
                    --causal_conv_dec ${causal_conv_dec} \
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --n_weight_emb ${n_weight_emb} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --s_conv_flag ${s_conv_flag} \
                    --seg_conv_flag ${seg_conv_flag} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities ${densities_cycvae_dec} \
                    --n_stage ${n_stage} \
                    --fftl ${fftl} \
                    --fs ${fs} \
                    --batch_size ${batch_size_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --lpc ${lpc} \
                    --right_size_wave ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --s_dim ${s_dim} \
                    --mid_dim ${mid_dim} \
                    --gen_model ${expdir_ft}/checkpoint-${min_idx_ft}.pkl \
                    --resume ${expdir_sp}/checkpoint-${idx_resume_sp}.pkl \
                    --GPU_device ${GPU_device}
        else
            echo ""
            echo "vc decoder fine-tuning is in training, please use less/vim to monitor the training log: ${expdir_sp}/log/train.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_sp}/log/train.log \
                train_sparse-gru-cycle-melsp-spk-vae-ftdec-gauss-smpl_weightemb_mwdlp_smpl_v2.py \
                    --feats ${feats} \
                    --feats_eval_list $feats_list_eval_list \
                    --waveforms ${waveforms} \
                    --waveforms_eval_list $wavs_list_eval_list \
                    --stats data/${trn}/stats_jnt.h5 \
                    --expdir ${expdir_sp} \
                    --lr ${lr} \
                    --do_prob ${do_prob} \
                    --step_count ${step_count_wave} \
                    --mel_dim ${mel_dim} \
                    --lat_dim ${lat_dim} \
                    --lat_dim_e ${lat_dim_e} \
                    --stats_list ${stats_list_list} \
                    --spk_list ${spk_list} \
                    --hidden_units_enc ${hidden_units_enc} \
                    --hidden_layers_enc ${hidden_layers_enc} \
                    --hidden_units_dec ${hidden_units_dec} \
                    --hidden_layers_dec ${hidden_layers_dec} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --causal_conv_enc ${causal_conv_enc} \
                    --causal_conv_dec ${causal_conv_dec} \
                    --n_half_cyc ${n_half_cyc} \
                    --n_workers ${n_workers} \
                    --pad_len ${pad_len} \
                    --spkidtr_dim ${spkidtr_dim} \
                    --n_weight_emb ${n_weight_emb} \
                    --right_size_enc ${right_size_enc} \
                    --right_size_dec ${right_size_dec} \
                    --s_conv_flag ${s_conv_flag} \
                    --seg_conv_flag ${seg_conv_flag} \
                    --t_start ${t_start} \
                    --t_end ${t_end} \
                    --interval ${interval} \
                    --densities ${densities_cycvae_dec} \
                    --n_stage ${n_stage} \
                    --fftl ${fftl} \
                    --fs ${fs} \
                    --batch_size ${batch_size_wave} \
                    --upsampling_factor ${upsampling_factor} \
                    --hidden_units_wave ${hidden_units_wave} \
                    --hidden_units_wave_2 ${hidden_units_wave_2} \
                    --kernel_size_wave ${kernel_size_wave} \
                    --dilation_size_wave ${dilation_size_wave} \
                    --lpc ${lpc} \
                    --right_size_wave ${right_size_wave} \
                    --n_bands ${n_bands} \
                    --s_dim ${s_dim} \
                    --mid_dim ${mid_dim} \
                    --gen_model ${expdir_ft}/checkpoint-${min_idx_ft}.pkl \
                    --GPU_device ${GPU_device}
        fi
        echo ""
        echo "vc decoder fine-tuning finished, please check the log file, and try to decode/compile real-time demo"
        echo ""
    fi
fi
# }}}


# STAGE 8 {{{
if [ `echo ${stage} | grep 8` ] || [ `echo ${stage} | grep 9` ];then
for spk_src in ${spks_dec[@]};do
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb_v2" ]; then
        outdir=${expdir_wave}/${mdl_name_wave}-${data_name}_dev-${hidden_units_wave}-${step_count_wave}-${lpc}-${n_bands}-${min_idx_wave}
        #outdir=${expdir_wave}/${mdl_name_wave}-${data_name}_tst-${hidden_units_wave}-${step_count_wave}-${lpc}-${n_bands}-${min_idx_wave}
    fi
if [ `echo ${stage} | grep 8` ];then
    echo $spk_src $min_idx_wave $data_name $mdl_name_wave
    echo $outdir
    echo "###################################################"
    echo "#               DECODING STEP                     #"
    echo "###################################################"
    echo ${setting_wave}

    checkpoint=${expdir_wave}/checkpoint-${min_idx_wave}.pkl
    config=${expdir_wave}/model.conf

    feats=data/${dev}/feats.scp
    #feats=data/${tst}/feats.scp

    feats_scp=${expdir_wave}/feats_${min_idx_wave}_${spk_src}.scp
    cat $feats | grep "\/${spk_src}\/" | sort | head -n ${n_wav_decode} > ${feats_scp}

    # decode
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb_v2" ]; then
        echo ""
        #echo "now synthesizing ${spk_src}, log here:  ${expdir_wave}/log/decode_tst_${min_idx_wave}_${spk_src}.log"
        #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_wave}_${spk_src}.log \
        echo "now synthesizing ${spk_src}, log here:  ${expdir_wave}/log/decode_dev_${min_idx_wave}_${spk_src}.log"
        ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_wave}_${spk_src}.log \
            decode_wavernn_dualgru_compact_lpc_mband_cf.py \
                --feats ${feats_scp} \
                --outdir ${outdir}/${spk_src} \
                --checkpoint ${checkpoint} \
                --config ${config} \
                --fs ${fs} \
                --batch_size ${decode_batch_size} \
                --n_gpus ${n_gpus} \
                --GPU_device_str ${GPU_device_str}
        echo ""
        echo "synthesizing ${spk_src} is finished, pre-emphasized synthesized waveform here: $outdir/${spk_src}"
        echo ""
    fi
fi
# }}}


# STAGE 9 {{{
if [ `echo ${stage} | grep 9` ];then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    scp=${expdir_wave}/wav_generated_${min_idx_wave}_${spk_src}.scp
    find ${outdir}/${spk_src} -name "*.wav" | grep "\/${spk_src}\/" | sort > ${scp}

    # restore noise shaping
    ${train_cmd} --num-threads ${n_jobs} \
        ${expdir_wave}/${log}/noise_shaping_restore_${min_idx_wave}_${spk_src}.log \
        noise_shaping_emph.py \
            --waveforms ${scp} \
            --writedir ${outdir}_restored/${spk_src} \
            --alpha ${alpha} \
            --fs ${fs} \
            --inv true \
            --n_jobs ${n_jobs}
     echo ""
     echo "de-emphasis ${spk_src} is finished, synthesized waveform here: ${outdir}_restored/${spk_src}"
     echo ""
fi
# }}}
done
fi


#model=${expdir_vc}/checkpoint-${min_idx_cycvae}.pkl
#config=${expdir_vc}/model.conf
#outdir=${expdir_vc}/spkidtr-${min_idx_cycvae}
#mkdir -p $outdir
#${cuda_cmd} ${expdir_vc}/log/decode_spkidtr_${min_idx_cycvae}.log \
#    decode_spkidtr_map.py \
#        --outdir ${outdir} \
#        --model ${model} \
#        --config ${config}
#exit


# STAGE a {{{
if [ `echo ${stage} | grep a` ];then
for spkr in ${spks_src_dec[@]};do
for spk_trg in ${spks_trg_dec[@]};do
if [ $spkr != $spk_trg ]; then
    echo $spkr $spk_trg $min_idx_cycvae $min_idx_wave
    echo $data_name $mdl_name_vc $mdl_name_wave
    echo $n_gpus $GPU_device $GPU_device_str

    config=${expdir_vc}/model.conf
    model=${expdir_vc}/checkpoint-${min_idx_cycvae}.pkl

    h5outdir=hdf5/${dev}/${spkr}-${spk_trg}
    echo $h5outdir
    #if true; then
    ##if false; then
    echo "######################################################"
    echo "#                DECODING CONV. FEAT DEV             #"
    echo "######################################################"
    outdir=${expdir_vc}/wav_cv_${mdl_name_vc}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}_${spkr}-${spk_trg}_dev
    mkdir -p ${outdir}
    feats_scp=${outdir}/feats.scp
    cat data/${dev}/feats.scp | grep "\/${spkr}\/" | head -n ${n_wav_decode} > ${feats_scp}
    if [ $mdl_name_vc == "cycmelspxlf0capspkvae-gauss-smpl_sparse_weightemb_v2" ]; then
        echo ""
        echo "now decoding vc ${spkr}-to-${spk_trg}..., log here: ${expdir_vc}/log/decode_dev_${min_idx_cycvae}_${spkr}-${spk_trg}.log"
        ${cuda_cmd} ${expdir_vc}/log/decode_dev_${min_idx_cycvae}_${spkr}-${spk_trg}.log \
            decode_gru-cycle-melspxlf0capspkvae-gauss-smpl_spk.py \
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
        echo ""
        echo "decoding vc ${spkr}-to-${spk_trg} is finished, griffin-lim synthesized waveform here: ${outdir}, please try to synthesize with mwdlp"
        echo ""
    fi
    find ${h5outdir} -name "*.h5" | sort > data/${dev}/feats_cv_${spkr}-${spk_trg}.scp
    #fi

    #h5outdir=hdf5/${tst}/${spkr}-${spk_trg}
    #echo $h5outdir
    ##if true; then
    ###if false; then
    #echo "######################################################"
    #echo "#                DECODING CONV. FEAT TST             #"
    #echo "######################################################"
    #outdir=${expdir_vc}/wav_cv_${mdl_namevc}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}_${spkr}-${spk_trg}_tst
    #mkdir -p ${outdir}
    #feats_scp=${outdir}/feats.scp
    #cat data/${tst}/feats.scp | grep "\/${spkr}\/" | head -n ${n_wav_decode} > ${feats_scp}
    #if [ $mdl_name_vc == "cycmelspxlf0capspkvae-gauss-smpl_sparse_weightemb_v2" ]; then
    #    echo ""
    #    echo "now decoding vc ${spkr}-to-${spk_trg}..., log here: ${expdir_vc}/log/decode_tst_${min_idx_cycvae}_${spkr}-${spk_trg}.log"
    #    $densities_cycvae{cuda_cmd} ${expdir_vc}/log/decode_tst_${min_idx_cycvae}_${spkr}-${spk_trg}.log \
    #        decode_gru-cycle-melspxlf0capspkvae-gauss-smpl_spk.py \
    #            --feats ${feats_scp} \
    #            --spk_trg ${spk_trg} \
    #            --outdir ${outdir} \
    #            --model ${model} \
    #            --config ${config} \
    #            --fs ${fs} \
    #            --winms ${winms} \
    #            --fftl ${fftl} \
    #            --shiftms ${shiftms} \
    #            --n_gpus ${n_gpus} \
    #            --string_path ${string_path_cv} \
    #            --GPU_device_str ${GPU_device_str}
    #            #--GPU_device ${GPU_device} \
    #    echo ""
    #    echo "decoding vc ${spkr}-to-${spk_trg} is finished, griffin-lim synthesized waveform here: ${outdir}, please try to synthesize with mwdlp"
    #    echo ""
    #fi
    #find ${h5outdir} -name "*.h5" | sort > data/${tst}/feats_cv_${spkr}-${spk_trg}.scp
fi
done
done
fi
# }}}


# STAGE b {{{
if [ `echo ${stage} | grep b` ] || [ `echo ${stage} | grep c` ];then
for spk_src in ${spks_src_dec[@]};do
for spk_trg in ${spks_trg_dec[@]};do
if [ $spk_src != $spk_trg ]; then
outdir=${expdir_wave}/${mdl_name_vc}-${mdl_name_wave}-${data_name}_dev-${lat_dim}-${lat_dim_e}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}
#outdir=${expdir_wave}/${mdl_name_vc}-${mdl_name_wave}-${data_name}_tst-${lat_dim}-${lat_dim_e}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}
echo $outdir
if [ `echo ${stage} | grep b` ];then
    echo $spk_src $spk_trg $min_idx_cycvae $min_idx_wave
    echo $data_name $mdl_name_vc $mdl_name_wave
    echo $n_gpus $GPU_device $GPU_device_str
    echo $outdir
    echo "###################################################"
    echo "#               DECODING STEP                     #"
    echo "###################################################"
    checkpoint=${expdir_wave}/checkpoint-${min_idx_wave}.pkl
    config=${expdir_wave}/model.conf

    feats=data/${dev}/feats_cv_${spk_src}-${spk_trg}.scp
    #feats=data/${tst}/feats_cv_${spk_src}-${spk_trg}.scp

    feats_scp=${expdir_wave}/feats_${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}.scp
    #cat $feats | grep "\/${spk_src}-${spk_trg}\/" > ${feats_scp}
    cat $feats | grep "\/${spk_src}-${spk_trg}\/" | head -n ${n_wav_decode} > ${feats_scp}

    # decode
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb_v2" ]; then
        echo ""
        #echo "now synthesizing ${spk_src}-${spk_trg}..., log here: ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}.log"
        #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
        echo "now synthesizing ${spk_src}-${spk_trg}..., log here: ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}.log"
        ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
            decode_wavernn_dualgru_compact_lpc_mband_cf.py \
                --feats ${feats_scp} \
                --outdir ${outdir} \
                --checkpoint ${checkpoint} \
                --config ${config} \
                --fs ${fs} \
                --batch_size ${decode_batch_size} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --GPU_device_str ${GPU_device_str}
        echo ""
        echo "synthesizing ${spk_src}-${spk_trg} is finished, pre-emphasized synthesized waveform here: $outdir"
        echo ""
    fi
fi
# }}}


# STAGE c {{{
if [ `echo ${stage} | grep c` ];then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    scp=${expdir_wave}/wav_generated_${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}.scp
    find ${outdir} -name "*.wav" > ${scp}

    # restore noise shaping
    ${train_cmd} --num-threads ${n_jobs} \
        ${expdir_wave}/${log}/noise_shaping_restore_${min_idx_cycvae}-${min_idx_wave}_${spk_src}-${spk_trg}.log \
        noise_shaping_emph.py \
            --waveforms ${scp} \
            --writedir ${outdir}_restored \
            --alpha ${alpha} \
            --fs ${fs} \
            --inv true \
            --n_jobs ${n_jobs}
        echo ""
        echo "de-emphasis ${spk_src}-${spk_trg} is finished, synthesized waveform here: ${outdir}_restored"
        echo ""
fi
# }}}
fi
done
done
fi


# STAGE d {{{
if [ `echo ${stage} | grep d` ];then
for spkr in ${spks_src_dec[@]};do
for spk_trg in ${spks_trg_dec[@]};do
if [ $spkr != $spk_trg ]; then
    echo $spkr $spk_trg $min_idx_cycvae $min_idx_wave $min_idx_ft
    echo $data_name $mdl_name_vc $mdl_name_wave $mdl_name_ft
    echo $n_gpus $GPU_device $GPU_device_str

    config=${expdir_ft}/model.conf
    model=${expdir_ft}/checkpoint-${min_idx_ft}.pkl

    h5outdir=hdf5/${dev}/${spkr}-${spk_trg}
    echo $h5outdir
    #if true; then
    ##if false; then
    echo "######################################################"
    echo "#                DECODING CONV. FEAT DEV             #"
    echo "######################################################"
    outdir=${expdir_ft}/wav_cv_${mdl_name_ft}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spkr}-${spk_trg}_dev
    mkdir -p ${outdir}
    feats_scp=${outdir}/feats.scp
    cat data/${dev}/feats.scp | grep "\/${spkr}\/" | head -n ${n_wav_decode} > ${feats_scp}
    if [ $mdl_name_ft == "cycmelspspkvae-gauss-smpl_sparse_weightemb_mwdlp_smpl_v2" ]; then
        echo ""
        echo "now decoding fine-tuned vc ${spkr}-to-${spk_trg}..., log here: ${expdir_ft}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spkr}-${spk_trg}.log"
        ${cuda_cmd} ${expdir_ft}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spkr}-${spk_trg}.log \
            decode_gru-cycle-melspspkvae-gauss-smpl_ft_spk.py \
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
                --string_path ${string_path_ft} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
        echo ""
        echo "decoding fine-tuned vc ${spkr}-to-${spk_trg} is finished, griffin-lim synthesized waveform here: ${outdir}, please try to synthesize with mwdlp"
        echo ""
    fi
    find ${h5outdir} -name "*.h5" | sort > data/${dev}/feats_cv_${spkr}-${spk_trg}.scp
    #fi

    #h5outdir=hdf5/${tst}/${spkr}-${spk_trg}
    #echo $h5outdir
    ##if true; then
    ###if false; then
    #echo "######################################################"
    #echo "#                DECODING CONV. FEAT TST             #"
    #echo "######################################################"
    #outdir=${expdir_ft}/wav_cv_${mdl_name_ft}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spkr}-${spk_trg}_tst
    #mkdir -p ${outdir}
    #feats_scp=${outdir}/feats.scp
    #cat data/${tst}/feats.scp | grep "\/${spkr}\/" | head -n ${n_wav_decode} > ${feats_scp}
    #if [ $mdl_name_ft == "cycmelspspkvae-gauss-smpl_sparse_weightemb_mwdlp_smpl_v2" ]; then
    #    echo ""
    #    echo "now decoding fine-tuned vc ${spkr}-to-${spk_trg}..., log here: ${expdir_ft}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spkr}-${spk_trg}.log"
    #    ${cuda_cmd} ${expdir_ft}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spkr}-${spk_trg}.log \
    #        decode_gru-cycle-melspspkvae-gauss-smpl_mwdlp_ft_spk.py \
    #            --feats ${feats_scp} \
    #            --spk_trg ${spk_trg} \
    #            --outdir ${outdir} \
    #            --model ${model} \
    #            --config ${config} \
    #            --fs ${fs} \
    #            --winms ${winms} \
    #            --fftl ${fftl} \
    #            --shiftms ${shiftms} \
    #            --n_gpus ${n_gpus} \
    #            --string_path ${string_path_ft} \
    #            --GPU_device_str ${GPU_device_str}
    #            #--GPU_device ${GPU_device} \
    #    echo ""
    #    echo "decoding fine-tuned vc ${spkr}-to-${spk_trg} is finished, griffin-lim synthesized waveform here: ${outdir}, please try to synthesize with mwdlp"
    #    echo ""
    #fi
    #find ${h5outdir} -name "*.h5" | sort > data/${tst}/feats_cv_${spkr}-${spk_trg}.scp
fi
done
#fi
done
fi
# }}}


# STAGE e {{{
if [ `echo ${stage} | grep e` ] || [ `echo ${stage} | grep f` ];then
for spk_src in ${spks_src_dec[@]};do
for spk_trg in ${spks_trg_dec[@]};do
if [ $spk_src != $spk_trg ]; then
    outdir=${expdir_wave}/${mdl_name_ft}-${mdl_name_wave}-${data_name}_dev-${lat_dim}-${lat_dim_e}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}
    #outdir=${expdir_wave}/${mdl_name_ft}-${mdl_name_wave}-${data_name}_tst-${lat_dim}-${lat_dim_e}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}
if [ `echo ${stage} | grep e` ];then
    echo $spk_src $spk_trg $min_idx_cycvae $min_idx_wave $min_idx_ft
    echo $data_name $mdl_name_vc $mdl_name_wave $mdl_name_ft
    echo $n_gpus $GPU_device $GPU_device_str
    echo $outdir
    echo "###################################################"
    echo "#               DECODING STEP                     #"
    echo "###################################################"
    checkpoint=${expdir_wave}/checkpoint-${min_idx_wave}.pkl
    config=${expdir_wave}/model.conf

    feats=data/${dev}/feats_cv_${spk_src}-${spk_trg}.scp
    #feats=data/${tst}/feats_cv_${spk_src}-${spk_trg}.scp

    feats_scp=${expdir_wave}/feats_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}.scp
    #cat $feats | grep "\/${spk_src}-${spk_trg}\/" > ${feats_scp}
    cat $feats | grep "\/${spk_src}-${spk_trg}\/" | head -n ${n_wav_decode} > ${feats_scp}

    # decode
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb_v2" ]; then
        echo ""
        #echo "now synthesizing ${spk_src}-${spk_trg}..., log here: ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}.log"
        #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}.log \
        echo "now synthesizing ${spk_src}-${spk_trg}..., log here: ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}.log"
        ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}.log \
            decode_wavernn_dualgru_compact_lpc_mband_cf.py \
                --feats ${feats_scp} \
                --outdir ${outdir} \
                --checkpoint ${checkpoint} \
                --config ${config} \
                --fs ${fs} \
                --batch_size ${decode_batch_size} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_ft} \
                --GPU_device_str ${GPU_device_str}
        echo ""
        echo "synthesizing ${spk_src}-${spk_trg} is finished, pre-emphasized synthesized waveform here: $outdir"
        echo ""
    fi
fi
# }}}


# STAGE f {{{
if [ `echo ${stage} | grep f` ];then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    scp=${expdir_wave}/wav_generated_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}.scp
    find ${outdir} -name "*.wav" > ${scp}

    # restore noise shaping
    ${train_cmd} --num-threads ${n_jobs} \
        ${expdir_wave}/${log}/noise_shaping_restore_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}_${spk_src}-${spk_trg}.log \
        noise_shaping_emph.py \
            --waveforms ${scp} \
            --writedir ${outdir}_restored \
            --alpha ${alpha} \
            --fs ${fs} \
            --inv true \
            --n_jobs ${n_jobs}
    echo ""
    echo "de-emphasis ${spk_src}-${spk_trg} is finished, synthesized waveform here: ${outdir}_restored"
    echo ""
fi
# }}}
fi
done
done
fi


if [ `echo ${stage} | grep g` ] || [ `echo ${stage} | grep h` ] || [ `echo ${stage} | grep j` ]; then
    echo $expdir_sp
    if [ -f "${expdir_sp}/checkpoint-last.pkl" ]; then
        ${train_cmd} ${expdir_sp}/get_model_indices.log \
            get_model_indices.py \
                --expdir ${expdir_sp} \
                --confdir conf/${data_name}_sp
        min_idx_sp=`cat conf/${data_name}_sp.idx | awk '{print $2}'`
        echo "${data_name}: min_idx_sp=${min_idx_sp}"
        string_path_sp=/feat_cv_${mdl_name_sp}-${hidden_units_wave}-${step_count_wave}-${step_count}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}
    else
        echo "fine-tuned vc decoder checkpoints not found, please run vc decoder fine-tuning step"
        exit
    fi
fi


# STAGE g {{{
if [ `echo ${stage} | grep g` ];then
for spkr in ${spks_src_dec[@]};do
for spk_trg in ${spks_trg_dec[@]};do
if [ $spkr != $spk_trg ]; then
    echo $spkr $spk_trg $min_idx_cycvae $min_idx_wave $min_idx_ft $min_idx_sp
    echo $data_name $mdl_name_vc $mdl_name_wave $mdl_name_ft $mdl_name_sp
    echo $n_gpus $GPU_device $GPU_device_str

    config=${expdir_sp}/model.conf
    model=${expdir_sp}/checkpoint-${min_idx_sp}.pkl

    h5outdir=hdf5/${dev}/${spkr}-${spk_trg}
    echo $h5outdir
    #if true; then
    ##if false; then
    echo "######################################################"
    echo "#                DECODING CONV. FEAT DEV             #"
    echo "######################################################"
    outdir=${expdir_sp}/wav_cv_${mdl_name_sp}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spkr}-${spk_trg}_dev
    mkdir -p ${outdir}
    feats_scp=${outdir}/feats.scp
    cat data/${dev}/feats.scp | grep "\/${spkr}\/" | head -n ${n_wav_decode} > ${feats_scp}
    if [ $mdl_name_sp == "cycmelspspkvae-ftdec-gauss-smpl_sparse_wemb_mwdlp_smpl_v2" ]; then
        echo ""
        echo "now decoding fine-tuned vc decoder ${spkr}-to-${spk_trg}..., log here: ${expdir_sp}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spkr}-${spk_trg}.log"
        ${cuda_cmd} ${expdir_sp}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spkr}-${spk_trg}.log \
            decode_gru-cycle-melspspkvae-gauss-smpl_ft_spk.py \
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
                --string_path ${string_path_sp} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
        echo ""
        echo "decoding fine-tuned vc decoder ${spkr}-to-${spk_trg} is finished, griffin-lim synthesized waveform here: ${outdir}, please try to synthesize with mwdlp"
        echo ""
    fi
    find ${h5outdir} -name "*.h5" | sort > data/${dev}/feats_cv_${spkr}-${spk_trg}.scp
    #fi

    #h5outdir=hdf5/${tst}/${spkr}-${spk_trg}
    #echo $h5outdir
    ##if true; then
    ###if false; then
    #echo "######################################################"
    #echo "#                DECODING CONV. FEAT TST             #"
    #echo "######################################################"
    #outdir=${expdir_sp}/wav_cv_${mdl_name_sp}-${data_name}-${lat_dim}-${lat_dim_e}-${spkidtr_dim}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spkr}-${spk_trg}_tst
    #mkdir -p ${outdir}
    #feats_scp=${outdir}/feats.scp
    #cat data/${tst}/feats.scp | grep "\/${spkr}\/" | head -n ${n_wav_decode} > ${feats_scp}
    #if [ $mdl_name_sp == "cycmelspspkvae-ftdec-gauss-smpl_sparse_wemb_mwdlp_smpl_v2" ]; then
    #    echo ""
    #    echo "now decoding fine-tuned vc decoder ${spkr}-to-${spk_trg}..., log here: ${expdir_sp}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spkr}-${spk_trg}.log"
    #    ${cuda_cmd} ${expdir_sp}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spkr}-${spk_trg}.log \
    #        decode_gru-cycle-melspspkvae-gauss-smpl_mwdlp_ft_spk.py \
    #            --feats ${feats_scp} \
    #            --spk_trg ${spk_trg} \
    #            --outdir ${outdir} \
    #            --model ${model} \
    #            --config ${config} \
    #            --fs ${fs} \
    #            --winms ${winms} \
    #            --fftl ${fftl} \
    #            --shiftms ${shiftms} \
    #            --n_gpus ${n_gpus} \
    #            --string_path ${string_path_sp} \
    #            --GPU_device_str ${GPU_device_str}
    #            #--GPU_device ${GPU_device} \
    #    echo ""
    #    echo "decoding fine-tuned vc decoder ${spkr}-to-${spk_trg} is finished, griffin-lim synthesized waveform here: ${outdir}, please try to synthesize with mwdlp"
    #    echo ""
    #fi
    #find ${h5outdir} -name "*.h5" | sort > data/${tst}/feats_cv_${spkr}-${spk_trg}.scp
fi
done
#fi
done
fi
# }}}


# STAGE h {{{
if [ `echo ${stage} | grep h` ] || [ `echo ${stage} | grep j` ];then
for spk_src in ${spks_src_dec[@]};do
for spk_trg in ${spks_trg_dec[@]};do
if [ $spk_src != $spk_trg ]; then
    outdir=${expdir_wave}/${mdl_name_sp}-${mdl_name_wave}-${data_name}_dev-${lat_dim}-${lat_dim_e}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}
    #outdir=${expdir_wave}/${mdl_name_sp}-${mdl_name_wave}-${data_name}_tst-${lat_dim}-${lat_dim_e}-${n_half_cyc}-${hidden_units_wave}-${lpc}-${n_bands}-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}
if [ `echo ${stage} | grep h` ];then
    echo $spk_src $spk_trg $min_idx_cycvae $min_idx_wave $min_idx_ft $min_idx_sp
    echo $data_name $mdl_name_vc $mdl_name_wave $mdl_name_ft $mdl_name_sp
    echo $n_gpus $GPU_device $GPU_device_str
    echo $outdir
    echo "###################################################"
    echo "#               DECODING STEP                     #"
    echo "###################################################"
    checkpoint=${expdir_wave}/checkpoint-${min_idx_wave}.pkl
    config=${expdir_wave}/model.conf

    feats=data/${dev}/feats_cv_${spk_src}-${spk_trg}.scp
    #feats=data/${tst}/feats_cv_${spk_src}-${spk_trg}.scp

    feats_scp=${expdir_wave}/feats_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}.scp
    #cat $feats | grep "\/${spk_src}-${spk_trg}\/" > ${feats_scp}
    cat $feats | grep "\/${spk_src}-${spk_trg}\/" | head -n ${n_wav_decode} > ${feats_scp}

    # decode
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb_v2" ]; then
        echo ""
        #echo "now synthesizing ${spk_src}-${spk_trg}..., log here: ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}.log"
        #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}.log \
        echo "now synthesizing ${spk_src}-${spk_trg}..., log here: ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}.log"
        ${cuda_cmd} ${expdir_wave}/log/decode_dev_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}.log \
            decode_wavernn_dualgru_compact_lpc_mband_cf.py \
                --feats ${feats_scp} \
                --outdir ${outdir} \
                --checkpoint ${checkpoint} \
                --config ${config} \
                --fs ${fs} \
                --batch_size ${decode_batch_size} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_sp} \
                --GPU_device_str ${GPU_device_str}
        echo ""
        echo "synthesizing ${spk_src}-${spk_trg} is finished, pre-emphasized synthesized waveform here: $outdir"
        echo ""
    fi
fi
# }}}


# STAGE j {{{
if [ `echo ${stage} | grep j` ];then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    scp=${expdir_wave}/wav_generated_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}.scp
    find ${outdir} -name "*.wav" > ${scp}

    # restore noise shaping
    ${train_cmd} --num-threads ${n_jobs} \
        ${expdir_wave}/${log}/noise_shaping_restore_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}-${min_idx_sp}_${spk_src}-${spk_trg}.log \
        noise_shaping_emph.py \
            --waveforms ${scp} \
            --writedir ${outdir}_restored \
            --alpha ${alpha} \
            --fs ${fs} \
            --inv true \
            --n_jobs ${n_jobs}
    echo ""
    echo "de-emphasis ${spk_src}-${spk_trg} is finished, synthesized waveform here: ${outdir}_restored"
    echo ""
fi
# }}}
fi
done
done
fi
