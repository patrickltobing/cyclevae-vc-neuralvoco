#!/#bin/bash
##################################################################################################
#   SCRIPT FOR NON-PARALLEL VOICE CONVERSION based on CycleVAE/CycleVQVAE and WaveNet/WaveRNN    #
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
# 3: apply noise shaping step [pre-emphasis] [for neural vocoder waveform training data]
# 4: vc training step
# 5: reconstruction decoding step [for GV computation and neural vocoder training data augmentation]
# 6: conversion decoding step [converted waveform with conventional vocoder]
# 7: wavernn/wavenet training step
# 8: vc/copy-synthesis decoding step with wavernn/wavenet [converted/original waveform with neural vocoder]
# 9: restore noise shaping step [de-emphasis]
# }}}
#stage=0
#stage=init
#stage=012
#stage=0123
#stage=1
#stage=2
#stage=3
#stage=123
#stage=4
#stage=5
#stage=56
#stage=6
#stage=7
##stage=89
##stage=8
#stage=9

##number of parallel jobs in feature extraction / statistics calculation
n_jobs=1
#n_jobs=10
#n_jobs=25
#n_jobs=35
#n_jobs=45
#n_jobs=50
n_jobs=60

#######################################
#          TRAINING SETTING           #
#######################################

spks=(VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 VCC2SF3 VCC2SM3 VCC2TF1 VCC2TM1 VCC2TF2 VCC2TM2 VCC2SM4 VCC2SF4)
data_name=vcc2018

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
    data_name_wav=${data_name_wav}_22.05kHz
    mcep_alpha=0.455 #22.05k ## frequency warping based on pysptk.util.mcepalpha
    fftl=2048
    if [ $shiftms -eq 5 ]; then
        shiftms=4.9886621315192743764172335600907 #22.05k rounding 110/22050 5ms shift
    elif [ $shiftms -eq 10 ]; then
        shiftms=9.9773242630385487528344671201814 #22.05k rounding 220/22050 10ms shift
    fi
elif [ $fs -eq 24000 ]; then
    wav_org_dir=wav_24kHz
    data_name=${data_name}_24kHz
    data_name_wav=${data_name_wav}_24kHz
    mcep_alpha=0.466 #24k
    fftl=2048
elif [ $fs -eq 48000 ]; then
    wav_org_dir=wav_48kHz
    data_name=${data_name}_48kHz
    data_name_wav=${data_name_wav}_48kHz
    mcep_alpha=0.554 #48k
    fftl=4096
elif [ $fs -eq 44100 ]; then
    wav_org_dir=wav_44.1kHz
    data_name=${data_name}_44.1kHz
    data_name_wav=${data_name_wav}_44.1kHz
    mcep_alpha=0.544 #44.1k
    fftl=4096
    if [ $shiftms -eq 5 ]; then
        shiftms=4.9886621315192743764172335600907 #44.1k rounding 220/44100 5ms shift
    elif [ $shiftms -eq 10 ]; then
        shiftms=9.9773242630385487528344671201814 #44.1k rounding 440/44100 10ms shift
    fi
elif [ $fs -eq 16000 ]; then
    wav_org_dir=wav_16kHz
    data_name=${data_name}_16kHz
    data_name_wav=${data_name_wav}_16kHz
    mcep_alpha=0.41000000000000003 #16k
    fftl=1024
else
    echo "sampling rate not available"
    exit 1
fi
## from WORLD: number of code-aperiodicities = min(15000,fs/2-3000)/3000
## [https://github.com/mmorise/World/blob/master/src/codec.cpp] line 212


## winms: window length analysis for mel-spectrogram extraction
winms=`awk '{if ($1 == "winms:") print $2}' conf/config.yml`
## mel_dim: number of mel-spectrogram dimension
mel_dim=`awk '{if ($1 == "mel_dim:") print $2}' conf/config.yml`
## highpass_cutoff: cutoff frequency for low-cut filter to remove DC-component in recording
highpass_cutoff=`awk '{if ($1 == "highpass_cutoff:") print $2}' conf/config.yml`
## alpha: coefficient for pre-emphasis
alpha=`awk '{if ($1 == "alpha:") print $2}' conf/config.yml`

use_mcep=`awk '{if ($1 == "use_mcep:") print $2}' conf/config.yml`

## mcep_dim: number of mel-cepstrum dimension
mcep_dim=`awk '{if ($1 == "mcep_dim:") print $2}' conf/config.yml`
## powmcep_dim: 0th power + mcep_dim
if [ $use_mcep == "true" ]; then
    powmcep_dim=`expr ${mcep_dim} + 1`
else
    powmcep_dim=$mel_dim
fi

trn=tr_${data_name}
dev=dv_${data_name}
tst=ts_${data_name}

GPU_device=0
#GPU_device=1
#GPU_device=2

## Please see the conf/config.yml for explanation of the rest of variables
mdl_name=`awk '{if ($1 == "mdl_name:") print $2}' conf/config.yml`
epoch_count=`awk '{if ($1 == "epoch_count:") print $2}' conf/config.yml`
n_half_cyc=`awk '{if ($1 == "n_half_cyc:") print $2}' conf/config.yml`

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
kernel_size_enc=`awk '{if ($1 == "kernel_size_enc:") print $2}' conf/config.yml`
dilation_size_enc=`awk '{if ($1 == "dilation_size_enc:") print $2}' conf/config.yml`
kernel_size_dec=`awk '{if ($1 == "kernel_size_dec:") print $2}' conf/config.yml`
dilation_size_dec=`awk '{if ($1 == "dilation_size_dec:") print $2}' conf/config.yml`
kernel_size_lf0=`awk '{if ($1 == "kernel_size_lf0:") print $2}' conf/config.yml`
dilation_size_lf0=`awk '{if ($1 == "dilation_size_lf0:") print $2}' conf/config.yml`
causal_conv_enc=`awk '{if ($1 == "causal_conv_enc:") print $2}' conf/config.yml`
causal_conv_dec=`awk '{if ($1 == "causal_conv_dec:") print $2}' conf/config.yml`
causal_conv_lf0=`awk '{if ($1 == "causal_conv_lf0:") print $2}' conf/config.yml`
bi_enc=`awk '{if ($1 == "bi_enc:") print $2}' conf/config.yml`
bi_dec=`awk '{if ($1 == "bi_dec:") print $2}' conf/config.yml`
bi_lf0=`awk '{if ($1 == "bi_lf0:") print $2}' conf/config.yml`
do_prob=`awk '{if ($1 == "do_prob:") print $2}' conf/config.yml`
ar_enc=`awk '{if ($1 == "ar_enc:") print $2}' conf/config.yml`
ar_dec=`awk '{if ($1 == "ar_dec:") print $2}' conf/config.yml`
ar_f0=`awk '{if ($1 == "ar_f0:") print $2}' conf/config.yml`
diff=`awk '{if ($1 == "diff:") print $2}' conf/config.yml`
f0in=`awk '{if ($1 == "f0in:") print $2}' conf/config.yml`
n_workers=`awk '{if ($1 == "n_workers:") print $2}' conf/config.yml`
pad_len=`awk '{if ($1 == "pad_len:") print $2}' conf/config.yml`
detach=`awk '{if ($1 == "detach:") print $2}' conf/config.yml`
spkidtr_dim=`awk '{if ($1 == "spkidtr_dim:") print $2}' conf/config.yml`
right_size=`awk '{if ($1 == "right_size:") print $2}' conf/config.yml`

### settings for neural vocoder
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
batch_size_wave=`awk '{if ($1 == "batch_size_wave:") print $2}' conf/config.yml`
batch_size_utt_wave=`awk '{if ($1 == "batch_size_utt_wave:") print $2}' conf/config.yml`
batch_size_utt_eval_wave=`awk '{if ($1 == "batch_size_utt_eval_wave:") print $2}' conf/config.yml`
t_start=`awk '{if ($1 == "t_start:") print $2}' conf/config.yml`
t_end=`awk '{if ($1 == "t_end:") print $2}' conf/config.yml`
interval=`awk '{if ($1 == "interval:") print $2}' conf/config.yml`
densities=`awk '{if ($1 == "densities:") print $2}' conf/config.yml`
n_stage=`awk '{if ($1 == "n_stage:") print $2}' conf/config.yml`
seg=`awk '{if ($1 == "seg:") print $2}' conf/config.yml`
lpc=`awk '{if ($1 == "lpc:") print $2}' conf/config.yml`


#######################################
#     DECODING/FINE-TUNING SETTING    #
#######################################

idx_resume=
idx_resume_wave=

#min_idx=65 #spec,vae-laplace,vcc18
#min_idx=67 #spec-excit,vae-laplace,vcc18
min_idx=72 #spec-excit-spktr,vae-laplace,vcc18
#min_idx=75 #spec,vae-vq,vcc18
#min_idx=75 #spec-excit,vae-vq,vcc18
#min_idx=81 #spec-excit-spktr,vae-vq,vcc18

n_interp=0
#n_interp=10 #for speaker interpolation in 2-dim space with spec-excit cyclevae/cyclevqvae
#n_interp=20

min_idx_wave=

gv_coeff=`awk '{if ($1 == "gv_coeff:") print $2}' conf/config.yml`

if [ $mdl_name == "cycmceplf0capvae-laplace" ]; then
    string_path_rec=/feat_rec_${mdl_name}${detach}-${epoch_count}-${lat_dim}-${spkidtr_dim}-${n_half_cyc}-${min_idx}
    string_path_cv=/feat_cv_${mdl_name}${detach}-${epoch_count}-${lat_dim}-${spkidtr_dim}-${n_half_cyc}-${min_idx}
elif [ $mdl_name == "cycmceplf0capvae-vq" ]; then
    string_path_rec=/feat_rec_${mdl_name}${detach}-${epoch_count}-${lat_dim}-${spkidtr_dim}-${n_half_cyc}-${min_idx}
    string_path_cv=/feat_cv_${mdl_name}${detach}-${epoch_count}-${lat_dim}-${spkidtr_dim}-${n_half_cyc}-${min_idx}
elif [ $mdl_name == "cycmcepvae-laplace" ]; then
    string_path_rec=/feat_rec_${mdl_name}${detach}-${epoch_count}-${lat_dim}-${n_half_cyc}-${min_idx}
    string_path_cv=/feat_cv_${mdl_name}${detach}-${epoch_count}-${lat_dim}-${n_half_cyc}-${min_idx}
elif [ $mdl_name == "cycmcepvae-vq" ]; then
    string_path_rec=/feat_rec_${mdl_name}${detach}-${epoch_count}-${lat_dim}-${n_half_cyc}-${min_idx}
    string_path_cv=/feat_cv_${mdl_name}${detach}-${epoch_count}-${lat_dim}-${n_half_cyc}-${min_idx}
fi

GPU_device_str="0,1,2"
GPU_device_str="1,2,0"
GPU_device_str="2,0,1"
#GPU_device_str="2,1,0"
#GPU_device_str="0,2,1"
#GPU_device_str="1,0,2"
#GPU_device_str="1,2"
#GPU_device_str="2,0"
#GPU_device_str="0"
#GPU_device_str="1"
#GPU_device_str="2"

#n_gpus=1
#n_gpus=2
n_gpus=3

spks_trg_rec=(VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 VCC2SF3 VCC2SM3 VCC2TF1 VCC2TM1 VCC2TF2 VCC2TM2 VCC2SM4 VCC2SF4)
#spks_trg_rec=(VCC2TF1)
#spks_trg_rec=(VCC2SF3)

#spks_src_dec=(VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 VCC2SF3 VCC2SM3 VCC2TF1 VCC2TM1 VCC2TF2 VCC2TM2 VCC2SM4 VCC2SF4)
#spks_trg_dec=(VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 VCC2SF3 VCC2SM3 VCC2TF1 VCC2TM1 VCC2TF2 VCC2TM2 VCC2SM4 VCC2SF4)
spks_src_dec=(VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 VCC2SF3 VCC2SM3 VCC2SM4 VCC2SF4)
spks_trg_dec=(VCC2TF1 VCC2TM1 VCC2TF2 VCC2TM2)
#spks_src_dec=(VCC2SF1)
#spks_src_dec=(VCC2SM2)
#spks_trg_dec=(VCC2TF1)
#spks_trg_dec=(VCC2SF3)

#spk_ft=VCC2TF1

echo $GPU_device $GPU_device_str
echo $min_idx $idx_resume $epoch_count $batch_size $batch_size_utt

init_acc=false
#init_acc=true

decode_batch_size=1
#decode_batch_size=2
decode_batch_size=5
#decode_batch_size=10

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
        if [ -n "$(echo $spk | sed -n 's/\(bdl\)/\1/p')" ] \
            || [ -n "$(echo $spk | sed -n 's/\(slt\)/\1/p')" ]; then
            echo arctic $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 20 >> data/${dev}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n +21 | head -n -104 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 104 >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(VCC\)/\1/p')" ]; then
            echo vcc18 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n +11 >> data/${trn}/wav.scp
            find ${wav_org_dir}/test/${spk} -name "*.wav" \
                | sort >> data/${tst}/wav.scp
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
            echo $set
            expdir=exp/feature_extract/${set}
            mkdir -p $expdir
            rm -f $expdir/feature_extract.log
            for spk in ${spks[@]}; do
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
    echo "###########################################################"
    echo "#                   NOISE SHAPING STEP                    #"
    echo "###########################################################"
    if true; then
    #if false; then
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
fi
# }}}


stats_list=()
feats_eval_list=()
for spk in ${spks[@]};do
    stats_list+=(data/${trn}/stats_spk-${spk}.h5)
    if [ -n "$(echo $spk | sed -n 's/\(p\)/\1/p')" ]; then
        touch data/${dev}/feats_spk-${spk}.scp
    fi
    feats_eval_list+=(data/${dev}/feats_spk-${spk}.scp)
done

stats_list_list="$(IFS="@"; echo "${stats_list[*]}")"
feats_list_eval_list="$(IFS="@"; echo "${feats_eval_list[*]}")"

spk_list="$(IFS="@"; echo "${spks[*]}")"
echo ${spk_list}

echo $mdl_name
if [ $mdl_name == "cycmcepvae-laplace" ]; then
    setting=${mdl_name}_${data_name}_lr${lr}_bs${batch_size}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}_lat${lat_dim}_hue${hidden_units_enc}_hle${hidden_layers_enc}_hud${hidden_units_dec}_hld${hidden_layers_dec}_kse${kernel_size_enc}_dse${dilation_size_enc}_ksd${kernel_size_dec}_dsd${dilation_size_dec}_rs${right_size}_bie${bi_enc}_bid${bi_dec}_do${do_prob}_ep${epoch_count}_mcep${powmcep_dim}_nhcyc${n_half_cyc}_are${ar_enc}_ard${ar_dec}_f0in${f0in}_detach${detach}_di${diff}
elif [ $mdl_name == "cycmceplf0capvae-laplace" ]; then
    setting=${mdl_name}_${data_name}_lr${lr}_bs${batch_size}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hle${hidden_layers_enc}_hud${hidden_units_dec}_hld${hidden_layers_dec}_hlf${hidden_layers_lf0}_kse${kernel_size_enc}_dse${dilation_size_enc}_ksd${kernel_size_dec}_ksf${kernel_size_lf0}_dsd${dilation_size_dec}_dsf${dilation_size_lf0}_rs${right_size}_bie${bi_enc}_bid${bi_dec}_bif${bi_lf0}_do${do_prob}_ep${epoch_count}_mcep${powmcep_dim}_nhcyc${n_half_cyc}_are${ar_enc}_ard${ar_dec}_arf0${ar_f0}_d${detach}_s${spkidtr_dim}_di${diff}
elif [ $mdl_name == "cycmcepvae-vq" ]; then
    setting=${mdl_name}_${data_name}_lr${lr}_bs${batch_size}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}_lat${lat_dim}_hue${hidden_units_enc}_hle${hidden_layers_enc}_hud${hidden_units_dec}_hld${hidden_layers_dec}_kse${kernel_size_enc}_dse${dilation_size_enc}_ksd${kernel_size_dec}_dsd${dilation_size_dec}_rs${right_size}_bie${bi_enc}_bid${bi_dec}_do${do_prob}_ep${epoch_count}_mcep${powmcep_dim}_nhcyc${n_half_cyc}_are${ar_enc}_ard${ar_dec}_d${detach}_ctr${ctr_size}
elif [ $mdl_name == "cycmceplf0capvae-vq" ]; then
    setting=${mdl_name}_${data_name}_lr${lr}_bs${batch_size}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hle${hidden_layers_enc}_hud${hidden_units_dec}_hld${hidden_layers_dec}_hlf${hidden_layers_lf0}_kse${kernel_size_enc}_dse${dilation_size_enc}_ksd${kernel_size_dec}_ksf${kernel_size_lf0}_dsd${dilation_size_dec}_dsf${dilation_size_lf0}_rs${right_size}_bie${bi_enc}_bid${bi_dec}_bif${bi_lf0}_do${do_prob}_ep${epoch_count}_mcep${powmcep_dim}_nhcyc${n_half_cyc}_are${ar_enc}_ard${ar_dec}_arf0${ar_f0}_d${detach}_s${spkidtr_dim}_ctr${ctr_size}
fi

# STAGE 4 {{
# set variables
expdir=exp/tr_${setting}
mkdir -p $expdir
if [ `echo ${stage} | grep 4` ];then
    echo "###########################################################"
    echo "#               FEATURE MODELING STEP                     #"
    echo "###########################################################"
    echo $expdir

    if [ $mdl_name == "cycmcepvae-laplace" ];then
        feats=data/${trn}/feats.scp
        #${cuda_cmd} ${expdir}/log/train_resume-${idx_resume}.log \
        ${cuda_cmd} ${expdir}/log/train.log \
            train_gru-cycle-mcepvae-laplace.py \
                --feats ${feats} \
                --feats_eval_list $feats_list_eval_list \
                --stats data/${trn}/stats_jnt.h5 \
                --expdir ${expdir} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --mcep_dim ${powmcep_dim} \
                --lat_dim ${lat_dim} \
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
                --bi_enc ${bi_enc} \
                --bi_dec ${bi_dec} \
                --ar_enc ${ar_enc} \
                --ar_dec ${ar_dec} \
                --f0in ${f0in} \
                --causal_conv_enc ${causal_conv_enc} \
                --causal_conv_dec ${causal_conv_dec} \
                --batch_size ${batch_size} \
                --batch_size_utt ${batch_size_utt} \
                --batch_size_utt_eval ${batch_size_utt_eval} \
                --n_half_cyc ${n_half_cyc} \
                --n_workers ${n_workers} \
                --pad_len ${pad_len} \
                --string_path ${string_path} \
                --detach ${detach} \
                --diff ${diff} \
                --right_size ${right_size} \
                --GPU_device ${GPU_device}
                #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
    elif [ $mdl_name == "cycmceplf0capvae-laplace" ];then
        feats=data/${trn}/feats.scp
        #${cuda_cmd} ${expdir}/log/train_resume-${idx_resume}.log \
        ${cuda_cmd} ${expdir}/log/train.log \
            train_gru-cycle-mcep-lf0cap-vae-laplace.py \
                --feats ${feats} \
                --feats_eval_list $feats_list_eval_list \
                --stats data/${trn}/stats_jnt.h5 \
                --expdir ${expdir} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --mcep_dim ${powmcep_dim} \
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
                --bi_enc ${bi_enc} \
                --bi_dec ${bi_dec} \
                --bi_lf0 ${bi_lf0} \
                --ar_enc ${ar_enc} \
                --ar_dec ${ar_dec} \
                --ar_f0 ${ar_f0} \
                --diff ${diff} \
                --causal_conv_enc ${causal_conv_enc} \
                --causal_conv_dec ${causal_conv_dec} \
                --causal_conv_lf0 ${causal_conv_lf0} \
                --batch_size ${batch_size} \
                --batch_size_utt ${batch_size_utt} \
                --batch_size_utt_eval ${batch_size_utt_eval} \
                --n_half_cyc ${n_half_cyc} \
                --n_workers ${n_workers} \
                --pad_len ${pad_len} \
                --string_path ${string_path} \
                --detach ${detach} \
                --spkidtr_dim ${spkidtr_dim} \
                --right_size ${right_size} \
                --GPU_device ${GPU_device}
                #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
    elif [ $mdl_name == "cycmcepvae-vq" ];then
        feats=data/${trn}/feats.scp
        #${cuda_cmd} ${expdir}/log/train_resume-${idx_resume}.log \
        ${cuda_cmd} ${expdir}/log/train.log \
            train_gru-cycle-mcepvae-vq.py \
                --feats ${feats} \
                --feats_eval_list $feats_list_eval_list \
                --stats data/${trn}/stats_jnt.h5 \
                --expdir ${expdir} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --mcep_dim ${powmcep_dim} \
                --lat_dim ${lat_dim} \
                --ctr_size ${ctr_size} \
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
                --bi_enc ${bi_enc} \
                --bi_dec ${bi_dec} \
                --ar_enc ${ar_enc} \
                --ar_dec ${ar_dec} \
                --causal_conv_enc ${causal_conv_enc} \
                --causal_conv_dec ${causal_conv_dec} \
                --batch_size ${batch_size} \
                --batch_size_utt ${batch_size_utt} \
                --batch_size_utt_eval ${batch_size_utt_eval} \
                --n_half_cyc ${n_half_cyc} \
                --n_workers ${n_workers} \
                --pad_len ${pad_len} \
                --string_path ${string_path} \
                --detach ${detach} \
                --right_size ${right_size} \
                --GPU_device ${GPU_device}
                #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
    elif [ $mdl_name == "cycmceplf0capvae-vq" ];then
        feats=data/${trn}/feats.scp
        #${cuda_cmd} ${expdir}/log/train_resume-${idx_resume}.log \
        ${cuda_cmd} ${expdir}/log/train.log \
            train_gru-cycle-mcep-lf0cap-vae-vq.py \
                --feats ${feats} \
                --feats_eval_list $feats_list_eval_list \
                --stats data/${trn}/stats_jnt.h5 \
                --expdir ${expdir} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --mcep_dim ${powmcep_dim} \
                --lat_dim ${lat_dim} \
                --lat_dim_e ${lat_dim_e} \
                --ctr_size ${ctr_size} \
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
                --bi_enc ${bi_enc} \
                --bi_dec ${bi_dec} \
                --bi_lf0 ${bi_lf0} \
                --ar_enc ${ar_enc} \
                --ar_dec ${ar_dec} \
                --ar_f0 ${ar_f0} \
                --causal_conv_enc ${causal_conv_enc} \
                --causal_conv_dec ${causal_conv_dec} \
                --causal_conv_lf0 ${causal_conv_lf0} \
                --batch_size ${batch_size} \
                --batch_size_utt ${batch_size_utt} \
                --batch_size_utt_eval ${batch_size_utt_eval} \
                --n_half_cyc ${n_half_cyc} \
                --n_workers ${n_workers} \
                --pad_len ${pad_len} \
                --string_path ${string_path} \
                --detach ${detach} \
                --spkidtr_dim ${spkidtr_dim} \
                --right_size ${right_size} \
                --GPU_device ${GPU_device}
                #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
    fi
fi
# }}}


# STAGE 5 {{{
if [ `echo ${stage} | grep 5` ];then
    echo $expdir $n_gpus $GPU_device $GPU_device_str
    config=${expdir}/model.conf
    model=${expdir}/checkpoint-${min_idx}.pkl
    for spk_trg in ${spks_trg_rec[@]};do
        if true; then
        #if false; then
                echo "########################################################"
                echo "#          DECODING RECONST. FEAT and GV stat          #"
                echo "########################################################"
                echo $spk_trg $spk_src $min_idx
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
                if [ $mdl_name == "cycmcepvae-laplace" ];then
                    ${cuda_cmd} ${expdir}/log/decode_rec-cycrec_${spk_trg}_${min_idx}.log \
                        calc_rec-cycrec-gv_gru-cycle-mcepvae-laplace.py \
                            --feats ${feats_scp} \
                            --spk ${spk_trg} \
                            --outdir ${outdir} \
                            --model ${model} \
                            --config ${config} \
                            --GPU_device_str ${GPU_device_str} \
                            --string_path ${string_path_rec} \
                            --n_gpus ${n_gpus}
                            #--GPU_device ${GPU_device} \
                elif [ $mdl_name == "cycmcepvae-vq" ];then
                    ${cuda_cmd} ${expdir}/log/decode_rec-cycrec_${spk_trg}_${min_idx}.log \
                        calc_rec-cycrec-gv_gru-cycle-mcepvae-vq.py \
                            --feats ${feats_scp} \
                            --spk ${spk_trg} \
                            --outdir ${outdir} \
                            --model ${model} \
                            --config ${config} \
                            --GPU_device_str ${GPU_device_str} \
                            --string_path ${string_path_rec} \
                            --n_gpus ${n_gpus}
                            #--GPU_device ${GPU_device} \
                elif [ $mdl_name == "cycmceplf0capvae-laplace" ];then
                    ${cuda_cmd} ${expdir}/log/decode_rec-cycrec_${spk_trg}_${min_idx}.log \
                        calc_rec-cycrec-gv_gru-cycle-mceplf0capvae-laplace.py \
                            --feats ${feats_scp} \
                            --spk ${spk_trg} \
                            --outdir ${outdir} \
                            --model ${model} \
                            --config ${config} \
                            --GPU_device_str ${GPU_device_str} \
                            --string_path ${string_path_rec} \
                            --n_gpus ${n_gpus}
                            #--GPU_device ${GPU_device} \
                elif [ $mdl_name == "cycmceplf0capvae-vq" ];then
                    ${cuda_cmd} ${expdir}/log/decode_rec-cycrec_${spk_trg}_${min_idx}.log \
                        calc_rec-cycrec-gv_gru-cycle-mceplf0capvae-vq.py \
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
        ## org_trn/rec_trn
        find hdf5/${trn}/${spk} -name "*.h5" | sort >> ${feats_ft_scp}
        cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ft_scp}
        find hdf5/${trn}/${spk}-${spk} -name "*.h5" | sort >> ${feats_ft_scp}
        cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ft_scp}
        find hdf5/${trn}/${spk}-${spk}-${spk} -name "*.h5" | sort >> ${feats_ft_scp}
        cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ft_scp}
        n_feats=`find hdf5/${dev}/${spk} -name "*.h5" | wc -l `
        if [ $n_feats -gt 0 ]; then
            find hdf5/${dev}/${spk} -name "*.h5" | sort >> ${feats_ft_eval_scp}
            cat ${waveforms_eval} | grep "\/${spk}\/" >> ${waveforms_ft_eval_scp}
        fi
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
    if true; then
    #if false; then
    if [ $mdl_name == "cycmceplf0capvae-laplace" ]; then
        echo "######################################################"
        echo "#                PLOTTING SPEAKER SPACE              #"
        echo "######################################################"
        config=${expdir}/model.conf
        model=${expdir}/checkpoint-${min_idx}.pkl
        outdir=${expdir}/spk-map_${mdl_name}-${data_name}-${lat_dim}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}
            ${cuda_cmd} ${expdir}/log/decode_spk-map_${min_idx}.log \
                decode_gru-cycle-mceplf0capvae-laplace_spkidtr_map.py \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config}
    elif [ $mdl_name == "cycmceplf0capvae-vq" ]; then
        echo "######################################################"
        echo "#                PLOTTING SPEAKER SPACE              #"
        echo "######################################################"
        config=${expdir}/model.conf
        model=${expdir}/checkpoint-${min_idx}.pkl
        outdir=${expdir}/spk-map_${mdl_name}-${data_name}-${lat_dim}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}
            ${cuda_cmd} ${expdir}/log/decode_spk-map_${min_idx}.log \
                decode_gru-cycle-mceplf0capvae-vq_spkidtr_map.py \
                    --outdir ${outdir} \
                    --model ${model} \
                    --config ${config}

    fi
    fi
for spkr in ${spks_src_dec[@]};do
#if [ -n "$(echo $spkr | sed -n 's/\(S\)/\1/p')" ]; then
for spk_trg in ${spks_trg_dec[@]};do
if [ $spkr != $spk_trg ]; then
echo $spkr $spk_trg $min_idx
    echo $expdir $n_gpus $GPU_device $GPU_device_str

    config=${expdir}/model.conf
    model=${expdir}/checkpoint-${min_idx}.pkl

    h5outdir=hdf5/${dev}/${spkr}-${spk_trg}
    echo $h5outdir
    if true; then
    #if false; then
    echo "######################################################"
    echo "#                DECODING CONV. FEAT DEV             #"
    echo "######################################################"
    if [ $mdl_name == "cycmceplf0capvae-laplace" ]; then
        outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}_${spkr}-${spk_trg}-${n_interp}_dev
    elif [ $mdl_name == "cycmceplf0capvae-vq" ]; then
        outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${ctr_size}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}_${spkr}-${spk_trg}-${n_interp}_dev
    elif [ $mdl_name == "cycmcepvae-laplace" ]; then
        outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}_${spkr}-${spk_trg}_dev
    elif [ $mdl_name == "cycmcepvae-vq" ]; then
        outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${ctr_size}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}_${spkr}-${spk_trg}_dev
    fi
    mkdir -p ${outdir}
    feats_scp=${outdir}/feats.scp
    cat data/${dev}/feats.scp | grep "\/${spkr}\/" > ${feats_scp}
    if [ $mdl_name == "cycmcepvae-laplace" ]; then
        ${cuda_cmd} ${expdir}/log/decode_dev_${min_idx}_${spkr}-${spk_trg}.log \
            decode_gru-cycle-mcepvae-laplace.py \
                --feats ${feats_scp} \
                --spk_trg ${spk_trg} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --gv_coeff ${gv_coeff} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
    elif [ $mdl_name == "cycmceplf0capvae-laplace" ]; then
        ${cuda_cmd} ${expdir}/log/decode_dev_${min_idx}_${spkr}-${spk_trg}-${n_interp}.log \
            decode_gru-cycle-mceplf0capvae-laplace.py \
                --feats ${feats_scp} \
                --spk_trg ${spk_trg} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --n_interp ${n_interp} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
    elif [ $mdl_name == "cycmcepvae-vq" ]; then
        ${cuda_cmd} ${expdir}/log/decode_dev_${min_idx}_${spkr}-${spk_trg}.log \
            decode_gru-cycle-mcepvae-vq.py \
                --feats ${feats_scp} \
                --spk_trg ${spk_trg} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --gv_coeff ${gv_coeff} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
    elif [ $mdl_name == "cycmceplf0capvae-vq" ]; then
        ${cuda_cmd} ${expdir}/log/decode_dev_${min_idx}_${spkr}-${spk_trg}-${n_interp}.log \
            decode_gru-cycle-mceplf0capvae-vq.py \
                --feats ${feats_scp} \
                --spk_trg ${spk_trg} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --n_interp ${n_interp} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
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
    if [ $mdl_name == "cycmceplf0capvae-laplace" ]; then
        outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}_${spkr}-${spk_trg}-${n_interp}_tst
    elif [ $mdl_name == "cycmceplf0capvae-vq" ]; then
        outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${ctr_size}-${spkidtr_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}_${spkr}-${spk_trg}-${n_interp}_tst
    elif [ $mdl_name == "cycmcepvae-laplace" ]; then
        outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}_${spkr}-${spk_trg}_tst
    elif [ $mdl_name == "cycmcepvae-vq" ]; then
        outdir=${expdir}/wav_cv_${mdl_name}-${data_name}-${lat_dim}-${ctr_size}-${n_half_cyc}-${epoch_count}-${batch_size}-${batch_size_utt}-${min_idx}_${spkr}-${spk_trg}_tst
    fi
    mkdir -p ${outdir}
    feats_scp=${outdir}/feats.scp
    cat data/${tst}/feats.scp | grep "\/${spkr}\/" > ${feats_scp}
    if [ $mdl_name == "cycmcepvae-laplace" ]; then
        ${cuda_cmd} ${expdir}/log/decode_tst_${min_idx}_${spkr}-${spk_trg}.log \
            decode_gru-cycle-mcepvae-laplace.py \
                --feats ${feats_scp} \
                --spk_trg ${spk_trg} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
    elif [ $mdl_name == "cycmceplf0capvae-laplace" ]; then
        ${cuda_cmd} ${expdir}/log/decode_tst_${min_idx}_${spkr}-${spk_trg}-${n_interp}.log \
            decode_gru-cycle-mceplf0capvae-laplace.py \
                --feats ${feats_scp} \
                --spk_trg ${spk_trg} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --n_interp ${n_interp} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
    elif [ $mdl_name == "cycmcepvae-vq" ]; then
        ${cuda_cmd} ${expdir}/log/decode_tst_${min_idx}_${spkr}-${spk_trg}.log \
            decode_gru-cycle-mcepvae-vq.py \
                --feats ${feats_scp} \
                --spk_trg ${spk_trg} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
    elif [ $mdl_name == "cycmceplf0capvae-vq" ]; then
        ${cuda_cmd} ${expdir}/log/decode_tst_${min_idx}_${spkr}-${spk_trg}-${n_interp}.log \
            decode_gru-cycle-mceplf0capvae-vq.py \
                --feats ${feats_scp} \
                --spk_trg ${spk_trg} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --string_path ${string_path_cv} \
                --n_interp ${n_interp} \
                --GPU_device_str ${GPU_device_str}
                #--GPU_device ${GPU_device} \
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
if [ $mdl_name_wave == "wavenet" ]; then
    setting_wave=${mdl_name_wave}_${data_name}_lr${lr}_bs${batch_size}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}_dd${dilation_depth}_dr${dilation_repeat}_ks${kernel_size}_hc${hid_chn}_sc${skip_chn}_ksw${kernel_size_wave}_dsw${dilation_size_wave}_do${do_prob}_ep${epoch_count}_mcep${use_mcep}
elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpc" ]; then
    setting_wave=${mdl_name_wave}_${data_name}_lr${lr}_bs${batch_size_wave}_bsu${batch_size_utt_wave}_bsue${batch_size_utt_eval_wave}_huw${hidden_units_wave}_hu2w${hidden_units_wave_2}_ksw${kernel_size_wave}_dsw${dilation_size_wave}_do${do_prob}_ep${epoch_count}_mcep${use_mcep}_ts${t_start}_te${t_end}_i${interval}_d${densities}_ns${n_stage}_lpc${lpc}
elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpcseg" ]; then
    setting_wave=${mdl_name_wave}_${data_name}_lr${lr}_bs${batch_size_wave}_bsu${batch_size_utt_wave}_bsue${batch_size_utt_eval_wave}_huw${hidden_units_wave}_hu2w${hidden_units_wave_2}_ksw${kernel_size_wave}_dsw${dilation_size_wave}_do${do_prob}_ep${epoch_count}_mcep${use_mcep}_ts${t_start}_te${t_end}_i${interval}_d${densities}_ns${n_stage}_lpc${lpc}
fi

# STAGE 7 {{
# set variables
expdir_wave=exp/tr_${setting_wave}
mkdir -p $expdir_wave
if [ `echo ${stage} | grep 7` ];then
    echo "###########################################################"
    echo "#               WAVEFORM MODELING STEP                    #"
    echo "###########################################################"
    echo $expdir_wave
    
    if [ $mdl_name_wave == "wavenet" ];then
        feats=${expdir_wave}/feats_tr.scp
        feats_eval=${expdir_wave}/feats_ev.scp
        waveforms=${expdir_wave}/wavs_tr.scp
        waveforms_eval=${expdir_wave}/wavs_ev.scp
        ### Use these if not using reconst./cyclic reconst. feats
        #cat data/${trn}/feats.scp | sort > ${feats}
        #cat data/${dev}/feats.scp | sort > ${feats_eval}
        #cat data/${trn}/wav_ns.scp | sort > ${waveforms}
        #cat data/${dev}/wav_ns.scp | sort > ${waveforms_eval}
        ### Use these if using reconst./cyclic reconst. feats
        cat data/${trn}/feats_ft.scp | sort > ${feats}
        cat data/${dev}/feats_ft.scp | sort > ${feats_eval}
        cat data/${trn}/wav_ns_ft.scp | sort > ${waveforms}
        cat data/${dev}/wav_ns_ft.scp | sort > ${waveforms_eval}
                #--init true \
        #${cuda_cmd} ${expdir_wave}/log/train_resume-${idx_resume}.log \
        ${cuda_cmd} ${expdir_wave}/log/train.log \
            train_wavenet.py \
                --waveforms ${waveforms} \
                --waveforms_eval $waveforms_eval \
                --feats ${feats} \
                --feats_eval $feats_eval \
                --stats data/${trn}/stats_jnt.h5 \
                --expdir ${expdir_wave} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --upsampling_factor ${upsampling_factor} \
                --dilation_depth ${dilation_depth} \
                --dilation_repeat ${dilation_repeat} \
                --kernel_size ${kernel_size} \
                --hid_chn ${hid_chn} \
                --skip_chn ${skip_chn} \
                --batch_size ${batch_size} \
                --string_path ${string_path_rec} \
                --mcep_dim ${powmcep_dim} \
                --kernel_size_wave ${kernel_size_wave} \
                --dilation_size_wave ${dilation_size_wave} \
                --batch_size_utt ${batch_size_utt} \
                --batch_size_utt_eval ${batch_size_utt_eval} \
                --n_workers ${n_workers} \
                --pad_len ${pad_len} \
                --GPU_device ${GPU_device}
                #--string_path ${string_path} \
                #--resume ${expdir_wave}/checkpoint-${idx_resume}.pkl \
                #--mcep_dim ${mel_dim} \
                #--n_aux ${n_aux} \
                #--aux_kernel_size ${kernel_size_wave} \
                #--aux_dilation_size ${dilation_size_wave} \
    elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpc" ];then
        feats=${expdir_wave}/feats_tr.scp
        feats_eval=${expdir_wave}/feats_ev.scp
        waveforms=${expdir_wave}/wavs_tr.scp
        waveforms_eval=${expdir_wave}/wavs_ev.scp
        ### Use these if not using reconst./cyclic reconst. feats
        cat data/${trn}/feats.scp | sort > ${feats}
        cat data/${dev}/feats.scp | sort > ${feats_eval}
        cat data/${trn}/wav_ns.scp | sort > ${waveforms}
        cat data/${dev}/wav_ns.scp | sort > ${waveforms_eval}
        ### Use these if using reconst./cyclic reconst. feats
        #cat data/${trn}/feats_ft.scp | sort > ${feats}
        #cat data/${dev}/feats_ft.scp | sort > ${feats_eval}
        #cat data/${trn}/wav_ns_ft.scp | sort > ${waveforms}
        #cat data/${dev}/wav_ns_ft.scp | sort > ${waveforms_eval}
                #--init true \
        #${cuda_cmd} ${expdir_wave}/log/train_resume-${idx_resume_wave}.log \
        ${cuda_cmd} ${expdir_wave}/log/train.log \
            train_nstages-sparse-wavernn_dualgru_compact_lpc.py \
                --waveforms ${waveforms} \
                --waveforms_eval $waveforms_eval \
                --feats ${feats} \
                --feats_eval $feats_eval \
                --stats data/${trn}/stats_jnt.h5 \
                --expdir ${expdir_wave} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --upsampling_factor ${upsampling_factor} \
                --hidden_units_wave ${hidden_units_wave} \
                --hidden_units_wave_2 ${hidden_units_wave_2} \
                --batch_size ${batch_size_wave} \
                --string_path ${string_path} \
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
                --GPU_device ${GPU_device}
                #--resume ${expdir_wave}/checkpoint-${idx_resume_wave}.pkl \
                #--string_path ${string_path_rec} \
                #--mcep_dim ${mel_dim} \
    elif [ $mdl_name_wave == "wavernn_dualgru_compact_lpcseg" ];then
        feats=${expdir_wave}/feats_tr.scp
        feats_eval=${expdir_wave}/feats_ev.scp
        waveforms=${expdir_wave}/wavs_tr.scp
        waveforms_eval=${expdir_wave}/wavs_ev.scp
        ### Use these if not using reconst./cyclic reconst. feats
        cat data/${trn}/feats.scp | sort > ${feats}
        cat data/${dev}/feats.scp | sort > ${feats_eval}
        cat data/${trn}/wav_ns.scp | sort > ${waveforms}
        cat data/${dev}/wav_ns.scp | sort > ${waveforms_eval}
        ### Use these if using reconst./cyclic reconst. feats
        #cat data/${trn}/feats_ft.scp | sort > ${feats}
        #cat data/${dev}/feats_ft.scp | sort > ${feats_eval}
        #cat data/${trn}/wav_ns_ft.scp | sort > ${waveforms}
        #cat data/${dev}/wav_ns_ft.scp | sort > ${waveforms_eval}
                #--init true \
        #${cuda_cmd} ${expdir_wave}/log/train_resume-${idx_resume_wave}.log \
        ${cuda_cmd} ${expdir_wave}/log/train.log \
            train_nstages-sparse-wavernn_dualgru_compact_lpcseg.py \
                --waveforms ${waveforms} \
                --waveforms_eval $waveforms_eval \
                --feats ${feats} \
                --feats_eval $feats_eval \
                --stats data/${trn}/stats_jnt.h5 \
                --expdir ${expdir_wave} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --upsampling_factor ${upsampling_factor} \
                --hidden_units_wave ${hidden_units_wave} \
                --hidden_units_wave_2 ${hidden_units_wave_2} \
                --batch_size ${batch_size_wave} \
                --string_path ${string_path} \
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
                --seg ${seg} \
                --GPU_device ${GPU_device}
                #--resume ${expdir_wave}/checkpoint-${idx_resume_wave}.pkl \
                #--string_path ${string_path_rec} \
                #--mcep_dim ${mel_dim} \
    fi
fi
# }}}
