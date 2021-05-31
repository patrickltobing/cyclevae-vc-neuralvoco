#!/bin/bash
###################################################################################################
#        SCRIPT FOR High-Fidelity and Low-Latency Universal Neural Vocoder based on               #
#        Multiband WaveRNN with Data-driven Linear Prediction (MWDLP)                             #
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
# 4: mwdlp training step
# 5: copy-synthesis mwdlp using gpu
# 6: restore noise-shaping copy-synthesis mwdlp
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
#stage=4
#stage=5
#stage=56
#stage=6

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
#GPU_device=1
#GPU_device=2
#GPU_device=3
GPU_device=4
#GPU_device=5
GPU_device=6
#GPU_device=7
#GPU_device=8
#GPU_device=9

string_path="/log_1pmelmagsp"

lr=`awk '{if ($1 == "lr:") print $2}' conf/config.yml`
do_prob=`awk '{if ($1 == "do_prob:") print $2}' conf/config.yml`
n_workers=`awk '{if ($1 == "n_workers:") print $2}' conf/config.yml`

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
mid_dim=`awk '{if ($1 == "mid_dim:") print $2}' conf/config.yml`


#######################################
#          DECODING SETTING           #
#######################################

### Set GPU_device_str and n_gpus for GPU decoding with synchronized values
GPU_device_str="0"
GPU_device_str="4,6"
#GPU_device_str="0,1,2"
GPU_device_str="0,5,2,7,6"

n_gpus=1
n_gpus=2
#n_gpus=3
n_gpus=5
###

### This is for speakers that will be used in analysis-synthesis
spks_dec=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
spks_dec=(TEM2 SEF2)
spks_dec=(TEM2)
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

#echo $data_name

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
            fi
            if [ -f "conf/${spk}.pow" ]; then
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
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


if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb" ]; then
    setting_wave=${mdl_name_wave}_${data_name}_lr${lr}_bs${batch_size_wave}_huw${hidden_units_wave}_hu2w${hidden_units_wave_2}_ksw${kernel_size_wave}_dsw${dilation_size_wave}_do${do_prob}_st${step_count_wave}_mel${mel_dim}_ts${t_start}_te${t_end}_i${interval}_d${densities}_ns${n_stage}_lpc${lpc}_rs${right_size_wave}_nb${n_bands}_m${mid_dim}
fi


# STAGE 4 {{
# set variables
expdir_wave=exp/tr_${setting_wave}
if [ `echo ${stage} | grep 4` ];then
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
    feats=data/${trn}/feats.scp
    feats_eval=data/${dev}/feats.scp
    waveforms=data/${trn}/wav_ns.scp
    waveforms_eval=data/${dev}/wav_ns.scp
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb" ];then
        if [ $idx_resume_wave -gt 0 ]; then
            echo ""
            echo "mwdlp model is in training, please use less/vim to monitor the training log: ${expdir_wave}/log/train_resume-${idx_resume_wave}.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_wave}/log/train_resume-${idx_resume_wave}.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_10bit_cf_smpl_orgx_emb.py \
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
                    --mid_dim ${mid_dim} \
                    --resume ${expdir_wave}/checkpoint-${idx_resume_wave}.pkl \
                    --GPU_device ${GPU_device}
        else
            echo ""
            echo "mwdlp model is in training, please use less/vim to monitor the training log: ${expdir_wave}/log/train.log"
            echo ""
            echo "while opening the log file, please use phrase 'sme' or 'average' to quickly search for the summary on each epoch"
            ${cuda_cmd} ${expdir_wave}/log/train.log \
                train_nstages-sparse-wavernn_dualgru_compact_lpc_mband_10bit_cf_smpl_orgx_emb.py \
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
                    --mid_dim ${mid_dim} \
                    --GPU_device ${GPU_device}
        fi
        echo ""
        echo "mwdlp training finished, please check the log file, and try to compile/decode real-time"
        echo ""
    fi
fi
# }}}


if [ `echo ${stage} | grep 5` ] || [ `echo ${stage} | grep 6` ]; then
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
fi


# STAGE 5 {{{
if [ `echo ${stage} | grep 5` ] || [ `echo ${stage} | grep 6` ];then
for spk_src in ${spks_dec[@]};do
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb" ]; then
        outdir=${expdir_wave}/${mdl_name_wave}-${data_name}_dev-${hidden_units_wave}-${step_count_wave}-${lpc}-${n_bands}-${min_idx_wave}
        #outdir=${expdir_wave}/${mdl_name_wave}-${data_name}_tst-${hidden_units_wave}-${step_count_wave}-${lpc}-${n_bands}-${min_idx_wave}
    fi
if [ `echo ${stage} | grep 5` ];then
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
    if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb" ]; then
        echo ""
        echo "now synthesizing ${spk_src}, log here:  ${expdir_wave}/log/decode_dev_${min_idx_wave}_${spk_src}.log"
        #${cuda_cmd} ${expdir_wave}/log/decode_tst_${min_idx_wave}_${spk_src}.log \
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


# STAGE 6 {{{
if [ `echo ${stage} | grep 6` ];then
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
