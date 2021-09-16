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
# 0: dump model and compile C program
# 1: run analysis-synthesis with real-time demo using cpu
# 2: run analysis-synthesis and mel-spectrogram output/input with real-time demo using cpu
# }}}
stage=0
#stage=1
#stage=2

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
seg_conv_flag_wave=`awk '{if ($1 == "seg_conv_flag_wave:") print $2}' conf/config.yml`
s_dim=`awk '{if ($1 == "s_dim:") print $2}' conf/config.yml`
mid_dim=`awk '{if ($1 == "mid_dim:") print $2}' conf/config.yml`


#######################################
#          DECODING SETTING           #
#######################################

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


# parse options
. parse_options.sh

# stop when error occured
set -e
# }}}


if [ `echo ${stage} | grep 0` ]; then
echo $mdl_name_wave
if [ $mdl_name_wave == "wavernn_dualgru_compact_lpc_mband_10bit_cf_stft_emb_v2" ]; then
    setting_wave=${mdl_name_wave}_${data_name}_lr${lr}_bs${batch_size_wave}_huw${hidden_units_wave}_hu2w${hidden_units_wave_2}_ksw${kernel_size_wave}_dsw${dilation_size_wave}_do${do_prob}_st${step_count_wave}_mel${mel_dim}_ts${t_start}_te${t_end}_i${interval}_d${densities}_ns${n_stage}_lpc${lpc}_rs${right_size_wave}_nb${n_bands}_s${s_dim}_m${mid_dim}_ss${seg_conv_flag_wave}
fi
expdir_wave=exp/tr_${setting_wave}
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
fi
fi


demo_dir=demo_realtime

# STAGE 0 {{{
if [ `echo ${stage} | grep 0` ];then
    echo "###########################################################"
    echo "#        DUMP MODEL AND COMPILE REAL-TIME DEMO STEP       #"
    echo "###########################################################"

    echo ""
    echo "model is been dumping, please check ${expdir_wave}/dump_model.log"
    echo ""
    ${train_cmd} ${expdir_wave}/dump_model.log \
        dump_mwdlp-10b.py \
            ${expdir_wave}/model.conf \
            ${expdir_wave}/checkpoint-${min_idx_wave}.pkl \
            --fs ${fs} \
            --shiftms ${shiftms} \
            --winms ${winms} \
            --fftl ${fftl} \
            --highpass_cutoff ${highpass_cutoff}
    mv -v *.h ${demo_dir}/inc
    mv -v *.c ${demo_dir}/src
    echo ""
    echo "dump model finished"
    echo ""
    echo "now compiling..."
    echo ""
    cd ${demo_dir}
    make clean
    make
    cd ..
    echo ""
    echo "compile finished, please try to run real-time decoding"
    echo ""
fi
# }}}


# STAGE 1 {{{
if [ `echo ${stage} | grep 1` ];then
    echo "###########################################################"
    echo "#        ANALYSIS-SYNTHESIS WITH REAL-TIME DEMO STEP      #"
    echo "###########################################################"

    out_dir=wav_anasyn_realtime
    mkdir -p ${out_dir}
    out_dir=${out_dir}/${data_name}
    mkdir -p ${out_dir}
    for spk_src in ${spks_dec[@]};do
        out_spk_dir=${out_dir}/${spk_src}
        mkdir -p ${out_spk_dir}

        out_spk_dv_dir=${out_spk_dir}/dev
        mkdir -p ${out_spk_dv_dir}
        dv_list=data/${dev}/wav.scp
        wav_dv_scp=${out_spk_dv_dir}/wav.scp
        cat $dv_list | grep "\/${spk_src}\/" | sort | head -n ${n_wav_decode} > ${wav_dv_scp}

        rm -f "${out_spk_dv_dir}/log.txt"
        echo ""
        echo "waveforms are being synthesized, please see the log in ${out_spk_dv_dir}/log.txt"
        echo ""
        while read line;do
            name=`basename $line`
            echo $line ${out_spk_dv_dir}/log.txt
            echo $line >> ${out_spk_dv_dir}/log.txt
            echo ${out_spk_dv_dir}/$name >> ${out_spk_dv_dir}/log.txt
            ./${demo_dir}/bin/test_mwdlp $line ${out_spk_dv_dir}/$name >> ${out_spk_dv_dir}/log.txt
        done < ${wav_dv_scp}

        rm -f ${wav_dv_scp}

        out_spk_ts_dir=${out_spk_dir}/test
        mkdir -p ${out_spk_ts_dir}
        ts_list=data/${tst}/wav.scp
        wav_ts_scp=${out_spk_ts_dir}/wav.scp
        cat $ts_list | grep "\/${spk_src}\/" | sort | head -n ${n_wav_decode} > ${wav_ts_scp}

        rm -f "${out_spk_ts_dir}/log.txt"
        echo ""
        echo "waveforms are being synthesized, please see the log in ${out_spk_ts_dir}/log.txt"
        echo ""
        while read line;do
            name=`basename $line`
            echo $line ${out_spk_ts_dir}/log.txt
            echo $line >> ${out_spk_ts_dir}/log.txt
            echo ${out_spk_ts_dir}/$name >> ${out_spk_ts_dir}/log.txt
            ./${demo_dir}/bin/test_mwdlp $line ${out_spk_ts_dir}/$name >> ${out_spk_ts_dir}/log.txt
        done < ${wav_ts_scp}

        rm -f ${wav_ts_scp}

        echo ""
        echo "synthesis of ${spk_src} finished, outputs are located in ${out_spk_dv_dir} and ${out_spk_ts_dir}"
    done
    echo ""
    echo "synthesis of all speakers finished, outputs are located in respective directories of ${out_dir}"
    echo ""
fi
# }}}


# STAGE 2 {{{
if [ `echo ${stage} | grep 2` ];then
    echo "###########################################################"
    echo "#    SYNTHESIS AND MEL-SPEC OUT/IN WITH REAL-TIME DEMO    #"
    echo "###########################################################"

    out_dir=wav_melsp_realtime
    mkdir -p ${out_dir}
    out_dir=${out_dir}/${data_name}
    mkdir -p ${out_dir}
    for spk_src in ${spks_dec[@]};do
        out_spk_dir=${out_dir}/${spk_src}
        mkdir -p ${out_spk_dir}

        out_spk_dv_dir=${out_spk_dir}/dev
        mkdir -p ${out_spk_dv_dir}
        dv_list=data/${dev}/wav.scp
        wav_dv_scp=${out_spk_dv_dir}/wav.scp
        cat $dv_list | grep "\/${spk_src}\/" | sort | head -n ${n_wav_decode} > ${wav_dv_scp}

        rm -f "${out_spk_dv_dir}/log.txt"
        echo ""
        echo "waveforms & melsp are being synthesized & generated, please see the log in ${out_spk_dv_dir}/log.txt"
        echo ""
        while read line;do
            name=`basename $line .wav`
            echo $line ${out_spk_dv_dir}/log.txt
            echo $line >> ${out_spk_dv_dir}/log.txt
            echo ${out_spk_dv_dir}/${name}_anasyn.wav >> ${out_spk_dv_dir}/log.txt
            ./${demo_dir}/bin/test_mwdlp -o ${out_spk_dv_dir}/${name}_melsp.bin ${out_spk_dv_dir}/${name}_melsp.txt \
                $line ${out_spk_dv_dir}/${name}_anasyn.wav >> ${out_spk_dv_dir}/log.txt
            echo $line >> ${out_spk_dv_dir}/log.txt
            echo ${out_spk_dv_dir}/${name}_binsyn.wav >> ${out_spk_dv_dir}/log.txt
            ./${demo_dir}/bin/test_mwdlp -b ${out_spk_dv_dir}/${name}_melsp.bin ${out_spk_dv_dir}/${name}_binsyn.wav >> ${out_spk_dv_dir}/log.txt
            echo $line >> ${out_spk_dv_dir}/log.txt
            echo ${out_spk_dv_dir}/${name}_txtsyn.wav >> ${out_spk_dv_dir}/log.txt
            ./${demo_dir}/bin/test_mwdlp -t ${out_spk_dv_dir}/${name}_melsp.txt ${out_spk_dv_dir}/${name}_txtsyn.wav >> ${out_spk_dv_dir}/log.txt
        done < ${wav_dv_scp}

        rm -f ${wav_dv_scp}

        out_spk_ts_dir=${out_spk_dir}/test
        mkdir -p ${out_spk_ts_dir}
        ts_list=data/${tst}/wav.scp
        wav_ts_scp=${out_spk_ts_dir}/wav.scp
        cat $ts_list | grep "\/${spk_src}\/" | sort | head -n ${n_wav_decode} > ${wav_ts_scp}

        rm -f "${out_spk_ts_dir}/log.txt"
        echo ""
        echo "waveforms & melsp are being synthesized & generated, please see the log in ${out_spk_ts_dir}/log.txt"
        echo ""
        while read line;do
            name=`basename $line .wav`
            echo $line ${out_spk_ts_dir}/log.txt
            echo $line >> ${out_spk_ts_dir}/log.txt
            echo ${out_spk_ts_dir}/$name >> ${out_spk_ts_dir}/log.txt
            echo ${out_spk_ts_dir}/${name}_anasyn.wav >> ${out_spk_ts_dir}/log.txt
            ./${demo_dir}/bin/test_mwdlp -o ${out_spk_ts_dir}/${name}_melsp.bin ${out_spk_ts_dir}/${name}_melsp.txt \
                $line ${out_spk_ts_dir}/${name}_anasyn.wav >> ${out_spk_ts_dir}/log.txt
            echo $line >> ${out_spk_ts_dir}/log.txt
            echo ${out_spk_ts_dir}/${name}_binsyn.wav >> ${out_spk_ts_dir}/log.txt
            ./${demo_dir}/bin/test_mwdlp -b ${out_spk_ts_dir}/${name}_melsp.bin ${out_spk_ts_dir}/${name}_binsyn.wav >> ${out_spk_ts_dir}/log.txt
            echo $line >> ${out_spk_ts_dir}/log.txt
            echo ${out_spk_ts_dir}/${name}_txtsyn.wav >> ${out_spk_ts_dir}/log.txt
            ./${demo_dir}/bin/test_mwdlp -t ${out_spk_ts_dir}/${name}_melsp.txt ${out_spk_ts_dir}/${name}_txtsyn.wav >> ${out_spk_ts_dir}/log.txt
        done < ${wav_ts_scp}

        rm -f ${wav_ts_scp}

        echo ""
        echo "synthesis and melsp out & in of ${spk_src} finished, outputs are located in ${out_spk_dv_dir} and ${out_spk_ts_dir}"
    done
    echo ""
    echo "synthesis and melsp out & in of all speakers finished, outputs are located in respective directories of ${out_dir}"
    echo ""
fi
# }}}
