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
# 3: run vc using speaker point target with real-time demo using cpu
# 4: run vc using interpolated speaker target with real-time demo using cpu
# }}}
stage=0
#stage=1
#stage=2
#stage=3
#stage=4

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
kernel_size_spk=`awk '{if ($1 == "kernel_size_spk:") print $2}' conf/config.yml`
dilation_size_spk=`awk '{if ($1 == "dilation_size_spk:") print $2}' conf/config.yml`
kernel_size_dec=`awk '{if ($1 == "kernel_size_dec:") print $2}' conf/config.yml`
dilation_size_dec=`awk '{if ($1 == "dilation_size_dec:") print $2}' conf/config.yml`
kernel_size_lf0=`awk '{if ($1 == "kernel_size_lf0:") print $2}' conf/config.yml`
dilation_size_lf0=`awk '{if ($1 == "dilation_size_lf0:") print $2}' conf/config.yml`
causal_conv_enc=`awk '{if ($1 == "causal_conv_enc:") print $2}' conf/config.yml`
causal_conv_dec=`awk '{if ($1 == "causal_conv_dec:") print $2}' conf/config.yml`
causal_conv_lf0=`awk '{if ($1 == "causal_conv_lf0:") print $2}' conf/config.yml`
spkidtr_dim=`awk '{if ($1 == "spkidtr_dim:") print $2}' conf/config.yml`
emb_spk_dim=`awk '{if ($1 == "emb_spk_dim:") print $2}' conf/config.yml`
n_weight_emb=`awk '{if ($1 == "n_weight_emb:") print $2}' conf/config.yml`
right_size_spk=`awk '{if ($1 == "right_size_spk:") print $2}' conf/config.yml`
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

### This is for VC source-target pairs
spks_src_dec=(SEM1 SEF2 SEM2 SEF1)
spks_src_dec=(SEM1 SEF1)
spks_src_dec=(SEF2)
spks_trg_dec=(TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
spks_trg_dec=(TEF1 TEM2)
spks_trg_dec=(TEM2)
###

###
#n_interp=1
#n_interp=2
n_interp=4
#n_interp=8
#n_interp=10
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


# parse options
. parse_options.sh

# stop when error occured
set -e
# }}}


if [ `echo ${stage} | grep 0` ] || [ `echo ${stage} | grep 4` ];then
echo $mdl_name_vc
if [ $mdl_name_vc == "cycmelspxlf0capspkvae-gauss-smpl_sparse_weightemb_v2" ]; then
    setting_vc=${mdl_name_vc}_${data_name}_lr${lr}_bs${batch_size}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hud${hidden_units_dec}_huf${hidden_units_lf0}_do${do_prob}_st${step_count}_mel${mel_dim}_nhcyc${n_half_cyc}_s${spkidtr_dim}_w${n_weight_emb}_ts${t_start_cycvae}_te${t_end_cycvae}_i${interval_cycvae}_de${densities_cycvae_enc}_dd${densities_cycvae_dec}_ns${n_stage_cycvae}_sc${s_conv_flag}_ss${seg_conv_flag}
fi
expdir_vc=exp/tr_${setting_vc}
echo $expdir_vc
if [ -f "${expdir_vc}/checkpoint-last.pkl" ]; then
    ${train_cmd} ${expdir_vc}/get_model_indices.log \
        get_model_indices.py \
            --expdir ${expdir_vc} \
            --confdir conf/${data_name}_vc
    min_idx_cycvae=`cat conf/${data_name}_vc.idx | awk '{print $2}'`
    echo "${data_name}: min_idx_cycvae=${min_idx_cycvae}"
else
    echo "vc checkpoints not found, please run vc training step"
fi


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


echo $mdl_name_ft
if [ $mdl_name_ft == "cycmelspspkvae-gauss-smpl_sparse_weightemb_mwdlp_smpl_v2" ]; then
    setting_ft=${mdl_name_ft}_${data_name}_lr${lr}_bs${batch_size_wave}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hud${hidden_units_dec}_huw${hidden_units_wave}_stc${step_count}_st${step_count_wave}_s${spkidtr_dim}_w${n_weight_emb}_de${densities_cycvae_enc}_dd${densities_cycvae_dec}_nb${n_bands}_sc${s_conv_flag}_ss${seg_conv_flag}_ssw${seg_conv_flag_wave}_${min_idx_cycvae}-${min_idx_wave}
fi
expdir_ft=exp/tr_${setting_ft}
echo $expdir_ft
if [ -f "${expdir_ft}/checkpoint-last.pkl" ]; then
    ${train_cmd} ${expdir_ft}/get_model_indices.log \
        get_model_indices.py \
            --expdir ${expdir_ft} \
            --confdir conf/${data_name}_ft
    min_idx_ft=`cat conf/${data_name}_ft.idx | awk '{print $2}'`
    echo "${data_name}: min_idx_ft=${min_idx_ft}"
else
    echo "fine-tune vc checkpoints not found, please run vc fine-tune training step"
fi


echo $mdl_name_sp
if [ $mdl_name_sp == "cycmelspspkvae-ftdec-gauss-smpl_sparse_wemb_mwdlp_smpl_v2" ]; then
    setting_sp=${mdl_name_sp}_${data_name}_lr${lr}_bs${batch_size_wave}_lat${lat_dim}_late${lat_dim_e}_hue${hidden_units_enc}_hud${hidden_units_dec}_huw${hidden_units_wave}_stc${step_count}_st${step_count_wave}_s${spkidtr_dim}_w${n_weight_emb}_de${densities_cycvae_enc}_dd${densities_cycvae_dec}_nb${n_bands}_sc${s_conv_flag}_ss${seg_conv_flag}_ssw${seg_conv_flag_wave}_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}
fi
expdir_sp=exp/tr_${setting_sp}
echo $expdir_sp
if [ -f "${expdir_sp}/checkpoint-last.pkl" ]; then
    ${train_cmd} ${expdir_sp}/get_model_indices.log \
        get_model_indices.py \
            --expdir ${expdir_sp} \
            --confdir conf/${data_name}_sp
    min_idx_sp=`cat conf/${data_name}_sp.idx | awk '{print $2}'`
    echo "${data_name}: min_idx_sp=${min_idx_sp}"
else
    echo "fine-tuned vc decoder checkpoints not found, please run vc decoder fine-tuning step"
fi
fi


demo_dir=demo_realtime

# STAGE 0 {{{
if [ `echo ${stage} | grep 0` ];then
    echo "###########################################################"
    echo "#        DUMP MODEL AND COMPILE REAL-TIME DEMO STEP       #"
    echo "###########################################################"

    echo ""
    echo "model is been dumping, please check ${expdir_sp}/dump_model.log"
    echo ""
    ${train_cmd} ${expdir_sp}/dump_model.log \
        dump_sparse-cyclevae_jnt_mwdlp-10b.py \
            ${expdir_sp}/model.conf \
            ${expdir_sp}/checkpoint-${min_idx_sp}.pkl \
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
            ./${demo_dir}/bin/test_cycvae_mwdlp $line ${out_spk_dv_dir}/$name >> ${out_spk_dv_dir}/log.txt
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
            ./${demo_dir}/bin/test_cycvae_mwdlp $line ${out_spk_ts_dir}/$name >> ${out_spk_ts_dir}/log.txt
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
            ./${demo_dir}/bin/test_cycvae_mwdlp -o ${out_spk_dv_dir}/${name}_melsp.bin ${out_spk_dv_dir}/${name}_melsp.txt \
                $line ${out_spk_dv_dir}/${name}_anasyn.wav >> ${out_spk_dv_dir}/log.txt
            echo $line >> ${out_spk_dv_dir}/log.txt
            echo ${out_spk_dv_dir}/${name}_binsyn.wav >> ${out_spk_dv_dir}/log.txt
            ./${demo_dir}/bin/test_cycvae_mwdlp -b ${out_spk_dv_dir}/${name}_melsp.bin ${out_spk_dv_dir}/${name}_binsyn.wav >> ${out_spk_dv_dir}/log.txt
            echo $line >> ${out_spk_dv_dir}/log.txt
            echo ${out_spk_dv_dir}/${name}_txtsyn.wav >> ${out_spk_dv_dir}/log.txt
            ./${demo_dir}/bin/test_cycvae_mwdlp -t ${out_spk_dv_dir}/${name}_melsp.txt ${out_spk_dv_dir}/${name}_txtsyn.wav >> ${out_spk_dv_dir}/log.txt
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
            ./${demo_dir}/bin/test_cycvae_mwdlp -o ${out_spk_ts_dir}/${name}_melsp.bin ${out_spk_ts_dir}/${name}_melsp.txt \
                $line ${out_spk_ts_dir}/${name}_anasyn.wav >> ${out_spk_ts_dir}/log.txt
            echo $line >> ${out_spk_ts_dir}/log.txt
            echo ${out_spk_ts_dir}/${name}_binsyn.wav >> ${out_spk_ts_dir}/log.txt
            ./${demo_dir}/bin/test_cycvae_mwdlp -b ${out_spk_ts_dir}/${name}_melsp.bin ${out_spk_ts_dir}/${name}_binsyn.wav >> ${out_spk_ts_dir}/log.txt
            echo $line >> ${out_spk_ts_dir}/log.txt
            echo ${out_spk_ts_dir}/${name}_txtsyn.wav >> ${out_spk_ts_dir}/log.txt
            ./${demo_dir}/bin/test_cycvae_mwdlp -t ${out_spk_ts_dir}/${name}_melsp.txt ${out_spk_ts_dir}/${name}_txtsyn.wav >> ${out_spk_ts_dir}/log.txt
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


# STAGE 3 {{{
if [ `echo ${stage} | grep 3` ];then
    echo "###########################################################"
    echo "#   VC ON SPEAKER POINT TARGET WITH REAL-TIME DEMO STEP   #"
    echo "###########################################################"

    out_dir=wav_cv_point
    mkdir -p ${out_dir}
    out_dir=${out_dir}/${data_name}
    mkdir -p ${out_dir}
    for spk_src in ${spks_src_dec[@]};do
    for spk_trg in ${spks_trg_dec[@]};do
        spk_idx=1
        for spk_srch in ${spks[@]};do
            if [ "$spk_trg" == "$spk_srch" ]; then
                break
            fi
            spk_idx=$((${spk_idx}+1))
        done
        if [ $spk_idx -gt ${#spks[@]} ]; then
            echo error, $spk_trg not in spk_list
            exit
        fi

        out_spk_dir=${out_dir}/${spk_src}-${spk_trg}
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
            ./${demo_dir}/bin/test_cycvae_mwdlp -i ${spk_idx} $line ${out_spk_dv_dir}/$name >> ${out_spk_dv_dir}/log.txt
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
            ./${demo_dir}/bin/test_cycvae_mwdlp -i ${spk_idx} $line ${out_spk_ts_dir}/$name >> ${out_spk_ts_dir}/log.txt
        done < ${wav_ts_scp}

        rm -f ${wav_ts_scp}

        echo ""
        echo "synthesis of ${spk_src}-${spk_trg} finished, outputs are located in ${out_spk_dv_dir} and ${out_spk_ts_dir}"
    done
        echo ""
        echo "synthesis of ${spk_src} finished, outputs are located in ${out_spk_dv_dir} and ${out_spk_ts_dir}"
    done
    echo ""
    echo "synthesis of all speakers finished, outputs are located in respective directories of ${out_dir}"
    echo ""
fi
# }}}


# STAGE 4 {{{
if [ `echo ${stage} | grep 4` ];then
    echo "###########################################################"
    echo "#    VC ON INTERPOLATED POINT WITH REAL-TIME DEMO STEP    #"
    echo "###########################################################"

    model=${expdir_ft}/checkpoint-${min_idx_ft}.pkl
    config=${expdir_ft}/model.conf
    outdir=${expdir_ft}/spkidtr-${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}
    mkdir -p $outdir
    ${cuda_cmd} ${expdir_ft}/log/decode_spkidtr_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}.log \
        decode_spkidtr_map.py \
            --outdir ${outdir} \
            --model ${model} \
            --config ${config}
    #exit
    echo ""
    echo "speaker space has been mapped, please see the figure and coords here: ${outdir}"
    echo ""
    awk -v n_interp="${n_interp}" 'BEGIN {flag_spkid=0;} \
        { \
            if (flag_spkid) { \
                if ($1 != "(1,") { \
                    if ($1 > 1) { \
                        if ($3 < min_x) min_x = $3;
                        else if ($3 > max_x) max_x = $3;
                        if ($4 < min_y) min_y = $4;
                        else if ($4 > max_y) max_y = $4;
                    } else { \
                        min_x = $3;
                        max_x = $3;
                        min_y = $4;
                        max_y = $4;
                    } \
                } else { \
                    flag_spkid=0;
                } \
            } else { \
                if ($1 == "spk-id") flag_spkid=1; \
            } \
        } \
        END { \
            delta_max_min_x = (max_x - min_x) / n_interp;
            x=min_x;
            for (i=0;i<=n_interp;i++) {
                if (i < n_interp) printf "%lf ", x;
                else printf "%lf\n", x;
                x += delta_max_min_x;
            }
            y=min_y;
            delta_max_min_y = (max_y - min_y) / n_interp;
            for (i=0;i<=n_interp;i++) {
                if (i < n_interp) printf "%lf ", y;
                else printf "%lf\n", y;
                y += delta_max_min_y;
            }
        }' \
            ${expdir_ft}/log/decode_spkidtr_${min_idx_cycvae}-${min_idx_wave}-${min_idx_ft}.log \
                > conf/min_max_coord_${n_interp}_${data_name}.txt
    x_coords=(`cat conf/min_max_coord_${n_interp}_${data_name}.txt | head -n 1`)
    y_coords=(`cat conf/min_max_coord_${n_interp}_${data_name}.txt | tail -n 1`)
    echo "min_to_max x, with ${n_interp} interpolations:" ${x_coords[@]}
    echo "min_to_max y, with ${n_interp} interpolations:" ${y_coords[@]}
    #exit

    out_dir=wav_cv_interp
    mkdir -p ${out_dir}
    out_dir=${out_dir}/${data_name}
    mkdir -p ${out_dir}
    for spk_src in ${spks_src_dec[@]};do
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
            name=`basename $line .wav`
            for x_coord in ${x_coords[@]};do
            for y_coord in ${y_coords[@]};do
                echo $line ${out_spk_dv_dir}/log.txt $x_coord $y_coord
                echo $line ${x_coord} ${y_coord} >> ${out_spk_dv_dir}/log.txt
                echo ${out_spk_dv_dir}/${name}_${x_coord}_${y_coord}.wav >> ${out_spk_dv_dir}/log.txt
                ./${demo_dir}/bin/test_cycvae_mwdlp -c ${x_coord} ${y_coord} \
                    $line ${out_spk_dv_dir}/${name}_${x_coord}_${y_coord}.wav >> ${out_spk_dv_dir}/log.txt
            done
            done
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
            name=`basename $line .wav`
            for x_coord in ${x_coords[@]};do
            for y_coord in ${y_coords[@]};do
                echo $line ${out_spk_ts_dir}/log.txt $x_coord $y_coord
                echo $line ${x_coord} ${y_coord} >> ${out_spk_ts_dir}/log.txt
                echo ${out_spk_ts_dir}/${name}_${x_coord}_${y_coord}.wav >> ${out_spk_ts_dir}/log.txt
                ./${demo_dir}/bin/test_cycvae_mwdlp -c ${x_coord} ${y_coord} \
                    $line ${out_spk_ts_dir}/${name}_${x_coord}_${y_coord}.wav >> ${out_spk_ts_dir}/log.txt
            done
            done
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
