#!/bin/bash

spks=(tts morikawa okada otake uchino SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 VCC2SF3 VCC2SM3 bdl \
        p237 p245 p251 p252 p259 p274 p304 p311 p326 p345 p360 p363 nuct mizokuchi taga takada yamada \
            TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1 VCC2TF1 VCC2TM1 VCC2TF2 VCC2TM2 VCC2SM4 VCC2SF4 slt \
                p231 p238 p248 p253 p264 p265 p266 p276 p305 p308 p318 p335)

file_idx=001
src_spk=p326

in_dir=wav
out_dir=wav_cv_point

mkdir -p $out_dir

spk_idx=1

for spk in ${spks[@]};do
    echo $file_idx $src_spk to $spk $spk_idx
    ./bin/test_cycvae_mwdlp $spk_idx ${in_dir}/${file_idx}_${src_spk}.wav ${out_dir}/${spk_idx}_${file_idx}_${src_spk}-${spk}.wav 
    #./bin/test_cycvae_mwdlp.exe $spk_idx ${in_dir}/${file_idx}_${src_spk}.wav ${out_dir}/${spk_idx}_${file_idx}_${src_spk}-${spk}.wav 
    spk_idx=$(( ${spk_idx}+1  ))
done

