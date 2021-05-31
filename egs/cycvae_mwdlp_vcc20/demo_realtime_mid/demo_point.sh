#!/bin/bash

spks=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)

file_idx=E10061
src_spk=SEF2

in_dir=wav
out_dir=wav_cv_point

mkdir -p $out_dir

spk_idx=1

for spk in ${spks[@]};do
    echo $file_idx $src_spk to $spk $spk_idx
    ./bin/test_cycvae_mwdlp -i $spk_idx ${in_dir}/${file_idx}_${src_spk}.wav ${out_dir}/${spk_idx}_${file_idx}_${src_spk}-${spk}.wav 
    #./bin/test_cycvae_mwdlp.exe -i $spk_idx ${in_dir}/${file_idx}_${src_spk}.wav ${out_dir}/${spk_idx}_${file_idx}_${src_spk}-${spk}.wav 
    spk_idx=$(( ${spk_idx}+1  ))
done

