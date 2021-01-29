#!/bin/bash

x_coords=(0.00 -0.05 -0.10 -0.15 -0.20 -0.25 -0.30 -0.40)
y_coords=(0.00 -0.05 -0.10 -0.15 -0.20 -0.25 -0.30 -0.35 -0.40 -0.45)

file_idx=001
src_spk=p326

in_dir=wav
out_dir=wav_cv_interp

mkdir -p $out_dir

for x in ${x_coords[@]};do
for y in ${y_coords[@]};do
    echo $file_idx $src_spk to $x $y
    ./bin/test_cycvae_mwdlp -c $x $y ${in_dir}/${file_idx}_${src_spk}.wav ${out_dir}/${file_idx}_${src_spk}-interpolate_${x}_${y}.wav 
    #./bin/test_cycvae_mwdlp.exe -c $x $y ${in_dir}/${file_idx}_${src_spk}.wav ${out_dir}/${file_idx}_${src_spk}-interpolate_${x}_${y}.wav 
done
done

