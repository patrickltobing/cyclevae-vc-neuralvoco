#!/bin/bash


in_dir=wav
out_dir=wav_cv_point_spk

mkdir -p $out_dir

ls ${in_dir}/*.wav > tmp.list

while read line;do
    name=`basename $line .wav`
    echo $line $name
    ./bin/test_cycvae_mwdlp -i 40 $line ${out_dir}/${name}_TEM2.wav
    #./bin/test_cycvae_mwdlp.exe $line ${out_dir}/$name
done < tmp.list

rm -f tmp.list

#split=(${line// / })
#for spk in ${spks[@]};do
#    spk_idx=$(( ${spk_idx}+1  ))
#count=`expr $count + 1`
#done

