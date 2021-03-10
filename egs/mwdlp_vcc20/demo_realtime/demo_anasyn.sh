#!/bin/bash


#in_dir=wav_8kHz
#in_dir=wav_16kHz
in_dir=wav_24kHz
#out_dir=wav_anasyn_8kHz
#out_dir=wav_anasyn_16kHz
out_dir=wav_anasyn_24kHz

mkdir -p $out_dir

ls ${in_dir}/*.wav > tmp_anasyn.list

while read line;do
    name=`basename $line`
    echo $line $name
    ./bin/test_mwdlp $line ${out_dir}/$name
    #./bin/test_mwdlp.exe $line ${out_dir}/$name
done < tmp_anasyn.list

rm -f tmp_anasyn.list

#split=(${line// / })
#for spk in ${spks[@]};do
#    spk_idx=$(( ${spk_idx}+1  ))
#count=`expr $count + 1`
#done
