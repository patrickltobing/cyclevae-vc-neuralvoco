#!/bin/bash


in_dir=wav
out_dir=wav_melsp

mkdir -p $out_dir

#ls ${in_dir}/*.wav > tmp.list
#
#while read line;do
#    name=`basename $line .wav`
#    echo $line $name
#    ./bin/test_mwdlp -o melsp.bin melsp.txt $line ${out_dir}/${name}_anasyn.wav
#    ./bin/test_mwdlp -b melsp.bin ${out_dir}/${name}_binsyn.wav
#    ./bin/test_mwdlp -t melsp.txt ${out_dir}/${name}_txtsyn.wav
#    ./bin/test_mwdlp.exe -o melsp.bin melsp.txt $line ${out_dir}/${name}_anasyn.wav
#    ./bin/test_mwdlp.exe -b melsp.bin ${out_dir}/${name}_binsyn.wav
#    ./bin/test_mwdlp.exe -t melsp.txt ${out_dir}/${name}_txtsyn.wav
#done < tmp.list
#
#rm -f tmp.list

line=${in_dir}/001_p326.wav
name=`basename $line .wav`

./bin/test_mwdlp -o ${out_dir}/${name}_melsp.bin ${out_dir}/${name}_melsp.txt $line ${out_dir}/${name}_anasyn.wav
./bin/test_mwdlp -b ${out_dir}/${name}_melsp.bin ${out_dir}/${name}_binsyn.wav
./bin/test_mwdlp -t ${out_dir}/${name}_melsp.txt ${out_dir}/${name}_txtsyn.wav
#./bin/test_mwdlp.exe -o ${out_dir}/${name}_melsp.bin ${out_dir}/${name}_melsp.txt $line ${out_dir}/${name}_anasyn.wav
#./bin/test_mwdlp.exe -b ${out_dir}/${name}_melsp.bin ${out_dir}/${name}_binsyn.wav
#./bin/test_mwdlp.exe -t ${out_dir}/${name}_melsp.txt ${out_dir}/${name}_txtsyn.wav

#split=(${line// / })
#for spk in ${spks[@]};do
#    spk_idx=$(( ${spk_idx}+1  ))
#count=`expr $count + 1`
#done
