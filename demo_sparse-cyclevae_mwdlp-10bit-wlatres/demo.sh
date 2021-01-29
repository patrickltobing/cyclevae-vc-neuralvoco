#!/bin/bash


in_dir=wav
out_dir=wav_cv

mkdir -p $out_dir

src_spk=SEF2
trg_spk=TEM2

spk_id=40

x_coord=-0.240053
y_coord=-0.039851


in_file=E10061_${src_spk}
out_name_a=test_${spk_id}_${trg_spk}
out_name_b=test_${x_coord}_${y_coord}_${trg_spk}


./bin/test_cycvae_mwdlp -i ${spk_id} ${in_dir}/${in_file}.wav ${out_dir}/${out_name_a}.wav
./bin/test_cycvae_mwdlp -c ${x_coord} ${y_coord} ${in_dir}/${in_file}.wav ${out_dir}/${out_name_b}.wav
#./bin/test_cycvae_mwdlp.exe -i ${spk_id} ${in_dir}/${in_file}.wav ${out_dir}/${out_name_a}.wav
#./bin/test_cycvae_mwdlp.exe -c ${x_coord} ${y_coord} ${in_dir}/${in_file}.wav ${out_dir}/${out_name_b}.wav


#play ${in_dir}/${in_file}.wav
ref_file=E10061_${trg_spk}
#play ${in_dir}/${ref_file}.wav

echo "Input wav file is "${in_dir}/${in_file}.wav
echo "A reference wav file is "${in_dir}/${ref_file}.wav

echo "First converted wav file is "${out_dir}/${out_name_a}.wav
echo "Second converted wav file is "${out_dir}/${out_name_b}.wav

#play ${out_dir}/${out_name_a}.wav
#play ${out_dir}/${out_name_b}.wav


#rm ${out_dir}/${out_name_a}.wav
#rm ${out_dir}/${out_name_b}.wav
