#!/bin/bash

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

#http://vc-challenge.org/
#https://github.com/nii-yamagishilab/VCC2020-database

wget https://github.com/nii-yamagishilab/VCC2020-database/raw/master/vcc2020_database_training_source.zip
wget https://github.com/nii-yamagishilab/VCC2020-database/raw/master/vcc2020_database_training_target_task1.zip
wget https://github.com/nii-yamagishilab/VCC2020-database/raw/master/vcc2020_database_training_target_task2.zip
wget https://github.com/nii-yamagishilab/VCC2020-database/raw/master/vcc2020_database_evaluation.zip
wget https://github.com/nii-yamagishilab/VCC2020-database/raw/master/vcc2020_database_groundtruth.zip

unzip vcc2020_database_training_source.zip
rm -vf vcc2020_database_training_source.zip
unzip vcc2020_database_training_target_task1.zip
rm -vf vcc2020_database_training_target_task1.zip
unzip vcc2020_database_training_target_task2.zip
rm -vf vcc2020_database_training_target_task2.zip
unzip vcc2020_database_evaluation.zip
rm -vf vcc2020_database_evaluation.zip
unzip vcc2020_database_groundtruth.zip
rm -vf vcc2020_database_groundtruth.zip

rm -vfr __MACOSX

trg_dir=wav_24kHz

mkdir -p ${trg_dir}

mv -v source/S* ${trg_dir}
mv -v target_task1/T* ${trg_dir}
mv -v target_task2/T* ${trg_dir}

mkdir -p ${trg_dir}/test

mv -v vcc2020_database_evaluation/S* ${trg_dir}/test
mv -v vcc2020_database_groundtruth/T* ${trg_dir}/test

rm -vfr source
rm -vfr target_task1
rm -vfr target_task2
rm -vfr vcc2020_database_evaluation
rm -vfr vcc2020_database_groundtruth
