#!/bin/bash

#spks=(VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 VCC2SF3 VCC2SM3 bdl VCC2TF1 VCC2TM1 VCC2TF2 VCC2TM2 VCC2SM4 VCC2SF4 slt)
#data_name=vcc2018arctic_22.05kHz
spks=(VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 VCC2SF3 VCC2SM3 VCC2TF1 VCC2TM1 VCC2TF2 VCC2TM2 VCC2SM4 VCC2SF4)
data_name=vcc2018_22.05kHz

max_frame=0
max_spk=""

for spk in ${spks[*]}; do
    echo $spk tr
    max_frame_spk=`awk '{if ($1 == "max_frame:") print $2}' exp/feature_extract/tr_${data_name}/feature_extract_${spk}.log`
    echo $max_frame_spk
    if [[ $max_frame_spk -gt $max_frame ]]; then
        max_frame=$max_frame_spk
        max_spk=$spk
    fi 
    echo $spk dv
    max_frame_spk=`awk '{if ($1 == "max_frame:") print $2}' exp/feature_extract/dv_${data_name}/feature_extract_${spk}.log`
    echo $max_frame_spk
    if [[ $max_frame_spk -gt $max_frame ]]; then
        max_frame=$max_frame_spk
        max_spk=$spk
    fi 
done

echo $max_spk $max_frame
