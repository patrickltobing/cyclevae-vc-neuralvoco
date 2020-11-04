#!/bin/bash

#spks=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 p237 p245 p251 p252 p259 p274 p304 p311 p326 p345 p360 p363 TEF1 TEM1 TEF2 TEM2 \
#    TFF1 TGF1 TMF1 p231 p238 p248 p253 p264 p265 p266 p276 p305 p308 p318 p335)
#data_name=vcc2020vctk_24kHz
spks=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)
data_name=vcc2020_24kHz

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
