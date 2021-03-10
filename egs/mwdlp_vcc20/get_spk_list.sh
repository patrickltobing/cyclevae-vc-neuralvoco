#!/bin/bash

#ls wav_24kHz/train > tmp
ls wav_24kHz > tmp
a=(`cat tmp`)

echo ${a[@]}
echo ${#a[@]}

#ls wav_24kHz_unseen/test > tmp
#a=(`cat tmp`)
#
#echo ${a[@]}
#echo ${#a[@]}

rm -f tmp
