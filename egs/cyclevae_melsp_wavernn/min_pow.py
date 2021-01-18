#!/usr/bin/env/python

import os

import numpy as np

folder = 'exp/init_spk_stat/tr_vctk85spks_48kHz'
spks=['p376', 'p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p232', 'p233', 'p234', 'p236', 'p239', 'p240', 'p241', 'p243', 'p244', 'p246', 'p247', 'p249', 'p250', 'p254', 'p255', 'p256', 'p257', 'p258', 'p260', 'p261', 'p262', 'p263', 'p267', 'p268', 'p269', 'p270', 'p271', 'p272', 'p273', 'p275', 'p277', 'p278', 'p279', 'p280', 'p281', 'p282', 'p283', 'p284', 'p285', 'p286', 'p287', 'p288', 'p292', 'p293', 'p294', 'p295', 'p297', 'p298', 'p299', 'p300', 'p301', 'p302', 'p303', 'p306', 'p307', 'p310', 'p312', 'p313', 'p314', 'p315', 'p316', 'p317', 'p323', 'p329', 'p330', 'p333', 'p334', 'p336', 'p339', 'p340', 'p341', 'p343', 'p347', 'p351', 'p361', 'p362', 'p364', 'p374']
for spk in spks:
    print(spk)
    in_file = os.path.join(folder,spk+'_npowhistogram.txt')
    print(in_file)
    arr_data = np.loadtxt(in_file)

    length = arr_data.shape[0]
    peak_1 = -999999999
    peak_1_idx = 0
    global_min = 999999999
    global_min_idx = length // 2
    peak_2 = -999999999
    peak_2_idx = length-1
    list_min_global_idx = []

    for i in range(length // 2):
        if arr_data[i][1] > peak_1:
            peak_1_idx = i
            peak_1 = arr_data[i][1]
    for i in range(length-1,(length // 2)-1,-1):
        if arr_data[i][1] > peak_2:
            peak_2_idx = i
            peak_2 = arr_data[i][1]
    for i in range(length):
        if arr_data[i][1] <= global_min and i > peak_1_idx and i < peak_2_idx:
            global_min_idx = i
            if arr_data[i][1] == global_min:
                list_min_global_idx.append(arr_data[i][0])
            else:
                list_min_global_idx = []
                list_min_global_idx.append(arr_data[i][0])
            global_min = arr_data[i][1]
    min_pow = np.mean(list_min_global_idx)

    print('%d %d %lf' % (peak_1_idx, arr_data[peak_1_idx][0], peak_1))
    print('%d %d %lf' % (global_min_idx, arr_data[global_min_idx][0], global_min))
    print('%d %d %lf' % (peak_2_idx, arr_data[peak_2_idx][0], peak_2))
    print(list_min_global_idx)
    print(min_pow)
    out_file = os.path.join('conf',spk+'.pow')
    print(out_file)
    f = open(out_file, 'w')
    f.write('%.1f\n' % (min_pow))
    f.close()
