#!/usr/bin/env/python

import os

import numpy as np

folder = 'exp/init_spk_stat/tr_vctk85spks_48kHz'
spks=['p376', 'p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p232', 'p233', 'p234', 'p236', 'p239', 'p240', 'p241', 'p243', 'p244', 'p246', 'p247', 'p249', 'p250', 'p254', 'p255', 'p256', 'p257', 'p258', 'p260', 'p261', 'p262', 'p263', 'p267', 'p268', 'p269', 'p270', 'p271', 'p272', 'p273', 'p275', 'p277', 'p278', 'p279', 'p280', 'p281', 'p282', 'p283', 'p284', 'p285', 'p286', 'p287', 'p288', 'p292', 'p293', 'p294', 'p295', 'p297', 'p298', 'p299', 'p300', 'p301', 'p302', 'p303', 'p306', 'p307', 'p310', 'p312', 'p313', 'p314', 'p315', 'p316', 'p317', 'p323', 'p329', 'p330', 'p333', 'p334', 'p336', 'p339', 'p340', 'p341', 'p343', 'p347', 'p351', 'p361', 'p362', 'p364', 'p374']
for spk in spks:
    print(spk)
    in_file = os.path.join(folder,spk+'_f0histogram.txt')
    print(in_file)
    arr_data = np.loadtxt(in_file)

    length = arr_data.shape[0]
    left_min = 999999999
    left_min_idx = -1
    right_min = 999999999
    right_min_idx = -1
    peak = np.max(arr_data[:,1])
    peak_idx = np.argmax(arr_data[:,1])

    # left min
    if arr_data[peak_idx,0] > 90:
        left_left_f0 = arr_data[peak_idx,0]//2+40-15+1
        print(left_left_f0)
        if left_left_f0 > 150:
            left_left_f0 = 130
            left_left_f0_idx = int(left_left_f0)-40+1
            left_left_max = np.max(arr_data[:left_left_f0_idx,1])
            left_left_max_idx = np.argmax(arr_data[:left_left_f0_idx,1])
        else:
            left_left_f0_idx = int(left_left_f0)-40+1
            left_left_max = np.max(arr_data[:left_left_f0_idx,1])
            left_left_max_idx = np.argmax(arr_data[:left_left_f0_idx,1])
            if left_left_max >= 0.0045:
                left_left_f0 -= 20
                left_left_f0_idx = int(left_left_f0)-40+1
                left_left_max = np.max(arr_data[:left_left_f0_idx,1])
                left_left_max_idx = np.argmax(arr_data[:left_left_f0_idx,1])
                while left_left_max < 0.000045:
                    left_left_max_idx += 1
                    left_left_max = arr_data[left_left_max_idx,1]
        print('%lf %d %d' % (left_left_max, left_left_max_idx, arr_data[left_left_max_idx,0]))
        print('%lf %d %d' % (peak, peak_idx, arr_data[peak_idx,0]))
        left_right_min = np.min(arr_data[left_left_max_idx+1:peak_idx,1])
        left_right_min_idx = np.argmin(arr_data[left_left_max_idx+1:peak_idx,1])+left_left_max_idx
        if left_left_max - left_right_min >= 0.001: #saddle min
            left_min = left_right_min
            left_min_idx = left_right_min_idx
        else:
            for i in range(left_left_max_idx-1,-1,-1):
                if left_min_idx == -1 and arr_data[i,1] < 0.0006:
                    left_min = arr_data[i+1,1]
                    left_min_idx = i+1
                elif left_min_idx != -1 and arr_data[i,1] >= 0.001:
                    left_min = 999999999
                    left_min_idx = -1
        print('%lf %d %d' % (left_right_min, left_right_min_idx, arr_data[left_right_min_idx,0]))
    else:
        for i in range(peak_idx-1,-1,-1):
            if left_min_idx == -1 and arr_data[i,1] < 0.0006:
                left_min = arr_data[i+1,1]
                left_min_idx = i+1
            elif left_min_idx != -1 and arr_data[i,1] >= 0.001:
                left_min = 999999999
                left_min_idx = -1

    # right min
    for i in range(peak_idx+1,length):
        if right_min_idx == -1 and arr_data[i,1] < 0.00009:
            right_min = arr_data[i-1,1]
            right_min_idx = i-1
        elif right_min_idx != -1 and arr_data[i,1] >= 0.00013:
            right_min = 999999999
            right_min_idx = -1

    print('%d %d %lf' % (left_min_idx, arr_data[left_min_idx][0], left_min))
    print('%d %d %lf' % (peak_idx, arr_data[peak_idx][0], peak))
    print('%d %d %lf' % (right_min_idx, arr_data[right_min_idx][0], right_min))
    out_file = os.path.join('conf',spk+'.f0')
    print(out_file)
    f = open(out_file, 'w')
    f.write('%d %d\n' % (arr_data[left_min_idx,0], arr_data[right_min_idx,0]))
    f.close()
