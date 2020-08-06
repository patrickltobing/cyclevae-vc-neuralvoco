#!/bin/bash

outdir=acc_summary
mkdir -p $outdir

max_epoch=120

min_idx=58
min_idx=66
min_idx=74
min_idx=75
min_idx=77

lat_dim=32

n_cyc=2

prior="laplace"

detach=true

expdir_log=exp/tr_cycmcepvae-${prior}-detach_vcc2020_24kHz_lr1e-4_bs30_bsu6_bsue10_lat${lat_dim}_hue1024_hle1_hud1024_hld1_kse7_dse1_ksd7_dsd1_biefalse_bidfalse_do0.5_ep${max_epoch}_mcep50_nhcyc${n_cyc}_aretrue_ardfalse_f0intrue_detach${detach}/log

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEF2.log \
        > $outdir/sef-tef_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEM2.log \
        > $outdir/sef-tem_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEF2.log \
        > $outdir/sem-tef_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEM2.log \
        > $outdir/sem-tem_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TFF1.log \
        > $outdir/sef-tff_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TFM1.log \
        > $outdir/sef-tfm_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TGF1.log \
        > $outdir/sef-tgf_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TGM1.log \
        > $outdir/sef-tgm_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TMF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TMF1.log \
        > $outdir/sef-tmf_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TMM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TMM1.log \
        > $outdir/sef-tmm_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TFF1.log \
        > $outdir/sem-tff_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TFM1.log \
        > $outdir/sem-tfm_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TGF1.log \
        > $outdir/sem-tgf_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TGM1.log \
        > $outdir/sem-tgm_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TMF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TMF1.log \
        > $outdir/sef-tmf_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TMM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TMM1.log \
        > $outdir/sem-tmm_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TMF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TMF1.log \
        > $outdir/sf-tf_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TMM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TMM1.log \
        > $outdir/sf-tm_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TMF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TMF1.log \
        > $outdir/sm-tf_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TMM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TMM1.log \
        > $outdir/sm-tm_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

awk -f summary_acc.awk \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEF2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TEM2.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TFF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TFM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TGF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TGM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TMF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TMF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TMF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TMF1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF1-TMM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM1-TMM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEF2-TMM1.log \
    ${expdir_log}/decode_evl_${min_idx}_SEM2-TMM1.log \
        > $outdir/s-t_${lat_dim}-${n_cyc}_cycmcepvae-$prior-detach${detach}-${min_idx}.txt

