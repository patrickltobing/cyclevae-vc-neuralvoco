#!/bin/sh

#mkdir -p summary

#echo 40-2_cycvaemcep-vq_vcc2018_noarenc_pad_noardec
#awk -f local/proc_loss_log_vae-spec.awk \
#    exp/tr_cycmcepvae-vq_vcc2018_22.05kHz_lr1e-4_bs30_bsu6_bsue10_lat40_hue1024_hle1_hud1024_hld1_kse7_dse1_ksd7_dsd1_rs0_biefalse_bidfalse_do0.5_ep90_mcep50_nhcyc4_arefalse_ardfalse_dtrue_ctr128/log/train.log \
#        > summary/40-2_cycvaemcep-vq_vcc2018_noarenc_pad_noardec.txt

#echo 32-1_cycvaemceplf0-laplace_vcc2018_noarenc_pad_diff
#awk -f local/proc_loss_log_vae-spec-excit.awk \
#    ~/nas02home/Workspace/cyclevae-multispk-vc_test/egs/vcc18/exp/tr_cycmceplf0capvae-laplace_vcc2018_22.05kHz_lr1e-4_bs30_bsu6_bsue10_lat32_late32_hue1024_hle1_hud1024_hld1_hlf1_kse7_dse1_ksd7_ksf7_dsd1_dsf1_rs0_biefalse_bidfalse_biffalse_do0.5_ep90_mcep50_nhcyc2_arefalse_ardtrue_arf0true_dtrue_s0_ditrue/log/train.log \
#        > summary/32-1_cycvaemceplf0-laplace_vcc2018_noarenc_pad_diff.txt

#echo 32-1_cycvaemceplf0-laplace_vcc2018_noarenc_pad_diff_s2
#awk -f local/proc_loss_log_vae-spec-excit.awk \
#    exp/tr_cycmceplf0capvae-laplace_vcc2018_22.05kHz_lr1e-4_bs30_bsu6_bsue10_lat32_late32_hue1024_hle1_hud1024_hld1_hlf1_kse7_dse1_ksd7_ksf7_dsd1_dsf1_rs0_biefalse_bidfalse_biffalse_do0.5_ep90_mcep50_nhcyc2_arefalse_ardtrue_arf0true_dtrue_s2_ditrue/log/train.log \
#        > summary/32-1_cycvaemceplf0-laplace_vcc2018_noarenc_pad_diff_s2.txt

#echo 32-1_cycvaemcep-laplace_vcc2018_noarenc_pad_diff
#awk -f local/proc_loss_log_vae-spec.awk \
#    exp/tr_cycmcepvae-laplace_vcc2018_22.05kHz_lr1e-4_bs30_bsu6_bsue10_lat32_hue1024_hle1_hud1024_hld1_kse7_dse1_ksd7_dsd1_rs0_biefalse_bidfalse_do0.5_ep90_mcep50_nhcyc2_arefalse_ardtrue_f0intrue_detachtrue_ditrue/log/train.log \
#    exp/tr_cycmcepvae-laplace_vcc2018_22.05kHz_lr1e-4_bs30_bsu6_bsue10_lat32_hue1024_hle1_hud1024_hld1_kse7_dse1_ksd7_dsd1_rs0_biefalse_bidfalse_do0.5_ep90_mcep50_nhcyc2_arefalse_ardtrue_f0intrue_detachtrue_ditrue/log/train_resume-58.log \
#        > summary/32-1_cycvaemcep-laplace_vcc2018_noarenc_pad_diff.txt

#echo 50-2_cycvaemceplf0-vq_vcc2018_noarenc_pad_noardec_s2
#awk -f local/proc_loss_log_vae-spec-excit.awk \
#    exp/tr_cycmceplf0capvae-vq_vcc2018_22.05kHz_lr1e-4_bs30_bsu6_bsue10_lat50_late50_hue1024_hle1_hud1024_hld1_hlf1_kse7_dse1_ksd7_ksf7_dsd1_dsf1_rs0_biefalse_bidfalse_biffalse_do0.5_ep90_mcep50_nhcyc4_arefalse_ardfalse_arf0false_dtrue_s2_ctr128/log/train.log \
#        > summary/50-2_cycvaemceplf0-vq_vcc2018_noarenc_pad_noardec_s2.txt

#echo 50-2_cycvaemceplf0-vq_vcc2018_noarenc_pad_noardec
#awk -f local/proc_loss_log_vae-spec-excit.awk \
#    exp/tr_cycmceplf0capvae-vq_vcc2018_22.05kHz_lr1e-4_bs30_bsu6_bsue10_lat50_late50_hue1024_hle1_hud1024_hld1_hlf1_kse7_dse1_ksd7_ksf7_dsd1_dsf1_rs0_biefalse_bidfalse_biffalse_do0.5_ep90_mcep50_nhcyc4_arefalse_ardfalse_arf0false_dtrue_s0_ctr128/log/train.log \
#        > summary/50-2_cycvaemceplf0-vq_vcc2018_noarenc_pad_noardec.txt

