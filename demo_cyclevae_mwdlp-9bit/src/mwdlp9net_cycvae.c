/* Copyright (c) 2018 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/* Modified by Patrick Lumban Tobing (Nagoya University) on Dec. 2020 - Jan. 2021,
   marked by PLT_<Dec20/Jan21> */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>
#include "nnet_data.h"
#include "nnet_cv_data.h"
#include "nnet.h"
#include "common.h"
#include "arch.h"
#include "mwdlp9net_cycvae.h"
#include "mwdlp9net_cycvae_private.h"
#include "mu_law_9_table.h"

#define PREEMPH 0.85f


#if 0
static void print_vector(float *x, int N)
{
    int i;
    for (i=0;i<N;i++) printf("%f ", x[i]);
    printf("\n");
}
#endif


//PLT_Dec20
static void run_frame_network_cyclevae_post_melsp_excit_spk(CycleVAEPostMelspExcitSpkNNetState *net,
    const float *melsp_in, float *spk_code_aux, float *melsp_cv, short first_frame_flag)
    //const float *melsp_in, float *spk_code_aux, float *melsp_cv, short first_frame_flag, float *lat_tmp, float *spk_tmp, float *conv_tmp, float *gru_tmp, float *f0_tmp)
        //MWDLP9CycleVAEPostMelspExcitSpkNetState *net_st)
{
    int i;
    float in[FEATURE_DIM_MELSP];
    float enc_melsp_input[FEATURE_CONV_ENC_MELSP_OUT_SIZE];
    float enc_excit_input[FEATURE_CONV_ENC_EXCIT_OUT_SIZE];
    float lat_melsp[FEATURE_LAT_DIM_MELSP];
    float lat_excit[FEATURE_LAT_DIM_EXCIT];
    float lat_excit_melsp[FEATURE_LAT_DIM_EXCIT_MELSP];
    float spk_code_lat_excit_melsp[FEATURE_N_SPK_LAT_DIM_EXCIT_MELSP];
    float spk_input[FEATURE_CONV_SPK_OUT_SIZE];
    float time_varying_spk_code[FEATURE_N_SPK];
    float spk_code_aux_lat_excit[FEATURE_N_SPK_2_LAT_DIM_EXCIT];
    float dec_melsp_input[FEATURE_CONV_DEC_MELSP_OUT_SIZE];
    float dec_excit_input[FEATURE_CONV_DEC_EXCIT_OUT_SIZE];
    float dec_excit_out[FEATURE_DIM_EXCIT];
    float post_input[FEATURE_CONV_POST_OUT_SIZE];
    float uvf0, f0;
    float uvcap;
    float cap[FEATURE_CAP_DIM];
    float uvf0_f0[2];
    float uvcap_cap[FEATURE_UV_CAP_DIM];
    float spk_code_aux_f0_lat_excit_melsp[FEATURE_N_SPK_2_2_LAT_DIM_EXCIT_MELSP];
    float spk_code_aux_excit_melsp[FEATURE_N_SPK_2_DIM_EXCIT_MELSP];
    float melsp_pdf[FEATURE_DIM_MELSP_2];
    float melsp_loc[FEATURE_DIM_MELSP];
    float melsp_scale[FEATURE_DIM_MELSP];
    //int k;

    //handle pad first before calling this function
    //compute enc input here [segmental input conv.]
    //printf("enc_melsp_excit %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    RNN_COPY(in, melsp_in, FEATURE_DIM_MELSP);
    compute_normalize(&melsp_norm, in);
    compute_conv1d_linear(&feature_conv_enc_melsp, enc_melsp_input, net->feature_conv_enc_melsp_state, in);
    compute_conv1d_linear(&feature_conv_enc_excit, enc_excit_input, net->feature_conv_enc_excit_state, in);
    //printf("exc_conv");
    //for (k = 0; k < FEATURE_CONV_ENC_EXCIT_OUT_SIZE; k++) {
    ////    printf(" [%d] %f", k, enc_excit_input[k]);
    //    conv_tmp[k] = enc_excit_input[k];
    //}
    //printf("\n");
    //printf("msp_conv");
    //for (k = 0; k < FEATURE_CONV_ENC_MELSP_OUT_SIZE; k++) {
    ////    printf(" [%d] %f", k, enc_melsp_input[k]);
    //    conv_tmp[FEATURE_CONV_ENC_EXCIT_OUT_SIZE+k] = enc_melsp_input[k];
    //}
    //printf("\n");
    compute_gru_enc_melsp(&gru_enc_melsp, net->gru_enc_melsp_state, enc_melsp_input);
    compute_gru_enc_excit(&gru_enc_excit, net->gru_enc_excit_state, enc_excit_input);
    //printf("exc_gru");
    //for (k = 0; k < GRU_ENC_EXCIT_STATE_SIZE; k++) {
    //    printf(" [%d] %f", k, net->gru_enc_excit_state[k]);
    //    gru_tmp[k] = net->gru_enc_excit_state[k];
    //}
    //printf("\n");
    compute_dense(&fc_out_enc_melsp, lat_melsp, net->gru_enc_melsp_state);
    compute_dense(&fc_out_enc_excit, lat_excit, net->gru_enc_excit_state);
    //compute_dense_linear(&fc_out_enc_excit, lat_excit, net->gru_enc_excit_state);
    //for (k = 0; k < FEATURE_LAT_DIM_EXCIT; k++) {
    ////    printf("lat_exc [%d] %f ", k, lat_excit[k]);
    //    lat_tmp[k] = lat_excit[k];
    //}
    //for (k = 0; k < FEATURE_LAT_DIM_EXCIT_MELSP; k++) {
    //    printf("lat_melsp [%d] %f ", k, lat_melsp[k]);
    //    lat_tmp[k] = lat_melsp[k];
    //}
    RNN_COPY(lat_excit_melsp, lat_excit, FEATURE_LAT_DIM_EXCIT);
    RNN_COPY(&lat_excit_melsp[FEATURE_LAT_DIM_EXCIT], lat_melsp, FEATURE_LAT_DIM_MELSP);
    //for (k = 0; k < FEATURE_LAT_DIM_EXCIT_MELSP; k++) {
    ////    printf("lat [%d] %f ", k, lat_excit_melsp[k]);
    //    lat_tmp[k] = lat_excit_melsp[k];
    //}
    //printf("\n");

    //compute spk input here [concate spk-code,lat_excit,lat_melsp]
    //printf("spk_net %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    RNN_COPY(spk_code_lat_excit_melsp, spk_code_aux, FEATURE_N_SPK); //spk_code_aux=[1-hot-spk-code,time-vary-spk-code]
    RNN_COPY(&spk_code_lat_excit_melsp[FEATURE_N_SPK], lat_excit_melsp, FEATURE_LAT_DIM_EXCIT_MELSP);
    if (first_frame_flag) { //pad_first
        float *mem_spk = net->feature_conv_spk_state; //mem of stored input frames
        for (i=0;i<SPK_CONV_KERNEL_1;i++) //store first input with replicate padding kernel_size-1
            RNN_COPY(&mem_spk[i*FEATURE_N_SPK_2_LAT_DIM_EXCIT], spk_code_aux_lat_excit, FEATURE_N_SPK_2_LAT_DIM_EXCIT);
    }
    compute_conv1d_linear(&feature_conv_spk, spk_input, net->feature_conv_spk_state, spk_code_lat_excit_melsp);
    compute_gru_spk(&gru_spk, net->gru_spk_state, spk_input);
    compute_dense(&fc_out_spk, time_varying_spk_code, net->gru_spk_state);
    RNN_COPY(&spk_code_aux[FEATURE_N_SPK], time_varying_spk_code, FEATURE_N_SPK);
    //for (k = 0; k < FEATURE_N_SPK; k++) {
    ////    printf("spk [%d] %f ", k, spk_code_aux[FEATURE_N_SPK+k]);
    //    spk_tmp[FEATURE_N_SPK+k] = spk_code_aux[FEATURE_N_SPK+k];
    //}
    //printf("\n");

    //compute excit input here [concate spk-code,spk-code-aux,lat_excit]
    //printf("dec_excit %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    RNN_COPY(spk_code_aux_lat_excit, spk_code_aux, FEATURE_N_SPK_2);
    //printf("dec_excit a0 %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    RNN_COPY(&spk_code_aux_lat_excit[FEATURE_N_SPK_2], lat_excit, FEATURE_LAT_DIM_EXCIT);
    //printf("dec_excit a1 %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    if (first_frame_flag) { //pad_first
        float *mem_dec_excit = net->feature_conv_dec_excit_state; //mem of stored input frames
    //    printf("dec_excit a2 %d %d\n", net_st->frame_count, net_st->cv_frame_count);
        for (i=0;i<DEC_EXCIT_CONV_KERNEL_1;i++) //store first input with replicate padding kernel_size-1
            RNN_COPY(&mem_dec_excit[i*FEATURE_N_SPK_2_LAT_DIM_EXCIT], spk_code_aux_lat_excit, FEATURE_N_SPK_2_LAT_DIM_EXCIT);
    //    printf("dec_excit a3 %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    }
    //printf("dec_excit a %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    compute_conv1d_linear(&feature_conv_dec_excit, dec_excit_input, net->feature_conv_dec_excit_state, spk_code_aux_lat_excit);
    compute_gru_dec_excit(&gru_dec_excit, net->gru_dec_excit_state, dec_excit_input);
    compute_dense_linear(&fc_out_dec_excit, dec_excit_out, net->gru_dec_excit_state);
    //printf("dec_excit b %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    //printf("uv_f0_before %f %f %f %f\n", dec_excit_out[0], dec_excit_out[1], uvf0, f0);
    compute_activation(&uvf0, dec_excit_out, 1, ACTIVATION_SIGMOID);
    //printf("uv_f0_mid %f %f %f %f\n", dec_excit_out[0], dec_excit_out[1], uvf0, f0);
    compute_activation(&f0, &dec_excit_out[1], 1, ACTIVATION_TANHSHRINK);
    //f0_tmp[0] = uvf0;
    //f0_tmp[1] = exp(f0*0.70206316+4.96925691);
    //printf("uv_f0_after %f %f %f %f\n", dec_excit_out[0], dec_excit_out[1], uvf0, f0);
    if (dec_excit_out[2] > 32) dec_excit_out[2] = 32;
    //printf("\nuv_cap_before %f %f %f %f\n", dec_excit_out[2], dec_excit_out[3], uvcap, cap[0]);
    compute_activation(&uvcap, &dec_excit_out[2], 1, ACTIVATION_SIGMOID);
    //printf("uv_cap_mid %f %f %f %f\n", dec_excit_out[2], dec_excit_out[3], uvcap, cap[0]);
    compute_activation(cap, &dec_excit_out[3], FEATURE_CAP_DIM, ACTIVATION_TANHSHRINK);
    //printf("uv_cap_after %f %f %f %f\n", dec_excit_out[2], dec_excit_out[3], uvcap, cap[0]);
    compute_normalize(&uvf0_norm, &uvf0); //normalize only uv because output f0 is normalized
    compute_normalize(&uvcap_norm, &uvcap); //normalize only uv because output cap is normalized
    //printf("dec_excit c %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    //concate uvf0-f0
    RNN_COPY(uvf0_f0, &uvf0, 1);
    RNN_COPY(&uvf0_f0[1], &f0, 1);
    //printf("dec_excit d %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    //concate uvcap-cap
    RNN_COPY(uvcap_cap, &uvcap, 1);
    RNN_COPY(&uvcap_cap[1], cap, FEATURE_CAP_DIM);

    //compute melsp input here [concate spk-code,spk-code-aux,uv-f0,lat_excit,lat_melsp]
    //printf("dec_melsp %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    RNN_COPY(spk_code_aux_f0_lat_excit_melsp, spk_code_aux, FEATURE_N_SPK_2);
    RNN_COPY(&spk_code_aux_f0_lat_excit_melsp[FEATURE_N_SPK_2], uvf0_f0, 2);
    RNN_COPY(&spk_code_aux_f0_lat_excit_melsp[FEATURE_N_SPK_2_2], lat_excit_melsp, FEATURE_LAT_DIM_EXCIT_MELSP);
    if (first_frame_flag) { //pad_first
        float *mem_dec_melsp = net->feature_conv_dec_melsp_state; //mem of stored input frames
        for (i=0;i<DEC_MELSP_CONV_KERNEL_1;i++) //store first input with replicate padding kernel_size-1
            RNN_COPY(&mem_dec_melsp[i*FEATURE_N_SPK_2_2_LAT_DIM_EXCIT_MELSP], spk_code_aux_f0_lat_excit_melsp, FEATURE_N_SPK_2_2_LAT_DIM_EXCIT_MELSP);
    }
    compute_conv1d_linear(&feature_conv_dec_melsp, dec_melsp_input, net->feature_conv_dec_melsp_state, spk_code_aux_f0_lat_excit_melsp);
    compute_gru_dec_melsp(&gru_dec_melsp, net->gru_dec_melsp_state, dec_melsp_input);
    compute_dense(&fc_out_dec_melsp, melsp_cv, net->gru_dec_melsp_state);

    //compute post input here [concate spk-code,spk-code-aux,uv-f0,uv-codeap,melsp]
    //printf("post %d %d\n", net_st->frame_count, net_st->cv_frame_count);
    RNN_COPY(spk_code_aux_excit_melsp, spk_code_aux_f0_lat_excit_melsp, FEATURE_N_SPK_2_2);
    RNN_COPY(&spk_code_aux_excit_melsp[FEATURE_N_SPK_2_2], uvcap_cap, FEATURE_UV_CAP_DIM);
    RNN_COPY(&spk_code_aux_excit_melsp[FEATURE_N_SPK_2_EXCIT], melsp_cv, FEATURE_DIM_MELSP);
    if (first_frame_flag) { //pad_first
        float *mem_post = net->feature_conv_post_state; //mem of stored input frames
        for (i=0;i<POST_CONV_KERNEL_1;i++) //store first input with replicate padding kernel_size-1
            RNN_COPY(&mem_post[i*FEATURE_N_SPK_2_DIM_EXCIT_MELSP], spk_code_aux_excit_melsp, FEATURE_N_SPK_2_DIM_EXCIT_MELSP);
    }
    //printf("\npost_input");
    //for (i=0;i<FEATURE_N_SPK_2_DIM_EXCIT_MELSP;i++) {
    //    printf(" [%d] %f", i, spk_code_aux_excit_melsp[i]);
    //}
    //printf("\n");
    compute_conv1d_linear(&feature_conv_post, post_input, net->feature_conv_post_state, spk_code_aux_excit_melsp);
    //printf("\npost_conv");
    //for (i=0;i<FEATURE_CONV_POST_OUT_SIZE;i++) {
    //    printf(" [%d] %f", i, post_input[i]);
    //}
    //printf("\n");
    compute_gru_post(&gru_post, net->gru_post_state, post_input);
    //printf("\npost_gru");
    //for (i=0;i<GRU_POST_STATE_SIZE;i++) {
    //    printf(" [%d] %f", i, net->gru_post_state[i]);
    //}
    //printf("\n");
    compute_dense_linear(&fc_out_post, melsp_pdf, net->gru_post_state);
    compute_activation(melsp_loc, melsp_pdf, FEATURE_DIM_MELSP, ACTIVATION_TANHSHRINK);
    compute_activation(melsp_scale, &melsp_pdf[FEATURE_DIM_MELSP], FEATURE_DIM_MELSP, ACTIVATION_SIGMOID);
    //compute_activation(melsp_loc, melsp_pdf, FEATURE_DIM_MELSP, ACTIVATION_LINEAR);
    //compute_activation(melsp_scale, &melsp_pdf[FEATURE_DIM_MELSP], FEATURE_DIM_MELSP, ACTIVATION_LINEAR);
    compute_denormalize(&melsp_norm, melsp_cv);
    compute_denormalize(&melsp_norm, melsp_loc);
    compute_sampling_laplace(melsp_loc, melsp_scale, FEATURE_DIM_MELSP);
    //printf("\nmelsp_cv + post_res");
    for (i=0;i<FEATURE_DIM_MELSP;i++) {
    //    printf(" [%d] %f", i, melsp_cv[i]);
        melsp_cv[i] += melsp_loc[i];
    //    printf(" %f %f %f", melsp_loc[i], melsp_scale[i], melsp_cv[i]);
    }
    //printf("\n");
    //printf("post_end %d %d\n", net_st->frame_count, net_st->cv_frame_count);
}


//PLT_Jan21
static void run_frame_network_mwdlp9(MWDLP9NNetState *net, float *gru_a_condition, float *gru_b_condition, 
    const float *features, int flag_last_frame)
    //const float *features, int flag_last_frame, float *conv_tmp)
{
    float in[FEATURES_DIM];
    float conv_out[FEATURE_CONV_OUT_SIZE];
    float condition[FEATURE_DENSE_OUT_SIZE];
    RNN_COPY(in, features, FEATURES_DIM);
    //feature normalization if not last frame, just replicate if last frame
    if (!flag_last_frame) compute_normalize(&feature_norm, in);
    //segmental input conv. and fc layer with relu
    compute_conv1d_linear(&feature_conv, conv_out, net->feature_conv_state, in);
    //for (int i=0; i<FEATURE_CONV_OUT_SIZE; i++)
    //    conv_tmp[i] = conv_out[i];
    compute_dense(&feature_dense, condition, conv_out);
    //compute condition (input_vector_cond*input_matrix_cond+input_bias) for each gru_a and b; fixed for one frame
    compute_dense_linear(&gru_a_dense_feature, gru_a_condition, condition);
    //for (int i=0; i<RNN_MAIN_NEURONS_3; i++)
    //    conv_tmp[i] = gru_a_condition[i];
    compute_dense_linear(&gru_b_dense_feature, gru_b_condition, condition);
    //for (int i=0; i<RNN_SUB_NEURONS_3; i++)
    //    conv_tmp[i] = gru_b_condition[i];
}


//PLT_Jan21
static void run_sample_network_mwdlp9(MWDLP9NNetState *net, const EmbeddingLayer *a_embed,
    float *pdf, const float *gru_a_condition, const float *gru_b_condition, int *last_sample)
    //float *pdf, const float *gru_a_condition, const float *gru_b_condition, int *last_sample, float *gru_tmp)
{
    int i, j, idx_last_sample;
    //int k;
    float gru_a_input[RNN_MAIN_NEURONS_3];
    float gru_b_input[RNN_SUB_NEURONS_3];
    //copy input conditioning * GRU_input_cond_weights + input_bias contribution (a)
    RNN_COPY(gru_a_input, gru_a_condition, RNN_MAIN_NEURONS_3);
    //compute last embedding * GRU_input_embed_weights contribution (a)
    for (i=0;i<N_MBANDS;i++) {
        // stored embedding: n_bands x 512 x hidden_size_main
        //k = i*RNN_MAIN_NEURONS_3;
        //for (j=0,idx_last_sample=i*RNN_MAIN_NEURONS_3_N_QUANTIZE+last_sample[i]*RNN_MAIN_NEURONS_3;j<RNN_MAIN_NEURONS_3;j++) {
        for (j=0,idx_last_sample=i*RNN_MAIN_NEURONS_3_N_QUANTIZE+last_sample[i]*RNN_MAIN_NEURONS_3;j<RNN_MAIN_NEURONS_3;j++)
            gru_a_input[j] += a_embed->embedding_weights[idx_last_sample + j];
        //    if (i==0) {
        //        gru_tmp[k+j] = a_embed->embedding_weights[idx_last_sample + j];
    //            printf("last_sample %d %d %d %d %d %d %d %d %f %f\n", last_sample[i], i, j, RNN_MAIN_NEURONS_3_N_QUANTIZE, RNN_MAIN_NEURONS, N_QUANTIZE, idx_last_sample, idx_last_sample+j, a_embed->embedding_weights[idx_last_sample + j], gru_tmp[k+j]);
        //    }
        //}
    }
    //compute sparse gru_a
    compute_sparse_gru(&sparse_gru_a, net->gru_a_state, gru_a_input);
    //for (i=0;i<RNN_MAIN_NEURONS;i++)
    //    gru_tmp[i] = net->gru_a_state[i];
    //copy input conditioning * GRU_input_cond_weights + bias contribution (b)
    RNN_COPY(gru_b_input, gru_b_condition, RNN_SUB_NEURONS_3);
    //compute gru_a state contribution to gru_b
    sgemv_accum(gru_b_input, (&gru_b_dense_feature_state)->input_weights, RNN_SUB_NEURONS_3, RNN_MAIN_NEURONS,
        RNN_SUB_NEURONS_3, net->gru_a_state);
    //compute gru_b and output
    compute_gru3(&gru_b, net->gru_b_state, gru_b_input);
    compute_mdense_mwdlp9(&dual_fc, &fc_out, pdf, net->gru_b_state, last_sample);
}


//PLT_Jan21
MWDLP9NET_CYCVAE_EXPORT int mwdlp9cyclevaenet_get_size()
{
    return sizeof(MWDLP9CycleVAEPostMelspExcitSpkNetState);
}


//PLT_Jan21
MWDLP9NET_CYCVAE_EXPORT MWDLP9CycleVAEPostMelspExcitSpkNetState *mwdlp9cyclevaenet_create()
{
    MWDLP9CycleVAEPostMelspExcitSpkNetState *mwdlp9net;
    mwdlp9net = (MWDLP9CycleVAEPostMelspExcitSpkNetState *) calloc(1,mwdlp9cyclevaenet_get_size());
    if (mwdlp9net != NULL) {
        if (FIRST_N_OUTPUT == 0) mwdlp9net->first_flag = 1;
        int i, j, k;
        for (i=0, k=0;i<DLPC_ORDER;i++)
            for (j=0;j<N_MBANDS;j++,k++)
                mwdlp9net->last_sample[k] = INIT_LAST_SAMPLE;
        return mwdlp9net;
    }
    printf("Cannot allocate and initialize memory for MWDLP9CycleVAEPostMelspExcitSpkNetState.\n");
    exit(EXIT_FAILURE);
    return NULL;
}


//PLT_Jan21
MWDLP9NET_CYCVAE_EXPORT void mwdlp9cyclevaenet_destroy(MWDLP9CycleVAEPostMelspExcitSpkNetState *mwdlp9cyclevaenet)
{
    if (mwdlp9cyclevaenet != NULL) free(mwdlp9cyclevaenet);
}


//PLT_Jan21
MWDLP9NET_CYCVAE_EXPORT int mwdlp9net_get_size()
{
    return sizeof(MWDLP9NetState);
}


//PLT_Jan21
MWDLP9NET_CYCVAE_EXPORT MWDLP9NetState *mwdlp9net_create()
{
    MWDLP9NetState *mwdlp9net;
    mwdlp9net = (MWDLP9NetState *) calloc(1,mwdlp9net_get_size());
    if (mwdlp9net != NULL) {
        if (FIRST_N_OUTPUT == 0) mwdlp9net->first_flag = 1;
        int i, j, k;
        for (i=0, k=0;i<DLPC_ORDER;i++)
            for (j=0;j<N_MBANDS;j++,k++)
                mwdlp9net->last_sample[k] = INIT_LAST_SAMPLE;
        return mwdlp9net;
    }
    printf("Cannot allocate and initialize memory for MWDLP9NetState.\n");
    exit(EXIT_FAILURE);
    return NULL;
}


//PLT_Jan21
MWDLP9NET_CYCVAE_EXPORT void mwdlp9net_destroy(MWDLP9NetState *mwdlp9net)
{
    if (mwdlp9net != NULL) free(mwdlp9net);
}


//PLT_Jan21
MWDLP9NET_CYCVAE_EXPORT void cyclevae_post_melsp_excit_spk_convert_mwdlp9net_synthesize(
    MWDLP9CycleVAEPostMelspExcitSpkNetState *mwdlp9net, float *features,
        float *spk_code_aux, short *output, int *n_output, int flag_last_frame)
        //float *spk_code_aux, short *output, int *n_output, int flag_last_frame, short *pcm_1, short *pcm_2, short *pcm_3, short *pcm_4, float *gru_tmp)
        //float *spk_code_aux, short *output, int *n_output, int flag_last_frame, short *pcm_1, short *pcm_2, short *pcm_3, short *pcm_4, float *conv_tmp, float *gru_tmp)
        //float *spk_code_aux, short *output, int *n_output, int flag_last_frame, float *melsp_cv, float *lat_tmp, float *spk_tmp, float *conv_tmp, float *gru_tmp, float *f0_tmp)
        //float *spk_code_aux, short *output, int *n_output, int flag_last_frame, float *melsp_cv)
        //float *spk_code_aux, short *output, int *n_output, int flag_last_frame, float *melsp_cv, short *pcm_1, short *pcm_2, short *pcm_3, short *pcm_4)
    //MWDLP9CycleVAEPostMelspExcitSpkNetState *mwdlp9net, const float *features,
{
    int i, j, k, l, m;
    int sample[N_MBANDS];
    float melsp_cv[FEATURE_DIM_MELSP];
    float pdf[N_QUANTIZE_MBANDS];
    float gru_a_condition[RNN_MAIN_NEURONS_3];
    float gru_b_condition[RNN_SUB_NEURONS_3];
    const EmbeddingLayer *a_embed = &gru_a_embed;
    MWDLP9NNetState *nnet = &mwdlp9net->nnet;
    CycleVAEPostMelspExcitSpkNNetState *cv_nnet = &mwdlp9net->cv_nnet;
    int *last_sample_mb_pt = &mwdlp9net->last_sample[N_MBANDS];
    int *last_sample_0_pt = &mwdlp9net->last_sample[0];
    float tmp_out;
    float *pqmf_state_0_pt = &mwdlp9net->pqmf_state[0];
    float *pqmf_state_mbsqr_pt = &mwdlp9net->pqmf_state[N_MBANDS_SQR];
    float *pqmf_state_ordmb_pt = &mwdlp9net->pqmf_state[PQMF_ORDER_MBANDS];
    const float *pqmf_synth_filter = (&pqmf_synthesis)->input_weights;
    //printf("cv_synth a\n");
    //printf("frame_count: %d\n", mwdlp9net->frame_count);
    if (mwdlp9net->cv_frame_count < FEATURE_CONV_VC_DELAY) { //stored input frames not yet reach delay (cyclevae)
        float *mem_enc_melsp = cv_nnet->feature_conv_enc_melsp_state; //mem of stored input frames
        float *mem_enc_excit = cv_nnet->feature_conv_enc_excit_state; //mem of stored input frames
        float in_c[FEATURE_DIM_MELSP];
        RNN_COPY(in_c, features, FEATURE_DIM_MELSP);
        compute_normalize(&melsp_norm, in_c); //feature normalization
        if (mwdlp9net->cv_frame_count == 0) //pad_first
            for (i=0;i<ENC_CONV_KERNEL_1;i++) { //store first input with replicate padding kernel_size-1
                RNN_COPY(&mem_enc_melsp[i*FEATURE_DIM_MELSP], in_c, FEATURE_DIM_MELSP);
                RNN_COPY(&mem_enc_excit[i*FEATURE_DIM_MELSP], in_c, FEATURE_DIM_MELSP);
            }
        else {
            float tmp_c[FEATURE_CONV_ENC_STATE_SIZE];
            RNN_COPY(tmp_c, &mem_enc_melsp[FEATURE_DIM_MELSP], FEATURE_CONV_ENC_STATE_SIZE_1); //store previous input kernel_size-2
            RNN_COPY(&tmp_c[FEATURE_CONV_ENC_STATE_SIZE_1], in_c, FEATURE_DIM_MELSP); //add new input
            RNN_COPY(mem_enc_melsp, tmp_c, FEATURE_CONV_ENC_STATE_SIZE); //move to mem_enc_melsp
            RNN_COPY(mem_enc_excit, tmp_c, FEATURE_CONV_ENC_STATE_SIZE); //move to mem_enc_excit
        }
        mwdlp9net->cv_frame_count++;
        mwdlp9net->frame_count++;
        *n_output = 0;
        return ;
    } else if (mwdlp9net->cv_frame_count == FEATURE_CONV_VC_DELAY) {
        //printf("cv_synth a0 %d %d %d %d\n", mwdlp9net->cv_frame_count, mwdlp9net->frame_count, FEATURE_CONV_VC_DELAY, FEATURE_CONV_ALL_DELAY);
        //printf("cv_synth a1\n");
        //run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 1, lat_tmp, spk_tmp, conv_tmp, gru_tmp, f0_tmp); //convert melsp 1st frame
        run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 1); //convert melsp 1st frame
        //run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 1, mwdlp9net); //convert melsp 1st frame
        //printf("cv_synth a0 %d %d %d %d\n", mwdlp9net->cv_frame_count, mwdlp9net->frame_count, FEATURE_CONV_VC_DELAY, FEATURE_CONV_ALL_DELAY);
        //printf("cv_synth a2\n");
        //RNN_COPY(melsp_cv, features, FEATURE_DIM_MELSP);
        //RNN_COPY(features, melsp_cv, FEATURE_DIM_MELSP);
        compute_normalize(&feature_norm, melsp_cv); //feature normalization
        //printf("cv_synth a3\n");
        for (i=0;i<CONV_KERNEL_1;i++) //store first input with replicate padding kernel_size-1
            RNN_COPY(&nnet->feature_conv_state[i*FEATURES_DIM], melsp_cv, FEATURES_DIM);
        //compute_denormalize(&feature_norm, melsp_cv); //feature normalization
        //printf("cv_synth a4a\n");
        //printf("cv_synth a0 %d %d %d %d\n", mwdlp9net->cv_frame_count, mwdlp9net->frame_count, FEATURE_CONV_VC_DELAY, FEATURE_CONV_ALL_DELAY);
        if (mwdlp9net->frame_count < FEATURE_CONV_ALL_DELAY) { //stored input frames not yet reach delay (cyclevae+wvrnn)
        //    printf("cv_synth a5a\n");
            mwdlp9net->cv_frame_count++;
            mwdlp9net->frame_count++;
            *n_output = 0;
        //    printf("cv_synth a5\n");
            return;
        } //if cyclevae+wavernn delay is reached, then wavernn delay is causal
    } else if (mwdlp9net->frame_count < FEATURE_CONV_ALL_DELAY) { //stored input frames not yet reach cyclevae+wavernn delay
        //printf("cv_synth a6\n");
        //run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 0, lat_tmp, spk_tmp, conv_tmp, gru_tmp, f0_tmp); //convert melsp
        run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 0); //convert melsp
        //run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 0, mwdlp9net); //convert melsp
        //printf("cv_synth a7\n");
        //RNN_COPY(melsp_cv, features, FEATURE_DIM_MELSP);
        //RNN_COPY(features, melsp_cv, FEATURE_DIM_MELSP);
        compute_normalize(&feature_norm, melsp_cv); //feature normalization
        //printf("cv_synth a8\n");
        float *mem = nnet->feature_conv_state; //mem of stored input frames
        RNN_MOVE(mem, &mem[FEATURES_DIM], FEATURE_CONV_STATE_SIZE_1); //store previous input kernel_size-2
        RNN_COPY(&mem[FEATURE_CONV_STATE_SIZE_1], melsp_cv, FEATURES_DIM); //add new input
        mwdlp9net->cv_frame_count++;
        mwdlp9net->frame_count++;
        *n_output = 0;
        //compute_denormalize(&feature_norm, melsp_cv); //feature normalization
        //printf("cv_synth a9\n");
        return;
    }
    //printf("cv_synth b\n");
    //cyclevae+wavernn delay is reached
    if (!flag_last_frame) { //not last frame [decided by the section handling input waveform]
        //printf("cv_synth c\n");
        if (FEATURE_CONV_ALL_DELAY_FLAG) //if wavernn delay is not causal, then always convert melsp once reached this portion
            run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 0);
            //run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 0, lat_tmp, spk_tmp, conv_tmp, gru_tmp, f0_tmp);
            //run_frame_network_cyclevae_post_melsp_excit_spk(cv_nnet, features, spk_code_aux, melsp_cv, 0, mwdlp9net);
        //printf("cv_synth d\n");
        //RNN_COPY(melsp_cv, features, FEATURE_DIM_MELSP);
        //compute_normalize(&feature_norm, melsp_cv); //feature normalization
        //RNN_COPY(features, melsp_cv, FEATURE_DIM_MELSP);
        run_frame_network_mwdlp9(nnet, gru_a_condition, gru_b_condition, melsp_cv, 0);
        //run_frame_network_mwdlp9(nnet, gru_a_condition, gru_b_condition, melsp_cv, 0, conv_tmp);
        //printf("cv_synth e\n");
        //printf("\n");
        for (i=0,m=0,*n_output=0;i<N_SAMPLE_BANDS;i++) {
        //    printf("cv_synth f %d\n", i+1);
            //infer.
            run_sample_network_mwdlp9(nnet, a_embed, pdf,
                    gru_a_condition, gru_b_condition, mwdlp9net->last_sample);
                    //gru_a_condition, gru_b_condition, mwdlp9net->last_sample, gru_tmp);
            for (j=0;j<N_MBANDS;j++) {
                sample[j] = sample_from_pdf_mwdlp(&pdf[j*N_QUANTIZE], N_QUANTIZE);
                mwdlp9net->buffer_output[j] = mu_law_9_table[sample[j]]*N_MBANDS;
                //if (j==0) pcm_1[i] = mu_law_9_table[sample[j]]*32768;
                //else if (j==1) pcm_2[i] = mu_law_9_table[sample[j]]*32768;
                //else if (j==2) pcm_3[i] = mu_law_9_table[sample[j]]*32768;
                //else if (j==3) pcm_4[i] = mu_law_9_table[sample[j]]*32768;
                //printf(" [%d] %d %d %f", j+1, mwdlp9net->last_sample[j], sample[j], mwdlp9net->buffer_output[j]);
                //if (j==0) printf(" %d", pcm_1[i]);
                //else if (j==1) printf(" %d", pcm_2[i]);
                //else if (j==2) printf(" %d", pcm_3[i]);
                //else if (j==3) printf(" %d", pcm_4[i]);
            }
            //printf("\n");
            //update state of last_sample integer output
            //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; K: DLPC_ORDER
            RNN_MOVE(last_sample_mb_pt, last_sample_0_pt, LPC_ORDER_1_MBANDS);
            RNN_COPY(last_sample_0_pt, sample, N_MBANDS);
            //update state of pqmf synthesis input
            RNN_MOVE(pqmf_state_0_pt, pqmf_state_mbsqr_pt, PQMF_ORDER_MBANDS);
            RNN_COPY(pqmf_state_ordmb_pt, mwdlp9net->buffer_output, N_MBANDS_SQR);
            //pqmf synthesis if its delay sufficient [stored previous generated samples in a multiple of NBANDS]
            //previous samples at least PQMF_DELAY, because current output contribute 1 at each band
            //which means right_side + current sufficient to cover center+right side samples
            //printf("first_n_output: %d %d %d %d %d %d %d\n", PQMF_DELAY, FIRST_N_OUTPUT, FIRST_N_OUTPUT_MBANDS, PQMF_DELAY / N_MBANDS, PQMF_DELAY % N_MBANDS, (((PQMF_DELAY / N_MBANDS) + (PQMF_DELAY % N_MBANDS)) * N_MBANDS) / PQMF_DELAY, (140 % (PQMF_DELAY + 0)));
            if (mwdlp9net->sample_count >= PQMF_DELAY) {
                if (mwdlp9net->first_flag > 0) {
                    //synthesis n=n_bands samples
                    for (j=0;j<N_MBANDS;j++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp9net->pqmf_state[j*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp9net->deemph_mem;
                        mwdlp9net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        // 16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                //        printf("[%d] %d ", m, output[m]);
                    }
                } else {
                    //synthesis first n=(((pqmf_delay / n_bands + pqmf_delay % n_bands) * n_bands) % pqmf_delay) samples
                    //update state for first output t-(order//2),t,t+(order//2) [zero pad_left]
                    //shift from (center-first_n_output) in current pqmf_state to center (pqmf_delay_mbands)
                    //for (delay+first_n_output)*n_bands samples
                    RNN_COPY(&mwdlp9net->first_pqmf_state[PQMF_DELAY_MBANDS],
                        &mwdlp9net->pqmf_state[PQMF_DELAY_MBANDS-FIRST_N_OUTPUT_MBANDS],
                            PQMF_DELAY_MBANDS+FIRST_N_OUTPUT_MBANDS);
                    for (j=0;j<FIRST_N_OUTPUT;j++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp9net->first_pqmf_state[j*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp9net->deemph_mem;
                        mwdlp9net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //16bit pcm signed
                        output[m] = round(tmp_out * 32768);
               //         printf("[%d] %d ", m, output[m]);
                    }
                    mwdlp9net->first_flag = 1;
                    *n_output += FIRST_N_OUTPUT;
                    //synthesis n=n_bands samples
                    for (k=0;k<N_MBANDS;k++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp9net->pqmf_state[k*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp9net->deemph_mem;
                        mwdlp9net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                //        printf("[%d] %d ", m, output[m]);
                    }
                }
                //printf("\n");
                *n_output += N_MBANDS;
            }
            mwdlp9net->sample_count += N_MBANDS;
            //printf("cv_synth g\n");
        }    
    } else if (mwdlp9net->sample_count >= PQMF_DELAY) { //last frame [input waveform is ended,
        //    synthesize the delayed samples [frame-side and pqmf-side] if previous generated samples already reached PQMF delay]
        //synthesis n=pqmf_delay samples + (n_sample_bands x feature_pad_right) samples [replicate pad_right]
        //replicate_pad_right segmental_conv
        float *last_frame = &nnet->feature_conv_state[FEATURE_CONV_STATE_SIZE_1]; //for replicate pad_right
        for (l=0,m=0,*n_output=0;l<FEATURE_CONV_ALL_DELAY;l++) { //note that delay includes cyclevae+wavernn, if only neural vocoder discard cyclevae delay
            //printf("cv_synth h %d\n", l+1);
            run_frame_network_mwdlp9(nnet, gru_a_condition, gru_b_condition, last_frame, 1);
            //run_frame_network_mwdlp9(nnet, gru_a_condition, gru_b_condition, last_frame, 1, conv_tmp);
            //printf("cv_synth i\n");
            for (i=0;i<N_SAMPLE_BANDS;i++) {
            //    printf("cv_synth j %d\n", i+1);
                //infer.
                run_sample_network_mwdlp9(nnet, a_embed, pdf,
                        gru_a_condition, gru_b_condition, mwdlp9net->last_sample);
                        //gru_a_condition, gru_b_condition, mwdlp9net->last_sample, gru_tmp);
                for (j=0;j<N_MBANDS;j++) {
                    sample[j] = sample_from_pdf_mwdlp(&pdf[j*N_QUANTIZE], N_QUANTIZE);
                    mwdlp9net->buffer_output[j] = mu_law_9_table[sample[j]]*N_MBANDS;
                }
                //update state of last_sample integer output
                //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; K: DLPC_ORDER
                RNN_MOVE(last_sample_mb_pt, last_sample_0_pt, LPC_ORDER_1_MBANDS);
                RNN_COPY(last_sample_0_pt, sample, N_MBANDS);
                //update state of pqmf synthesis input
                //t-(order//2),t,t+(order//2), n_update = n_bands^2, n_stored_state = order*n_bands
                RNN_MOVE(pqmf_state_0_pt, pqmf_state_mbsqr_pt, PQMF_ORDER_MBANDS);
                RNN_COPY(pqmf_state_ordmb_pt, mwdlp9net->buffer_output, N_MBANDS_SQR);
                //synthesis n=n_bands samples
                for (j=0;j<N_MBANDS;j++,m++) {
                    tmp_out = 0;
                    //pqmf_synth
                    sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1, &mwdlp9net->pqmf_state[j*N_MBANDS]);
                    //clamp
                    if (tmp_out < -1) tmp_out = -1;
                    else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                    //de-emphasis
                    tmp_out += PREEMPH*mwdlp9net->deemph_mem;
                    mwdlp9net->deemph_mem = tmp_out;
                    //clamp
                    if (tmp_out < -1) tmp_out = -1;
                    else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                    //16bit pcm signed
                    output[m] = round(tmp_out * 32768);
                }
                *n_output += N_MBANDS;
            }    
        }
        //zero_pad_right pqmf [last pqmf state size is PQMF_ORDER+PQMF_DELAY, i.e., 2*DELAY+DELAY
        //    because we want to synthesize DELAY number of samples, themselves require
        //    2*DELAY samples covering left- and right-sides of kaiser window]
        //printf("cv_synth k\n");
        RNN_COPY(mwdlp9net->last_pqmf_state, &mwdlp9net->pqmf_state[N_MBANDS_SQR], PQMF_ORDER_MBANDS);
        //from [o_1,...,o_{2N},0] to [o_1,..,o_{N+1},0,...,0]; N=PQMF_DELAY=PQMF_ORDER//2=(TAPS-1)//2
        for  (i=0;i<PQMF_DELAY;i++,m++) {
        //    printf("cv_synth l %d\n", i+1);
            tmp_out = 0;
            //pqmf_synth
            sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1, &mwdlp9net->last_pqmf_state[i*N_MBANDS]);
            //clamp
            if (tmp_out < -1) tmp_out = -1;
            else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
            //de-emphasis
            tmp_out += PREEMPH*mwdlp9net->deemph_mem;
            mwdlp9net->deemph_mem = tmp_out;
            //clamp
            if (tmp_out < -1) tmp_out = -1;
            else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
            //16bit pcm signed
            output[m] = round(tmp_out * 32768);
        }
        *n_output += PQMF_DELAY;
        //printf("cv_synth m\n");
    }
    return;
}


//PLT_Jan21
MWDLP9NET_CYCVAE_EXPORT void mwdlp9net_synthesize(MWDLP9NetState *mwdlp9net, const float *features,
    short *output, int *n_output, int flag_last_frame)
{
    int i, j, k, l, m;
    int sample[N_MBANDS];
    float pdf[N_QUANTIZE_MBANDS];
    float gru_a_condition[RNN_MAIN_NEURONS_3];
    float gru_b_condition[RNN_SUB_NEURONS_3];
    //float conv_tmp[FEATURE_CONV_OUT_SIZE];
    //float conv_tmp[FEATURE_DENSE_OUT_SIZE];
    //float conv_tmp[RNN_MAIN_NEURONS_3];
    //float conv_tmp[RNN_SUB_NEURONS_3];
    //float gru_tmp[RNN_MAIN_NEURONS_3];
    const EmbeddingLayer *a_embed = &gru_a_embed;
    MWDLP9NNetState *nnet = &mwdlp9net->nnet;
    int *last_sample_mb_pt = &mwdlp9net->last_sample[N_MBANDS];
    int *last_sample_0_pt = &mwdlp9net->last_sample[0];
    float tmp_out;
    float *pqmf_state_0_pt = &mwdlp9net->pqmf_state[0];
    float *pqmf_state_mbsqr_pt = &mwdlp9net->pqmf_state[N_MBANDS_SQR];
    float *pqmf_state_ordmb_pt = &mwdlp9net->pqmf_state[PQMF_ORDER_MBANDS];
    const float *pqmf_synth_filter = (&pqmf_synthesis)->input_weights;
    if (mwdlp9net->frame_count < FEATURE_CONV_DELAY) { //stored input frames not yet reach delay
        float *mem = nnet->feature_conv_state; //mem of stored input frames
        float in[FEATURES_DIM];
        RNN_COPY(in, features, FEATURES_DIM);
        compute_normalize(&feature_norm, in); //feature normalization
        if (mwdlp9net->frame_count == 0) //pad_first
            for (i=0;i<CONV_KERNEL_1;i++) //store first input with replicate padding kernel_size-1
                RNN_COPY(&mem[i*FEATURES_DIM], in, FEATURES_DIM);
        else {
            RNN_MOVE(mem, &mem[FEATURES_DIM], FEATURE_CONV_STATE_SIZE_1); //store previous input kernel_size-2
            RNN_COPY(&mem[FEATURE_CONV_STATE_SIZE_1], in, FEATURES_DIM); //add new input
        }
        mwdlp9net->frame_count++;
        *n_output = 0;
        return;
    }
    if (!flag_last_frame) {
        run_frame_network_mwdlp9(nnet, gru_a_condition, gru_b_condition, features, 0);
        //run_frame_network_mwdlp9(nnet, gru_a_condition, gru_b_condition, features, 0, conv_tmp);
        for (i=0,m=0,*n_output=0;i<N_SAMPLE_BANDS;i++) {
            //infer.
            run_sample_network_mwdlp9(nnet, a_embed, pdf,
                    gru_a_condition, gru_b_condition, mwdlp9net->last_sample);
                    //gru_a_condition, gru_b_condition, mwdlp9net->last_sample, gru_tmp);
            for (j=0;j<N_MBANDS;j++) {
                sample[j] = sample_from_pdf_mwdlp(&pdf[j*N_QUANTIZE], N_QUANTIZE);
                mwdlp9net->buffer_output[j] = mu_law_9_table[sample[j]]*N_MBANDS;
            }
            //update state of last_sample integer output
            //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; K: DLPC_ORDER
            RNN_MOVE(last_sample_mb_pt, last_sample_0_pt, LPC_ORDER_1_MBANDS);
            RNN_COPY(last_sample_0_pt, sample, N_MBANDS);
            //update state of pqmf synthesis input
            RNN_MOVE(pqmf_state_0_pt, pqmf_state_mbsqr_pt, PQMF_ORDER_MBANDS);
            RNN_COPY(pqmf_state_ordmb_pt, mwdlp9net->buffer_output, N_MBANDS_SQR);
            //pqmf synthesis if its delay sufficient
            if (mwdlp9net->sample_count >= PQMF_DELAY) {
                if (mwdlp9net->first_flag > 0) {
                    //synthesis n=n_bands samples
                    for (j=0;j<N_MBANDS;j++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp9net->pqmf_state[j*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp9net->deemph_mem;
                        mwdlp9net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        // 16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                    }
                } else {
                    //synthesis first n=((pqmf_order+1) % n_bands) samples
                    //update state for first output t-(order//2),t,t+(order//2) [zero pad_left]
                    //shift from (center-first_n_output) in current pqmf_state to center (pqmf_delay_mbands)
                    //for (delay+first_n_output)*n_bands samples
                    RNN_COPY(&mwdlp9net->first_pqmf_state[PQMF_DELAY_MBANDS],
                        &mwdlp9net->pqmf_state[PQMF_DELAY_MBANDS-FIRST_N_OUTPUT_MBANDS],
                            PQMF_DELAY_MBANDS+FIRST_N_OUTPUT_MBANDS);
                    for (j=0;j<FIRST_N_OUTPUT;j++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp9net->first_pqmf_state[j*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp9net->deemph_mem;
                        mwdlp9net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                    }
                    mwdlp9net->first_flag = 1;
                    *n_output += FIRST_N_OUTPUT;
                    //synthesis n=n_bands samples
                    for (k=0;k<N_MBANDS;k++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp9net->pqmf_state[k*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp9net->deemph_mem;
                        mwdlp9net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                    }
                }
                *n_output += N_MBANDS;
            }
            mwdlp9net->sample_count += N_MBANDS;
        }    
    } else if (mwdlp9net->sample_count >= PQMF_DELAY) {
        //synthesis n=pqmf_order samples + (n_sample_bands x feature_pad_right) samples [replicate pad_right]
        //replicate_pad_right segmental_conv
        float *last_frame = &nnet->feature_conv_state[FEATURE_CONV_STATE_SIZE_1]; //for replicate pad_right
        for (l=0,m=0,*n_output=0;l<FEATURE_CONV_DELAY;l++) {
            run_frame_network_mwdlp9(nnet, gru_a_condition, gru_b_condition, last_frame, 1);
            //run_frame_network_mwdlp9(nnet, gru_a_condition, gru_b_condition, last_frame, 1, conv_tmp);
            for (i=0;i<N_SAMPLE_BANDS;i++) {
                //infer
                run_sample_network_mwdlp9(nnet, a_embed, pdf,
                        gru_a_condition, gru_b_condition, mwdlp9net->last_sample);
                        //gru_a_condition, gru_b_condition, mwdlp9net->last_sample, gru_tmp);
                for (j=0;j<N_MBANDS;j++) {
                    sample[j] = sample_from_pdf_mwdlp(&pdf[j*N_QUANTIZE], N_QUANTIZE);
                    mwdlp9net->buffer_output[j] = mu_law_9_table[sample[j]]*N_MBANDS;
                }
                //update state of last_sample integer output
                //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; K: DLPC_ORDER
                RNN_MOVE(last_sample_mb_pt, last_sample_0_pt, LPC_ORDER_1_MBANDS);
                RNN_COPY(last_sample_0_pt, sample, N_MBANDS);
                //update state of pqmf synthesis input
                //t-(order//2),t,t+(order//2), n_update = n_bands^2, n_stored_state = order*n_bands
                RNN_MOVE(pqmf_state_0_pt, pqmf_state_mbsqr_pt, PQMF_ORDER_MBANDS);
                RNN_COPY(pqmf_state_ordmb_pt, mwdlp9net->buffer_output, N_MBANDS_SQR);
                //synthesis n=n_bands samples
                for (j=0;j<N_MBANDS;j++,m++) {
                    tmp_out = 0;
                    //pqmf_synth
                    sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1, &mwdlp9net->pqmf_state[j*N_MBANDS]);
                    //clamp
                    if (tmp_out < -1) tmp_out = -1;
                    else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                    //de-emphasis
                    tmp_out += PREEMPH*mwdlp9net->deemph_mem;
                    mwdlp9net->deemph_mem = tmp_out;
                    //clamp
                    if (tmp_out < -1) tmp_out = -1;
                    else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                    //16bit pcm signed
                    output[m] = round(tmp_out * 32768);
                }
                *n_output += N_MBANDS;
            }    
        }
        //zero_pad_right pqmf
        RNN_MOVE(mwdlp9net->last_pqmf_state, &mwdlp9net->pqmf_state[N_MBANDS_SQR], PQMF_ORDER_MBANDS);
        //from [o_1,...,o_{2N},0] to [o_1,..,o_{N+1},0,...,0]; N=PQMF_DELAY=PQMF_ORDER//2=(TAPS-1)//2
        for  (i=0;i<PQMF_DELAY;i++,m++) {
            tmp_out = 0;
            //pqmf_synth
            sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1, &mwdlp9net->last_pqmf_state[i*N_MBANDS]);
            //clamp
            if (tmp_out < -1) tmp_out = -1;
            else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
            //de-emphasis
            tmp_out += PREEMPH*mwdlp9net->deemph_mem;
            mwdlp9net->deemph_mem = tmp_out;
            //clamp
            if (tmp_out < -1) tmp_out = -1;
            else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
            //16bit pcm signed
            output[m] = round(tmp_out * 32768);
        }
        *n_output += PQMF_DELAY;
    }
    return;
}
