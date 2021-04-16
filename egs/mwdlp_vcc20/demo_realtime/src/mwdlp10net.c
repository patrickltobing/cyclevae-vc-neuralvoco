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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Dec. 2020 - Mar. 2021,
   marked by PLT_<Dec20/Jan21/Mar21> */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>
#include <time.h>
#include "nnet_data.h"
#include "nnet.h"
#include "common.h"
#include "arch.h"
#include "mwdlp10net.h"
#include "mwdlp10net_private.h"
#include "mu_law_10_table.h"

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
static void run_frame_network_mwdlp10(MWDLP10NNetState *net, float *gru_a_condition, float *gru_b_condition, 
    float *gru_c_condition, const float *features, int flag_last_frame)
{
    float conv_out[FEATURE_CONV_OUT_SIZE];
    float condition[FEATURE_DENSE_OUT_SIZE];
    //clock_t t;
    //double time_taken;
    //feature normalization if not last frame, just replicate if last frame
    if (!flag_last_frame) {
        float in[FEATURES_DIM];
        //float conv_in[FEATURES_DIM];
        //float conv_in_in[FEATURE_CONV_IN_IN_OUT_SIZE];
        //float conv_in_out[FEATURES_DIM];
        RNN_COPY(in, features, FEATURES_DIM);
        compute_normalize(&feature_norm, in);
        compute_conv1d_linear(&feature_conv, conv_out, net->feature_conv_state, in);
        //compute_dense(&feature_conv_in, conv_in, in);
        //compute_dense(&feature_conv_in_in, conv_in_in, in);
        //compute_dense(&feature_conv_in_out, conv_in_out, conv_in_in);
        //compute_conv1d_linear(&feature_conv, conv_out, net->feature_conv_state, conv_in);
        //compute_conv1d_linear(&feature_conv, conv_out, net->feature_conv_state, conv_in_out);
    } else {
        compute_conv1d_linear(&feature_conv, conv_out, net->feature_conv_state, features);
    }
    //segmental input conv. and fc layer with relu
    //t = clock();
    //time_taken = ((double)(clock()-t))/CLOCKS_PER_SEC;
    //printf("conv_mwdlp %f sec.\n", time_taken);
    compute_dense(&feature_dense, condition, conv_out);
    //compute condition (input_vector_cond*input_matrix_cond+input_bias) for each gru_a, b, and c; fixed for one frame
    compute_dense_linear(&gru_a_dense_feature, gru_a_condition, condition);
    compute_dense_linear(&gru_b_dense_feature, gru_b_condition, condition);
    compute_dense_linear(&gru_c_dense_feature, gru_c_condition, condition);
}


//PLT_Mar21
static void run_sample_network_mwdlp10_coarse(MWDLP10NNetState *net, const EmbeddingLayer *a_embed_coarse,
    const EmbeddingLayer *a_embed_fine, const EmbeddingLayer *prev_logits_coarse, float *pdf,
        const float *gru_a_condition, const float *gru_b_condition, int *last_coarse, int *last_fine)
{
    int i, j, idx_bands, idx_coarse, idx_fine;
    float gru_a_input[RNN_MAIN_NEURONS_3];
    float gru_b_input[RNN_SUB_NEURONS_3];
    //copy input conditioning * GRU_input_cond_weights + input_bias contribution (a)
    RNN_COPY(gru_a_input, gru_a_condition, RNN_MAIN_NEURONS_3);
    //compute last coarse / last fine embedding * GRU_input_embed_weights contribution (a)
    for (i=0;i<N_MBANDS;i++) {
        // stored embedding: n_bands x 32 x hidden_size_main
        for (j=0,idx_bands=i*RNN_MAIN_NEURONS_3_SQRT_QUANTIZE,
                    idx_coarse=idx_bands+last_coarse[i]*RNN_MAIN_NEURONS_3,
                        idx_fine=idx_bands+last_fine[i]*RNN_MAIN_NEURONS_3;j<RNN_MAIN_NEURONS_3;j++)
            gru_a_input[j] += a_embed_coarse->embedding_weights[idx_coarse + j]
                                + a_embed_fine->embedding_weights[idx_fine + j];
    }
    //compute sparse gru_a
    compute_sparse_gru(&sparse_gru_a, net->gru_a_state, gru_a_input);
    //copy input conditioning * GRU_input_cond_weights + bias contribution (b)
    RNN_COPY(gru_b_input, gru_b_condition, RNN_SUB_NEURONS_3);
    //compute gru_a state contribution to gru_b
    sgemv_accum16_(gru_b_input, (&gru_b_dense_feature_state)->input_weights, RNN_SUB_NEURONS_3, RNN_MAIN_NEURONS,
        RNN_SUB_NEURONS_3, net->gru_a_state);
    //compute gru_b and coarse_output
    compute_gru3(&gru_b, net->gru_b_state, gru_b_input);
    compute_mdense_mwdlp10(&dual_fc_coarse, &fc_out_coarse, prev_logits_coarse->embedding_weights, pdf, net->gru_b_state, last_coarse);
}


//PLT_Mar21
static void run_sample_network_mwdlp10_fine(MWDLP10NNetState *net, const EmbeddingLayer *c_embed_coarse,
    const EmbeddingLayer *prev_logits_fine, float *pdf, const float *gru_c_condition, int *coarse, int *last_fine)
{
    int i, j, idx_coarse;
    float gru_c_input[RNN_SUB_NEURONS_3];
    //copy input conditioning * GRU_input_cond_weights + input_bias contribution (c)
    RNN_COPY(gru_c_input, gru_c_condition, RNN_SUB_NEURONS_3);
    //compute current coarse embedding * GRU_input_embed_weights contribution (c)
    for (i=0;i<N_MBANDS;i++) {
        // stored embedding: n_bands x 32 x hidden_size_sub
        for (j=0,idx_coarse=i*RNN_SUB_NEURONS_3_SQRT_QUANTIZE+coarse[i]*RNN_SUB_NEURONS_3;j<RNN_SUB_NEURONS_3;j++)
            gru_c_input[j] += c_embed_coarse->embedding_weights[idx_coarse + j];
    }
    //compute gru_b state contribution to gru_c
    sgemv_accum16_(gru_c_input, (&gru_c_dense_feature_state)->input_weights, RNN_SUB_NEURONS_3, RNN_SUB_NEURONS,
        RNN_SUB_NEURONS_3, net->gru_b_state);
    //compute gru_c and fine_output
    compute_gru3(&gru_c, net->gru_c_state, gru_c_input);
    compute_mdense_mwdlp10(&dual_fc_fine, &fc_out_fine, prev_logits_fine->embedding_weights, pdf, net->gru_c_state, last_fine);
}


//PLT_Mar21
static void run_sample_network_mwdlp10_coarse_nodlpc(MWDLP10NNetState *net, const EmbeddingLayer *a_embed_coarse,
    const EmbeddingLayer *a_embed_fine, float *pdf, const float *gru_a_condition, const float *gru_b_condition,
        int *last_coarse, int *last_fine)
{
    int i, j, idx_bands, idx_coarse, idx_fine;
    float gru_a_input[RNN_MAIN_NEURONS_3];
    float gru_b_input[RNN_SUB_NEURONS_3];
    //copy input conditioning * GRU_input_cond_weights + input_bias contribution (a)
    RNN_COPY(gru_a_input, gru_a_condition, RNN_MAIN_NEURONS_3);
    //compute last coarse / last fine embedding * GRU_input_embed_weights contribution (a)
    for (i=0;i<N_MBANDS;i++) {
        // stored embedding: n_bands x 32 x hidden_size_main
        for (j=0,idx_bands=i*RNN_MAIN_NEURONS_3_SQRT_QUANTIZE,
                    idx_coarse=idx_bands+last_coarse[i]*RNN_MAIN_NEURONS_3,
                        idx_fine=idx_bands+last_fine[i]*RNN_MAIN_NEURONS_3;j<RNN_MAIN_NEURONS_3;j++)
            gru_a_input[j] += a_embed_coarse->embedding_weights[idx_coarse + j]
                                + a_embed_fine->embedding_weights[idx_fine + j];
    }
    //compute sparse gru_a
    compute_sparse_gru(&sparse_gru_a, net->gru_a_state, gru_a_input);
    //copy input conditioning * GRU_input_cond_weights + bias contribution (b)
    RNN_COPY(gru_b_input, gru_b_condition, RNN_SUB_NEURONS_3);
    //compute gru_a state contribution to gru_b
    sgemv_accum16_(gru_b_input, (&gru_b_dense_feature_state)->input_weights, RNN_SUB_NEURONS_3, RNN_MAIN_NEURONS,
        RNN_SUB_NEURONS_3, net->gru_a_state);
    //compute gru_b and coarse_output
    compute_gru3(&gru_b, net->gru_b_state, gru_b_input);
    compute_mdense_mwdlp10_nodlpc(&dual_fc_coarse, &fc_out_coarse, pdf, net->gru_b_state);
}


//PLT_Mar21
static void run_sample_network_mwdlp10_fine_nodlpc(MWDLP10NNetState *net, const EmbeddingLayer *c_embed_coarse, float *pdf,
    const float *gru_c_condition, int *coarse)
{
    int i, j, idx_coarse;
    float gru_c_input[RNN_SUB_NEURONS_3];
    //copy input conditioning * GRU_input_cond_weights + input_bias contribution (c)
    RNN_COPY(gru_c_input, gru_c_condition, RNN_SUB_NEURONS_3);
    //compute current coarse embedding * GRU_input_embed_weights contribution (c)
    for (i=0;i<N_MBANDS;i++) {
        // stored embedding: n_bands x 32 x hidden_size_sub
        for (j=0,idx_coarse=i*RNN_SUB_NEURONS_3_SQRT_QUANTIZE+coarse[i]*RNN_SUB_NEURONS_3;j<RNN_SUB_NEURONS_3;j++)
            gru_c_input[j] += c_embed_coarse->embedding_weights[idx_coarse + j];
    }
    //compute gru_b state contribution to gru_c
    sgemv_accum16_(gru_c_input, (&gru_c_dense_feature_state)->input_weights, RNN_SUB_NEURONS_3, RNN_SUB_NEURONS,
        RNN_SUB_NEURONS_3, net->gru_b_state);
    //compute gru_c and fine_output
    compute_gru3(&gru_c, net->gru_c_state, gru_c_input);
    compute_mdense_mwdlp10_nodlpc(&dual_fc_fine, &fc_out_fine, pdf, net->gru_c_state);
}


//PLT_Dec20
MWDLP10NET_EXPORT int mwdlp10net_get_size()
{
    return sizeof(MWDLP10NetState);
}


//PLT_Dec20
MWDLP10NET_EXPORT MWDLP10NetState *mwdlp10net_create()
{
    MWDLP10NetState *mwdlp10net;
    mwdlp10net = (MWDLP10NetState *) calloc(1,mwdlp10net_get_size());
    if (mwdlp10net != NULL) {
        if (FIRST_N_OUTPUT == 0) mwdlp10net->first_flag = 1;
        int i, j, k;
        if (!NO_DLPC) {
            for (i=0, k=0;i<DLPC_ORDER;i++)
                for (j=0;j<N_MBANDS;j++,k++)
                    mwdlp10net->last_coarse[k] = INIT_LAST_SAMPLE;
        } else {
            for (j=0;j<N_MBANDS;j++)
                mwdlp10net->last_coarse[j] = INIT_LAST_SAMPLE;
        }
        for (i=0;i<N_QUANTIZE;i++)
            mwdlp10net->mu_law_10_table[i] = mu_law_10_table[i];
        return mwdlp10net;
    }
    printf("Cannot allocate and initialize memory for MWDLP10NetState.\n");
    exit(EXIT_FAILURE);
    return NULL;
}


//PLT_Dec20
MWDLP10NET_EXPORT void mwdlp10net_destroy(MWDLP10NetState *mwdlp10net)
{
    if (mwdlp10net != NULL) free(mwdlp10net);
}


//PLT_Jan21
MWDLP10NET_EXPORT void mwdlp10net_synthesize(MWDLP10NetState *mwdlp10net, const float *features,
    short *output, int *n_output, int flag_last_frame)
{
    int i, j, k, l, m;
    int coarse[N_MBANDS];
    int fine[N_MBANDS];
    float pdf[SQRT_QUANTIZE_MBANDS];
    float gru_a_condition[RNN_MAIN_NEURONS_3];
    float gru_b_condition[RNN_SUB_NEURONS_3];
    float gru_c_condition[RNN_SUB_NEURONS_3];
    const EmbeddingLayer *a_embed_coarse = &gru_a_embed_coarse;
    const EmbeddingLayer *a_embed_fine = &gru_a_embed_fine;
    const EmbeddingLayer *c_embed_coarse = &gru_c_embed_coarse;
    const EmbeddingLayer *prev_logits_c = &prev_logits_coarse;
    const EmbeddingLayer *prev_logits_f = &prev_logits_fine;
    MWDLP10NNetState *nnet = &mwdlp10net->nnet;
    int *last_coarse_mb_pt = &mwdlp10net->last_coarse[N_MBANDS];
    int *last_coarse_0_pt = &mwdlp10net->last_coarse[0];
    int *last_fine_mb_pt = &mwdlp10net->last_fine[N_MBANDS];
    int *last_fine_0_pt = &mwdlp10net->last_fine[0];
    float tmp_out;
    float *pqmf_state_0_pt = &mwdlp10net->pqmf_state[0];
    float *pqmf_state_mbsqr_pt = &mwdlp10net->pqmf_state[N_MBANDS_SQR];
    float *pqmf_state_ordmb_pt = &mwdlp10net->pqmf_state[PQMF_ORDER_MBANDS];
    const float *pqmf_synth_filter = (&pqmf_synthesis)->input_weights;
    if (mwdlp10net->frame_count < FEATURE_CONV_DELAY) { //stored input frames not yet reach delay
        float *mem = nnet->feature_conv_state; //mem of stored input frames
        float in[FEATURES_DIM];
        RNN_COPY(in, features, FEATURES_DIM);
        compute_normalize(&feature_norm, in); //feature normalization
        //float conv_in[FEATURES_DIM];
        //float conv_in_in[FEATURE_CONV_IN_IN_OUT_SIZE];
        //float conv_in_out[FEATURES_DIM];
        //compute_dense(&feature_conv_in, conv_in, in);
        //compute_dense(&feature_conv_in_in, conv_in_in, in);
        //compute_dense(&feature_conv_in_out, conv_in_out, conv_in_in);
        if (mwdlp10net->frame_count == 0) //pad_first
            for (i=0;i<CONV_KERNEL_1;i++) //store first input with replicate padding kernel_size-1
                //RNN_COPY(&mem[i*FEATURES_DIM], conv_in, FEATURES_DIM);
                //RNN_COPY(&mem[i*FEATURES_DIM], conv_in_out, FEATURES_DIM);
                RNN_COPY(&mem[i*FEATURES_DIM], in, FEATURES_DIM);
        else {
            RNN_MOVE(mem, &mem[FEATURES_DIM], FEATURE_CONV_STATE_SIZE_1); //store previous input kernel_size-2
            RNN_COPY(&mem[FEATURE_CONV_STATE_SIZE_1], in, FEATURES_DIM); //add new input
        }
        mwdlp10net->frame_count++;
        *n_output = 0;
        return;
    }
    if (!flag_last_frame) {
        run_frame_network_mwdlp10(nnet, gru_a_condition, gru_b_condition, gru_c_condition, features, 0);
        for (i=0,m=0,*n_output=0;i<N_SAMPLE_BANDS;i++) {
            //coarse
            run_sample_network_mwdlp10_coarse(nnet, a_embed_coarse, a_embed_fine, prev_logits_c, pdf,
                    gru_a_condition, gru_b_condition, mwdlp10net->last_coarse, mwdlp10net->last_fine);
            for (j=0;j<N_MBANDS;j++)
                coarse[j] = sample_from_pdf_mwdlp(&pdf[j*SQRT_QUANTIZE], SQRT_QUANTIZE);
            //fine
            run_sample_network_mwdlp10_fine(nnet, c_embed_coarse, prev_logits_f, pdf, gru_c_condition, coarse, mwdlp10net->last_fine);
            //printf("\n");
            for (j=0;j<N_MBANDS;j++) {
                fine[j] = sample_from_pdf_mwdlp(&pdf[j*SQRT_QUANTIZE], SQRT_QUANTIZE);
                mwdlp10net->buffer_output[j] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*N_MBANDS;
            //    printf("[%d] %d %d %d %f ", j+1, coarse[j], fine[j], coarse[j]*SQRT_QUANTIZE+fine[j], mwdlp10net->buffer_output[j]);
                //if (j==0) pcm_1[i] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*32768;
                //else if (j==1) pcm_2[i] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*32768;
                //else if (j==2) pcm_3[i] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*32768;
                //else if (j==3) pcm_4[i] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*32768;
            }
            //printf("\n");
            //update state of last_coarse and last_fine integer output
            //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; K: DLPC_ORDER
            RNN_MOVE(last_coarse_mb_pt, last_coarse_0_pt, LPC_ORDER_1_MBANDS);
            RNN_COPY(last_coarse_0_pt, coarse, N_MBANDS);
            RNN_MOVE(last_fine_mb_pt, last_fine_0_pt, LPC_ORDER_1_MBANDS);
            RNN_COPY(last_fine_0_pt, fine, N_MBANDS);
            //update state of pqmf synthesis input
            RNN_MOVE(pqmf_state_0_pt, pqmf_state_mbsqr_pt, PQMF_ORDER_MBANDS);
            RNN_COPY(pqmf_state_ordmb_pt, mwdlp10net->buffer_output, N_MBANDS_SQR);
            //pqmf synthesis if its delay sufficient
            if (mwdlp10net->sample_count >= PQMF_DELAY) {
                if (mwdlp10net->first_flag > 0) {
                    //synthesis n=n_bands samples
                    for (j=0;j<N_MBANDS;j++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp10net->pqmf_state[j*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp10net->deemph_mem;
                        mwdlp10net->deemph_mem = tmp_out;
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
                    RNN_COPY(&mwdlp10net->first_pqmf_state[PQMF_DELAY_MBANDS],
                        &mwdlp10net->pqmf_state[PQMF_DELAY_MBANDS-FIRST_N_OUTPUT_MBANDS],
                            PQMF_DELAY_MBANDS+FIRST_N_OUTPUT_MBANDS);
                    for (j=0;j<FIRST_N_OUTPUT;j++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp10net->first_pqmf_state[j*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp10net->deemph_mem;
                        mwdlp10net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                    }
                    mwdlp10net->first_flag = 1;
                    *n_output += FIRST_N_OUTPUT;
                    //synthesis n=n_bands samples
                    for (k=0;k<N_MBANDS;k++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp10net->pqmf_state[k*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp10net->deemph_mem;
                        mwdlp10net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                    }
                }
                *n_output += N_MBANDS;
            }
            mwdlp10net->sample_count += N_MBANDS;
        }    
    } else if (mwdlp10net->sample_count >= PQMF_DELAY) {
        //synthesis n=pqmf_order samples + (n_sample_bands x feature_pad_right) samples [replicate pad_right]
        //replicate_pad_right segmental_conv
        float *last_frame = &nnet->feature_conv_state[FEATURE_CONV_STATE_SIZE_1]; //for replicate pad_right
        for (l=0,m=0,*n_output=0;l<FEATURE_CONV_DELAY;l++) {
            run_frame_network_mwdlp10(nnet, gru_a_condition, gru_b_condition, gru_c_condition, last_frame, 1);
            for (i=0;i<N_SAMPLE_BANDS;i++) {
                //coarse
                run_sample_network_mwdlp10_coarse(nnet, a_embed_coarse, a_embed_fine, prev_logits_c, pdf,
                        gru_a_condition, gru_b_condition, mwdlp10net->last_coarse, mwdlp10net->last_fine);
                for (j=0;j<N_MBANDS;j++)
                    coarse[j] = sample_from_pdf_mwdlp(&pdf[j*SQRT_QUANTIZE], SQRT_QUANTIZE);
                //fine
                run_sample_network_mwdlp10_fine(nnet, c_embed_coarse, prev_logits_f, pdf, gru_c_condition, coarse,
                    mwdlp10net->last_fine);
                for (j=0;j<N_MBANDS;j++) {
                    fine[j] = sample_from_pdf_mwdlp(&pdf[j*SQRT_QUANTIZE], SQRT_QUANTIZE);
                    //float,[-1,1),upsample-bands(x n_bands)
                    mwdlp10net->buffer_output[j] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*N_MBANDS;
                }
                //update state of last_coarse and last_fine integer output
                //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; K: DLPC_ORDER
                RNN_MOVE(last_coarse_mb_pt, last_coarse_0_pt, LPC_ORDER_1_MBANDS);
                RNN_COPY(last_coarse_0_pt, coarse, N_MBANDS);
                RNN_MOVE(last_fine_mb_pt, last_fine_0_pt, LPC_ORDER_1_MBANDS);
                RNN_COPY(last_fine_0_pt, fine, N_MBANDS);
                //update state of pqmf synthesis input
                //t-(order//2),t,t+(order//2), n_update = n_bands^2, n_stored_state = order*n_bands
                RNN_MOVE(pqmf_state_0_pt, pqmf_state_mbsqr_pt, PQMF_ORDER_MBANDS);
                RNN_COPY(pqmf_state_ordmb_pt, mwdlp10net->buffer_output, N_MBANDS_SQR);
                //synthesis n=n_bands samples
                for (j=0;j<N_MBANDS;j++,m++) {
                    tmp_out = 0;
                    //pqmf_synth
                    sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1, &mwdlp10net->pqmf_state[j*N_MBANDS]);
                    //clamp
                    if (tmp_out < -1) tmp_out = -1;
                    else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                    //de-emphasis
                    tmp_out += PREEMPH*mwdlp10net->deemph_mem;
                    mwdlp10net->deemph_mem = tmp_out;
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
        RNN_MOVE(mwdlp10net->last_pqmf_state, &mwdlp10net->pqmf_state[N_MBANDS_SQR], PQMF_ORDER_MBANDS);
        //from [o_1,...,o_{2N},0] to [o_1,..,o_{N+1},0,...,0]; N=PQMF_DELAY=PQMF_ORDER//2=(TAPS-1)//2
        for  (i=0;i<PQMF_DELAY;i++,m++) {
            tmp_out = 0;
            //pqmf_synth
            sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1, &mwdlp10net->last_pqmf_state[i*N_MBANDS]);
            //clamp
            if (tmp_out < -1) tmp_out = -1;
            else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
            //de-emphasis
            tmp_out += PREEMPH*mwdlp10net->deemph_mem;
            mwdlp10net->deemph_mem = tmp_out;
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


//PLT_Mar21
MWDLP10NET_EXPORT void mwdlp10net_synthesize_nodlpc(MWDLP10NetState *mwdlp10net, const float *features,
    short *output, int *n_output, int flag_last_frame)
{
    int i, j, k, l, m;
    //int coarse[N_MBANDS];
    //int fine[N_MBANDS];
    float pdf[SQRT_QUANTIZE_MBANDS];
    float gru_a_condition[RNN_MAIN_NEURONS_3];
    float gru_b_condition[RNN_SUB_NEURONS_3];
    float gru_c_condition[RNN_SUB_NEURONS_3];
    const EmbeddingLayer *a_embed_coarse = &gru_a_embed_coarse;
    const EmbeddingLayer *a_embed_fine = &gru_a_embed_fine;
    const EmbeddingLayer *c_embed_coarse = &gru_c_embed_coarse;
    MWDLP10NNetState *nnet = &mwdlp10net->nnet;
    int *last_coarse_0_pt = &mwdlp10net->last_coarse[0];
    int *last_fine_0_pt = &mwdlp10net->last_fine[0];
    float tmp_out;
    float *pqmf_state_0_pt = &mwdlp10net->pqmf_state[0];
    float *pqmf_state_mbsqr_pt = &mwdlp10net->pqmf_state[N_MBANDS_SQR];
    float *pqmf_state_ordmb_pt = &mwdlp10net->pqmf_state[PQMF_ORDER_MBANDS];
    const float *pqmf_synth_filter = (&pqmf_synthesis)->input_weights;
    if (mwdlp10net->frame_count < FEATURE_CONV_DELAY) { //stored input frames not yet reach delay
        float *mem = nnet->feature_conv_state; //mem of stored input frames
        float in[FEATURES_DIM];
        RNN_COPY(in, features, FEATURES_DIM);
        compute_normalize(&feature_norm, in); //feature normalization
        //float conv_in[FEATURES_DIM];
        //float conv_in_in[FEATURE_CONV_IN_IN_OUT_SIZE];
        //float conv_in_out[FEATURES_DIM];
        //compute_dense(&feature_conv_in, conv_in, in);
        //compute_dense(&feature_conv_in_in, conv_in_in, in);
        //compute_dense(&feature_conv_in_out, conv_in_out, conv_in_in);
        if (mwdlp10net->frame_count == 0) //pad_first
            for (i=0;i<CONV_KERNEL_1;i++) //store first input with replicate padding kernel_size-1
                //RNN_COPY(&mem[i*FEATURES_DIM], conv_in, FEATURES_DIM);
                //RNN_COPY(&mem[i*FEATURES_DIM], conv_in_out, FEATURES_DIM);
                RNN_COPY(&mem[i*FEATURES_DIM], in, FEATURES_DIM);
        else {
            RNN_MOVE(mem, &mem[FEATURES_DIM], FEATURE_CONV_STATE_SIZE_1); //store previous input kernel_size-2
            RNN_COPY(&mem[FEATURE_CONV_STATE_SIZE_1], in, FEATURES_DIM); //add new input
        }
        mwdlp10net->frame_count++;
        *n_output = 0;
        return;
    }
    if (!flag_last_frame) {
        run_frame_network_mwdlp10(nnet, gru_a_condition, gru_b_condition, gru_c_condition, features, 0);
        for (i=0,m=0,*n_output=0;i<N_SAMPLE_BANDS;i++) {
            //coarse
            run_sample_network_mwdlp10_coarse_nodlpc(nnet, a_embed_coarse, a_embed_fine, pdf,
                    gru_a_condition, gru_b_condition, last_coarse_0_pt, last_fine_0_pt);
            for (j=0;j<N_MBANDS;j++)
                last_coarse_0_pt[j] = sample_from_pdf_mwdlp(&pdf[j*SQRT_QUANTIZE], SQRT_QUANTIZE);
            //fine
            run_sample_network_mwdlp10_fine_nodlpc(nnet, c_embed_coarse, pdf, gru_c_condition, last_coarse_0_pt);
            //printf("\n");
            for (j=0;j<N_MBANDS;j++) {
                last_fine_0_pt[j] = sample_from_pdf_mwdlp(&pdf[j*SQRT_QUANTIZE], SQRT_QUANTIZE);
                mwdlp10net->buffer_output[j] = mwdlp10net->mu_law_10_table[last_coarse_0_pt[j] * SQRT_QUANTIZE + last_fine_0_pt[j]]*N_MBANDS;
            //    printf("[%d] %d %d %d %f ", j+1, coarse[j], fine[j], coarse[j]*SQRT_QUANTIZE+fine[j], mwdlp10net->buffer_output[j]);
                //if (j==0) pcm_1[i] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*32768;
                //else if (j==1) pcm_2[i] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*32768;
                //else if (j==2) pcm_3[i] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*32768;
                //else if (j==3) pcm_4[i] = mwdlp10net->mu_law_10_table[coarse[j] * SQRT_QUANTIZE + fine[j]]*32768;
            }
            //printf("\n");
            //update state of last_coarse and last_fine integer output
            //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; K: DLPC_ORDER
            //RNN_COPY(last_coarse_0_pt, coarse, N_MBANDS);
            //RNN_COPY(last_fine_0_pt, fine, N_MBANDS);
            //update state of pqmf synthesis input
            RNN_MOVE(pqmf_state_0_pt, pqmf_state_mbsqr_pt, PQMF_ORDER_MBANDS);
            RNN_COPY(pqmf_state_ordmb_pt, mwdlp10net->buffer_output, N_MBANDS_SQR);
            //pqmf synthesis if its delay sufficient
            if (mwdlp10net->sample_count >= PQMF_DELAY) {
                if (mwdlp10net->first_flag > 0) {
                    //synthesis n=n_bands samples
                    for (j=0;j<N_MBANDS;j++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp10net->pqmf_state[j*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp10net->deemph_mem;
                        mwdlp10net->deemph_mem = tmp_out;
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
                    RNN_COPY(&mwdlp10net->first_pqmf_state[PQMF_DELAY_MBANDS],
                        &mwdlp10net->pqmf_state[PQMF_DELAY_MBANDS-FIRST_N_OUTPUT_MBANDS],
                            PQMF_DELAY_MBANDS+FIRST_N_OUTPUT_MBANDS);
                    for (j=0;j<FIRST_N_OUTPUT;j++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp10net->first_pqmf_state[j*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp10net->deemph_mem;
                        mwdlp10net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                    }
                    mwdlp10net->first_flag = 1;
                    *n_output += FIRST_N_OUTPUT;
                    //synthesis n=n_bands samples
                    for (k=0;k<N_MBANDS;k++,m++) {
                        tmp_out = 0;
                        //pqmf_synth
                        sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1,
                            &mwdlp10net->pqmf_state[k*N_MBANDS]);
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //de-emphasis
                        tmp_out += PREEMPH*mwdlp10net->deemph_mem;
                        mwdlp10net->deemph_mem = tmp_out;
                        //clamp
                        if (tmp_out < -1) tmp_out = -1;
                        else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                        //16bit pcm signed
                        output[m] = round(tmp_out * 32768);
                    }
                }
                *n_output += N_MBANDS;
            }
            mwdlp10net->sample_count += N_MBANDS;
        }    
    } else if (mwdlp10net->sample_count >= PQMF_DELAY) {
        //synthesis n=pqmf_order samples + (n_sample_bands x feature_pad_right) samples [replicate pad_right]
        //replicate_pad_right segmental_conv
        float *last_frame = &nnet->feature_conv_state[FEATURE_CONV_STATE_SIZE_1]; //for replicate pad_right
        for (l=0,m=0,*n_output=0;l<FEATURE_CONV_DELAY;l++) {
            run_frame_network_mwdlp10(nnet, gru_a_condition, gru_b_condition, gru_c_condition, last_frame, 1);
            for (i=0;i<N_SAMPLE_BANDS;i++) {
                //coarse
                run_sample_network_mwdlp10_coarse_nodlpc(nnet, a_embed_coarse, a_embed_fine, pdf,
                        gru_a_condition, gru_b_condition, last_coarse_0_pt, last_fine_0_pt);
                for (j=0;j<N_MBANDS;j++)
                    last_coarse_0_pt[j] = sample_from_pdf_mwdlp(&pdf[j*SQRT_QUANTIZE], SQRT_QUANTIZE);
                //fine
                run_sample_network_mwdlp10_fine_nodlpc(nnet, c_embed_coarse, pdf, gru_c_condition, last_coarse_0_pt);
                for (j=0;j<N_MBANDS;j++) {
                    last_fine_0_pt[j] = sample_from_pdf_mwdlp(&pdf[j*SQRT_QUANTIZE], SQRT_QUANTIZE);
                    //float,[-1,1),upsample-bands(x n_bands)
                    mwdlp10net->buffer_output[j] = mwdlp10net->mu_law_10_table[last_coarse_0_pt[j] * SQRT_QUANTIZE + last_fine_0_pt[j]]*N_MBANDS;
                }
                //update state of last_coarse and last_fine integer output
                //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; K: DLPC_ORDER
                //RNN_COPY(last_coarse_0_pt, coarse, N_MBANDS);
                //RNN_COPY(last_fine_0_pt, fine, N_MBANDS);
                //update state of pqmf synthesis input
                //t-(order//2),t,t+(order//2), n_update = n_bands^2, n_stored_state = order*n_bands
                RNN_MOVE(pqmf_state_0_pt, pqmf_state_mbsqr_pt, PQMF_ORDER_MBANDS);
                RNN_COPY(pqmf_state_ordmb_pt, mwdlp10net->buffer_output, N_MBANDS_SQR);
                //synthesis n=n_bands samples
                for (j=0;j<N_MBANDS;j++,m++) {
                    tmp_out = 0;
                    //pqmf_synth
                    sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1, &mwdlp10net->pqmf_state[j*N_MBANDS]);
                    //clamp
                    if (tmp_out < -1) tmp_out = -1;
                    else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
                    //de-emphasis
                    tmp_out += PREEMPH*mwdlp10net->deemph_mem;
                    mwdlp10net->deemph_mem = tmp_out;
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
        RNN_MOVE(mwdlp10net->last_pqmf_state, &mwdlp10net->pqmf_state[N_MBANDS_SQR], PQMF_ORDER_MBANDS);
        //from [o_1,...,o_{2N},0] to [o_1,..,o_{N+1},0,...,0]; N=PQMF_DELAY=PQMF_ORDER//2=(TAPS-1)//2
        for  (i=0;i<PQMF_DELAY;i++,m++) {
            tmp_out = 0;
            //pqmf_synth
            sgemv_accum(&tmp_out, pqmf_synth_filter, 1, TAPS_MBANDS, 1, &mwdlp10net->last_pqmf_state[i*N_MBANDS]);
            //clamp
            if (tmp_out < -1) tmp_out = -1;
            else if (tmp_out > 0.999969482421875) tmp_out = 0.999969482421875;
            //de-emphasis
            tmp_out += PREEMPH*mwdlp10net->deemph_mem;
            mwdlp10net->deemph_mem = tmp_out;
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
