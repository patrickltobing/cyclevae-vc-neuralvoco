/* Copyright (c) 2018 Mozilla
                 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Sept./Dec. 2020 - Mar./Jul. 2021,
   marked by PLT_<Sep20/Dec20/Jan21/Mar21/Jul21> */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "opus_types.h"
#include "arch.h"
#include "common.h"
#include "nnet.h"
#include "nnet_data.h"
#include "nnet_cv_data.h"
#include "mwdlp10net_cycvae_private.h"

#if defined(WINDOWS_SYS)
#define RAND_MAX_FLT_MIN_FLT_MIN (UINT_MAX + FLT_MIN + FLT_MIN)
#else
#define RAND_MAX_FLT_MIN_FLT_MIN (NRAND48_MAX + FLT_MIN + FLT_MIN)
#endif
//#define RAND_MAX_FLT_MIN_FLT_MIN (RAND_MAX + FLT_MIN + FLT_MIN)

#define SOFTMAX_HACK

#ifdef __AVX__
#include "vec_avx.h"
#elif __ARM_NEON__
#include "vec_neon.h"
#else
#warning Compiling without any vectorization. This code will be very slow
#include "vec.h"
#endif

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}


//PLT_Mar21
void sgemv_accum16_(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
 sgemv_accum16(out, weights, rows, cols, col_stride, x);
}

void sgemv_accum(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
 int i, j;
 for (i=0;i<rows;i++)
 {
    for (j=0;j<cols;j++) {
       out[i] += weights[j*col_stride + i]*x[j];
   }
 }
}

void compute_activation(float *output, float *input, int N, int activation)
{
   int i;
   if (activation == ACTIVATION_SIGMOID) {
      vec_sigmoid(output, input, N);
   } else if (activation == ACTIVATION_TANH) {
      vec_tanh(output, input, N);
   } else if (activation == ACTIVATION_TANHSHRINK) { //PLT_Sep20
      vec_tanhshrink(output, input, N);
   } else if (activation == ACTIVATION_EXP) { //PLT_Feb21
      vec_exp(output, input, N);
   } else if (activation == ACTIVATION_SIGMOID_EXP) { //PLT_Feb21
      vec_sigmoid_exp(output, input, N);
   } else if (activation == ACTIVATION_TANH_EXP) { //PLT_Feb21
      vec_tanh_exp(output, input, N);
   } else if (activation == ACTIVATION_RELU) {
      for (i=0;i<N;i++)
         output[i] = relu(input[i]);
   } else if (activation == ACTIVATION_SOFTMAX) {
#ifdef SOFTMAX_HACK
      for (i=0;i<N;i++)
         output[i] = input[i];
#else
      float sum = 0;
      softmax(output, input, N);
      for (i=0;i<N;i++) {
         sum += output[i];
      }
      sum = 1.f/(sum+1e-30);
      for (i=0;i<N;i++)
         output[i] = sum*output[i];
#endif
   } else {
      //celt_assert(activation == ACTIVATION_LINEAR);
      for (i=0;i<N;i++)
         output[i] = input[i];
   }
}

void compute_dense(const DenseLayer *layer, float *output, const float *input)
{
   int i, N;
   //int N, M;
   //int stride;
   //M = layer->nb_inputs;
   N = layer->nb_neurons;
   //stride = N;
   //celt_assert(input != output);
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   // sgemv_accum16(output, layer->input_weights, N, M, stride, input);
   sgemv_accum16(output, layer->input_weights, N, layer->nb_inputs, N, input);
   compute_activation(output, output, N, layer->activation);
}

//PLT_Dec20
void compute_dense_linear(const DenseLayer *layer, float *output, const float *input)
{
   int i, N;
   //int N, M;
   //int stride;
   //M = layer->nb_inputs;
   N = layer->nb_neurons;
   //stride = N;
   //celt_assert(input != output);
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   // sgemv_accum16(output, layer->input_weights, N, M, stride, input);
   sgemv_accum16(output, layer->input_weights, N, layer->nb_inputs, N, input);
}

//PLT_Mar21
void compute_mdense_mwdlp10(const MDenseLayerMWDLP10 *layer, const DenseLayer *fc_layer,
    const float *prev_logits, float *output, const float *input, const int *last_output)
{
    //int i, j, k, l, m, n, last_idx;
    int i, j, k, n, last_idx;
    float vec_out[MDENSE_OUT_DUALFC_2_MBANDS], dualfc_out[MDENSE_OUT_DUALFC_MBANDS], fc_out[MDENSE_OUT_FC_MBANDS];
    float signs[LPC_ORDER_MBANDS], mags[LPC_ORDER_MBANDS];

    //compute dualfc output vectors
    for (i=0;i<MDENSE_OUT_DUALFC_2_MBANDS;i++)
       vec_out[i] = layer->bias[i];
    sgemv_accum16(vec_out, layer->input_weights, MDENSE_OUT_DUALFC_2_MBANDS, RNN_SUB_NEURONS, MDENSE_OUT_DUALFC_2_MBANDS, input);
    compute_activation(vec_out, vec_out, MDENSE_OUT_DUALFC_2_MBANDS, layer->activation);
    //combine dualfc channels
    // [[[K,K,32]_1,...,[K,K,32]_NB]_1,[[K,K,32]_1,...,[K,K,32]_NB]_2]
    sgev_dualfc8(dualfc_out, layer->factors, MDENSE_OUT_DUALFC_2_MBANDS, vec_out);

    //final fc layer
    // [[K,K,32]_1,...,[K,K,32]_NB]
    for (i=0;i<MDENSE_OUT_FC_MBANDS;i++)
       fc_out[i] = fc_layer->bias[i];
    //sgemv_fclogits16(fc_out, fc_layer->input_weights, MDENSE_OUT_FC, MDENSE_OUT_DUALFC, N_MBANDS, dualfc_out);
    sgemv_fcout32(fc_out, fc_layer->input_weights, MDENSE_OUT_FC, MDENSE_OUT_DUALFC, N_MBANDS, dualfc_out);
    //for (n=0,k=0;n<N_MBANDS;n++) {
    //    compute_activation(&fc_out[k], &fc_out[k], DLPC_ORDER, layer->activation_signs); //signs
    //    k += DLPC_ORDER;
    //    compute_activation(&fc_out[k], &fc_out[k], DLPC_ORDER, layer->activation_mags); //mags
    //    k += DLPC_ORDER;
    //    compute_activation(&fc_out[k], &fc_out[k], SQRT_QUANTIZE, layer->activation_logits); //mags
    //    RNN_COPY(&output[n*SQRT_QUANTIZE], &fc_out[k], SQRT_QUANTIZE);
    //    k += SQRT_QUANTIZE;
    //}
    //compute_activation(fc_out, fc_out, LPC_ORDER_MBANDS, layer->activation_signs); //signs
    //compute_activation(&fc_out[LPC_ORDER_MBANDS], &fc_out[LPC_ORDER_MBANDS], LPC_ORDER_MBANDS, layer->activation_mags); //mags
    //compute_activation(&fc_out[LPC_ORDER_MBANDS_2], &fc_out[LPC_ORDER_MBANDS_2], SQRT_QUANTIZE_MBANDS, layer->activation_logits); //logits
    //RNN_COPY(output, &fc_out[LPC_ORDER_MBANDS_2], SQRT_QUANTIZE_MBANDS);
    for (n=0,i=0,j=0,k=0;n<N_MBANDS;n++) {
        RNN_COPY(&signs[i], &fc_out[k], DLPC_ORDER);
        k += DLPC_ORDER;
        RNN_COPY(&mags[i], &fc_out[k], DLPC_ORDER);
        k += DLPC_ORDER;
        RNN_COPY(&output[j], &fc_out[k], SQRT_QUANTIZE);
        i += DLPC_ORDER;
        j += SQRT_QUANTIZE;
        k += SQRT_QUANTIZE;
    }
    compute_activation(signs, signs, LPC_ORDER_MBANDS, layer->activation_signs); //signs
    compute_activation(mags, mags, LPC_ORDER_MBANDS, layer->activation_mags); //mags
    compute_activation(output, output, SQRT_QUANTIZE_MBANDS, layer->activation_logits); //logits

    //refine logits with data-driven linear prediction procedure
    //for (n=0,k=0;n<N_MBANDS;n++) {
    for (n=0,k=0;n<N_MBANDS;n++) {
        //compute_activation(&fc_out[k], &fc_out[k], DLPC_ORDER, layer->activation_signs); //signs
        //k += DLPC_ORDER;
        //compute_activation(&fc_out[k], &fc_out[k], DLPC_ORDER, layer->activation_mags); //mags
        //k += DLPC_ORDER;
        //compute_activation(&fc_out[k], &fc_out[k], SQRT_QUANTIZE, layer->activation_logits); //mags
        //RNN_COPY(&output[n*SQRT_QUANTIZE], &fc_out[k], SQRT_QUANTIZE);
        //k += SQRT_QUANTIZE;
        //for (i=0,j=n*SQRT_QUANTIZE,m=n*MDENSE_OUT_FC,l=m+DLPC_ORDER;i<DLPC_ORDER;i++) {
        for (i=0,j=n*SQRT_QUANTIZE;i<DLPC_ORDER;i++,k++) {
            last_idx = last_output[i*N_MBANDS+n];
            output[j+last_idx] += signs[k]*mags[k]*prev_logits[last_idx];
        }
    }
}

//PLT_Mar21
void compute_mdense_mwdlp10_nodlpc(const MDenseLayerMWDLP10 *layer, const DenseLayer *fc_layer, float *output,
    const float *input)
{
    int i, j, n;
    float vec_out[MID_OUT_MBANDS_2];
    float dualfc_out[MID_OUT_MBANDS];

    //mid-logits by dualfc
    for (i=0;i<MID_OUT_MBANDS_2;i++)
       vec_out[i] = layer->bias[i];
    sgemv_accum16(vec_out, layer->input_weights, MID_OUT_MBANDS_2, RNN_SUB_NEURONS, MID_OUT_MBANDS_2, input);
    compute_activation(vec_out, vec_out, MID_OUT_MBANDS_2, layer->activation);
    //combine dualfc channels
    sgev_dualfc8(dualfc_out, layer->factors, MDENSE_OUT_DUALFC_2_MBANDS, vec_out);

    //logits by last fc-layer
    for (n=0;n<N_MBANDS;n++)
        for (i=0,j=n*SQRT_QUANTIZE;i<SQRT_QUANTIZE;i++)
            output[j+i] = fc_layer->bias[i];
    sgemv_fclogits16(output, fc_layer->input_weights, SQRT_QUANTIZE, MID_OUT, N_MBANDS, dualfc_out);
    compute_activation(output, output, SQRT_QUANTIZE_MBANDS, layer->activation_logits);
}

//PLT_Sep20
void compute_gru3(const GRULayer *gru, float *state, const float *input)
{
   int i;
   //int N;
   //int stride;
   float zrh[RNN_SUB_NEURONS_3]; //reduce memory, set with this GRU's units
   float recur[RNN_SUB_NEURONS_3];
   float *z;
   float *r;
   float *h;
   //N = gru->nb_neurons;
   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_SUB_NEURONS];
   h = &zrh[RNN_SUB_NEURONS_2];
   //celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   //celt_assert(input != state);
   //celt_assert(gru->reset_after);
   //stride = 3*N;
   RNN_COPY(zrh, input, RNN_SUB_NEURONS_3);
   for (i=0;i<RNN_SUB_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum16(recur, gru->recurrent_weights, RNN_SUB_NEURONS_3, RNN_SUB_NEURONS, RNN_SUB_NEURONS_3, state);
   for (i=0;i<RNN_SUB_NEURONS_2;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, RNN_SUB_NEURONS_2, ACTIVATION_SIGMOID);
   //compute_activation(zrh, zrh, RNN_SUB_NEURONS_2, ACTIVATION_SIGMOID_EXP);
   for (i=0;i<RNN_SUB_NEURONS;i++)
      h[i] += recur[RNN_SUB_NEURONS_2+i]*z[i];
      //h[i] += recur[RNN_SUB_NEURONS_2+i]*r[i];
   compute_activation(h, h, RNN_SUB_NEURONS, gru->activation);
   for (i=0;i<RNN_SUB_NEURONS;i++)
      state[i] = r[i]*state[i] + (1-r[i])*h[i];
}

//PLT_Sep20
void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input)
{
   int i, j, k;
   //int N;
   float zrh[RNN_MAIN_NEURONS_3];
   float recur[RNN_MAIN_NEURONS_3];
   float *z;
   float *r;
   float *h;
   //N = gru->nb_neurons;
   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_MAIN_NEURONS];
   h = &zrh[RNN_MAIN_NEURONS_2];
   //celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   //celt_assert(input != state);
   //celt_assert(gru->reset_after);
   RNN_COPY(zrh, input, RNN_MAIN_NEURONS_3);
   for (i=0;i<RNN_MAIN_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   for (k=0;k<3;k++)
      for (i=0,j=k*RNN_MAIN_NEURONS;i<RNN_MAIN_NEURONS;i++)
         recur[j + i] += gru->diag_weights[j + i]*state[i];
   sparse_sgemv_accum16(recur, gru->recurrent_weights, RNN_MAIN_NEURONS_3, gru->idx, state);
   //for (i=0;i<RNN_MAIN_NEURONS;i++)
   //     printf("is[%d] %f\n", i, state[i]);
   //for (i=0;i<RNN_MAIN_NEURONS_3;i++)
   //     printf("sg[%d] %f\n", i, recur[i]);
   for (i=0;i<RNN_MAIN_NEURONS_2;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, RNN_MAIN_NEURONS_2, ACTIVATION_SIGMOID);
   //compute_activation(zrh, zrh, RNN_MAIN_NEURONS_2, ACTIVATION_SIGMOID_EXP);
   for (i=0;i<RNN_MAIN_NEURONS;i++)
      h[i] += recur[RNN_MAIN_NEURONS_2+i]*z[i];
      //h[i] += recur[RNN_MAIN_NEURONS_2+i]*r[i];
   compute_activation(h, h, RNN_MAIN_NEURONS, gru->activation);
   for (i=0;i<RNN_MAIN_NEURONS;i++)
      state[i] = r[i]*state[i] + (1-r[i])*h[i];
}

//PLT_Jun21
void compute_conv1d_linear_enc_melsp(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   //int stride;
   float tmp[FEATURE_CONV_ENC_MELSP_INPUT_SIZE]; //set to input_size*kernel_size
   //celt_assert(input != output);
   RNN_COPY(tmp, mem, FEATURE_CONV_ENC_MELSP_STATE_SIZE); //get state_size of last frame (in*(kernel_size-1))
   RNN_COPY(&tmp[FEATURE_CONV_ENC_MELSP_STATE_SIZE], input, FEATURE_DIM_MELSP); //append current input frame
   //for (int j=0;j<layer->kernel_size;j++) {
   //    for (i=0;i<layer->nb_inputs;i++) {
   //        printf("tmp [%d][%d] %f\n", j, i, tmp[j*layer->nb_inputs+i]);
   //    }
   //}
   // compute conv
   for (int i=0;i<FEATURE_CONV_ENC_MELSP_OUT_SIZE;i++)
      output[i] = layer->bias[i];
   sgemv_accum16(output, layer->input_weights, FEATURE_CONV_ENC_MELSP_OUT_SIZE, FEATURE_CONV_ENC_MELSP_INPUT_SIZE,
                    FEATURE_CONV_ENC_MELSP_OUT_SIZE, tmp);
   //no activation (linear)
   RNN_COPY(mem, &tmp[FEATURE_DIM_MELSP], FEATURE_CONV_ENC_MELSP_STATE_SIZE); //set state size for next frame
}

//PLT_Jun21
void compute_conv1d_linear_enc_excit(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   //int stride;
   float tmp[FEATURE_CONV_ENC_EXCIT_INPUT_SIZE]; //set to input_size*kernel_size
   //celt_assert(input != output);
   RNN_COPY(tmp, mem, FEATURE_CONV_ENC_EXCIT_STATE_SIZE); //get state_size of last frame (in*(kernel_size-1))
   RNN_COPY(&tmp[FEATURE_CONV_ENC_EXCIT_STATE_SIZE], input, FEATURE_DIM_MELSP); //append current input frame
   //for (int j=0;j<layer->kernel_size;j++) {
   //    for (i=0;i<layer->nb_inputs;i++) {
   //        printf("tmp [%d][%d] %f\n", j, i, tmp[j*layer->nb_inputs+i]);
   //    }
   //}
   // compute conv
   for (int i=0;i<FEATURE_CONV_ENC_EXCIT_OUT_SIZE;i++)
      output[i] = layer->bias[i];
   sgemv_accum16(output, layer->input_weights, FEATURE_CONV_ENC_EXCIT_OUT_SIZE, FEATURE_CONV_ENC_EXCIT_INPUT_SIZE,
                    FEATURE_CONV_ENC_EXCIT_OUT_SIZE, tmp);
   //no activation (linear)
   RNN_COPY(mem, &tmp[FEATURE_DIM_MELSP], FEATURE_CONV_ENC_EXCIT_STATE_SIZE); //set state size for next frame
}

//PLT_Jun21
void compute_conv1d_linear_spk(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   //int stride;
   float tmp[FEATURE_CONV_SPK_INPUT_SIZE]; //set to input_size*kernel_size
   //celt_assert(input != output);
   RNN_COPY(tmp, mem, FEATURE_CONV_SPK_STATE_SIZE); //get state_size of last frame (in*(kernel_size-1))
   RNN_COPY(&tmp[FEATURE_CONV_SPK_STATE_SIZE], input, FEATURE_RED_DIM); //append current input frame
   //for (int j=0;j<layer->kernel_size;j++) {
   //    for (i=0;i<layer->nb_inputs;i++) {
   //        printf("tmp [%d][%d] %f\n", j, i, tmp[j*layer->nb_inputs+i]);
   //    }
   //}
   // compute conv
   for (int i=0;i<FEATURE_CONV_SPK_OUT_SIZE;i++)
      output[i] = layer->bias[i];
   sgemv_accum16(output, layer->input_weights, FEATURE_CONV_SPK_OUT_SIZE, FEATURE_CONV_SPK_INPUT_SIZE,
                    FEATURE_CONV_SPK_OUT_SIZE, tmp);
   //no activation (linear)
   RNN_COPY(mem, &tmp[FEATURE_RED_DIM], FEATURE_CONV_SPK_STATE_SIZE); //set state size for next frame
}

//PLT_Jun21
void compute_conv1d_linear_dec_melsp(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   //int stride;
   float tmp[FEATURE_CONV_DEC_MELSP_INPUT_SIZE]; //set to input_size*kernel_size
   //celt_assert(input != output);
   RNN_COPY(tmp, mem, FEATURE_CONV_DEC_MELSP_STATE_SIZE); //get state_size of last frame (in*(kernel_size-1))
   RNN_COPY(&tmp[FEATURE_CONV_DEC_MELSP_STATE_SIZE], input, FEATURE_RED_DIM); //append current input frame
   //for (int j=0;j<layer->kernel_size;j++) {
   //    for (i=0;i<layer->nb_inputs;i++) {
   //        printf("tmp [%d][%d] %f\n", j, i, tmp[j*layer->nb_inputs+i]);
   //    }
   //}
   // compute conv
   for (int i=0;i<FEATURE_CONV_DEC_MELSP_OUT_SIZE;i++)
      output[i] = layer->bias[i];
   sgemv_accum16(output, layer->input_weights, FEATURE_CONV_DEC_MELSP_OUT_SIZE, FEATURE_CONV_DEC_MELSP_INPUT_SIZE,
                    FEATURE_CONV_DEC_MELSP_OUT_SIZE, tmp);
   //no activation (linear)
   RNN_COPY(mem, &tmp[FEATURE_RED_DIM], FEATURE_CONV_DEC_MELSP_STATE_SIZE); //set state size for next frame
}

//PLT_Jun21
void compute_conv1d_linear_frame_in(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   //int stride;
   float tmp[FEATURE_CONV_INPUT_SIZE]; //set to input_size*kernel_size
   //celt_assert(input != output);
   RNN_COPY(tmp, mem, FEATURE_CONV_STATE_SIZE); //get state_size of last frame (in*(kernel_size-1))
   RNN_COPY(&tmp[FEATURE_CONV_STATE_SIZE], input, FEATURE_RED_DIM); //append current input frame
   //for (int j=0;j<layer->kernel_size;j++) {
   //    for (i=0;i<layer->nb_inputs;i++) {
   //        printf("tmp [%d][%d] %f\n", j, i, tmp[j*layer->nb_inputs+i]);
   //    }
   //}
   // compute conv
   for (int i=0;i<FEATURE_CONV_OUT_SIZE;i++)
      output[i] = layer->bias[i];
   sgemv_accum16(output, layer->input_weights, FEATURE_CONV_OUT_SIZE, FEATURE_CONV_INPUT_SIZE,
                    FEATURE_CONV_OUT_SIZE, tmp);
   //no activation (linear)
   RNN_COPY(mem, &tmp[FEATURE_RED_DIM], FEATURE_CONV_STATE_SIZE); //set state size for next frame
}

//PLT_Jul21
int sample_from_pdf_mwdlp(const float *pdf, int N, RNGState *rng_state)
{
    int i;
    float r;
#if defined(WINDOWS_SYS)
    UINT buffer = 0;
    float tmp[SQRT_QUANTIZE], cdf[SQRT_QUANTIZE], sum, norm;
    for (i=0;i<N;i++)
        tmp[i] = pdf[i];
    softmax(tmp, tmp, N);
    for (i=0,sum=0;i<N;i++)
        sum += tmp[i];
    norm = 1.f/sum;
    /* Convert tmp to a CDF (sum of all previous probs., init. with 0) */
    cdf[0] = 0;
    for (i=1;i<N;i++)
        cdf[i] = cdf[i-1] + norm*tmp[i-1];
    /* Do the sampling (from the cdf). */
    BCryptGenRandom(rng_state->rng_prov, (PUCHAR)(&buffer), sizeof(buffer), 0);
    r = (float)buffer / UINT_MAX;
    for (i=N-1;i>0;i--)
        if (r >= cdf[i]) return i; //largest cdf that is less/equal than r
    return 0;
#else
    long int rand_num = 0;
    float tmp[SQRT_QUANTIZE], cdf[SQRT_QUANTIZE], sum, norm;
    for (i=0;i<N;i++)
        tmp[i] = pdf[i];
    softmax(tmp, tmp, N);
    for (i=0,sum=0;i<N;i++)
        sum += tmp[i];
    norm = 1.f/sum;
    /* Convert tmp to a CDF (sum of all previous probs., init. with 0) */
    cdf[0] = 0;
    for (i=1;i<N;i++)
        cdf[i] = cdf[i-1] + norm*tmp[i-1];
    /* Do the sampling (from the cdf). */
    nrand48_r(rng_state->xsubi, rng_state->drand_buffer, &rand_num); // res ~ [0,2^31-1]
    r = (float) rand_num / NRAND48_MAX; //r ~ [0,1]
    //r = (float) rand() / RAND_MAX; //r ~ [0,1]
    for (i=N-1;i>0;i--)
        if (r >= cdf[i]) return i; //largest cdf that is less/equal than r
    return 0;
#endif
}

//PLT_Dec20
void compute_normalize(const NormStats *norm_stats, float *input_output)
{
  for (int i=0;i<norm_stats->n_dim;i++)
    input_output[i] = (input_output[i] - norm_stats->mean[i]) / norm_stats->std[i];
}

//PLT_Dec20
void compute_denormalize(const NormStats *norm_stats, float *input_output)
{
  for (int i=0;i<norm_stats->n_dim;i++)
    input_output[i] = input_output[i] * norm_stats->std[i] + norm_stats->mean[i];
}

//PLT_Jan21
void compute_sparse_gru_enc_melsp(const SparseFrameGRULayer *gru, float *state, const float *input)
{
   int i, j, k;
   float zrh[RNN_ENC_MELSP_NEURONS_3];
   float recur[RNN_ENC_MELSP_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_ENC_MELSP_NEURONS];
   h = &zrh[RNN_ENC_MELSP_NEURONS_2];

   for (i=0;i<RNN_ENC_MELSP_NEURONS_3;i++) {
      recur[i] = gru->recurrent_bias[i];
      zrh[i] = gru->input_bias[i];
   }
   for (k=0;k<3;k++)
      for (i=0,j=k*RNN_ENC_MELSP_NEURONS;i<RNN_ENC_MELSP_NEURONS;i++)
         recur[j + i] += gru->diag_weights[j + i]*state[i];
   sparse_sgemv_accum16(recur, gru->recurrent_weights, RNN_ENC_MELSP_NEURONS_3, gru->idx, state);
   sgemv_accum16(zrh, gru->input_weights, RNN_ENC_MELSP_NEURONS_3, FEATURE_CONV_ENC_MELSP_OUT_SIZE, RNN_ENC_MELSP_NEURONS_3, input);

   for (i=0;i<RNN_ENC_MELSP_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   //compute_activation(zrh, zrh, RNN_ENC_MELSP_NEURONS_2, ACTIVATION_SIGMOID);
   compute_activation(zrh, zrh, RNN_ENC_MELSP_NEURONS_2, ACTIVATION_SIGMOID_EXP);

   for (i=0;i<RNN_ENC_MELSP_NEURONS;i++)
      h[i] += recur[RNN_ENC_MELSP_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   //compute_activation(h, h, RNN_ENC_MELSP_NEURONS, ACTIVATION_TANH);
   compute_activation(h, h, RNN_ENC_MELSP_NEURONS, gru->activation);

   for (i=0;i<RNN_ENC_MELSP_NEURONS;i++)
      state[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
}

//PLT_Jan21
void compute_sparse_gru_enc_excit(const SparseFrameGRULayer *gru, float *state, const float *input)
{
   int i, j, k;
   float zrh[RNN_ENC_EXCIT_NEURONS_3];
   float recur[RNN_ENC_EXCIT_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_ENC_EXCIT_NEURONS];
   h = &zrh[RNN_ENC_EXCIT_NEURONS_2];

   for (i=0;i<RNN_ENC_EXCIT_NEURONS_3;i++) {
      recur[i] = gru->recurrent_bias[i];
      zrh[i] = gru->input_bias[i];
   }
   for (k=0;k<3;k++)
      for (i=0,j=k*RNN_ENC_EXCIT_NEURONS;i<RNN_ENC_EXCIT_NEURONS;i++)
         recur[j + i] += gru->diag_weights[j + i]*state[i];
   sparse_sgemv_accum16(recur, gru->recurrent_weights, RNN_ENC_EXCIT_NEURONS_3, gru->idx, state);
   sgemv_accum16(zrh, gru->input_weights, RNN_ENC_EXCIT_NEURONS_3, FEATURE_CONV_ENC_EXCIT_OUT_SIZE, RNN_ENC_EXCIT_NEURONS_3, input);

   for (i=0;i<RNN_ENC_EXCIT_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   //compute_activation(zrh, zrh, RNN_ENC_EXCIT_NEURONS_2, ACTIVATION_SIGMOID);
   compute_activation(zrh, zrh, RNN_ENC_EXCIT_NEURONS_2, ACTIVATION_SIGMOID_EXP);

   for (i=0;i<RNN_ENC_EXCIT_NEURONS;i++)
      h[i] += recur[RNN_ENC_EXCIT_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   //compute_activation(h, h, RNN_ENC_EXCIT_NEURONS, ACTIVATION_TANH);
   compute_activation(h, h, RNN_ENC_EXCIT_NEURONS, gru->activation);

   for (i=0;i<RNN_ENC_EXCIT_NEURONS;i++)
      state[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
}

//PLT_Dec20
void compute_gru_spk(const FrameGRULayer *gru, float *state, const float *input)
{
   int i;
   float zrh[RNN_SPK_NEURONS_3];
   float recur[RNN_SPK_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_SPK_NEURONS];
   h = &zrh[RNN_SPK_NEURONS_2];

   for (i=0;i<RNN_SPK_NEURONS_3;i++) {
      recur[i] = gru->recurrent_bias[i];
      zrh[i] = gru->input_bias[i];
   }
   sgemv_accum16(recur, gru->recurrent_weights, RNN_SPK_NEURONS_3, RNN_SPK_NEURONS, RNN_SPK_NEURONS_3, state);
   sgemv_accum16(zrh, gru->input_weights, RNN_SPK_NEURONS_3, FEATURE_CONV_SPK_OUT_SIZE, RNN_SPK_NEURONS_3, input);

   for (i=0;i<RNN_SPK_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   //compute_activation(zrh, zrh, RNN_SPK_NEURONS_2, ACTIVATION_SIGMOID);
   compute_activation(zrh, zrh, RNN_SPK_NEURONS_2, ACTIVATION_SIGMOID_EXP);

   for (i=0;i<RNN_SPK_NEURONS;i++)
      h[i] += recur[RNN_SPK_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   //compute_activation(h, h, RNN_SPK_NEURONS, ACTIVATION_TANH);
   compute_activation(h, h, RNN_SPK_NEURONS, gru->activation);

   for (i=0;i<RNN_SPK_NEURONS;i++)
      state[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
}

//PLT_Jul21
void compute_sparse_gru_dec_melsp(const SparseFrameGRULayer *gru, float *state, const float *input)
{
   int i, j, k;
   float zrh[RNN_DEC_MELSP_NEURONS_3];
   float recur[RNN_DEC_MELSP_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_DEC_MELSP_NEURONS];
   h = &zrh[RNN_DEC_MELSP_NEURONS_2];

   for (i=0;i<RNN_DEC_MELSP_NEURONS_3;i++) {
      recur[i] = gru->recurrent_bias[i];
      zrh[i] = gru->input_bias[i];
   }
   for (k=0;k<3;k++)
      for (i=0,j=k*RNN_DEC_MELSP_NEURONS;i<RNN_DEC_MELSP_NEURONS;i++)
         recur[j + i] += gru->diag_weights[j + i]*state[i];
   sparse_sgemv_accum16(recur, gru->recurrent_weights, RNN_DEC_MELSP_NEURONS_3, gru->idx, state);
   sgemv_accum16(zrh, gru->input_weights, RNN_DEC_MELSP_NEURONS_3, FEATURE_CONV_DEC_MELSP_OUT_SIZE, RNN_DEC_MELSP_NEURONS_3, input);

   for (i=0;i<RNN_DEC_MELSP_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   //compute_activation(zrh, zrh, RNN_DEC_MELSP_NEURONS_2, ACTIVATION_SIGMOID);
   compute_activation(zrh, zrh, RNN_DEC_MELSP_NEURONS_2, ACTIVATION_SIGMOID_EXP);

   for (i=0;i<RNN_DEC_MELSP_NEURONS;i++)
      h[i] += recur[RNN_DEC_MELSP_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   //compute_activation(h, h, RNN_DEC_MELSP_NEURONS, ACTIVATION_TANH);
   compute_activation(h, h, RNN_DEC_MELSP_NEURONS, gru->activation);

   for (i=0;i<RNN_DEC_MELSP_NEURONS;i++)
      state[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
}

//PLT_Jul21
void compute_sampling_gauss(float *mu, const float *std, int dim, RNGState *rng_state)
{
    float u1, u2 = 0, mag = 0;
#if defined(WINDOWS_SYS)
    UINT buffer = 0;
    for (int i=0;i<dim;i++) {
        if (i % 2 == 0) {
            BCryptGenRandom(rng_state->rng_prov, (PUCHAR)(&buffer), sizeof(buffer), 0);
            u1 = ((float) buffer + FLT_MIN) / RAND_MAX_FLT_MIN_FLT_MIN; //u1 ~ (0,1)
            BCryptGenRandom(rng_state->rng_prov, (PUCHAR)(&buffer), sizeof(buffer), 0);
            u2 = ((float) buffer + FLT_MIN) / RAND_MAX_FLT_MIN_FLT_MIN; //u1 ~ (0,1)
            mag = sqrt(-2*log(u1));
            u2 *= 6.283185307179586476925286766559;
            ////temperature sampling: 0.675
            mu[i] += 0.675*std[i]*mag*cos(u2);
        } else mu[i] += 0.675*std[i]*mag*sin(u2);
    }
#else
    long int rand_num = 0;
    for (int i=0;i<dim;i++) {
        if (i % 2 == 0) {
            nrand48_r(rng_state->xsubi, rng_state->drand_buffer, &rand_num); // res ~ [0,2^31-1]
            u1 = ((float) rand_num + FLT_MIN) / RAND_MAX_FLT_MIN_FLT_MIN; //u1 ~ (0,1)
            //u1 = ((float) rand() + FLT_MIN) / RAND_MAX_FLT_MIN_FLT_MIN; //u1 ~ (0,1)
            nrand48_r(rng_state->xsubi, rng_state->drand_buffer, &rand_num); // res ~ [0,2^31-1]
            u2 = ((float) rand_num + FLT_MIN) / RAND_MAX_FLT_MIN_FLT_MIN; //u1 ~ (0,1)
            //u2 = ((float) rand() + FLT_MIN) / RAND_MAX_FLT_MIN_FLT_MIN; //u2 ~ (0,1)
            mag = sqrt(-2*log(u1));
            u2 *= 6.283185307179586476925286766559;
            ////temperature sampling: 0.675
            mu[i] += 0.675*std[i]*mag*cos(u2);
        } else mu[i] += 0.675*std[i]*mag*sin(u2);
    }
    return;
#endif
}

// PLT_Dec20
void compute_spkidtr(const DenseLayer * in_emb_layer, const DenseLayer * in_layer, const DenseLayer * out_layer,
    float* output, float* coeff, const float* input)
{
    int i, j, k;

    //transform to N-dim
    //N = in_emb_layer->nb_neurons;
    float tmp_in[FC_IN_SPK_CODE_OUT_SIZE];
    //printf("\n");
    for (i = 0; i < FC_IN_SPK_CODE_OUT_SIZE; i++) {
        tmp_in[i] = in_emb_layer->bias[i];
        //   printf("[%d] %f ", i, tmp_in[i]);
    }
    //printf("\n");
    sgemv_accum16(tmp_in, in_emb_layer->input_weights, FC_IN_SPK_CODE_OUT_SIZE, in_emb_layer->nb_inputs, FC_IN_SPK_CODE_OUT_SIZE, input);
    //for (i=0;i<N;i++) {
    //   printf("[%d] %f ", i, tmp_in[i]);
    //}
    //printf("\n");
    compute_activation(tmp_in, tmp_in, FC_IN_SPK_CODE_OUT_SIZE, in_emb_layer->activation);
    //for (i=0;i<N;i++) {
    //   printf("[%d] %f ", i, tmp_in[i]);
    //}
    //printf("\n");

    //transform to 2-dim
    //N = in_layer->nb_neurons;
    float tmp[FC_IN_SPK_CODE_TRANSFORM_OUT_SIZE];
    for (i = 0; i < FC_IN_SPK_CODE_TRANSFORM_OUT_SIZE; i++) {
        tmp[i] = in_layer->bias[i];
        //   printf("[%d] %f ", i, tmp[i]);
    }
    //printf("\n");
    sgemv_accum16(tmp, in_layer->input_weights, FC_IN_SPK_CODE_TRANSFORM_OUT_SIZE, in_layer->nb_inputs, FC_IN_SPK_CODE_TRANSFORM_OUT_SIZE, tmp_in);
    //printf("2-dim %f %f\n", tmp[0], tmp[1]);
    compute_activation(tmp, tmp, FC_IN_SPK_CODE_TRANSFORM_OUT_SIZE, in_layer->activation);
    //printf("2-dim tanh %f %f\n", tmp[0], tmp[1]);
    printf("2-dim spk-coord: %f %f\n", tmp[0], tmp[1]);

    //transform to n-coeff
    //N = out_layer->nb_neurons;
    for (i = 0; i < FC_OUT_SPK_CODE_TRANSFORM_OUT_SIZE; i++) {
        coeff[i] = out_layer->bias[i];
        //   printf("[%d] %f ", i, output[i]);
    }
    //printf("\n");
    sgemv_accum16(coeff, out_layer->input_weights, FC_OUT_SPK_CODE_TRANSFORM_OUT_SIZE, out_layer->nb_inputs, FC_OUT_SPK_CODE_TRANSFORM_OUT_SIZE, tmp);
    compute_activation(coeff, coeff, FC_OUT_SPK_CODE_TRANSFORM_OUT_SIZE, out_layer->activation);

    printf("%d-dim spk-embed-coeff:", FEATURE_N_WEIGHT_EMBED_SPK);
    //multiply with coeff
    for (i = 0, k = 0; i < FEATURE_N_WEIGHT_EMBED_SPK; i++) {
        printf(" %f", coeff[i]);
        for (j = 0; j < FEATURE_DIM_EMBED_SPK; j++, k++)
            output[k] *= coeff[i];
    }
    printf("\n");
}

//PLT_Dec20
void compute_spkidtr_coord(const DenseLayer* layer, float* output, float* coeff, const float* input)
{
    int i, N, j, k;
    //transform to n-coeff [from input 2-dim]
    N = layer->nb_neurons;
    for (i = 0; i < N; i++)
        coeff[i] = layer->bias[i];
    sgemv_accum16(coeff, layer->input_weights, N, layer->nb_inputs, N, input);
    compute_activation(coeff, coeff, N, layer->activation);

    printf("%d-dim spk-embed-coeff:", FEATURE_N_WEIGHT_EMBED_SPK);
    //multiply with coeff
    for (i = 0, k = 0; i < FEATURE_N_WEIGHT_EMBED_SPK; i++) {
        printf(" %f", coeff[i]);
        for (j = 0; j < FEATURE_DIM_EMBED_SPK; j++, k++)
            output[k] *= coeff[i];
    }
    printf("\n");
}
