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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Sept.-Dec. 2020,
   marked by PLT_<Sep/Dec>20 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

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

#define HALF_RAND_MAX (RAND_MAX / 2)
#define HALF_RAND_MAX_FLT_MIN (HALF_RAND_MAX + FLT_MIN)

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


void sgemv_accum(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   if (rows % 16 == 0)
   {
      sgemv_accum16(out, weights, rows, cols, col_stride, x);
   } else {
      for (i=0;i<rows;i++)
      {
         for (j=0;j<cols;j++) {
         //   printf("sg: %d %d %f %f %f\n", i, j, weights[j*col_stride+i], x[j], out[i]);
            out[i] += weights[j*col_stride + i]*x[j];
        //    printf("%f\n", out[i]);
        }
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
   // printf("tanhshrink\n");
   } else if (activation == ACTIVATION_EXP) { //PLT_Sep20
      softmax(output, input, N); //softmax functions in vec*.h only compute exp() 
    //  printf("activation exp");
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
   // sgemv_accum(output, layer->input_weights, N, M, stride, input);
   sgemv_accum(output, layer->input_weights, N, layer->nb_inputs, N, input);
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
   // sgemv_accum(output, layer->input_weights, N, M, stride, input);
   sgemv_accum(output, layer->input_weights, N, layer->nb_inputs, N, input);
}

//PLT_Dec20
void compute_mdense_mwdlp10(const MDenseLayerMWDLP10 *layer, const DenseLayer *fc_layer, float *output,
    const float *input, const int *last_output)
{
    int i, c, n;
    int n_lpcbands, n_c_lpcbands2;
    int n_logitsbands, n_c_logitsbands2;
    float tmp[MDENSE_OUT];
    float signs[LPC_ORDER_MBANDS_2], mags[LPC_ORDER_MBANDS_2], mids[MID_OUT_MBANDS_2];
    float lpc_signs[LPC_ORDER_MBANDS], lpc_mags[LPC_ORDER_MBANDS], res_mids[MID_OUT_MBANDS];
    float *logits;
    //celt_assert(input != output);
    for (i=0;i<MDENSE_OUT;i++)
       tmp[i] = layer->bias[i];
    sgemv_accum(tmp, layer->input_weights, MDENSE_OUT, RNN_SUB_NEURONS, MDENSE_OUT, input);
    //for (i=0;i<MDENSE_OUT;i++) {
    //    printf("mdense_out [%d] %f\n", i, tmp[i]);
    //}
    //exit(0);
    //output_dim:2*N_MBANDS*(DLPC_ORDER*2+MID_OUT),e.g.,DLPC_ORDER=6,N_MBANDS=5,MID_OUT=32->440
    //signs, 1st 6*5*2
    RNN_COPY(signs, tmp, LPC_ORDER_MBANDS_2);
    //-1,1
    compute_activation(signs, signs, LPC_ORDER_MBANDS_2, layer->activation_signs);
    //for (i=0;i<LPC_ORDER_MBANDS_2;i++) {
    //    printf("signs [%d] %f\n", i, signs[i]);
    //}
    //exit(0);
    //mags, 2nd 6*5*2
    RNN_COPY(mags, &tmp[LPC_ORDER_MBANDS_2], LPC_ORDER_MBANDS_2);
    //>= 0
    compute_activation(mags, mags, LPC_ORDER_MBANDS_2, layer->activation_mags);
    //for (i=0;i<LPC_ORDER_MBANDS_2;i++) {
    //    printf("mags [%d] %f\n", i, mags[i]);
    //}
    //exit(0);
    //mids, last 32*5*2
    RNN_COPY(mids, &tmp[LPC_ORDER_MBANDS_4], MID_OUT_MBANDS_2);
    compute_activation(mids, mids, MID_OUT_MBANDS_2, layer->activation_mids);
    //for (i=0;i<MID_OUT_MBANDS_2;i++) {
    //    printf("mids [%d] %f\n", i, mids[i]);
    //}
    //exit(0);
    for (i=0;i<LPC_ORDER_MBANDS;i++) {
        lpc_signs[i] = 0;
        lpc_mags[i] = 0;
    }
    for (i=0;i<SQRT_QUANTIZE_MBANDS;i++)
        res_mids[i] = 0;
    for (n=0;n<N_MBANDS;n++) { //n_bands x 2 x n_lpc or 32, loop order is n_bands --> 2 --> n_lpc/32
        //lpc: n_b x 2 x n_lpc
        for (c=0,n_lpcbands=n*DLPC_ORDER,n_c_lpcbands2=n_lpcbands*2;c<2;c++) {
            for (i=0;i<DLPC_ORDER;i++,n_c_lpcbands2++) {
                //previous code uses shared factors for signs/mags between bands [c*DLPC_ORDER + i]
                //changed into band-dependent factors for signs/mags [n*DLPC_ORDER*2 + c*DLPC_ORDER + i]
                lpc_signs[n_lpcbands+i] += signs[n_c_lpcbands2]*layer->factor_signs[n_c_lpcbands2]
                lpc_mags[n_lpcbands+i] += mags[n_c_lpcbands2]*layer->factor_mags[n_c_lpcbands2];
            }
        }
        //mids: n_b x 32
        for (c=0,n_logitsbands=n*SQRT_QUANTIZE,n_c_logitsbands2=n_logitsbands*2;c<2;c++) {
            //factor of mids also band-dependent, indexing similar as above signs/mags with 32-dim
            for (i=0;i<SQRT_QUANTIZE;i++,n_c_logitsbands2++)
                res_mids[n_logitsbands+i] += mids[n_c_logitsbands2]*layer->factor_mids[n_c_logitsbands2];
        }
        //logits: 32 x 32 [[o_1,...,o_256]_1,...,[o_1,...,o_256]_N]
        logits = &output[n*SQRT_QUANTIZE];
        for (i=0;i<SQRT_QUANTIZE;i++)
            logits[i] = fc_layer->bias[i];
        sgemv_accum(logits, fc_layer->input_weights, SQRT_QUANTIZE, MID_OUT, SQRT_QUANTIZE, &res_mids[n_logitsbands]);
        compute_activation(logits, logits, SQRT_QUANTIZE, fc_layer->activation);
        //for (i=0;i<MID_OUT;i++) {
        //    printf("res_mids [%d][%d] %f\n", n, i, res_mids[n_logitsbands+i]);
        //}
        //for (i=0;i<SQRT_QUANTIZE;i++) {
        //    printf("logits_out [%d][%d] %f\n", n, i, logits[i]);
        //}
        //refine logits using linear prediction with one-hot basis of previous samples and data-driven lpc
        //last_output: [[o_1,...,o_N]_1,...,[o_1,...,o_N]_K]; lpc: [[o_1,...,o_K]_1,...,[o_1,...,o_K]_N]
        for (i=0;i<DLPC_ORDER;i++)
            logits[last_output[i*N_MBANDS+n]] += lpc_signs[n_lpcbands+i]*lpc_mags[n_lpcbands+i];
        //    //logits[last_output[i*N_MBANDS+n]] += lpc[n_lpcbands+i];
   }
   //for (i=0;i<MID_OUT_MBANDS;i++) {
   //    printf("res_mids [%d] %f\n", i, res_mids[i]);
   //}
   //for (i=0;i<SQRT_QUANTIZE_MBANDS;i++) {
   //    printf("logits_outs [%d] %f\n", i, output[i]);
   //}
   //for (i=0;i<LPC_ORDER_MBANDS;i++) {
   //    printf("lpc_signs [%d] %f\n", i, lpc_signs[i]);
   //}
   //for (i=0;i<LPC_ORDER_MBANDS;i++) {
   //    printf("lpc_mags [%d] %f\n", i, lpc_mags[i]);
   //}
   //exit(0);
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
   sgemv_accum(recur, gru->recurrent_weights, RNN_SUB_NEURONS_3, RNN_SUB_NEURONS, RNN_SUB_NEURONS_3, state);
   for (i=0;i<RNN_SUB_NEURONS_2;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, RNN_SUB_NEURONS_2, ACTIVATION_SIGMOID);
   for (i=0;i<RNN_SUB_NEURONS;i++)
      h[i] += recur[RNN_SUB_NEURONS_2+i]*z[i];
      //h[i] += recur[RNN_SUB_NEURONS_2+i]*r[i];
   compute_activation(h, h, RNN_SUB_NEURONS, gru->activation);
   for (i=0;i<RNN_SUB_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i];
      //h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<RNN_SUB_NEURONS;i++)
      state[i] = h[i];
}

//PLT_Sep20
void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input)
{
   int i, k;
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
      for (i=0;i<RNN_MAIN_NEURONS;i++)
         recur[k*RNN_MAIN_NEURONS + i] += gru->diag_weights[k*RNN_MAIN_NEURONS + i]*state[i];
   sparse_sgemv_accum16(recur, gru->recurrent_weights, RNN_MAIN_NEURONS_3, gru->idx, state);
   //for (i=0;i<RNN_MAIN_NEURONS;i++)
   //     printf("is[%d] %f\n", i, state[i]);
   //for (i=0;i<RNN_MAIN_NEURONS_3;i++)
   //     printf("sg[%d] %f\n", i, recur[i]);
   for (i=0;i<RNN_MAIN_NEURONS_2;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, RNN_MAIN_NEURONS_2, ACTIVATION_SIGMOID);
   for (i=0;i<RNN_MAIN_NEURONS;i++)
      h[i] += recur[RNN_MAIN_NEURONS_2+i]*z[i];
      //h[i] += recur[RNN_MAIN_NEURONS_2+i]*r[i];
   compute_activation(h, h, RNN_MAIN_NEURONS, gru->activation);
   for (i=0;i<RNN_MAIN_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i];
      //h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<RNN_MAIN_NEURONS;i++)
      state[i] = h[i];
}

//PLT_Sep20
void compute_conv1d_linear(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   int i;
   int N, M, state_size;
   //int stride;
   M = layer->nb_inputs*layer->kernel_size;
   N = layer->nb_neurons;
   state_size = layer->nb_inputs*(layer->kernel_size-1);
   float tmp[M]; //set to input_size*kernel_size
   //celt_assert(input != output);
   RNN_COPY(tmp, mem, state_size); //get state_size of last frame (in*(kernel_size-1))
   RNN_COPY(&tmp[state_size], input, layer->nb_inputs); //append current input frame
   //for (int j=0;j<layer->kernel_size;j++) {
   //    for (i=0;i<layer->nb_inputs;i++) {
   //        printf("tmp [%d][%d] %f\n", j, i, tmp[j*layer->nb_inputs+i]);
   //    }
   //}
   // compute conv
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   sgemv_accum(output, layer->input_weights, N, M, N, tmp);
   //no activation (linear)
   RNN_COPY(mem, &tmp[layer->nb_inputs], state_size); //set state size for next frame
}

//PLT_Sep20
int sample_from_pdf_mwdlp(const float *pdf, int N)
{
    int i;
    float r;
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
    r = (float) rand() / RAND_MAX; //r ~ [0,1]
    for (i=N-1;i>0;i--)
        if (r >= cdf[i]) return i; //largest cdf that is less/equal than r
    return 0;
}

//PLT_Dec20
void compute_normalize(const NormStats *norm_stats, float *input_output)
{
  for (int i=0;i<norm_stats->n_dim;i++)
    input_output[i] = (input_output[i] - norm_stats->mean[i]) / norm_stats->std[i]
}

//PLT_Dec20
void compute_denormalize(const NormStats *norm_stats, float *input_output)
{
  for (int i=0;i<norm_stats->n_dim;i++)
    input_output[i] = input_output[i] * norm_stats->std[i] + norm_stats->mean[i]
}

//PLT_Dec20
void compute_gru_enc_melsp(const GRULayer *gru, float *state, const float *input)
{
   int i;
   float zrh[RNN_ENC_MELSP_NEURONS_3];
   float recur[RNN_ENC_MELSP_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_ENC_MELSP_NEURONS];
   h = &zrh[RNN_ENC_MELSP_NEURONS_2];

   for (i=0;i<RNN_ENC_MELSP_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_ENC_MELSP_NEURONS_3, RNN_ENC_MELSP_NEURONS, RNN_ENC_MELSP_NEURONS_3, state);

   compute_dense_linear(gru->weights, gru_input, input);

   for (i=0;i<RNN_ENC_MELSP_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   compute_activation(zrh, zrh, RNN_ENC_MELSP_NEURONS_2, ACTIVATION_SIGMOID);

   for (i=0;i<RNN_ENC_MELSP_NEURONS;i++)
      h[i] += recur[RNN_ENC_MELSP_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   compute_activation(h, h, RNN_ENC_MELSP_NEURONS, gru->activation);

   for (i=0;i<RNN_ENC_MELSP_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
   for (i=0;i<RNN_ENC_MELSP_NEURONS;i++)
      state[i] = h[i];
}

//PLT_Dec20
void compute_gru_enc_excit(const GRULayer *gru, float *state, const float *input)
{
   int i;
   float zrh[RNN_ENC_EXCIT_NEURONS_3];
   float recur[RNN_ENC_EXCIT_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_ENC_EXCIT_NEURONS];
   h = &zrh[RNN_ENC_EXCIT_NEURONS_2];

   for (i=0;i<RNN_ENC_EXCIT_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_ENC_EXCIT_NEURONS_3, RNN_ENC_EXCIT_NEURONS, RNN_ENC_EXCIT_NEURONS_3, state);

   compute_dense_linear(gru->weights, zrh, input);

   for (i=0;i<RNN_ENC_EXCIT_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   compute_activation(zrh, zrh, RNN_ENC_EXCIT_NEURONS_2, ACTIVATION_SIGMOID);

   for (i=0;i<RNN_ENC_EXCIT_NEURONS;i++)
      h[i] += recur[RNN_ENC_EXCIT_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   compute_activation(h, h, RNN_ENC_EXCIT_NEURONS, gru->activation);

   for (i=0;i<RNN_ENC_EXCIT_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
   for (i=0;i<RNN_ENC_EXCIT_NEURONS;i++)
      state[i] = h[i];
}

//PLT_Dec20
void compute_gru_spk(const GRULayer *gru, float *state, const float *input)
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

   RNN_COPY(zrh, input, RNN_SPK_NEURONS_3);

   for (i=0;i<RNN_SPK_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_SPK_NEURONS_3, RNN_SPK_NEURONS, RNN_SPK_NEURONS_3, state);

   compute_dense_linear(gru->weights, zrh, input);

   for (i=0;i<RNN_SPK_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   compute_activation(zrh, zrh, RNN_SPK_NEURONS_2, ACTIVATION_SIGMOID);

   for (i=0;i<RNN_SPK_NEURONS;i++)
      h[i] += recur[RNN_SPK_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   compute_activation(h, h, RNN_SPK_NEURONS, gru->activation);

   for (i=0;i<RNN_SPK_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
   for (i=0;i<RNN_SPK_NEURONS;i++)
      state[i] = h[i];
}

//PLT_Dec20
void compute_gru_dec_excit(const GRULayer *gru, float *state, const float *input)
{
   int i;
   float zrh[RNN_DEC_EXCIT_NEURONS_3];
   float recur[RNN_DEC_EXCIT_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_DEC_EXCIT_NEURONS];
   h = &zrh[RNN_DEC_EXCIT_NEURONS_2];

   for (i=0;i<RNN_DEC_EXCIT_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_DEC_EXCIT_NEURONS_3, RNN_DEC_EXCIT_NEURONS, RNN_DEC_EXCIT_NEURONS_3, state);

   compute_dense_linear(gru->weights, zrh, input);

   for (i=0;i<RNN_DEC_EXCIT_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   compute_activation(zrh, zrh, RNN_DEC_EXCIT_NEURONS_2, ACTIVATION_SIGMOID);

   for (i=0;i<RNN_DEC_EXCIT_NEURONS;i++)
      h[i] += recur[RNN_DEC_EXCIT_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   compute_activation(h, h, RNN_DEC_EXCIT_NEURONS, gru->activation);

   for (i=0;i<RNN_DEC_EXCIT_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
   for (i=0;i<RNN_DEC_EXCIT_NEURONS;i++)
      state[i] = h[i];
}

//PLT_Dec20
void compute_gru_dec_melsp(const GRULayer *gru, float *state, const float *input)
{
   int i;
   float zrh[RNN_DEC_MELSP_NEURONS_3];
   float recur[RNN_DEC_MELSP_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_DEC_MELSP_NEURONS];
   h = &zrh[RNN_DEC_MELSP_NEURONS_2];

   for (i=0;i<RNN_DEC_MELSP_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_DEC_MELSP_NEURONS_3, RNN_DEC_MELSP_NEURONS, RNN_DEC_MELSP_NEURONS_3, state);

   compute_dense_linear(gru->weights, zrh, input);

   for (i=0;i<RNN_DEC_MELSP_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   compute_activation(zrh, zrh, RNN_DEC_MELSP_NEURONS_2, ACTIVATION_SIGMOID);

   for (i=0;i<RNN_DEC_MELSP_NEURONS;i++)
      h[i] += recur[RNN_DEC_MELSP_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   compute_activation(h, h, RNN_DEC_MELSP_NEURONS, gru->activation);

   for (i=0;i<RNN_DEC_MELSP_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
   for (i=0;i<RNN_DEC_MELSP_NEURONS;i++)
      state[i] = h[i];
}

//PLT_Dec20
void compute_gru_post(const GRULayer *gru, float *state, const float *input)
{
   int i;
   float zrh[RNN_POST_NEURONS_3];
   float recur[RNN_POST_NEURONS_3];
   float *z;
   float *r;
   float *h;

   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_POST_NEURONS];
   h = &zrh[RNN_POST_NEURONS_2];

   for (i=0;i<RNN_POST_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_POST_NEURONS_3, RNN_POST_NEURONS, RNN_POST_NEURONS_3, state);

   compute_dense_linear(gru->weights, zrh, input);

   for (i=0;i<RNN_POST_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   compute_activation(zrh, zrh, RNN_POST_NEURONS_2, ACTIVATION_SIGMOID);

   for (i=0;i<RNN_POST_NEURONS;i++)
      h[i] += recur[RNN_POST_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   compute_activation(h, h, RNN_POST_NEURONS, gru->activation);

   for (i=0;i<RNN_POST_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
   for (i=0;i<RNN_POST_NEURONS;i++)
      state[i] = h[i];
}

//PLT_Dec20
void compute_sampling_laplace(float *loc, const float *scale, int dim)
{
    float r;
    for (int i=0;i<dim;i++) {
        r = ((float) rand() - HALF_RAND_MAX) / HALF_RAND_MAX_FLT_MIN; //r ~ (-1,1)
        // loc - sign(r)*scale*log(1-2|r/2|)
        if (r > 0) loc[i] -= scale[i] * log(1-r);
        else loc[i] += scale[i] * log(1+r);
    }
}

//PLT_Dec20
void compute_spkidtr(const DenseLayer *in_layer, const DenseLayer *out_layer, float *output, const float *input)
{
   int i, N;
   //transform to 2-dim
   N = in_layer->nb_neurons;
   float tmp[N];
   for (i=0;i<N;i++)
      tmp[i] = in_layer->bias[i];
   sgemv_accum(tmp, in_layer->input_weights, N, in_layer->nb_inputs, N, input);
   //transform to N_SPK-dim
   N = out_layer->nb_neurons;
   for (i=0;i<N;i++)
      output[i] = out_layer->bias[i];
   sgemv_accum(output, out_layer->input_weights, N, out_layer->nb_inputs, N, tmp);
}

//PLT_Dec20
void compute_spkidtr_coord(const DenseLayer *out_layer, float *output, const float *input)
{
   int i, N;
   //transform to N_SPK-dim [from input 2-dim]
   N = layer->nb_neurons;
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   sgemv_accum(output, layer->input_weights, N, layer->nb_inputs, N, input);
}
