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
#include "tansig_table.h"
#include "nnet.h"
#include "nnet_data.h"
#include "lpcnet_private.h"
#include "gumbel_table.h"

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


//static void sgemv_accum(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
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

//PLT_Sep20
void compute_mdense_mwdlp(const MDenseLayerMBDLP *layer, const DenseLayer *fc_layer, float *output,
    const float *input, const int *last_output)
{
    int i, c, n;
    int n_lpcbands, n_lpcbands2, n_c_lpcbands2, c_lpc;
    int n_logitsbands, n_logitsbands2, c_logits;
    float tmp[MDENSE_OUT];
    float signs[LPC_ORDER_MBANDS_2], mags[LPC_ORDER_MBANDS_2], mids[MID_OUT_MBANDS_2];
    //float lpc[LPC_ORDER_MBANDS], res_mids[MID_OUT_MBANDS];
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
    for (i=0;i<MID_OUT_MBANDS;i++)
        res_mids[i] = 0;
    for (n=0;n<N_MBANDS;n++) { //n_bands x 2 x n_lpc or 32
        //lpc: n_b x 2 x n_lpc
        for (c=0,n_lpcbands=n*DLPC_ORDER,n_lpcbands2=n_lpcbands*2;c<2;c++) {
            //factors of lpcs shared between bands
            //for (i=0,c_lpc=c*DLPC_ORDER,n_c_lpcbands2=n_lpcbands2+c_lpc;i<DLPC_ORDER;i++)
            //    lpc[n_lpcbands+i] += signs[n_c_lpcbands2 + i]*layer->factor_signs[c_lpc + i] 
            //                            * mags[n_c_lpcbands2 + i]*layer->factor_mags[c_lpc + i];
            for (i=0,c_lpc=c*DLPC_ORDER,n_c_lpcbands2=n_lpcbands2+c_lpc;i<DLPC_ORDER;i++) {
                lpc_signs[n_lpcbands+i] += signs[n_c_lpcbands2 + i]*layer->factor_signs[c_lpc + i];
                lpc_mags[n_lpcbands+i] += mags[n_c_lpcbands2 + i]*layer->factor_mags[c_lpc + i];
            }
        }
        //mids: n_b x 2 x 32
        for (c=0,n_logitsbands=n*MID_OUT,n_logitsbands2=n_logitsbands*2;c<2;c++) {
            //factor of mids band-dependent
            for (i=0,c_logits=n_logitsbands2+c*MID_OUT;i<MID_OUT;i++)
                res_mids[n_logitsbands+i] += mids[c_logits + i]*layer->factor_mids[c_logits + i];
        }
        //logits: 32 x 256 [[o_1,...,o_256]_1,...,[o_1,...,o_256]_N]
        logits = &output[n*LAST_FC_OUT];
        for (i=0;i<LAST_FC_OUT;i++)
            logits[i] = fc_layer->bias[i];
        sgemv_accum(logits, fc_layer->input_weights, LAST_FC_OUT, MID_OUT, LAST_FC_OUT, &res_mids[n_logitsbands]);
        compute_activation(logits, logits, LAST_FC_OUT, fc_layer->activation);
        //for (i=0;i<MID_OUT;i++) {
        //    printf("res_mids [%d][%d] %f\n", n, i, res_mids[n_logitsbands+i]);
        //}
        //for (i=0;i<LAST_FC_OUT;i++) {
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
   //for (i=0;i<LAST_FC_OUT_MBANDS;i++) {
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

void compute_gru(const GRULayer *gru, float *state, const float *input)
{
   int i;
   int N, M;
   int stride;
   float tmp[MAX_RNN_NEURONS];
   float z[MAX_RNN_NEURONS];
   float r[MAX_RNN_NEURONS];
   float h[MAX_RNN_NEURONS];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   stride = 3*N;
   /* Compute update gate. */
   for (i=0;i<N;i++)
      z[i] = gru->bias[i];
   if (gru->reset_after)
   {
      for (i=0;i<N;i++)
         z[i] += gru->bias[3*N + i];
   }
   sgemv_accum(z, gru->input_weights, N, M, stride, input);
   sgemv_accum(z, gru->recurrent_weights, N, N, stride, state);
   compute_activation(z, z, N, ACTIVATION_SIGMOID);

   /* Compute reset gate. */
   for (i=0;i<N;i++)
      r[i] = gru->bias[N + i];
   if (gru->reset_after)
   {
      for (i=0;i<N;i++)
         r[i] += gru->bias[4*N + i];
   }
   sgemv_accum(r, &gru->input_weights[N], N, M, stride, input);
   sgemv_accum(r, &gru->recurrent_weights[N], N, N, stride, state);
   compute_activation(r, r, N, ACTIVATION_SIGMOID);

   /* Compute output. */
   for (i=0;i<N;i++)
      h[i] = gru->bias[2*N + i];
   if (gru->reset_after)
   {
      for (i=0;i<N;i++)
         tmp[i] = gru->bias[5*N + i];
      sgemv_accum(tmp, &gru->recurrent_weights[2*N], N, N, stride, state);
      for (i=0;i<N;i++)
         h[i] += tmp[i] * r[i];
      sgemv_accum(h, &gru->input_weights[2*N], N, M, stride, input);
   } else {
      for (i=0;i<N;i++)
         tmp[i] = state[i] * r[i];
      sgemv_accum(h, &gru->input_weights[2*N], N, M, stride, input);
      sgemv_accum(h, &gru->recurrent_weights[2*N], N, N, stride, tmp);
   }
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<N;i++)
      state[i] = h[i];
}

void compute_gru2(const GRULayer *gru, float *state, const float *input)
{
   int i;
   int N, M;
   int stride;
   float zrh[3*MAX_RNN_NEURONS];
   float recur[3*MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   stride = 3*N;
   /* Compute update gate. */
   for (i=0;i<3*N;i++)
      zrh[i] = gru->bias[i];
   sgemv_accum(zrh, gru->input_weights, 3*N, M, stride, input);
   for (i=0;i<3*N;i++)
      recur[i] = gru->bias[3*N + i];
   sgemv_accum(recur, gru->recurrent_weights, 3*N, N, stride, state);
   for (i=0;i<2*N;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID);
   for (i=0;i<N;i++)
      h[i] += recur[2*N+i]*r[i];
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<N;i++)
      state[i] = h[i];
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
   {
      for (i=0;i<RNN_MAIN_NEURONS;i++)
         recur[k*RNN_MAIN_NEURONS + i] += gru->diag_weights[k*RNN_MAIN_NEURONS + i]*state[i];
   }
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
void compute_conv1d_mwdlp(const Conv1DLayer *layer, float *output, float *mem, const float *input)
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
    for (i=0;i<N;i++) {
        tmp[i] = pdf[i];
    }
    softmax(tmp, tmp, N);
    for (i=0,sum=0;i<N;i++) {
        sum += tmp[i];
    }
    norm = 1.f/sum;
    /* Convert tmp to a CDF (sum of all previous probs., init. with 0) */
    cdf[0] = 0;
    for (i=1;i<N;i++) {
        cdf[i] = cdf[i-1] + norm*tmp[i-1];
    }
    /* Do the sampling (from the cdf). */
    r = (float) rand() / RAND_MAX; //r ~ [0,1]
    for (i=N-1;i>0;i--) {
        if (r >= cdf[i]) return i; //largest cdf that is less/equal than r
    }
    return 0;
}

//PLT_Dec20
void compute_gru_enc(const GRULayer *gru, float *state, const float *input)
{
   int i;
   float zrh[RNN_ENC_NEURONS_3];
   float recur[RNN_ENC_NEURONS_3];
   float *z;
   float *r;
   float *h;
   z = zrh; //swap with r, pytorch rzh, keras zrh
   r = &zrh[RNN_ENC_NEURONS];
   h = &zrh[RNN_ENC_NEURONS_2];
   RNN_COPY(zrh, input, RNN_ENC_NEURONS_3);
   for (i=0;i<RNN_ENC_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_ENC_NEURONS_3, RNN_ENC_NEURONS, RNN_ENC_NEURONS_3, state);
   for (i=0;i<RNN_ENC_NEURONS_2;i++)
      zrh[i] += recur[i]; //z_t and r_t computed in a similar way : sigmoid(in_t + W_z*h_{t-1})
   compute_activation(zrh, zrh, RNN_ENC_NEURONS_2, ACTIVATION_SIGMOID);
   for (i=0;i<RNN_ENC_NEURONS;i++)
      h[i] += recur[RNN_ENC_NEURONS_2+i]*z[i]; //n_t = tanh(in_t + r_t o W_n*h_{t-1})
   compute_activation(h, h, RNN_ENC_NEURONS, gru->activation);
   for (i=0;i<RNN_ENC_NEURONS;i++)
      h[i] = r[i]*state[i] + (1-r[i])*h[i]; //h_t = z_t o h_{t-1} + (1-z_t) o n_t
   for (i=0;i<RNN_ENC_NEURONS;i++)
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
   RNN_COPY(zrh, input, RNN_DEC_EXCIT_NEURONS_3);
   for (i=0;i<RNN_DEC_EXCIT_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_DEC_EXCIT_NEURONS_3, RNN_DEC_EXCIT_NEURONS, RNN_DEC_EXCIT_NEURONS_3, state);
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
   RNN_COPY(zrh, input, RNN_DEC_MELSP_NEURONS_3);
   for (i=0;i<RNN_DEC_MELSP_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_DEC_MELSP_NEURONS_3, RNN_DEC_MELSP_NEURONS, RNN_DEC_MELSP_NEURONS_3, state);
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
   RNN_COPY(zrh, input, RNN_POST_NEURONS_3);
   for (i=0;i<RNN_POST_NEURONS_3;i++)
      recur[i] = gru->bias[i];
   sgemv_accum(recur, gru->recurrent_weights, RNN_POST_NEURONS_3, RNN_POST_NEURONS, RNN_POST_NEURONS_3, state);
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

