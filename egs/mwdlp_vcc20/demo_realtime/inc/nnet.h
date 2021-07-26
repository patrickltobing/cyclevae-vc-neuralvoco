/* Copyright (c) 2018 Mozilla
   Copyright (c) 2017 Jean-Marc Valin */
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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Sept.-Dec. 2020 - Jul. 2021,
   marked by PLT_<Sep20/Dec20/Mar21/Jul21> */

#ifndef _NNET_H_
#define _NNET_H_

#include "arch.h"

#define ACTIVATION_LINEAR  0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH    2
#define ACTIVATION_RELU    3
#define ACTIVATION_SOFTMAX 4
#define ACTIVATION_EXP 5 //PLT_Sep20
#define ACTIVATION_TANHSHRINK 6 //PLT_Sep20
#define ACTIVATION_TANH_EXP 7 //PLT_Sep20

typedef struct {
  const float *bias;
  const float *input_weights;
  int nb_inputs;
  int nb_neurons;
  int activation;
} DenseLayer;

//PLT_Dec20
typedef struct {
  const float *mean;
  const float *std;
  int n_dim;
} NormStats;

//PLT_May21
typedef struct {
  const float *bias;
  const float *input_weights;
  const float *factors;
  int activation;
  int activation_signs;
  int activation_mags;
  int activation_logits;
} MDenseLayerMWDLP10;

typedef struct {
  const float *bias;
  const float *input_weights;
  const float *recurrent_weights;
  int nb_inputs;
  int nb_neurons;
  int activation;
  int reset_after;
} GRULayer;

typedef struct {
  const float *bias;
  const float *diag_weights;
  const float *recurrent_weights;
  const int *idx;
  int nb_neurons;
  int activation;
  int reset_after;
} SparseGRULayer;

typedef struct {
  const float *bias;
  const float *input_weights;
  int nb_inputs;
  int kernel_size;
  int nb_neurons;
  int activation;
} Conv1DLayer;

typedef struct {
  const float *embedding_weights;
  int nb_inputs;
  int dim;
} EmbeddingLayer;

//PLT_Jul21
typedef struct {
#ifdef WINDOWS_SYS
    BCRYPT_ALG_HANDLE rng_prov;
#else
    unsigned short int xsubi[3];
    struct drand48_data drand_buffer[1];
#endif
} RNGState;

//PLT_Mar21
void sgemv_accum16_(float *out, const float *weights, int rows, int cols, int col_stride, const float *x);
void sgemv_accum(float *out, const float *weights, int rows, int cols, int col_stride, const float *x);

void compute_activation(float *output, float *input, int N, int activation);

void compute_dense(const DenseLayer *layer, float *output, const float *input);

//PLT_Dec20
void compute_dense_linear(const DenseLayer *layer, float *output, const float *input);

//PLT_Mar21
void compute_mdense_mwdlp10(const MDenseLayerMWDLP10 *layer, const DenseLayer *fc_layer, const float *prev_logits,
    float *output, const float *input, const int *last_output);
    //float *output, const float *input, const int *last_output, float* ddlpc);

//PLT_Mar21
void compute_mdense_mwdlp10_nodlpc(const MDenseLayerMWDLP10 *layer, const DenseLayer *fc_layer, float *output,
    const float *input);

void compute_gru3(const GRULayer *gru, float *state, const float *input);

void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input);

//PLT_Jun21
void compute_conv1d_linear_frame_in(const Conv1DLayer *layer, float *output, float *mem, const float *input);

//PLT_Jul21
int sample_from_pdf_mwdlp(const float *pdf, int N, RNGState *rng_state);

//PLT_Dec20
void compute_normalize(const NormStats *norm_stats, float *input_output);

#endif /* _MLP_H_ */
