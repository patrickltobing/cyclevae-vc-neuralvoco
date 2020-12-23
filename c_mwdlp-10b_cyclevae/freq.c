/* Copyright (c) 2017-2018 Mozilla */
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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Dec. 2020,
   marked by PLT_Dec20 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "freq.h"
#include "arch.h"
#include <assert.h>

//PLT_Dec20
#include "hpassfilt.h" //get high-pass filter coefficients w/ numpy from dump script
#include "halfwin.h" //get ((N-1)/2) hanning window coefficients [because of periodic symmetry] w/ numpy from dump script
#include "melfb.h" //get mel-filterbank w/ numpy from dump script


///*Taken from vec_avx.h, vec_neon.h, and vec.h for mel-filterbank & mag-spec matrix multiplication*/
//#ifdef __AVX__
//#include <immintrin.h>
//#ifndef __AVX2__
//static void sgemv_accum16_(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
//{
//   int i, j;
//   for (i=0;i<rows;i+=16)
//   {
//      printf("sg2 %d\n", i);
//      float * restrict y;
//      __m256 vy0, vy8;
//      y = &out[i];
//      vy0 = _mm256_loadu_ps(&y[0]);
//      vy8 = _mm256_loadu_ps(&y[8]);
//      for (j=0;j<cols;j++)
//      {
//         //   printf("sg: %d %d %f %f %f\n", i, j, weights[j*col_stride+i], x[j], out[i]);
//         __m256 vxj;
//         __m256 vw;
//         vxj = _mm256_broadcast_ss(&x[j]);
//
//         vw = _mm256_loadu_ps(&weights[j*col_stride + i]);
//         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);
//
//         vw = _mm256_loadu_ps(&weights[j*col_stride + i + 8]);
//         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
//      }
//      _mm256_storeu_ps (&y[0], vy0);
//      _mm256_storeu_ps (&y[8], vy8);
//     //       printf("%f\n", out[i]);
//   }
//}
//#else
//static void sgemv_accum16_(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
//{
//   int i, j;
//   for (i=0;i<rows;i+=16)
//   {
//      //printf("sg %d\n", i);
//      float * restrict y;
//      __m256 vy0, vy8;
//      y = &out[i];
//      vy0 = _mm256_loadu_ps(&y[0]);
//      vy8 = _mm256_loadu_ps(&y[8]);
//      for (j=0;j<cols;j++)
//      {
//         //printf("cols %d\n", j);
//         //printf("sg: %d %d %f %f %f\n", i, j, weights[j*col_stride+i], x[j], out[i]);
//         __m256 vxj;
//         __m256 vw;
//         vxj = _mm256_broadcast_ss(&x[j]);
//
//         vw = _mm256_loadu_ps(&weights[j*col_stride + i]);
//         vy0 = _mm256_add_ps(_mm256_mul_ps(vw, vxj), vy0);
//
//         vw = _mm256_loadu_ps(&weights[j*col_stride + i + 8]);
//         vy8 = _mm256_add_ps(_mm256_mul_ps(vw, vxj), vy8);
//      }
//      _mm256_storeu_ps (&y[0], vy0);
//      _mm256_storeu_ps (&y[8], vy8);
//     //printf("%f\n", out[i]);
//   }
//}
//#endif
//#elif __ARM_NEON__
//#include <arm_neon.h>
//static void sgemv_accum16_(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
//{
//    int i, j;
//    for (i=0;i<rows;i+=16)
//    {
//	float * restrict y = &out[i];
//      
//	/* keep y[0..15] in registers for duration of inner loop */
//      
//	float32x4_t y0_3 = vld1q_f32(&y[0]);
//	float32x4_t y4_7 = vld1q_f32(&y[4]);
//	float32x4_t y8_11 = vld1q_f32(&y[8]);
//	float32x4_t y12_15 = vld1q_f32(&y[12]);
//      
//	for (j=0;j<cols;j++)
//	{
//	    const float * restrict w;
//	    float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;
//	    float32x4_t xj;
//
//	    w = &weights[j*col_stride + i];
//	    wvec0_3 = vld1q_f32(&w[0]);
//	    wvec4_7 = vld1q_f32(&w[4]);
//	    wvec8_11 = vld1q_f32(&w[8]);
//	    wvec12_15 = vld1q_f32(&w[12]);
//	    
//	    xj = vld1q_dup_f32(&x[j]);
//	 
//	    y0_3 = vmlaq_f32(y0_3, wvec0_3, xj);
//	    y4_7 = vmlaq_f32(y4_7, wvec4_7, xj);
//	    y8_11 = vmlaq_f32(y8_11, wvec8_11, xj);
//	    y12_15 = vmlaq_f32(y12_15, wvec12_15, xj);
//	}
//
//	/* save y[0..15] back to memory */
//      
//	vst1q_f32(&y[0], y0_3);
//	vst1q_f32(&y[4], y4_7);
//	vst1q_f32(&y[8], y8_11);
//	vst1q_f32(&y[12], y12_15);
//      
//    }
//}
//#else
//#warning Compiling melspec matrix multiplication without any vectorization.
//static void sgemv_accum16_(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
//{
//   int i, j;
//   for (i=0;i<rows;i+=16)
//   {
//      for (j=0;j<cols;j++)
//      {
//         const float * restrict w;
//         float * restrict y;
//         float xj;
//         w = &weights[j*col_stride + i];
//         xj = x[j];
//         y = &out[i];
//         y[0] += w[0]*xj;
//         y[1] += w[1]*xj;
//         y[2] += w[2]*xj;
//         y[3] += w[3]*xj;
//         y[4] += w[4]*xj;
//         y[5] += w[5]*xj;
//         y[6] += w[6]*xj;
//         y[7] += w[7]*xj;
//         y[8] += w[8]*xj;
//         y[9] += w[9]*xj;
//         y[10] += w[10]*xj;
//         y[11] += w[11]*xj;
//         y[12] += w[12]*xj;
//         y[13] += w[13]*xj;
//         y[14] += w[14]*xj;
//         y[15] += w[15]*xj;
//      }
//   }
//}
//#endif


//PLT_Dec20
int dspstate_get_size()
{
    return sizeof(DSPState);
}


DSPState *dspstate_create()
{
    DSPState *dsp;
    dsp = (DSPState *) calloc(1,dspstate_get_size());
    dsp->kfft = opus_fft_alloc_twiddles(FFT_LENGTH, NULL, NULL, NULL, 0);
    if (dsp != NULL) {
        int i, j, k;
        for (i=0;i<HPASS_FILT_TAPS;i++)
            dsp->hpass_filt[i] = hpassfilt[i];
        for (i=0;i<HALF_WINDOW_LENGTH_1;i++)
            dsp->half_window[i] = halfwin[i];
        for (i=0;i<MEL_DIM;i++) {
            for (j=0,k=i*MAGSP_DIM;j<MAGSP_DIM;j++) {
                dsp->melfb[k+j] = melfb[k+j];
            //    printf("mfb %d %d %f\n", i, j, dsp->melfb[k+j]);
            }
        }
        return dsp;
    }
    printf("Cannot allocate and initialize memory for DSPState.\n");
    exit(EXIT_FAILURE);
    return NULL;
}


void dspstate_destroy(DSPState *dsp)
{
    if (dsp != NULL) free(dsp);
}


void shift_apply_hpassfilt(DSPState *dsp, float *x)
{
    //printf("in_a\n");
    RNN_MOVE(dsp->samples_hpass, &dsp->samples_hpass[1], HPASS_FILT_TAPS_1); //shift buffer to the left
    //printf("in_b\n");
    dsp->samples_hpass[HPASS_FILT_TAPS_1] = *x; //add new sample at the right
    //printf("in_c\n");
    *x = 0;
    for (int i=0;i<HPASS_FILT_TAPS;i++) {
    //    printf("in_d %d\n", i);
    //    printf("%f %f\n", dsp->samples_hpass[HPASS_FILT_TAPS_1-i], dsp->hpass_filt[i]);
        *x += dsp->samples_hpass[HPASS_FILT_TAPS_1-i] * dsp->hpass_filt[i]; //filter convolution
    }
    //printf("in_e\n");
}


void apply_window(DSPState *dsp)
{
    int i;
    printf("window\n");
    for (i=0;i<HALF_WINDOW_LENGTH_1;i++) { //window [symmetric] centered on FFT length
        //zero pad left
        dsp->in_fft[i+WIN_LEFT_IDX].r = dsp->samples_win[i]*dsp->half_window[i];
        //zero pad right [mirrored of left]
        dsp->in_fft[WIN_RIGHT_IDX-i].r = dsp->samples_win[WINDOW_LENGTH_2-i]*dsp->half_window[i];
    //    printf("%d %d %d %d %f %f %f %f %f\n", i, i+WIN_LEFT_IDX, WIN_RIGHT_IDX-i, WINDOW_LENGTH_2-i,
    //        dsp->samples_win[i], dsp->samples_win[WINDOW_LENGTH_2-i], dsp->half_window[i],
    //            dsp->in_fft[i+WIN_LEFT_IDX].r, dsp->in_fft[WIN_RIGHT_IDX-i].r);
    }
    //window length is even, exists coefficient 1 on ((N-1)/2+(N-1)%2)th bin because of periodic window
    dsp->in_fft[i+WIN_LEFT_IDX].r = dsp->samples_win[i];
}


void shift_apply_window(DSPState *dsp, const float *x)
{
    RNN_MOVE(dsp->samples_win, &dsp->samples_win[FRAME_SHIFT], BUFFER_LENGTH); //shift buffer to the left
    //pointer "->" precedes reference "&", no need bracket after &
    RNN_COPY(&dsp->samples_win[BUFFER_LENGTH], x, FRAME_SHIFT); //add new samples to the right
    int i;
    //printf("shift_window\n");
    for (i=0;i<HALF_WINDOW_LENGTH_1;i++) { //window [symmetric] centered on FFT length
        //zero pad left
        dsp->in_fft[i+WIN_LEFT_IDX].r = dsp->samples_win[i]*dsp->half_window[i];
        //zero pad right [mirrored of left]
        dsp->in_fft[WIN_RIGHT_IDX-i].r = dsp->samples_win[WINDOW_LENGTH_2-i]*dsp->half_window[i];
    //    printf("%d %d %d %d %f %f %f %f %f\n", i, i+WIN_LEFT_IDX, WIN_RIGHT_IDX-i, WINDOW_LENGTH_2-i,
    //        dsp->samples_win[i], dsp->samples_win[WINDOW_LENGTH_2-i], dsp->half_window[i],
    //            dsp->in_fft[i+WIN_LEFT_IDX].r, dsp->in_fft[WIN_RIGHT_IDX-i].r);
    }
    //window length is even, exists coefficient 1 on ((N-1)/2+(N-1)%2)th bin because of periodic window
    dsp->in_fft[i+WIN_LEFT_IDX].r = dsp->samples_win[i];
}


void mel_spec_extract(DSPState *dsp, float *melsp)
{
    int i, j, k;
    //printf("in_melsp_a\n");
    opus_fft(dsp->kfft, dsp->in_fft, dsp->out_fft, 0); //STFT
    //printf("in_melsp_b\n");
    //cplx -> mag
    for (i=0;i<MAGSP_DIM;i++) {
        //need to multiply by 10^3 here to match the output of librosa STFT
        dsp->magsp[i] = sqrt(pow((dsp->out_fft[i].r*1000), 2) + pow((dsp->out_fft[i].i*1000), 2));
    //    printf("in_melsp_c %d %f\n", i, dsp->magsp[i]);
    }
    //printf("in_melsp_d\n");
    //mag -> mel --> log(1+10000*mel)
    //if (MEL_DIM_16_BLOCK) { //compute melfb*magsp matmul in a block of 16
    ////    printf("in_melsp_e\n");
    //    sgemv_accum16_(melsp, dsp->melfb, MEL_DIM, MAGSP_DIM, MEL_DIM, dsp->magsp);
    ////    printf("in_melsp_f\n");
    ////    for (i=0;i<MEL_DIM;i++) {
    ////        printf("in_melsp_g %d\n", i);
    ////        melsp[i] = log(1+10000*melsp[i]);
    ////    }
    ////    printf("in_melsp_h\n");
    //} else {
    //    int j;
    ////    printf("in_melsp_e1\n");
    //    for (i=0;i<MEL_DIM;i++) {
    ////        printf("in_melsp_f1 %d\n", i);
    //        for (j=0,melsp[i]=0;j<MAGSP_DIM;j++) {
    ////            printf("in_melsp_g1 %d\n", j);
    //            melsp[i] += dsp->magsp[j]*dsp->melfb[i];
    //        }
    //        melsp[i] = log(1+10000*melsp[i]);
    //    }
    ////    printf("in_melsp_h1\n");
    //}
    //int j, k;
    for (i=0;i<MEL_DIM;i++) {
    //  printf("in_melsp_f1 %d\n", i);
      for (j=0,k=i*MAGSP_DIM,melsp[i]=0;j<MAGSP_DIM;j++) {
    //        printf("in_melsp_g1 %d\n", j);
          melsp[i] += dsp->magsp[j]*dsp->melfb[k+j];
      //    printf("in_melsp_g1 %d %d %d %f %f\n", i, j, k, dsp->magsp[j], dsp->melfb[k+j]);
      }
      melsp[i] = log(1+10000*melsp[i]);
      //melsp[i] *= 10000;
    }
    //melsp[i] = log(1+10000*melsp[i]);
    //printf("done_melsp\n");
}
