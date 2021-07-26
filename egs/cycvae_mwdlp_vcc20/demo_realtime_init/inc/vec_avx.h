/* Copyright (c) 2018 Mozilla
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
/*
  AVX implementation of vector operations, compile with -mavx
  AVX2/FMA implementation of vector operations, compile with -mavx2 -mfma
*/
/* Modified by Patrick Lumban Tobing (Nagoya University) on Sept. 2020 - Feb, Mar. 2021,
   marked by PLT_<Sep20/Feb21/Mar21> */

#ifndef VEC_AVX_H
#define VEC_AVX_H

#include <immintrin.h>

//PLT_Mar21
#ifndef __AVX2__
#define _mm256_fmadd_ps(a,b,c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define _mm_fmadd_ps(a,b,c) _mm_add_ps(_mm_mul_ps(a, b), c)
#endif

//PLT_Mar21
// this more accurate exp function is taken from https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx/48869291#48869291
// and slightly modified to use fmadd_ps, which is defined as a macro if not use AVX2
static __m256 exp256_ps(__m256 x)
{
/* Modified code from this source: https://github.com/reyoung/avx_mathfun

   AVX implementation of exp
   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/
   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.
  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:
  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  (this is the zlib license)

*/
/* 
  To increase the compatibility across different compilers the original code is
  converted to plain AVX2 intrinsics code without ingenious macro's,
  gcc style alignment attributes etc.
  Moreover, the part "express exp(x) as exp(g + n*log(2))" has been significantly simplified.
  This modified code is not thoroughly tested!
*/


__m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
__m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

__m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
__m256   inv_LOG2EF    = _mm256_set1_ps(0.693147180559945f);

__m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
__m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
__m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
__m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
__m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
__m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
__m256   fx;
__m256i  imm0;
__m256   one           = _mm256_set1_ps(1.0f);

        x     = _mm256_min_ps(x, exp_hi);
        x     = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
        fx     = _mm256_mul_ps(x, cephes_LOG2EF);
        fx     = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
__m256  z      = _mm256_mul_ps(fx, inv_LOG2EF);
        x      = _mm256_sub_ps(x, z);
        z      = _mm256_mul_ps(x,x);

__m256  y      = cephes_exp_p0;
        y      = _mm256_fmadd_ps (y, x, cephes_exp_p1);
        y      = _mm256_fmadd_ps (y, x, cephes_exp_p2);
        y      = _mm256_fmadd_ps (y, x, cephes_exp_p3);
        y      = _mm256_fmadd_ps (y, x, cephes_exp_p4);
        y      = _mm256_fmadd_ps (y, x, cephes_exp_p5);
        y      = _mm256_fmadd_ps (y, z, x);
        y      = _mm256_add_ps(y, one);

  /* build 2^n */
        imm0   = _mm256_cvttps_epi32(fx);
        imm0   = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
        imm0   = _mm256_slli_epi32(imm0, 23);
__m256  pow2n  = _mm256_castsi256_ps(imm0);
        y      = _mm256_mul_ps(y, pow2n);
        return y;
}


//PLT_Mar21
static float celt_exp(float x)
{
   float out[8];
   __m256 X, Y;
   X = _mm256_set1_ps(x);
   Y = exp256_ps(X);
   _mm256_storeu_ps(out, Y);
   return out[0];
}

//PLT_Mar21
static void softmax(float *y, const float *x, int N)
{
    for (int i=0;i<N;i++) {
        //if (x[i] < -32)
        //    y[i] = exp(-32);
        //else if (x[i] > 32)
        //    y[i] = exp(32);
        //else
        //    y[i] = exp(x[i]);
        //y[i] = celt_exp_approx(x[i]);
        //if (x[i] > 88 || x[i] < -88) printf("%lf\n", x[i]);
        //y[i] = exp(x[i]);
        //y[i] = celt_exp(x[i]);
        //if (x[i] > -32) {
        //    if (x[i] < 32) y[i] = exp(x[i]);
        //    else y[i] = 78962960182680.695160978022635108;
        //} else y[i] = 1.2664165549094175723120904155965e-14;
        if (x[i] > -103) {
            if (x[i] < 85) y[i] = exp(x[i]);
            else y[i] = 8.223013E+36;
        } else y[i] = 1.401298E-45;
        //if (x[i] > -700) {
        //    if (x[i] < 700) y[i] = exp(x[i]);
        //    else y[i] = 1.0142320547350045094553295952313e+304;
        //} else y[i] = 9.8596765437597708567053729478495e-305;
        //if (x[i] > -745.132) {
        //    if (x[i] < 709.78268) y[i] = exp(x[i]);
        //    else y[i] = 1.797587E+308;
        //} else y[i] = 4.940656E-324;
        //if (x[i] > -88) {
        //    if (x[i] < 88) y[i] = exp(x[i]);
        //    else y[i] = 1.6516362549940018555283297962649e+38;
        //} else y[i] = 6.0546018954011858845318605338106e-39;
        //if (x[i] < 32)
        //    y[i] = exp(x[i]);
        //else
        //    y[i] = 78962960182680.687500;
    }
}

//PLT_Feb21
static void vec_exp(float *y, const float *x, int N)
{
    //int i;
    //__m256 X, Y;
    //for (i=0;i<N-7;i+=8)
    //{
    //    X = _mm256_loadu_ps(&x[i]);
    //    Y = exp8_approx(X);
    //    _mm256_storeu_ps(&y[i], Y);
    //}
    for (int i=0;i<N;i++) {
    //for (;i<N;i++)
        //if (x[i] < -32)
        //    y[i] = exp(-32);
        //else if (x[i] > 32)
        //    y[i] = exp(32);
        //else
        //    y[i] = exp(x[i]);
        //if (x[i] > 88 || x[i] < -88) printf("%lf\n", x[i]);
        //if (x[i] > 38 || x[i] < -38) printf("exp %lf\n", x[i]);
        //y[i] = exp(x[i]);
        //if (x[i] > -32) {
        //    if (x[i] < 32) y[i] = exp(x[i]);
        //    else y[i] = 78962960182680.695160978022635108;
        //} else y[i] = 1.2664165549094175723120904155965e-14;
        //if (x[i] > -38) {
        //    if (x[i] < 38) y[i] = exp(x[i]);
        //    else y[i] = 31855931757113756.220328671701299;
        //} else y[i] = 3.1391327920480296287089646522319e-17;
        //if (x[i] > -103) {
        //    if (x[i] < 85) y[i] = exp(x[i]);
        //    else y[i] = 8.223013E+36;
        //} else y[i] = 1.401298E-45;
        y[i] = exp(x[i]);
        //if (x[i] > -700) {
        //    if (x[i] < 700) y[i] = exp(x[i]);
        //    else y[i] = 1.0142320547350045094553295952313e+304;
        //} else y[i] = 9.8596765437597708567053729478495e-305;
        //if (x[i] > -745.132) {
        //    if (x[i] < 709.78268) y[i] = exp(x[i]);
        //    else y[i] = 1.797587E+308;
        //} else y[i] = 4.940656E-324;
        //if (x[i] > -88) {
        //    if (x[i] < 88) y[i] = exp(x[i]);
        //    else y[i] = 1.6516362549940018555283297962649e+38;
        //} else y[i] = 6.0546018954011858845318605338106e-39;
        //if (x[i] < 32)
        //    y[i] = exp(x[i]);
        //else
        //    y[i] = 78962960182680.687500;
    }
}

//PLT_Mar21
static void vec_tanh(float *y, const float *x, int N)
{
    int i;
    __m256 Y, rcp_Y;
    for (i=0;i<N-7;i+=8)
    {
        Y = exp256_ps(_mm256_loadu_ps(&x[i]));
        rcp_Y = _mm256_rcp_ps(Y);
        _mm256_storeu_ps(&y[i], _mm256_mul_ps(_mm256_sub_ps(Y, rcp_Y),  _mm256_rcp_ps(_mm256_add_ps(Y, rcp_Y))));
    }
    for (;i<N;i++)
    {
        float ex;
        ex = celt_exp(x[i]);
        y[i] = (ex-1/ex)/(ex+1/ex);
    }
}

//PLT_Feb21
static void vec_tanh_exp(float *y, const float *x, int N)
{
    //int i;
    //const __m256 two = _mm256_set1_ps(2.f);
    //const __m256 one = _mm256_set1_ps(1.f);
    //__m256 X, Y;
    //for (i=0;i<N-7;i+=8)
    //{
    //    X = _mm256_loadu_ps(&x[i]);
    //    X = _mm256_mul_ps(X, two);
    //    Y = exp8_approx(X);
    //    Y = _mm256_mul_ps(_mm256_sub_ps(Y, one),  _mm256_rcp_ps(_mm256_add_ps(Y, one)));
    //    _mm256_storeu_ps(&y[i], Y);
    //}
    float ex;
    //float ex2;
    //for (;i<N;i++)
    for (int i=0;i<N;i++)
    {
        //if (x[i] > -32) {
        //    if (x[i] < 32) {
        //        ex2 = exp(2*x[i]);
        //        y[i] = (ex2-1)/(ex2+1);
        //    } else y[i] = 0.99999999999999999999999999967924;
        //} else y[i] = -0.99999999999999999999999999967924;
        //if (x[i] > 38 || x[i] < -38) printf("tanh exp %lf\n", x[i]);
        //if (x[i] > -10) {
        //    if (x[i] < 10) {
        //        //ex2 = exp(2*x[i]);
        //        //y[i] = (ex2-1)/(ex2+1);
        //        ex = exp(x[i]);
        //        y[i] = (ex-1/ex)/(ex+1/ex);
        //    } else y[i] = 1;
        //} else y[i] = -1;
        ex = exp(x[i]);
        y[i] = (ex-1/ex)/(ex+1/ex);
    //    y[i] = tanh(x[i]);
        //if (x[i] < 32)
    }
}

//PLT_Sep20
static void vec_tanhshrink(float *y, const float *x, int N)
{
    //int i;
    //const __m256 two = _mm256_set1_ps(2.f);
    //const __m256 one = _mm256_set1_ps(1.f);
    //__m256 X, Y;
    //for (i=0;i<N-7;i+=8)
    //{
    //    X = _mm256_loadu_ps(&x[i]);
    //    Y = exp8_approx(_mm256_mul_ps(X, two));
    //    Y = _mm256_sub_ps(X, _mm256_mul_ps(_mm256_sub_ps(Y, one),  _mm256_rcp_ps(_mm256_add_ps(Y, one))));
    //    _mm256_storeu_ps(&y[i], Y);
    //}
    float ex;
    //float ex2;
    //for (;i<N;i++)
    for (int i=0;i<N;i++)
    {
        //if (x[i] < -32)
        //else if (x[i] > 32)
        //    ex2 = exp(32);
        //else
        //    ex2 = exp(2*x[i]);
        //ex2 = exp(2*x[i]);
        //y[i] = x[i]-(ex2-1)/(ex2+1);
        //if (x[i] > -32) {
        //    if (x[i] < 32) {
        //        ex2 = exp(2*x[i]);
        //        y[i] = x[i]-(ex2-1)/(ex2+1);
        //    } else y[i] = x[i]-0.99999999999999999999999999967924;
        //} else y[i] = x[i]+0.99999999999999999999999999967924;
        //if (x[i] > 38 || x[i] < -38) printf("tanhshrink exp %lf\n", x[i]);
        //if (x[i] > -10) {
        //    if (x[i] < 10) {
        //        //ex2 = exp(2*x[i]);
        //        //y[i] = x[i]-(ex2-1)/(ex2+1);
        //        ex = exp(x[i]);
        //        y[i] = x[i]-(ex-1/ex)/(ex+1/ex);
        //    } else y[i] = x[i]-1;
        //} else y[i] = x[i]+1;
        ex = exp(x[i]);
        y[i] = x[i]-(ex-1/ex)/(ex+1/ex);
    //    y[i] = x[i]-tanh(x[i]);
    }
}

//PLT_Mar21
static void vec_sigmoid(float *y, const float *x, int N)
{
    int i;
    const __m256 one = _mm256_set1_ps(1.f);
    for (i=0;i<N-7;i+=8)
    {
        _mm256_storeu_ps(&y[i], _mm256_rcp_ps(_mm256_add_ps(_mm256_rcp_ps(exp256_ps(_mm256_loadu_ps(&x[i]))), one)));
    }
    for (;i<N;i++)
    {
        y[i] = 1/(1/celt_exp(x[i])+1);
    }
}

//PLT_Feb21
static void vec_sigmoid_exp(float *y, const float *x, int N)
{
    float ex;
    for (int i=0;i<N;i++)
    {
        //if (x[i] > -37) {
        //    if (x[i] < 29) y[i] = 1-1/(exp(x[i])+1);
        //    else y[i] = 1;
        //} y[i] = 0;
        ex = exp(x[i]);
        y[i] = 1/(1/ex+1);
        //y[i] = 1/(1/exp(x[i])+1);
        //y[i] = 1-1/(exp(x[i])+1);
    }
}

//PLT_Mar21
//col_stride is the dimension of output, because weight vector is written as kernel_size x in x out
static void sgemv_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j, k;
   float * restrict y;
   __m256 vy0, vy8, vxj;
   for (i=0;i<(rows-(rows%16));i+=16)
   {
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      for (j=0;j<cols;j++)
      {
         vxj = _mm256_broadcast_ss(&x[j]);

         k = j*col_stride + i;
         vy0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k]), vxj, vy0);
         vy8 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 8]), vxj, vy8);
      }
      _mm256_storeu_ps (&y[0], vy0);
      _mm256_storeu_ps (&y[8], vy8);
   }
   for (;i<rows;i++)
   {
      for (j=0;j<cols;j++)
         out[i] += weights[j*col_stride + i]*x[j];
   }
}

//PLT_Mar21
static void sgev_dualfc8(float *out, const float *factors, int rows, const float *x)
{
   //int i, half_rows;
   int i, j, half_rows;
   half_rows = rows / 2;
   if (half_rows % 8 == 0) {
      for (i=0;i<half_rows;i+=8) //1st channel
         _mm256_storeu_ps (&out[i], _mm256_mul_ps(_mm256_loadu_ps(&factors[i]), _mm256_loadu_ps(&x[i])));
      for (j=0;i<rows;i+=8,j+=8) //2nd channel
         _mm256_storeu_ps (&out[j], _mm256_fmadd_ps(_mm256_loadu_ps(&factors[i]), _mm256_loadu_ps(&x[i]), _mm256_loadu_ps(&out[j])));
   } else {
      int half_rows_8 = half_rows/8 * 8;
      for (i=0;i<half_rows_8;i+=8) //1st channel
         _mm256_storeu_ps (&out[i], _mm256_mul_ps(_mm256_loadu_ps(&factors[i]), _mm256_loadu_ps(&x[i])));
      for (;i<half_rows;i++)
         out[i] = factors[i]*x[i];
      for (j=0;i<(rows-(half_rows%8));i+=8,j+=8) //2nd channel
         _mm256_storeu_ps (&out[j], _mm256_fmadd_ps(_mm256_loadu_ps(&factors[i]), _mm256_loadu_ps(&x[i]), _mm256_loadu_ps(&out[j])));
      for (;i<rows;i++,j++)
         out[j] += factors[i]*x[i];
   }
   ////printf("%d %d\n",half_rows,rows);
   //for (i=0;i<half_rows;i++) {
   //   out[i] = factors[i]*x[i] + factors[i+half_rows]*x[i+half_rows];
   //   //out[i] = factors[i]*x[i];
   ////   printf("[%d] %lf %lf %lf\n",i,out[i],factors[i],x[i]);
   //}
   ////printf("%lf %lf %lf\n",out[i-1],factors[i-1],x[i-1]);
   ////for (j=0;i<rows;i++,j++) {
   ////   out[j] += factors[i]*x[i];
   //////   printf("[%d-%d] %lf %lf %lf\n",i,j,out[j],factors[i],x[i]);
   ////}
   ////printf("%lf %lf %lf\n",out[j-1],factors[i-1],x[i-1]);
}

//PLT_Mar21
//weights are shared between bands
static void sgemv_fclogits16(float *out, const float *weights, int rows, int cols, int n_bands, const float *x)
{
   //int i, j, n, row_bands, col_bands;
   int i, j, k, n, row_bands, col_bands;
   float * restrict y;
   __m256 vy0, vy8, vxj;
   for (n=0,row_bands=0;n<n_bands;n++)
   {
      for (i=0,col_bands=n*cols;i<(rows-(rows%16));i+=16,row_bands+=16)
      {
         y = &out[row_bands];
         vy0 = _mm256_loadu_ps(&y[0]);
         vy8 = _mm256_loadu_ps(&y[8]);
         for (j=0;j<cols;j++)
         {
            vxj = _mm256_broadcast_ss(&x[col_bands+j]);

            k = j*rows + i;
            vy0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k]), vxj, vy0);
            vy8 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 8]), vxj, vy8);
         }
         _mm256_storeu_ps (&y[0], vy0);
         _mm256_storeu_ps (&y[8], vy8);
      }
      for (;i<rows;i++,row_bands++)
      {
         for (j=0;j<cols;j++)
            out[row_bands] += weights[j*rows + i]*x[col_bands+j];
      }
   }
   //for (n=0,row_bands=0;n<n_bands;n++)
   //   for (i=0,col_bands=n*cols;i<rows;i++,row_bands++)
   //      for (j=0;j<cols;j++)
   //         out[row_bands] += weights[j*rows + i]*x[col_bands+j];
}

//PLT_Mar21
//weights are shared between bands
static void sgemv_fcout32(float *out, const float *weights, int rows, int cols, int n_bands, const float *x)
{
   //int i, j, n, row_bands, col_bands;
   int i, j, k, n, row_bands, col_bands;
   float * restrict y;
   __m256 vy0, vy8, vy16, vy24, vxj;
   for (n=0,row_bands=0;n<n_bands;n++)
   {
      for (i=0,col_bands=n*cols;i<(rows-(rows%32));i+=32,row_bands+=32)
      {
         y = &out[row_bands];
         vy0 = _mm256_loadu_ps(&y[0]);
         vy8 = _mm256_loadu_ps(&y[8]);
         vy16 = _mm256_loadu_ps(&y[16]);
         vy24 = _mm256_loadu_ps(&y[24]);
         for (j=0;j<cols;j++)
         {
            vxj = _mm256_broadcast_ss(&x[col_bands+j]);

            k = j*rows + i;
            vy0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k]), vxj, vy0);
            vy8 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 8]), vxj, vy8);
            vy16 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 16]), vxj, vy16);
            vy24 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 24]), vxj, vy24);
         }
         _mm256_storeu_ps (&y[0], vy0);
         _mm256_storeu_ps (&y[8], vy8);
         _mm256_storeu_ps (&y[16], vy16);
         _mm256_storeu_ps (&y[24], vy24);
      }
      for (;i<rows;i++,row_bands++)
      {
         for (j=0;j<cols;j++)
            out[row_bands] += weights[j*rows + i]*x[col_bands+j];
      }
   }
   //for (n=0,row_bands=0;n<n_bands;n++)
   //   for (i=0,col_bands=n*cols;i<rows;i++,row_bands++)
   //      for (j=0;j<cols;j++)
   //         out[row_bands] += weights[j*rows + i]*x[col_bands+j];
}

//PLT_Mar21
static void sparse_sgemv_accum16(float *out, const float *weights, int rows, const int *idx, const float *x)
{
   int i, j, cols;
   float * restrict y;
   __m256 vy0, vy8, vxj;
   for (i=0;i<(rows-(rows%16));i+=16) //output side, 3*hidden_size, simultaneous computation in a block of 16 (2 256-bit registers)
   {
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      cols = *idx++;
      //input side, non-zero indices (sum(block_16) > 1e-10) recorded with dump_lpcnet.py printSparseVector
      for (j=0;j<cols;j++)
      {
         vxj = _mm256_broadcast_ss(&x[*idx++]);

         vy0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[0]), vxj, vy0);
         vy8 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[8]), vxj, vy8);

         weights += 16;
      }
      _mm256_storeu_ps (&y[0], vy0);
      _mm256_storeu_ps (&y[8], vy8);
   }
   for (;i<rows;i++)
   {
      for (j=0;j<cols;j++)
         out[i] += weights[j]*x[j];
   }
}

#endif /* VEC_AVX_H */
