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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Sept. 2020 - Aug. 2021,
   marked by PLT_<MonthYear> */

#ifndef VEC_AVX_H
#define VEC_AVX_H

#include <immintrin.h>

//PLT_Jul21
//#ifndef __AVX2__
//#define _mm256_fmadd_ps(a,b,c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
//#define _mm_fmadd_ps(a,b,c) _mm_add_ps(_mm_mul_ps(a, b), c)
//#endif

//PLT_Aug21
/*
    based on https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx/48869291#48869291
    slightly modified to bypass intermediate variables
    throughput:
    6 fmadd ~3c, 2 mul ~1c, 2 add_ps ~1c, 2 add_epi32 ~0.67c, 1 slli_epi32 ~0.5c, 1 cvttps_epi32 ~0.5c, 1 max ~0.5c, 1 round ~1c --> ~8.17 C
    latency (dependency):
    1 max ~4l -> 1 round ~8l -> 1 sub ~4l --> [3 fmadd ~12l -> 1 mul ~4l -> 1 fmadd ~4l -> 1 add ~4l] || [1 cvttps_epi32 0.5l -> add_epi32 1l -> 1 slli_epi32 ~1] -> 1 mul ~4l ~44 C latency
*/ 
static inline __m256 exp256_ps(__m256 x, __m256 fx)
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
      converted to plain AVX2 intrinsics code without ingenious macro's, gcc style alignment attributes etc.
      expression exp(x) as exp(g + n*log(2))" has been significantly simplified.
    */

    x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f)), _mm256_set1_ps(-88.3762626647949f));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693147180559945f)));

    /* build 2^n */
    return _mm256_mul_ps(_mm256_add_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(1.9875691500E-4), x, _mm256_set1_ps(1.3981999507E-3)), x, _mm256_set1_ps(8.3334519073E-3)), x, _mm256_set1_ps(4.1665795894E-2)), x, _mm256_set1_ps(1.6666665459E-1)), x, _mm256_set1_ps(5.0000001201E-1)), _mm256_mul_ps(x, x), x), _mm256_set1_ps(1.0f)),
                            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(0x7f)), 23)));
}

//PLT_Aug21
/*
    reciprocal correction based on polynomial third order https://github.com/stgatilov/recip_rsqrt_benchmark/blob/master/routines_sse.h#L133
    https://github.com/stgatilov/recip_rsqrt_benchmark
    recip_float4_ieee: maximal error = 0 [_mm_div_ps]
    recip_float4_ieee: cycles per call = 7.30
    recip_float4_fast: maximal error = 0.000300257 [_mm_rcp_ps]
    recip_float4_fast: cycles per call = 1.13
    recip_double2_ieee: maximal error = 0 [_mm256_div_pd]
    recip_double2_ieee: cycles per call = 14.17
    recip_double2_fast: maximal error = 0.000300175 [_mm256_rcp_ps w/ conversion]
    recip_double2_fast: cycles per call = 3.36
    recip_double2_r3: maximal error = 2.70471e-011
    recip_double2_r3: cycles per call = 8.94
    throughput:
    1 rcp ~1c, 3 mul ~1.5c, 2 add ~1c, 1 sub ~0.5c --> ~4 C throughput
    latency (dependency):
    1 rcp ~4c -> 1 mul ~4c -> 1 sub ~4c -> 1 mul ~4c -> 1 add ~4c -> 1 mul ~4c -> 1 add ~4c --> ~28 C latency
*/
static inline __m256 rcp256_r3_ps(__m256 x, __m256 a, __m256 r, const __m256 one)
{
    a = _mm256_rcp_ps(x);
    r = _mm256_sub_ps(one, _mm256_mul_ps(x, a));
    return _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(r, r), r), a), a); // (r^2 + r) * a + a
}

//PLT_Aug21
static inline void vec_exp(float* x, int N)
{
    for (register int i = 0; i < N; i++) {
        if (x[i] > -102.87347) {
            if (x[i] < 88.72283) x[i] = exp(x[i]);
            else x[i] = 3.40279851902147610656242037972608745472E+38;
        } else x[i] = 1.40129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125E-45;
    }
}

//PLT_Aug21
static inline void vec_tanh(float* x, int N)
{
    register int i;
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 min_one = _mm256_set1_ps(-1.f);
    const __m256 max_eps = _mm256_set1_ps(8.31777f);
    const __m256 min_eps = _mm256_set1_ps(-8.31777f);
    __m256 x8, Y, rcp_Y, fx, a, r;
    for (i = 0; i < N - 7; i += 8)
    {
        x8 = _mm256_loadu_ps(&x[i]);
        Y = exp256_ps(x8, fx);
        rcp_Y = rcp256_r3_ps(Y, a, r, one);
        _mm256_storeu_ps(&x[i], _mm256_blendv_ps(one,
                                    _mm256_blendv_ps(min_one,
                                            _mm256_mul_ps(_mm256_sub_ps(Y, rcp_Y), rcp256_r3_ps(_mm256_add_ps(Y, rcp_Y), a, r, one)),
                                                _mm256_cmp_ps(x8, min_eps, _CMP_GT_OQ)),
                                                    _mm256_cmp_ps(x8, max_eps, _CMP_LT_OQ)));
    }
    for (; i < N; i++)
    {
        if (x[i] < 9.01092) {
            if (x[i] > -9.01092) {
                x[i] = tanh(x[i]);
            } else x[i] = -1;
        } else x[i] = 1;
    }
}

//PLT_Aug21
static inline void vec_tanh_exp(float* x, int N)
{
    for (register int i = 0; i < N; i++)
    {
        //x = 9.01092 v1 tanh(x) 1 max
        //x = 8.31777 v2 (e^x-1/e^x) / (e^x+1/e^x) 1 max
        //x = -8.31777 v2 -1 min
        //x = -9.01092 v1 -1 min
        if (x[i] < 9.01092) { //v1
            if (x[i] > -9.01092) {
                x[i] = tanh(x[i]);
            } else x[i] = -1;
        } else x[i] = 1;
    }
}

//PLT_Aug21
static inline void vec_tanhshrink(float* x, int N)
{
    for (register int i = 0; i < N; i++)
    {
        if (x[i] < 9.01092) {
            if (x[i] > -9.01092) {
                x[i] = x[i]-tanh(x[i]);
            } else x[i] = x[i]+1;
        } else x[i] = x[i]-1;
    }
}

//PLT_Aug21
static inline void vec_sigmoid(float* x, int N)
{
    register int i;
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 zero = _mm256_set1_ps(0.f);
    const __m256 min_eps = _mm256_set1_ps(-87.33656f);
    __m256 fx, a, r, x8;
    for (i = 0; i < N - 7; i += 8)
    { //v4
        //x = 7.62462; avx_v1 [1/(1+1/e^x)] 9.997559E-01 max
        //x = -87.33656; avx_v1 0 min
        //_mm256_storeu_ps(&y[i], _mm256_rcp_ps(_mm256_add_ps(_mm256_rcp_ps(exp256_ps(_mm256_loadu_ps(&x[i]), fx)), one)));
        //x = 17.32868; avx_v2 [1-1/(e^x+1)] 1 max
        //x = -7.62475; avx_v2 2.441406E-04 min
        //_mm256_storeu_ps(&y[i], _mm256_sub_ps(one, _mm256_rcp_ps(_mm256_add_ps(exp256_ps(_mm256_loadu_ps(&x[i]), fx), one))));
        //x = 16.63554; avx_v3 w/ 3rd order poly outer rcp [1/(1+1/e^x)] 1 max
        //x = -87.33656; avx_v3 1.175781476877627662986568809413861781633964882443978019590384729569786277647569505688807112164795398712158203125E-38 min --> blend with mask to zero
        //_mm256_rcp_ps of eps_min = 8.504982254280047655532952987262517248E+37
        //_mm256_rcp_ps of INF = 0
        //x = 16.63554; avx_v4 w/ 3rd order poly both rcp improves small number x [1/(1+1/e^x)] 1 max
        //x = -87.33656; avx_v4 + mask 0 min
        x8 = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&x[i], _mm256_blendv_ps(zero,
                                   rcp256_r3_ps(_mm256_add_ps(rcp256_r3_ps(exp256_ps(x8, fx), a, r, one), one), a, r, one),
                                       _mm256_cmp_ps(x8, min_eps, _CMP_GT_OQ)));
    }
    for (; i < N; i++)
    {
        if (x[i] < 17.32868) {
            if (x[i] > -103.97209) x[i] = 1 / (1 / exp(x[i]) + 1);
            else x[i] = 0;
        } else x[i] = 1;
    }
}

//PLT_Aug21
static inline void vec_sigmoid_exp(float* x, int N)
{
    for (register int i = 0; i < N; i++)
    {
        //x = 17.32868; v1 [1/(1/e^x+1)], v2 [1-1/(e^x+1)] 1 max
        //x = 16.63554; v1 store exp 1 max
        //x = -16.63554; v2 store exp 0 min
        //x = -36.7368; v2 0 min
        //x = -88.72284; v1 store exp 0 min
        //x = -103.97209; v1 0 min
        if (x[i] < 17.32868) { //v1 no store exp
            if (x[i] > -103.97209) x[i] = 1 / (1 / exp(x[i]) + 1);
            else x[i] = 0;
        } else x[i] = 1;
    }
}

//PLT_Aug21
//weight vector is written as (kernel_size x in) x out
static inline void sgemv_accum16(float* out, const float* weights, int rows, int cols, const float* x)
{
    register int i, j, k;
    register float* restrict y;
    __m256 vy0, vy8, vxj;
    for (i = 0; i < (rows - (rows % 16)); i += 16)
    {
        y = &out[i];
        vy0 = _mm256_loadu_ps(&y[0]);
        vy8 = _mm256_loadu_ps(&y[8]);
        for (j = 0; j < cols; j++)
        {
            vxj = _mm256_broadcast_ss(&x[j]);

            k = j * rows + i;
            vy0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k]), vxj, vy0);
            vy8 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 8]), vxj, vy8);
        }
        _mm256_storeu_ps(&y[0], vy0);
        _mm256_storeu_ps(&y[8], vy8);
    }
    for (; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
            out[i] += weights[j * rows + i] * x[j];
    }
}

//PLT_Aug21
static inline void sgev_dualfc8(float* out, const float* factors, int rows, const float* x)
{
    //int i, half_rows;
    register int i, j, half_rows;
    half_rows = rows / 2;
    if (half_rows % 8 == 0) {
        for (i = 0; i < half_rows; i += 8) //1st channel
            _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_loadu_ps(&factors[i]), _mm256_loadu_ps(&x[i])));
        for (j = 0; i < rows; i += 8, j += 8) //2nd channel
            _mm256_storeu_ps(&out[j], _mm256_fmadd_ps(_mm256_loadu_ps(&factors[i]), _mm256_loadu_ps(&x[i]), _mm256_loadu_ps(&out[j])));
    }
    else {
        int half_rows_8 = half_rows / 8 * 8;
        for (i = 0; i < half_rows_8; i += 8) //1st channel
            _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_loadu_ps(&factors[i]), _mm256_loadu_ps(&x[i])));
        for (; i < half_rows; i++)
            out[i] = factors[i] * x[i];
        for (j = 0; i < (rows - (half_rows % 8)); i += 8, j += 8) //2nd channel
            _mm256_storeu_ps(&out[j], _mm256_fmadd_ps(_mm256_loadu_ps(&factors[i]), _mm256_loadu_ps(&x[i]), _mm256_loadu_ps(&out[j])));
        for (; i < rows; i++, j++)
            out[j] += factors[i] * x[i];
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

//PLT_Aug21
//weights are shared between bands
static inline void sgemv_fclogits16(float* out, const float* weights, int rows, int cols, int n_bands, const float* x)
{
    //int i, j, n, row_bands, col_bands;
    register int i, j, k, n, row_bands, col_bands;
    register float* restrict y;
    __m256 vy0, vy8, vxj;
    for (n = 0, row_bands = 0; n < n_bands; n++)
    {
        for (i = 0, col_bands = n * cols; i < (rows - (rows % 16)); i += 16, row_bands += 16)
        {
            y = &out[row_bands];
            vy0 = _mm256_loadu_ps(&y[0]);
            vy8 = _mm256_loadu_ps(&y[8]);
            for (j = 0; j < cols; j++)
            {
                vxj = _mm256_broadcast_ss(&x[col_bands + j]);

                k = j * rows + i;
                vy0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k]), vxj, vy0);
                vy8 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 8]), vxj, vy8);
            }
            _mm256_storeu_ps(&y[0], vy0);
            _mm256_storeu_ps(&y[8], vy8);
        }
        for (; i < rows; i++, row_bands++)
        {
            for (j = 0; j < cols; j++)
                out[row_bands] += weights[j * rows + i] * x[col_bands + j];
        }
    }
    //for (n=0,row_bands=0;n<n_bands;n++)
    //   for (i=0,col_bands=n*cols;i<rows;i++,row_bands++)
    //      for (j=0;j<cols;j++)
    //         out[row_bands] += weights[j*rows + i]*x[col_bands+j];
}

//PLT_Aug21
//weights are shared between bands
static inline void sgemv_fcout32(float* out, const float* weights, int rows, int cols, int n_bands, const float* x)
{
    //int i, j, n, row_bands, col_bands;
    register int i, j, k, n, row_bands, col_bands;
    register float* restrict y;
    __m256 vy0, vy8, vy16, vy24, vxj;
    for (n = 0, row_bands = 0; n < n_bands; n++)
    {
        for (i = 0, col_bands = n * cols; i < (rows - (rows % 32)); i += 32, row_bands += 32)
        {
            y = &out[row_bands];
            vy0 = _mm256_loadu_ps(&y[0]);
            vy8 = _mm256_loadu_ps(&y[8]);
            vy16 = _mm256_loadu_ps(&y[16]);
            vy24 = _mm256_loadu_ps(&y[24]);
            for (j = 0; j < cols; j++)
            {
                vxj = _mm256_broadcast_ss(&x[col_bands + j]);

                k = j * rows + i;
                vy0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k]), vxj, vy0);
                vy8 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 8]), vxj, vy8);
                vy16 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 16]), vxj, vy16);
                vy24 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[k + 24]), vxj, vy24);
            }
            _mm256_storeu_ps(&y[0], vy0);
            _mm256_storeu_ps(&y[8], vy8);
            _mm256_storeu_ps(&y[16], vy16);
            _mm256_storeu_ps(&y[24], vy24);
        }
        for (; i < rows; i++, row_bands++)
        {
            for (j = 0; j < cols; j++)
                out[row_bands] += weights[j * rows + i] * x[col_bands + j];
        }
    }
    //for (n=0,row_bands=0;n<n_bands;n++)
    //   for (i=0,col_bands=n*cols;i<rows;i++,row_bands++)
    //      for (j=0;j<cols;j++)
    //         out[row_bands] += weights[j*rows + i]*x[col_bands+j];
}

//PLT_Aug21
static inline void sparse_sgemv_accum16(float* out, const float* weights, int rows, const int* idx, const float* x)
{
   register int i, j, cols;
   register float* restrict y;
   __m256 vy0, vy8, vxj;
   for (i = 0; i < (rows - (rows % 16)); i += 16) //output side, 3*hidden_size, simultaneous computation in a block of 16 (2 256-bit registers)
   {
       y = &out[i];
       vy0 = _mm256_loadu_ps(&y[0]);
       vy8 = _mm256_loadu_ps(&y[8]);
       cols = *idx++;
       //input side, non-zero indices (sum(block_16) > 1e-10) recorded with dump_lpcnet.py printSparseVector
       for (j = 0; j < cols; j++)
       {
           vxj = _mm256_broadcast_ss(&x[*idx++]);

           vy0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[0]), vxj, vy0);
           vy8 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[8]), vxj, vy8);

           weights += 16;
       }
       _mm256_storeu_ps(&y[0], vy0);
       _mm256_storeu_ps(&y[8], vy8);
   }
}

#endif /* VEC_AVX_H */
