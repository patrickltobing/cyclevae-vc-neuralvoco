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
#include "halfwin.h" //get half hanning window coefficients [because of symmetry] w/ numpy from dump script
#include "melfb.h" //get mel-filterbank w/ numpy from dump script

#ifdef __AVX__
#include "vec_avx.h"
#elif __ARM_NEON__
#include "vec_neon.h"
#else
#warning Compiling melspec matrix multiplication without any vectorization.
#include "vec.h"
#endif


//PLT_Dec20
int dspstate_getsize()
{
    return sizeof(DSPState);
}


DSPState *dspstate_create()
{
    DSPState *dsp;
    dsp = (DSPState *) calloc(1,dspstate_get_size());
    if (dsp != NULL) {
        int i;
        for (i=0;i<HPASS_FILT_TAPS;i++)
            dsp->hpass_filt[i] = hpassfilt[i]
        for (i=0;i<LEFT_REFLECT;i++)
            dsp->half_window[i] = halfwin[i]
        for (i=0;i<MEL_DIM;i++)
            dsp->melfb[i] = melfb[i]
        return dsp;
    } else {
        printf("Cannot allocate and initialize memory for DSPState.\n");
        exit(EXIT_FAILURE);
    }
}


void dspstate_destroy(DSPState *dsp)
{
    if (dsp != NULL) free(dsp);
}


void shift_apply_hpassfilt(DSPState *dsp, float *x)
{
    RNN_MOVE(dsp->samples_hpass, &dsp->samples_hpass[1], HPASS_FILT_TAPS_1); //shift buffer to the left
    dsp->samples_hpass[HPASS_FILT_TAPS_1] = *x; //add new sample at the right
    for (int i=0,*x=0;i<HPASS_FILT_TAPS;i++)
        *x += dsp->samples_hpass[HPASS_FILT_TAPS_1-i] * dsp->hpass_filt[i]; //filter convolution
}


void apply_window(DSPState *dsp)
{
    for (int i=0;i<LEFT_REFLECT;i++) { //window [symmetric] centered on FFT length
        //zero pad left
        dsp->in_fft[i+WIN_PAD_LEFT_IDX].r = dsp->samples_win[i]*dsp->half_window[i];
        //zero pad right [mirrored of left]
        dsp->in_fft[WIN_RIGHT_IDX-i].r = dsp->samples_win[WINDOW_LENGTH_1-i]*dsp->half_window[i];
    }
    if (MOD_WINDOW_LENGTH > 0) //window length is odd, exists 1 centered sample
        dsp->in_fft[i] = dsp->samples_win[i];
}


void shift_apply_window(DSPState *dsp, const float *x)
{
    RNN_MOVE(dsp->samples_win, dsp->samples_win, BUFFER_LENGTH); //shift buffer to the left
    RNN_COPY(&(dsp->samples_win[BUFFER_LENGTH]), x, FRAME_SHIFT); //add new samples to the right
    for (int i=0;i<LEFT_REFLECT;i++) { //window [symmetric] centered on FFT length
        //zero pad left
        dsp->in_fft[i+WIN_PAD_LEFT_IDX].r = dsp->samples_win[i]*dsp->half_window[i];
        //zero pad right [mirrored of left]
        dsp->in_fft[WIN_RIGHT_IDX-i].r = dsp->samples_win[WINDOW_LENGTH_1-i]*dsp->half_window[i];
    }
    if (MOD_WINDOW_LENGTH > 0) //window length is odd, exists 1 centered sample
        dsp->in_fft[i] = dsp->samples_win[i];
}


void mel_spec_extract(DSPState *dsp, float *melsp)
{
    int i;
    opus_fft(dsp->kfft, dsp->in_fft, dsp->out_fft, 0); //STFT w/ Cooley-Tukey FFT
    //cplx -> mag
    for (i=0;i<MAGSP_DIM;i++)
        dsp->magsp[i] = sqrt((dsp->out_fft[i].r)**2 + (dsp->out_fft[i].i)**2)
    //mag -> mel
    if (MEL_DIM_16_BLOCK) //compute melfb*magsp matmul in a block of 16
       sgemv_accum16(melsp, dsp->melfb, MEL_DIM, MAGSP_DIM, MEL_DIM, dsp->magsp);
    else {
        int j;
        for (i=0;i<MEL_DIM;i++)
            for (j=0,melsp[i]=0;j<MAGSP_DIM;j++)
                melsp[i] += dsp->magsp[j]*dsp->melfb[i];
    }
}
