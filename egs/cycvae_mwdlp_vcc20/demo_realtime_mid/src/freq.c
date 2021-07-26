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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Dec. 2020 - Jul. 2021,
   marked by PLT_<Dec20/Jul21> */

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


//PLT_Dec20
int dspstate_get_size()
{
    return sizeof(DSPState);
}


DSPState *dspstate_create()
{
    DSPState *dsp;
    dsp = (DSPState *) calloc(1,dspstate_get_size());
    if (dsp != NULL) {
        dsp->kfft = opus_fft_alloc_twiddles(FFT_LENGTH, NULL, NULL, NULL, 0);
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
    //printf("window\n");
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


//PLT_Jul21
void mel_spec_extract(DSPState *dsp, float *melsp)
{
    int i, j, k;
    //printf("in_melsp_a\n");
    opus_fft(dsp->kfft, dsp->in_fft, dsp->out_fft, 0); //STFT
    //printf("in_melsp_b\n");
    //cplx -> mag
    for (i=0;i<MAGSP_DIM;i++) {
        //need to multiply by 2.05*10^3 here to match the output of librosa STFT
        dsp->magsp[i] = sqrt(pow((dsp->out_fft[i].r*2050), 2) + pow((dsp->out_fft[i].i*2050), 2));
    //    printf("in_melsp_c %d %f\n", i, dsp->magsp[i]);
    }
    //printf("in_melsp_d\n");
    //mag -> mel --> log(1+10000*mel)
    //int j, k;
    for (i=0,k=0;i<MEL_DIM;i++) {
    //  printf("in_melsp_f1 %d\n", i);
      for (j=0,melsp[i]=0;j<MAGSP_DIM;j++,k++) {
    //        printf("in_melsp_g1 %d\n", j);
          melsp[i] += dsp->magsp[j]*dsp->melfb[k];
      //    printf("in_melsp_g1 %d %d %d %f %f\n", i, j, k, dsp->magsp[j], dsp->melfb[k+j]);
      }
      melsp[i] = log(1+10000*melsp[i]);
      //melsp[i] *= 10000;
    }
    //melsp[i] = log(1+10000*melsp[i]);
    //printf("done_melsp\n");
}
