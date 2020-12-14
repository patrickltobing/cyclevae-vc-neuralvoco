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

#include "kiss_fft.h"

//PLT_Dec20
#define SAMPLING_FREQUENCY 16000 //fs
#define FRAME_SHIFT 80 //int((fs/1000)*shiftms); shiftms = 5 ms
#define WINDOW_LENGTH 440 //int((fs/1000)*winms); winms = 27.5 ms
#define FFT_LENGTH 1024 //fs=8kHz-16kHz: 1024; 22.05kHz-24kHz: 2048; 44.1kHz-48kHz: 4096

#define WINDOW_LENGTH_1 (WINDOW_LENGTH - 1)
#define MOD_WINDOW_LENGTH (WINDOW_LENGTH % 2)

#define WIN_PAD (FFT_LENGTH - WINDOW_LENGTH) //window is centered on total FFT length
#define WIN_PAD_LEFT (WIN_PAD / 2)
#define WIN_PAD_RIGHT (WIN_PAD_LEFT + (WIN_PAD % 2)) //right pad is more than 1 if total pad is odd

#define LEFT_REFLECT (WINDOW_LENGTH / 2) //centered-position at t [0..T-1] -> t*frame_shift
#define RIGHT_REFLECT (LEFT_REFLECT - FRAME_SHIFT + (WINDOW_LENGTH % 2)) //add 1 more if win_length odd
#define FIRST_SAMPLES_DELAY (WINDOW_LENGTH - LEFT_REFLECT) //first minimum samples to receive [delay]

#define WIN_LEFT_IDX WIN_PAD_LEFT //0->439, index of centered 1st in total FFT-length
#define WIN_RIGHT_IDX (WIN_LEFT_IDX + WINDOW_LENGTH - 1) //0->439, index of centered 440th in total FFT-length

#define BUFFER_LENGTH (WINDOW_LENGTH - FRAME_SHIFT) //store samples for proceeding frame

#define HPASS_FILT_TAPS 1023 //order+1, has to be odd because high-pass filter passes nyq. freq.
#define HPASS_FILT_TAPS_1 (HPASS_FILT_TAPS - 1)  //order+1, has to be odd because high-pass filter passes nyq. freq.

#define MAGSP_DIM (FFT_LENGTH / 2 + 1)
#define MEL_DIM 80
#define MEL_DIM_16_BLOCK ((MEL_DIM % 16) == 0)


//PLT_Dec20
typedef struct {
    kiss_fft_state *kfft;
    float hpass_filt[HPASS_FILT_TAPS];
    float half_window[LEFT_REFLECT];
    float samples_hpass[HPASS_FILT_TAPS];
    float samples_win[WINDOW_LENGTH];
    kiss_fft_cpx in_fft[FFT_LENGTH]; //initialized with zeros, fill in only centered window_length
    kiss_fft_cpx out_fft[FFT_LENGTH];
    float magsp[MAGSP_DIM];
    float melfb[MEL_DIM];
} DSPState;

int dspstate_get_size();

DSPState *dspstate_create();

void dspstate_destroy(DSPState *dsp);

void shift_apply_hpassfilt(DSPState *dsp, float *x);

void apply_window(DSPState *dsp);

void shift_apply_window(DSPState *dsp, const float *x);

void mel_spec_extract(DSPState *dsp, float *melsp);

