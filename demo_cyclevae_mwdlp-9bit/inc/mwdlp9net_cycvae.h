/* Copyright (c) 2018 Mozilla */
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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Dec. 2020 - Jan. 2021,
   marked by PLT_<Dec20/Jan21> */

//PLT_Jan21
#ifndef _MWDLP9NET_CYCVAE_H_
#define _MWDLP9NET_CYCVAE_H_

#ifndef MWDLP9NET_CYCVAE_EXPORT
# if defined(WIN32)
#  if defined(MWDLP9NET_CYCVAE_BUILD) && defined(DLL_EXPORT)
#   define MWDLP9NET_CYCVAE_EXPORT __declspec(dllexport)
#  else
#   define MWDLP9NET_CYCVAE_EXPORT
#  endif
# elif defined(__GNUC__) && defined(MWDLP9NET_CYCVAE_BUILD)
#  define MWDLP9NET_CYCVAE_EXPORT __attribute__ ((visibility ("default")))
# else
#  define MWDLP9NET_CYCVAE_EXPORT
# endif
#endif


//PLT_Jan21
typedef struct MWDLP9CycleVAEPostMelspExcitSpkNetState MWDLP9CycleVAEPostMelspExcitSpkNetState;

typedef struct MWDLP9NetState MWDLP9NetState;

MWDLP9NET_CYCVAE_EXPORT int mwdlp9cyclevaenet_get_size();

MWDLP9NET_CYCVAE_EXPORT MWDLP9CycleVAEPostMelspExcitSpkNetState *mwdlp9cyclevaenet_create();

MWDLP9NET_CYCVAE_EXPORT void mwdlp9cyclevaenet_destroy(MWDLP9CycleVAEPostMelspExcitSpkNetState *mwdlp9cyclevaenet);

MWDLP9NET_CYCVAE_EXPORT int mwdlp9cyclevaenet_get_size();

MWDLP9NET_CYCVAE_EXPORT MWDLP9NetState *mwdlp9net_create();

MWDLP9NET_CYCVAE_EXPORT void mwdlp9net_destroy(MWDLP9NetState *mwdlp9net);

MWDLP9NET_CYCVAE_EXPORT void cyclevae_post_melsp_excit_spk_convert_mwdlp9net_synthesize(
    MWDLP9CycleVAEPostMelspExcitSpkNetState *st, float *features, float *spk_code_aux,
        short *output, int *n_output, int flag_last_frame);
        //short *output, int *n_output, int flag_last_frame, short *pcm_1, short *pcm_2, short *pcm_3, short *pcm_4, float *gru_tmp);
        //short *output, int *n_output, int flag_last_frame, short *pcm_1, short *pcm_2, short *pcm_3, short *pcm_4, float *conv_tmp, float *gru_tmp);
        //short *output, int *n_output, int flag_last_frame, float *melsp_cv, short *pcm_1, short *pcm_2, short *pcm_3, short *pcm_4);
        //short *output, int *n_output, int flag_last_frame, float *melsp_cv, float *lat_tmp, float *spk_tmp, float *conv_tmp, float *gru_tmp, float *f0_tmp);
        //short *output, int *n_output, int flag_last_frame, float *melsp_cv);
    //MWDLP9CycleVAEPostMelspExcitSpkNetState *st, const float *features, float *spk_code_aux,

MWDLP9NET_CYCVAE_EXPORT void mwdlp9net_synthesize(MWDLP9NetState *st, const float *features,
    short *output, int *n_output, int flag_last_frame);

#endif
