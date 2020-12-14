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
/* Modified by Patrick Lumban Tobing (Nagoya University) on Dec. 2020,
   marked by PLT_Dec20 */

#ifndef _MWDLP10NET_CYCVAE_H_
#define _MWDLP10NET_CYCVAE_H_

#ifndef MWDLP10NET_CYCVAE_EXPORT
# if defined(WIN32)
#  if defined(MWDLP10NET_CYCVAE_BUILD) && defined(DLL_EXPORT)
#   define MWDLP10NET_CYCVAE_EXPORT __declspec(dllexport)
#  else
#   define MWDLP10NET_CYCVAE_EXPORT
#  endif
# elif defined(__GNUC__) && defined(MWDLP10NET_CYCVAE_BUILD)
#  define MWDLP10NET_CYCVAE_EXPORT __attribute__ ((visibility ("default")))
# else
#  define MWDLP10NET_CYCVAE_EXPORT
# endif
#endif


//PLT_Dec20
typedef struct CycleVAEPostMelspExcitSpkNetState CycleVAEPostMelspExcitSpkNetState;

//PLT_Dec20
typedef struct MWDLP10NetState MWDLP10NetState;

//PLT_Dec20
MWDLP10NET_CYCVAE_EXPORT int mwdlp10net_get_size();

//PLT_Dec20
MWDLP10NET_CYCVAE_EXPORT MWDLP10NetState *mwdlp10net_create();

//PLT_Dec20
MWDLP10NET_CYCVAE_EXPORT void mwdlp10net_destroy(MWDLP10NetState *mwdlp10net);

//PLT_Dec20
MWDLP10NET_CYCVAE_EXPORT void cyclevae_post_melsp_excit_spk_convert_mwdlp10net_synthesize(MWDLP10NetState *st,
    CycleVAEPostMelspExcitSpkNetState *cvst, const float *features, const float *spk_code,
        short *output, int *n_output, int flag_last_frame)

//PLT_Dec20
MWDLP10NET_CYCVAE_EXPORT void mwdlp10net_synthesize(MWDLP10NetState *st, const float *features, short *output,
    int *n_output, int flag_last_frame);

#endif
