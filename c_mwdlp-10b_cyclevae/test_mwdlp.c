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
/* Based on test_lpcnet.c.
   Modified by Patrick Lumban Tobing (Nagoya University) on Sept.-Dec. 2020 */

#include <math.h>
#include <stdio.h>
#include <time.h>
#include "arch.h"
#include "lpcnet.h"
#include "freq.h"
#include "nnet_data.h"
#include "lpcnet_private.h"


int main(int argc, char **argv) {
    srand (time(NULL));
    FILE *fin, *fout;
    //FILE *fout_1, *fout_2, *fout_3, *fout_4, *fout_5;
    MBDLP16NetState *net;
    net = mbdlp16net_create();
    if (argc != 3)
    {
        fprintf(stderr, "usage: test_lpcnet <features.f32> <output.pcm>\n");
        return 0;
    }
    fin = fopen(argv[1], "rb");
    if (fin == NULL) {
	fprintf(stderr, "Can't open %s\n", argv[1]);
	exit(1);
    }

    fout = fopen(argv[2], "wb");
    if (fout == NULL) {
	fprintf(stderr, "Can't open %s\n", argv[2]);
	exit(1);
    }
    //fout_1 = fopen("b1.pcm", "wb");
    //fout_2 = fopen("b2.pcm", "wb");
    //fout_3 = fopen("b3.pcm", "wb");
    //fout_4 = fopen("b4.pcm", "wb");
    //fout_5 = fopen("b5.pcm", "wb");

    //fseek(fin, 0, SEEK_END);
    //int size = ftell(fin);
    //fseek(fin, 0, SEEK_SET);
    //int n_frames = (int) round((float) size / (FEATURES_DIM * sizeof(float)));
    float features[FEATURES_DIM];
    short pcm[MAX_N_OUTPUT];
    //short pcm_1[MAX_N_OUTPUT];
    //short pcm_2[MAX_N_OUTPUT];
    //short pcm_3[MAX_N_OUTPUT];
    //short pcm_4[MAX_N_OUTPUT];
    //short pcm_5[MAX_N_OUTPUT];
    int n_output;
    //int idx = 0, samples = 0;
    int samples = 0;
    //int i;
    //size_t result, result_test = FEATURES_DIM;
    clock_t t;
    t = clock();
    while (1) {
    //    printf("a %d\n", idx);
        //fread(features, sizeof(features[0]), FEATURES_DIM, fin);
        fread(features, sizeof(features[0]), FEATURES_DIM, fin);
        //for (i=0;i<FEATURES_DIM;i++) {
        //    printf("[%d][%d] %f\n", idx, i, features[i]);
        //}
    //    printf("b %d\n", idx);
        if (feof(fin)) break;
    //    if (result != result_test) printf("result: %ld != result_test: %ld\n", result, result_test);
    //    printf("c %d\n", idx);
        //mbdlp16net_synthesize(net, features, pcm, &n_output, 0);
        //mbdlp16net_synthesize(net, features, pcm, &n_output, 0, pcm_1, pcm_2, pcm_3, pcm_4, pcm_5);
        mbdlp16net_synthesize(net, features, pcm, &n_output, 0);
    //    printf("d %d\n", idx);
        if (n_output > 0)  {
    //        for (i=0;i<n_output;i++) printf("[%d] %d\n", i, pcm[i]);
            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
    //        break;
    //        fwrite(pcm_1, sizeof(pcm[0]), N_SAMPLE_BANDS, fout_1);
    //        fwrite(pcm_2, sizeof(pcm[0]), N_SAMPLE_BANDS, fout_2);
    //        fwrite(pcm_3, sizeof(pcm[0]), N_SAMPLE_BANDS, fout_3);
    //        fwrite(pcm_4, sizeof(pcm[0]), N_SAMPLE_BANDS, fout_4);
    //        fwrite(pcm_5, sizeof(pcm[0]), N_SAMPLE_BANDS, fout_5);
        }
   //     printf("e %d\n", idx);
   //     n_output = 0;
    //    idx += 1;
        samples += n_output;
   //     printf("f %d\n", idx);
    }
    //printf("g %d\n", idx);
    mbdlp16net_synthesize(net, features, pcm, &n_output, 1); //last_frame, synth pad_right
    fwrite(pcm, sizeof(pcm[0]), n_output, fout);
    //samples += n_output;
    //printf("h %d\n", idx);
    //if (n_output > 0) fwrite(pcm, sizeof(pcm[0]), n_output, fout);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    //printf("%d [frames], %d [samples] synthesis in %f seconds \n", idx, samples, time_taken);
    samples += n_output;
    printf("%d [frames] %d [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
        "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
        (int)((double)samples/120), samples, (double)samples/24000, time_taken,
            ((double)samples/24000)/time_taken, time_taken/((double)samples/24000),
                24*((double)samples/24000)/time_taken);
    //printf("synthesis in %f seconds \n", time_taken);
    //printf("i %d %d\n", idx, samples);
    fclose(fin);
    //printf("j %d %d\n", idx, samples);
    fclose(fout);
    //fclose(fout_1);
    //fclose(fout_2);
    //fclose(fout_3);
    //fclose(fout_4);
    //fclose(fout_5);
    //printf("k %d %d\n", idx, n_output);
    mbdlp16net_destroy(net);
    //printf("l %d %d\n", idx, n_output);
    return 0;
}
