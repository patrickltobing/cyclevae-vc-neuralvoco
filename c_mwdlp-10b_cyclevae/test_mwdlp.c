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
   Modified by Patrick Lumban Tobing (Nagoya University) on Sept.-Dec. 2020
   wav file read based on http://truelogic.org/wordpress/2015/09/04/parsing-a-wav-file-in-c
*/

#include <math.h>
#include <stdio.h>
#include <time.h>
#include "mwdlp10net_cycvae.h"
#include "freq.h"
#include "nnet_data.h"
#include "nnet_cv_data.h"
#include "mwdlp10net_cycvae_private.h"
#include "wave.h"


int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: test_mwdlp <cv_spk_idx> <input.wav> <output.pcm> or \n
                    test_mwdlp <cv_x_coord> <cv_y_coord> <input.wav> <output.pcm>\n");
        exit(1);
    } else {
        FILE *fin, *fout;
        if (argc == 4) { //exact point spk-code location
            fin = fopen(argv[2], "rb");
            if (fin == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[1]);
	            exit(1);
            }
            fout = fopen(argv[3], "wb");
            if (fout == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[2]);
	            exit(1);
            }
        } else { //interpolated spk-code location
            fin = fopen(argv[3], "rb");
            if (fin == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[1]);
	            exit(1);
            }
            fout = fopen(argv[4], "wb");
            if (fout == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[2]);
	            exit(1);
            }
        }

        //fseek(fin, 0, SEEK_END);
        //int size = ftell(fin);
        //fseek(fin, 0, SEEK_SET);
        //int n_frames = (int) round((float) size / (FEATURES_DIM * sizeof(float)));
        //size_t result, result_test = FEATURES_DIM;

        unsigned char buffer4[4];
        unsigned char buffer2[2];

        // WAVE header structure
        struct HEADER header;

        int read = 0;
        
        // read header parts
        read = fread(header.riff, sizeof(header.riff), 1, fin);
        printf("(1-4): %s \n", header.riff); 
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        
        // convert little endian to big endian 4 byte int
        header.overall_size  = buffer4[0] | 
                               (buffer4[1]<<8) | 
                               (buffer4[2]<<16) | 
                               (buffer4[3]<<24);
        
        printf("(5-8) Overall size: bytes:%u, Kb:%u \n", header.overall_size, header.overall_size/1024);
        
        read = fread(header.wave, sizeof(header.wave), 1, fin);
        printf("(9-12) Wave marker: %s\n", header.wave);
        
        read = fread(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fin);
        printf("(13-16) Fmt marker: %s\n", header.fmt_chunk_marker);
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        
        // convert little endian to big endian 4 byte integer
        header.length_of_fmt = buffer4[0] |
                                   (buffer4[1] << 8) |
                                   (buffer4[2] << 16) |
                                   (buffer4[3] << 24);
        printf("(17-20) Length of Fmt header: %u \n", header.length_of_fmt);
        
        read = fread(buffer2, sizeof(buffer2), 1, fin); printf("%u %u \n", buffer2[0], buffer2[1]);
        
        header.format_type = buffer2[0] | (buffer2[1] << 8);
        char format_name[10] = "";
        if (header.format_type == 1)
            strcpy(format_name,"PCM"); 
        else if (header.format_type == 6)
            strcpy(format_name, "A-law");
        else if (header.format_type == 7)
            strcpy(format_name, "Mu-law");
        
        printf("(21-22) Format type: %u %s \n", header.format_type, format_name);
        if (header.format_type != 1)
        {
            printf("Format is not PCM.\n");
            exit(1);
        }
        
        read = fread(buffer2, sizeof(buffer2), 1, fin);
        printf("%u %u \n", buffer2[0], buffer2[1]);
        
        header.channels = buffer2[0] | (buffer2[1] << 8);
        printf("(23-24) Channels: %u \n", header.channels);
        if (header.channels != 1)
        {
            printf("Channels are more than 1.\n");
            exit(1);
        }
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        
        header.sample_rate = buffer4[0] |
                               (buffer4[1] << 8) |
                               (buffer4[2] << 16) |
                               (buffer4[3] << 24);
        
        printf("(25-28) Sample rate: %u Hz\n", header.sample_rate);
        if (header.sample_rate != SAMPLING_RATE)
        {
            printf("Sampling rate is not %d Hz\n", SAMPLING_RATE);
            exit(1);
        }
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        
        header.byterate  = buffer4[0] |
                               (buffer4[1] << 8) |
                               (buffer4[2] << 16) |
                               (buffer4[3] << 24);
        printf("(29-32) Byte Rate: %u B/s , Bit Rate:%u b/s\n", header.byterate, header.byterate*8);
        
        read = fread(buffer2, sizeof(buffer2), 1, fin);
        printf("%u %u \n", buffer2[0], buffer2[1]);
        
        header.block_align = buffer2[0] |
                           (buffer2[1] << 8);
        printf("(33-34) Block Alignment: %u \n", header.block_align);
        
        read = fread(buffer2, sizeof(buffer2), 1, fin);
        printf("%u %u \n", buffer2[0], buffer2[1]);
        
        header.bits_per_sample = buffer2[0] |
                           (buffer2[1] << 8);
        printf("(35-36) Bits per sample: %u \n", header.bits_per_sample);
        
        read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fin);
        printf("(37-40) Data Marker: %s \n", header.data_chunk_header);
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        
        header.data_size = buffer4[0] |
                       (buffer4[1] << 8) |
                       (buffer4[2] << 16) | 
                       (buffer4[3] << 24 );
        printf("(41-44) Size of data chunk: %u B \n", header.data_size);
        
        // calculate no.of samples
        long num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);
        printf("Number of samples: %lu \n", num_samples);
        
        long size_of_each_sample = (header.channels * header.bits_per_sample) / 8;
        printf("Size of each sample: %ld B\n", size_of_each_sample);
        
        // calculate duration of file
        float duration_in_seconds = (float) header.overall_size / header.byterate;
        printf("Approx.Duration in seconds= %f\n", duration_in_seconds);
        //printf("Approx.Duration in h:m:s=%s\n", seconds_to_time(duration_in_seconds));
        
        // read each sample from data chunk if PCM
        if (header.format_type == 1 && header.channels == 1) { // PCM and mono
            long i =0;
            char data_buffer[size_of_each_sample];
            
            // make sure that the bytes-per-sample is completely divisible by num.of channels
            long bytes_in_each_channel = (size_of_each_sample / header.channels);
            if ((bytes_in_each_channel  * header.channels) != size_of_each_sample) {
                printf("Error: %ld x %ud <> %ld\n", bytes_in_each_channel, header.channels, size_of_each_sample);
                exit(1);
            }
            
            // the valid amplitude range for values based on the bits per sample
            long low_limit = 0l;
            long high_limit = 0l;
            
            switch (header.bits_per_sample) {
                case 8:
                    low_limit = -128;
                    high_limit = 127;
                    break;
                case 16:
                    low_limit = -32768;
                    high_limit = 32767;
                    break;
                case 32:
                    low_limit = -2147483648;
                    high_limit = 2147483647;
                    break;
            }                   
            
            int data_in_channel = 0;
            printf("nn.Valid range for data values : %ld to %ld \n", low_limit, high_limit);

            t = clock();
            srand (time(NULL));
            DSPState *dsp;
            MWDLP10CycleVAEPostMelspExcitSpkNetState *net;

            float features[FEATURES_DIM];
            short pcm[MAX_N_OUTPUT];
            int waveform_buffer_flag = 0;
            int n_output = 0;
            int samples = 0;
            clock_t t;

            // waveform-->features processing struct
            dsp = dspstate_create();

            // initialize mwdlp+cyclevae struct
            net = mwdlp10cyclevaenet_create();

            for (i =1; i <= num_samples; i++) {
                //printf("==========Sample %ld / %ld=============\n", i, num_samples);
                read = fread(data_buffer, sizeof(data_buffer), 1, fin);
                if (read == 1) {
                
                    /* Receives only mono 16-bit PCM */
                    data_in_channel = (data_buffer[0] & 0x00ff) | (data_buffer[1] << 8);
            
                    //printf("%d ", data_in_channel);
                    
                    //check waveform buffer here

                    if (waveform_buffer_flag) {
                        //extract melspectrogram here
                        mel_spec_extract(dsp, features);
            
                        cyclevae_post_melsp_excit_spk_convert_mwdlp10net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0);
                        //// check if value was in range
                        //if (data_in_channel < low_limit || data_in_channel > high_limit)
                        //    printf("**value out of range\n");
            
                        //printf("\n");
                        if (n_output > 0)  {
                            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        }
                        samples += n_output;
                    }
                } else {
                    printf("Error reading file. %d bytes\n", read);
                    fclose(fin);
                    fclose(fout);
                    dspstate_destroy(dsp);
                    mwdlp10cyclevaenet_destroy(net);
                    exit(1);
                }
            }
        }
        
        if (waveform_buffer_flag) {
            cyclevae_post_melsp_excit_spk_convert_mwdlp10net_synthesize(net, features, spk_code_aux, pcm, &n_output, 1); //last_frame, synth pad_right
            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
    
            t = clock() - t;
            double time_taken = ((double)t)/CLOCKS_PER_SEC;
            samples += n_output;
            //printf("%d [frames], %d [samples] synthesis in %f seconds \n", idx, samples, time_taken);
            printf("%d [frames] %d [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
                "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
                (int)((double)samples/120), samples, (double)samples/24000, time_taken,
                    ((double)samples/24000)/time_taken, time_taken/((double)samples/24000),
                        24*((double)samples/24000)/time_taken);
        }

        fclose(fin);
        fclose(fout);
        dspstate_destroy(dsp);
        mwdlp10cyclevaenet_destroy(net);

        return 0;
}
