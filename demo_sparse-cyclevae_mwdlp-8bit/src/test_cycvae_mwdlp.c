/*
   Copyright 2021 Patrick Lumban Tobing (Nagoya University)
   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
   WAV file read based on http://truelogic.org/wordpress/2015/09/04/parsing-a-wav-file-in-c
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mwdlp8net_cycvae.h"
#include "freq.h"
#include "nnet.h"
#include "nnet_data.h"
#include "nnet_cv_data.h"
#include "mwdlp8net_cycvae_private.h"
#include "wave.h"


int main(int argc, char **argv) {
    if (argc != 3 && argc != 4 && argc != 5) {
        fprintf(stderr, "usage: test_mwdlp <input.wav> <output.wav> or \n"\
                    "test_mwdlp <cv_spk_idx> <input.wav> <output.wav> or \n"\
                    "test_mwdlp <cv_x_coord> <cv_y_coord> <input.wav> <output.wav>\n");
        exit(1);
    } else {
        srand (time(NULL));
        FILE *fin, *fout;
        short spk_idx;
        if (argc == 3) { //analysis-synthesis
            fin = fopen(argv[1], "rb");
            if (fin == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[1]);
	            exit(1);
            }
            fout = fopen(argv[2], "wb");
            if (fout == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[2]);
	            fclose(fin);
	            exit(1);
            }
        } else if (argc == 4) { //exact point target spk-code location
            spk_idx = atoi(argv[1]);
            if (spk_idx > FEATURE_N_SPK) {
	            fprintf(stderr, "Speaker id %d is more than n_spk %d\n", spk_idx, FEATURE_N_SPK);
	            exit(1);
            }
            fin = fopen(argv[2], "rb");
            if (fin == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[2]);
	            exit(1);
            }
            fout = fopen(argv[3], "wb");
            if (fout == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[3]);
	            fclose(fin);
	            exit(1);
            }
        } else { //interpolated 2-dimensional target spk-code location
            fin = fopen(argv[3], "rb");
            if (fin == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[3]);
	            exit(1);
            }
            fout = fopen(argv[4], "wb");
            if (fout == NULL) {
	            fprintf(stderr, "Can't open %s\n", argv[4]);
	            fclose(fin);
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
        
        // read header parts [input wav]
        read = fread(header.riff, sizeof(header.riff), 1, fin);
        printf("(1-4): %s \n", header.riff); 
        // write header parts [output wav, following input reading]
        fwrite(header.riff, sizeof(header.riff), 1, fout);
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        fwrite(buffer4, sizeof(buffer4), 1, fout);
        
        // convert little endian to big endian 4 byte int
        header.overall_size  = buffer4[0] | 
                               (buffer4[1]<<8) | 
                               (buffer4[2]<<16) | 
                               (buffer4[3]<<24);
        
        printf("(5-8) Overall size: bytes:%u, Kb:%u \n", header.overall_size, header.overall_size/1024);
        
        read = fread(header.wave, sizeof(header.wave), 1, fin);
        printf("(9-12) Wave marker: %s\n", header.wave);
        fwrite(header.wave, sizeof(header.wave), 1, fout);
        
        read = fread(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fin);
        printf("(13-16) Fmt marker: %s\n", header.fmt_chunk_marker);
        fwrite(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fout);
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        fwrite(buffer4, sizeof(buffer4), 1, fout);
        
        // convert little endian to big endian 4 byte integer
        header.length_of_fmt = buffer4[0] |
                                   (buffer4[1] << 8) |
                                   (buffer4[2] << 16) |
                                   (buffer4[3] << 24);
        printf("(17-20) Length of Fmt header: %u \n", header.length_of_fmt);
        
        read = fread(buffer2, sizeof(buffer2), 1, fin);
        printf("%u %u \n", buffer2[0], buffer2[1]);
        fwrite(buffer2, sizeof(buffer2), 1, fout);
        
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
            fclose(fin);
            fclose(fout);
            exit(1);
        }
        
        read = fread(buffer2, sizeof(buffer2), 1, fin);
        printf("%u %u \n", buffer2[0], buffer2[1]);
        fwrite(buffer2, sizeof(buffer2), 1, fout);
        
        header.channels = buffer2[0] | (buffer2[1] << 8);
        printf("(23-24) Channels: %u \n", header.channels);
        if (header.channels != 1)
        {
            printf("Channels are more than 1.\n");
            fclose(fin);
            fclose(fout);
            exit(1);
        }
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        fwrite(buffer4, sizeof(buffer4), 1, fout);
        
        header.sample_rate = buffer4[0] |
                               (buffer4[1] << 8) |
                               (buffer4[2] << 16) |
                               (buffer4[3] << 24);
        
        printf("(25-28) Sample rate: %u Hz\n", header.sample_rate);
        if (header.sample_rate != SAMPLING_FREQUENCY)
        {
            printf("Sampling rate is not %d Hz\n", SAMPLING_FREQUENCY);
            fclose(fin);
            fclose(fout);
            exit(1);
        }
        
        read = fread(buffer4, sizeof(buffer4), 1, fin);
        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
        fwrite(buffer4, sizeof(buffer4), 1, fout);
        
        header.byterate  = buffer4[0] |
                               (buffer4[1] << 8) |
                               (buffer4[2] << 16) |
                               (buffer4[3] << 24);
        printf("(29-32) Byte Rate: %u B/s , Bit Rate:%u b/s\n", header.byterate, header.byterate*8);
        
        read = fread(buffer2, sizeof(buffer2), 1, fin);
        printf("%u %u \n", buffer2[0], buffer2[1]);
        fwrite(buffer2, sizeof(buffer2), 1, fout);
        
        header.block_align = buffer2[0] |
                           (buffer2[1] << 8);
        printf("(33-34) Block Alignment: %u \n", header.block_align);
        
        read = fread(buffer2, sizeof(buffer2), 1, fin);
        printf("%u %u \n", buffer2[0], buffer2[1]);
        fwrite(buffer2, sizeof(buffer2), 1, fout);
        
        header.bits_per_sample = buffer2[0] |
                           (buffer2[1] << 8);
        printf("(35-36) Bits per sample: %u \n", header.bits_per_sample);
        
        read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fin);
        printf("(37-40) Data Marker: %s \n", header.data_chunk_header);
        fwrite(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fout);
        
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
        // first samples are of the right-side hanning window, i.e., 1st frame,
        // the remaining samples are then divided by FRAME_SHIFT to get the remaining frames
        long num_frame = 1 + (num_samples - RIGHT_SAMPLES) / FRAME_SHIFT;
        // if theere exist remainder samples of the 2nd-Nth frame,
        // use reflected samples of the last buffer for the missing amount of samples to process one last frame
        long num_reflected_right_edge_samples = (num_samples - RIGHT_SAMPLES) % FRAME_SHIFT;
        if (num_reflected_right_edge_samples > 0) {
            num_reflected_right_edge_samples = FRAME_SHIFT - num_reflected_right_edge_samples;
            num_frame += 1; //use additional reflected samples for the trailing remainder samples on the right-edge
        }
        long num_samples_frame = num_frame * FRAME_SHIFT;
        printf("Number of samples in processed output wav: %lu \n", num_samples_frame);
        header.data_size = (num_samples_frame * header.channels * header.bits_per_sample) / 8;
        printf("(41-44) Size of data chunk to be written: %u B \n", header.data_size);
        fwrite(&header.data_size, sizeof(header.data_size), 1, fout);
        
        long size_of_each_sample = (header.channels * header.bits_per_sample) / 8;
        printf("Size of each sample: %ld B\n", size_of_each_sample);
        
        // calculate duration of file
        float duration_in_seconds = (float) header.overall_size / header.byterate;
        printf("Approx.Duration in seconds= %f\n", duration_in_seconds);
        
        // read each sample from data chunk if PCM
        if (header.format_type == 1 && header.channels == 1) { // PCM and mono
            clock_t t = clock();
            DSPState *dsp;
            if (argc == 3) { //analysis-synthesis
                MWDLP8NetState *net;

                float features[FEATURES_DIM];
                short pcm[MAX_N_OUTPUT]; //output is in short 2-byte (16-bit) format [-32768,32767]
                int first_buffer_flag = 0;
                int waveform_buffer_flag = 0;
                int n_output = 0;
                int samples = 0;
                float data_in_channel = 0;
                int i, j;
                int k;
                char data_buffer[size_of_each_sample];
                float x_buffer[FRAME_SHIFT];

                // initialize waveform-->features processing struct
                dsp = dspstate_create();

                // initialize mwdlp+cyclevae struct
                net = mwdlp8net_create();

                for (i = 0, j = 0, k = 0; i < num_samples; i++) {
                    //printf("==========Sample %d / %ld=============\n", i+1, num_samples);
                    read = fread(data_buffer, sizeof(data_buffer), 1, fin);
                    if (read == 1) {
                    
                        /* Receives only mono 16-bit PCM, convert to float [-1,0.999969482421875] */
                        data_in_channel = ((float) ((data_buffer[0] & 0x00ff) | (data_buffer[1] << 8))) / 32768;
                        //printf("%f\n", data_in_channel);

                        // high-pass filter to remove DC component of recording device
                        shift_apply_hpassfilt(dsp, &data_in_channel);
                
                        //check waveform buffer here
                        if (first_buffer_flag) { //first frame has been processed, now taking every FRAME_SHIFT amount of samples
                        //    printf("after first frame\n");
                            x_buffer[j] = data_in_channel;
                            j += 1;
                            if (j >= FRAME_SHIFT) {
                                shift_apply_window(dsp, x_buffer); //shift old FRAME_SHIFT amount for new FRAME_SHIFT amount and window
                                waveform_buffer_flag = 1;
                                j = 0;
                                k += 1;
                                printf(" [%d]", k);
                            }
                        } else { //take RIGHT_SAMPLES amount of samples as the first samples
                            //put only LEFT_SAMPLES and RIGHT_SAMPLES amount in window buffer, because zero-padding until FFT_LENGTH with centered window position
                            //(i//FRAME_SHIFT)th frame = i*FRAME_SHIFT [i: 0->(n_samples-1)]
                        //    printf("first frame\n");
                            dsp->samples_win[LEFT_SAMPLES_1+i] = data_in_channel;
                            if (i <= LEFT_SAMPLES_2) //reflect only LEFT_SAMPLES-1 amount because 0 value for the 1st coeff. of window
                                dsp->samples_win[LEFT_SAMPLES_2-i] = data_in_channel; 
                            if (i >= RIGHT_SAMPLES_1) { //process current buffer, and next, take for every FRAME_SHIFT amount samples
                                apply_window(dsp); //hanning window
                                first_buffer_flag = 1;
                                waveform_buffer_flag = 1;
                                k += 1;
                                printf("frame: [%d]", k);
                            }
                        }
                        //printf("wav buffer\n");

                        if (waveform_buffer_flag) {
                            //extract melspectrogram here
                            mel_spec_extract(dsp, features);
                            //printf("melsp extract, nth-frame: %d\n", k);

                            mwdlp8net_synthesize(net, features, pcm, &n_output, 0);
                
                            if (n_output > 0)  { //delay is reached, samples are generated
                                fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                                samples += n_output;
                            //    printf("write %d\n", samples);
                            }

                            waveform_buffer_flag = 0;
                        }
                        //printf("buffer processed\n");
                    } else {
                        printf("\nError reading file. %d bytes\n", read);
                        fclose(fin);
                        fclose(fout);
                        dspstate_destroy(dsp);
                        mwdlp8net_destroy(net);
                        exit(1);
                    }
                }
        
                if (!waveform_buffer_flag && j > 0) {
                    //set additional reflected samples for trailing remainder samples on the right edge here
                    int k;
                    for (i = 0, k=j-1; i < num_reflected_right_edge_samples; i++, j++)
                        x_buffer[j] = x_buffer[k-i];

                    if (j != FRAME_SHIFT) {
                        printf("\nError remainder right-edge samples calculation %d %d %ld\n", j, FRAME_SHIFT, num_reflected_right_edge_samples);
                        fclose(fin);
                        fclose(fout);
                        dspstate_destroy(dsp);
                        mwdlp8net_destroy(net);
                        exit(1);
                    } else {
                        printf(" [last frame]\n");
                    }
                    //}

                    mel_spec_extract(dsp, features);
                    mwdlp8net_synthesize(net, features, pcm, &n_output, 0); //last_frame, synth pad_right

                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    //    printf("write %d\n", samples);
                    }
                    mwdlp8net_synthesize(net, features, pcm, &n_output, 1); //synth pad_right
                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    //    printf("write pad_right %d\n", samples);
                    }
                }
    
                t = clock() - t;
                double time_taken = ((double)t)/CLOCKS_PER_SEC;
                //printf("%d [frames], %d [samples] synthesis in %f seconds \n", idx, samples, time_taken);
                printf("%d [frames] %d [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
                    "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
                    (int)((double)samples/FRAME_SHIFT), samples, (double)samples/SAMPLING_FREQUENCY, time_taken,
                        ((double)samples/SAMPLING_FREQUENCY)/time_taken, time_taken/((double)samples/SAMPLING_FREQUENCY),
                            N_SAMPLE_BANDS*((double)samples/SAMPLING_FREQUENCY)/time_taken);

                fclose(fin);
                fclose(fout);
                dspstate_destroy(dsp);
                mwdlp8net_destroy(net);
            } else {
                MWDLP8CycleVAEMelspExcitSpkNetState *net;

                float features[FEATURES_DIM];
                short pcm[MAX_N_OUTPUT]; //output is in short 2-byte (16-bit) format [-32768,32767]
                //short pcm_1[N_SAMPLE_BANDS], pcm_2[N_SAMPLE_BANDS], pcm_3[N_SAMPLE_BANDS], pcm_4[N_SAMPLE_BANDS];
                int first_buffer_flag = 0;
                int waveform_buffer_flag = 0;
                int n_output = 0;
                int samples = 0;
                float data_in_channel = 0;
                int i, j;
                int k;
                //int l;
                char data_buffer[size_of_each_sample];
                float x_buffer[FRAME_SHIFT];
                //float melsp_cv[FEATURE_DIM_MELSP];
                //float lat_tmp[FEATURE_LAT_DIM_EXCIT_MELSP];
                //float spk_tmp[FEATURE_N_SPK_2];
                //float conv_tmp[FEATURE_CONV_ENC_EXCIT_OUT_SIZE+FEATURE_CONV_ENC_MELSP_OUT_SIZE];
                //float gru_tmp[GRU_ENC_EXCIT_STATE_SIZE];
                //float f0_tmp[2];

                // initialize waveform-->features processing struct
                dsp = dspstate_create();

                // initialize mwdlp+cyclevae struct
                net = mwdlp8cyclevaenet_create();

                //printf("nn.Valid range for data values : %ld to %ld \n", low_limit, high_limit);

                // set spk-code here
                float spk_code_aux[FEATURE_N_SPK_2]; //if use gru-spk
                if (argc == 4) { //exact point spk-code location
                    float one_hot_code[FEATURE_N_SPK] = {0};
                    one_hot_code[spk_idx-1] = 1;
                    //N-dim 1-hot --> 2-dim --> N-dim [N_SPK]
                    printf("%d-dim 1-hot code: ", FEATURE_N_SPK);
                    for (k = 0; k < FEATURE_N_SPK; k++)
                        printf("[%d] %f ", k+1, one_hot_code[k]);
                    printf("\n");
                    compute_spkidtr(&fc_in_spk_code_transform, &fc_out_spk_code_transform, spk_code_aux, one_hot_code);
                    printf("%d-dim embed.: ", FEATURE_N_SPK);
                    for (k = 0; k < FEATURE_N_SPK; k++) {
                        printf("[%d] %f ", k+1, spk_code_aux[k]);
                    //    spk_tmp[k] = spk_code_aux[k];
                    }
                    printf("\n");
                    //exit(1);
                } else { //interpolated spk-code location
                    float spk_coord[2];
                    spk_coord[0] = atof(argv[1]);
                    spk_coord[1] = atof(argv[2]);
                    //2-dim --> N-dim [N_SPK]
                    printf("2-dim spk-coord: %f %f\n", spk_coord[0], spk_coord[1]);
                    compute_spkidtr_coord(&fc_out_spk_code_transform, spk_code_aux, spk_coord);
                    printf("%d-dim embed.: ", FEATURE_N_SPK);
                    for (k = 0; k < FEATURE_N_SPK; k++) {
                        printf("[%d] %f ", k+1, spk_code_aux[k]);
                    }
                    printf("\n");
                }

                //printf("nn.Valid range for data values : %ld to %ld \n", low_limit, high_limit);
                //FILE *tmp_file, *tmp_file2, *tmp_file3, *tmp_file4;
                //FILE *tmp_file5, *tmp_file6, *tmp_file7, *tmp_file8, *tmp_file9, *tmp_file10;
                //FILE *tmp_file5;
                //tmp_file = fopen("melsp.txt", "wt");
                //tmp_file = fopen("melsp.txt", "wt");
                //tmp_file2 = fopen("magsp.txt", "wt");
                //tmp_file3 = fopen("stft_real.txt", "wt");
                //tmp_file4 = fopen("stft_imag.txt", "wt");
                //tmp_file5 = fopen("melsp_cv.txt", "wt");
                //tmp_file6 = fopen("lat_est.txt", "wt");
                //tmp_file7 = fopen("spk_est.txt", "wt");
                //tmp_file8 = fopen("conv_est.txt", "wt");
                //tmp_file9 = fopen("gru_est.txt", "wt");
                //tmp_file10 = fopen("f0_est.txt", "wt");
                //FILE *band1, *band2, *band3, *band4;
                //band1 = fopen("band-1.pcm", "wb");
                //band2 = fopen("band-2.pcm", "wb");
                //band3 = fopen("band-3.pcm", "wb");
                //band4 = fopen("band-4.pcm", "wb");
                //for (i = 0, j = 0; i < num_samples; i++) {
                for (i = 0, j = 0, k = 0; i < num_samples; i++) {
                    //printf("==========Sample %d / %ld=============\n", i+1, num_samples);
                    read = fread(data_buffer, sizeof(data_buffer), 1, fin);
                    if (read == 1) {
                    
                        /* Receives only mono 16-bit PCM, convert to float [-1,0.999969482421875] */
                        data_in_channel = ((float) ((data_buffer[0] & 0x00ff) | (data_buffer[1] << 8))) / 32768;
                        //printf("%f\n", data_in_channel);

                        // high-pass filter to remove DC component of recording device
                        shift_apply_hpassfilt(dsp, &data_in_channel);
                        //pcm[0] = (short) (data_in_channel*32768);
                        //fwrite(&pcm[0], sizeof(pcm[0]), 1, fout);
                        //printf("hpassfilt\n");
                
                        //check waveform buffer here
                        if (first_buffer_flag) { //first frame has been processed, now taking every FRAME_SHIFT amount of samples
                        //    printf("after first frame\n");
                            x_buffer[j] = data_in_channel;
                            j += 1;
                            if (j >= FRAME_SHIFT) {
                                shift_apply_window(dsp, x_buffer); //shift old FRAME_SHIFT amount for new FRAME_SHIFT amount and window
                                waveform_buffer_flag = 1;
                                j = 0;
                                k += 1;
                                printf(" [%d]", k);
                            }
                        } else { //take RIGHT_SAMPLES amount of samples as the first samples
                            //put only LEFT_SAMPLES and RIGHT_SAMPLES amount in window buffer, because zero-padding until FFT_LENGTH with centered window position
                            //(i//FRAME_SHIFT)th frame = i*FRAME_SHIFT [i: 0->(n_samples-1)]
                        //    printf("first frame\n");
                            dsp->samples_win[LEFT_SAMPLES_1+i] = data_in_channel;
                            if (i <= LEFT_SAMPLES_2) //reflect only LEFT_SAMPLES-1 amount because 0 value for the 1st coeff. of window
                                dsp->samples_win[LEFT_SAMPLES_2-i] = data_in_channel; 
                            if (i >= RIGHT_SAMPLES_1) { //process current buffer, and next, take for every FRAME_SHIFT amount samples
                                apply_window(dsp); //hanning window
                                first_buffer_flag = 1;
                                waveform_buffer_flag = 1;
                                k += 1;
                                printf("frame: [%d]", k);
                            }
                        }
                        //printf("wav buffer\n");

                        if (waveform_buffer_flag) {
                            //for (l=0;l<MAGSP_DIM;l++) {
                            //    if (l < (MAGSP_DIM-1)) {
                            //        fprintf(tmp_file3, "%f ", dsp->out_fft[l].r);
                            //        fprintf(tmp_file4, "%f ", dsp->out_fft[l].i);
                            //    } else {
                            //        fprintf(tmp_file3, "%f\n", dsp->out_fft[l].r);
                            //        fprintf(tmp_file4, "%f\n", dsp->out_fft[l].i);
                            //    }
                            //}

                            //extract melspectrogram here
                            mel_spec_extract(dsp, features);
                            //for (l=0;l<MEL_DIM;l++)
                            ////    printf("[%d] %f\n", l+1, features[l]);
                            ////printf("melsp extract, nth-frame: %d\n", k);

                            ////    printf("[%d] %f\n", l+1, features[l]);
                            //    if (l < (MEL_DIM-1))
                            //        fprintf(tmp_file, "%f ", features[l]);
                            //    else
                            //        fprintf(tmp_file, "%f\n", features[l]);
                            //}
                            //for (l=0;l<MAGSP_DIM;l++) {
                            //    if (l < (MAGSP_DIM-1))
                            //        fprintf(tmp_file2, "%f ", dsp->magsp[l]);
                            //    else
                            //        fprintf(tmp_file2, "%f\n", dsp->magsp[l]);
                            //}
                            //fputs("};\n", f);
                
                            cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0);
                            //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0, 0);
                            //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0, melsp_cv);
                            //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0, melsp_cv, lat_tmp, spk_tmp, conv_tmp, gru_tmp, f0_tmp);
                            //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0, melsp_cv, pcm_1, pcm_2, pcm_3, pcm_4);
                            //printf("cv synth\n");
                
                            //if (k > FEATURE_CONV_VC_DELAY) {
                            //    for (l=0;l<MEL_DIM;l++)
                            //        if (l < (MEL_DIM-1))
                            //            fprintf(tmp_file5, "%f ", features[l]);
                            //        else
                            //            fprintf(tmp_file5, "%f\n", features[l]);
                            //    //for (l=0;l<FEATURE_LAT_DIM_EXCIT_MELSP;l++)
                            //    //    if (l < (FEATURE_LAT_DIM_EXCIT_MELSP-1))
                            //    //        fprintf(tmp_file6, "%f ", lat_tmp[l]);
                            //    //    else
                            //    //        fprintf(tmp_file6, "%f\n", lat_tmp[l]);
                            //    //for (l=0;l<FEATURE_N_SPK_2;l++)
                            //    //    if (l < (FEATURE_N_SPK_2-1))
                            //    //        fprintf(tmp_file7, "%f ", spk_tmp[l]);
                            //    //    else
                            //    //        fprintf(tmp_file7, "%f\n", spk_tmp[l]);
                            //    //for (l=0;l<FEATURE_CONV_ENC_EXCIT_OUT_SIZE+FEATURE_CONV_ENC_MELSP_OUT_SIZE;l++)
                            //    //    if (l < (FEATURE_CONV_ENC_EXCIT_OUT_SIZE+FEATURE_CONV_ENC_MELSP_OUT_SIZE-1))
                            //    //        fprintf(tmp_file8, "%f ", conv_tmp[l]);
                            //    //    else
                            //    //        fprintf(tmp_file8, "%f\n", conv_tmp[l]);
                            //    //for (l=0;l<GRU_ENC_EXCIT_STATE_SIZE;l++)
                            //    //    if (l < (GRU_ENC_EXCIT_STATE_SIZE-1))
                            //    //        fprintf(tmp_file9, "%f ", gru_tmp[l]);
                            //    //    else
                            //    //        fprintf(tmp_file9, "%f\n", gru_tmp[l]);
                            //    //fprintf(tmp_file10, "%f %f\n", f0_tmp[0], f0_tmp[1]);
                            //}
                            if (n_output > 0)  { //delay is reached, samples are generated
                                fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                            //    fwrite(pcm_1, sizeof(pcm_1[0]), N_SAMPLE_BANDS, band1);
                            //    fwrite(pcm_2, sizeof(pcm_2[0]), N_SAMPLE_BANDS, band2);
                            //    fwrite(pcm_3, sizeof(pcm_3[0]), N_SAMPLE_BANDS, band3);
                            //    fwrite(pcm_4, sizeof(pcm_4[0]), N_SAMPLE_BANDS, band4);
                                samples += n_output;
                            //    printf("write %d\n", samples);
                            }

                            waveform_buffer_flag = 0;
                        }
                        //printf("buffer processed\n");
                    } else {
                        printf("\nError reading file. %d bytes\n", read);
                        fclose(fin);
                        fclose(fout);
                        dspstate_destroy(dsp);
                        mwdlp8cyclevaenet_destroy(net);
                        exit(1);
                    }
                }
                //fclose(tmp_file);
                //fclose(tmp_file2);
                //fclose(tmp_file3);
                //fclose(tmp_file4);
                //fclose(band1);
                //fclose(band2);
                //fclose(band3);
                //fclose(band4);
        
                if (!waveform_buffer_flag && j > 0) {
                    //set additional reflected samples for trailing remainder samples on the right edge here
                    int k;
                    for (i = 0, k=j-1; i < num_reflected_right_edge_samples; i++, j++)
                        x_buffer[j] = x_buffer[k-i];

                    if (j != FRAME_SHIFT) {
                        printf("\nError remainder right-edge samples calculation %d %d %ld\n", j, FRAME_SHIFT, num_reflected_right_edge_samples);
                        fclose(fin);
                        fclose(fout);
                        dspstate_destroy(dsp);
                        mwdlp8cyclevaenet_destroy(net);
                        exit(1);
                    } else {
                        printf(" [last frame]\n");
                    }
                    //}

                    mel_spec_extract(dsp, features);
                    cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0); //last_frame, synth pad_right
                    //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0, 0); //last_frame, synth pad_right
                    //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0, melsp_cv); //last_frame
                    //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0, melsp_cv, lat_tmp, spk_tmp, conv_tmp, gru_tmp, f0_tmp); //last_frame

                    //for (l=0;l<MEL_DIM;l++)
                    //    if (l < (MEL_DIM-1))
                    //        fprintf(tmp_file5, "%f ", features[l]);
                    //    else
                    //        fprintf(tmp_file5, "%f\n", features[l]);
                    ////for (l=0;l<FEATURE_LAT_DIM_EXCIT_MELSP;l++)
                    ////    if (l < (FEATURE_LAT_DIM_EXCIT_MELSP-1))
                    ////        fprintf(tmp_file6, "%f ", lat_tmp[l]);
                    ////    else
                    ////        fprintf(tmp_file6, "%f\n", lat_tmp[l]);
                    ////for (l=0;l<FEATURE_N_SPK_2;l++)
                    ////    if (l < (FEATURE_N_SPK_2-1))
                    ////        fprintf(tmp_file7, "%f ", spk_tmp[l]);
                    ////    else
                    ////        fprintf(tmp_file7, "%f\n", spk_tmp[l]);
                    ////for (l=0;l<FEATURE_CONV_ENC_EXCIT_OUT_SIZE+FEATURE_CONV_ENC_MELSP_OUT_SIZE;l++)
                    ////    if (l < (FEATURE_CONV_ENC_EXCIT_OUT_SIZE+FEATURE_CONV_ENC_MELSP_OUT_SIZE-1))
                    ////        fprintf(tmp_file8, "%f ", conv_tmp[l]);
                    ////    else
                    ////        fprintf(tmp_file8, "%f\n", conv_tmp[l]);
                    ////for (l=0;l<GRU_ENC_EXCIT_STATE_SIZE;l++)
                    ////    if (l < (GRU_ENC_EXCIT_STATE_SIZE-1))
                    ////        fprintf(tmp_file9, "%f ", gru_tmp[l]);
                    ////    else
                    ////        fprintf(tmp_file9, "%f\n", gru_tmp[l]);
                    ////fprintf(tmp_file10, "%f %f\n", f0_tmp[0], f0_tmp[1]);
                    
                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    //    printf("write %d\n", samples);
                    }
                    cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 1); //synth pad_right
                    //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 1, 0); //synth pad_right
                    //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 1, melsp_cv); //synth pad_right
                    //cyclevae_melsp_excit_spk_convert_mwdlp8net_synthesize(net, features, spk_code_aux, pcm, &n_output, 1, melsp_cv, lat_tmp, spk_tmp, conv_tmp, gru_tmp, f0_tmp); //synth pad_right
                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    //    printf("write pad_right %d\n", samples);
                    }
                }
                //fclose(tmp_file5);
                //fclose(tmp_file6);
                //fclose(tmp_file7);
                //fclose(tmp_file8);
                //fclose(tmp_file9);
                //fclose(tmp_file10);
    
                t = clock() - t;
                double time_taken = ((double)t)/CLOCKS_PER_SEC;
                //printf("%d [frames], %d [samples] synthesis in %f seconds \n", idx, samples, time_taken);
                printf("%d [frames] %d [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
                    "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
                    (int)((double)samples/FRAME_SHIFT), samples, (double)samples/SAMPLING_FREQUENCY, time_taken,
                        ((double)samples/SAMPLING_FREQUENCY)/time_taken, time_taken/((double)samples/SAMPLING_FREQUENCY),
                            N_SAMPLE_BANDS*((double)samples/SAMPLING_FREQUENCY)/time_taken);

                fclose(fin);
                fclose(fout);
                dspstate_destroy(dsp);
                mwdlp8cyclevaenet_destroy(net);
            }
        } else {
            printf("Error not mono and 16-bit pcm\n");
            fclose(fin);
            fclose(fout);
            exit(1);
        }

        return 0;
    }
}

///*
//   Copyright 2021 Patrick Lumban Tobing (Nagoya University)
//   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
//
//   WAV file read/write is based on http://truelogic.org/wordpress/2015/09/04/parsing-a-wav-file-in-c
//
//   Argument parser is based on SPTK-3.11 [https://sourceforge.net/projects/sp-tk/files/SPTK/]
//*/
//
//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//
//#ifdef HAVE_STRING_H
//#include <string.h>
//#else
//#include <strings.h>
//#ifndef HAVE_STRRCHR
//#define strrchr rindex
//#endif
//#endif
//
//#include <time.h>
//#include <unistd.h>
//#include "mwdlp8net_cycvae.h"
//#include "freq.h"
//#include "nnet.h"
//#include "nnet_data.h"
//#include "nnet_cv_data.h"
//#include "mwdlp8net_cycvae_private.h"
//#include "wave.h"
//
//
///* Command Name  */
//char *cmnd;
//
//void usage(int status)
//{
//   fprintf(stderr, "\n");
//   fprintf(stderr, " %s - Multiband WaveRNN with data-driven Linear Prediction\n", cmnd);
//   fprintf(stderr, "                using 8-bit mu-law coarse-fine output architecture\n");
//   fprintf(stderr, "\n");
//   fprintf(stderr, "  usage:\n");
//   fprintf(stderr, "       %s [ options ] [ infile ] [ outfile ]\n", cmnd);
//   fprintf(stderr, "  options:\n");
//   fprintf(stderr, "       -e [outfile mel-spec] : print mel-spec [ignored if -d used]\n");
//   fprintf(stderr, "       -d    : flag to decode using input mel-spec\n");
//   fprintf(stderr, "       -h    : print this message\n");
//   fprintf(stderr, "  infile:\n");
//   fprintf(stderr, "       input wav-file/mel-spec       [default wav-file]\n");
//   fprintf(stderr, "  outfile:\n");
//   fprintf(stderr, "       output wav-file\n");
//   fprintf(stderr, "\n");
//   exit(status);
//}
//
//
//int main(int argc, char **argv) {
//    int wav_in_flag = 1;
//    int print_melsp_flag = 0;
//    FILE *fin, *fout_msp;
//
//    if ((cmnd = strrchr(argv[0], '/')) == NULL) {
//        cmnd = argv[0];
//    } else {
//        cmnd++;
//    }
//    while (--argc) {
//        if (**++argv == '-') {
//            switch (*(*argv + 1)) {
//                case 'e':
//                    print_melsp_flag = 1;
//                    //file handler melsp output
//                    fout_msp = fopen(*++argv, "rb");
//                    if (fout_msp == NULL) {
//                        fprintf(stderr, "Can't open %s\n", *argv);
//                        exit(1);
//                    }
//                    --argc;
//                    break;
//                case 'd':
//                    wav_in_flag = 0;
//                    print_melsp_flag = 0;
//                    break;
//                case 'h':
//                    usage(0);
//                    break;
//                default:
//                    fprintf(stderr, "%s : Invalid option '%c'!\n", cmnd, *(*argv + 1));
//                    usage(1);
//            }
//        } else {
//            //file handler wav/melsp input
//            fin = fopen(*++argv, "rb");
//            if (fin == NULL) {
//	            fprintf(stderr, "Can't open %s\n", *argv);
//                if (print_melsp_flag) {
//                    fclose(fout_msp);
//                }
//	            exit(1);
//            }
//        }
//    }
//
//    //file handler wav output
//    FILE *fout = fopen(*argv, "wb");
//    if (fout == NULL) {
//        fprintf(stderr, "Can't open %s\n", *argv);
//        if (print_melsp_flag) {
//            fclose(fout_msp);
//        }
//        fclose(fin);
//        exit(1);
//    }
//
//    srand (time(NULL));
//
//    if (wav_in_flag) { //input waveform
//        unsigned char buffer4[4];
//        unsigned char buffer2[2];
//    
//        // WAVE header structure
//        struct HEADER header;
//    
//        int read = 0;
//        
//        // read header parts [input wav]
//        read = fread(header.riff, sizeof(header.riff), 1, fin);
//        printf("(1-4): %s \n", header.riff); 
//        // write header parts [output wav, following input reading]
//        fwrite(header.riff, sizeof(header.riff), 1, fout);
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        fwrite(buffer4, sizeof(buffer4), 1, fout);
//        
//        // convert little endian to big endian 4 byte int
//        header.overall_size  = buffer4[0] | 
//                               (buffer4[1]<<8) | 
//                               (buffer4[2]<<16) | 
//                               (buffer4[3]<<24);
//        
//        printf("(5-8) Overall size: bytes:%u, Kb:%u \n", header.overall_size, header.overall_size/1024);
//        
//        read = fread(header.wave, sizeof(header.wave), 1, fin);
//        printf("(9-12) Wave marker: %s\n", header.wave);
//        fwrite(header.wave, sizeof(header.wave), 1, fout);
//        
//        read = fread(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fin);
//        printf("(13-16) Fmt marker: %s\n", header.fmt_chunk_marker);
//        fwrite(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fout);
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        fwrite(buffer4, sizeof(buffer4), 1, fout);
//        
//        // convert little endian to big endian 4 byte integer
//        header.length_of_fmt = buffer4[0] |
//                                   (buffer4[1] << 8) |
//                                   (buffer4[2] << 16) |
//                                   (buffer4[3] << 24);
//        printf("(17-20) Length of Fmt header: %u \n", header.length_of_fmt);
//        
//        read = fread(buffer2, sizeof(buffer2), 1, fin);
//        printf("%u %u \n", buffer2[0], buffer2[1]);
//        fwrite(buffer2, sizeof(buffer2), 1, fout);
//        
//        header.format_type = buffer2[0] | (buffer2[1] << 8);
//        char format_name[10] = "";
//        if (header.format_type == 1)
//            strcpy(format_name,"PCM"); 
//        else if (header.format_type == 6)
//            strcpy(format_name, "A-law");
//        else if (header.format_type == 7)
//            strcpy(format_name, "Mu-law");
//        
//        printf("(21-22) Format type: %u %s \n", header.format_type, format_name);
//        if (header.format_type != 1)
//        {
//            printf("Format is not PCM.\n");
//            fclose(fin);
//            fclose(fout);
//            exit(1);
//        }
//        
//        read = fread(buffer2, sizeof(buffer2), 1, fin);
//        printf("%u %u \n", buffer2[0], buffer2[1]);
//        fwrite(buffer2, sizeof(buffer2), 1, fout);
//        
//        header.channels = buffer2[0] | (buffer2[1] << 8);
//        printf("(23-24) Channels: %u \n", header.channels);
//        if (header.channels != 1)
//        {
//            printf("Channels are more than 1.\n");
//            fclose(fin);
//            fclose(fout);
//            exit(1);
//        }
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        fwrite(buffer4, sizeof(buffer4), 1, fout);
//        
//        header.sample_rate = buffer4[0] |
//                               (buffer4[1] << 8) |
//                               (buffer4[2] << 16) |
//                               (buffer4[3] << 24);
//        
//        printf("(25-28) Sample rate: %u Hz\n", header.sample_rate);
//        if (header.sample_rate != SAMPLING_FREQUENCY)
//        {
//            printf("Sampling rate is not %d Hz\n", SAMPLING_FREQUENCY);
//            fclose(fin);
//            fclose(fout);
//            exit(1);
//        }
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        fwrite(buffer4, sizeof(buffer4), 1, fout);
//        
//        header.byterate  = buffer4[0] |
//                               (buffer4[1] << 8) |
//                               (buffer4[2] << 16) |
//                               (buffer4[3] << 24);
//        printf("(29-32) Byte Rate: %u B/s , Bit Rate:%u b/s\n", header.byterate, header.byterate*8);
//        
//        read = fread(buffer2, sizeof(buffer2), 1, fin);
//        printf("%u %u \n", buffer2[0], buffer2[1]);
//        fwrite(buffer2, sizeof(buffer2), 1, fout);
//        
//        header.block_align = buffer2[0] |
//                           (buffer2[1] << 8);
//        printf("(33-34) Block Alignment: %u \n", header.block_align);
//        
//        read = fread(buffer2, sizeof(buffer2), 1, fin);
//        printf("%u %u \n", buffer2[0], buffer2[1]);
//        fwrite(buffer2, sizeof(buffer2), 1, fout);
//        
//        header.bits_per_sample = buffer2[0] |
//                           (buffer2[1] << 8);
//        printf("(35-36) Bits per sample: %u \n", header.bits_per_sample);
//        
//        read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fin);
//        printf("(37-40) Data Marker: %s \n", header.data_chunk_header);
//        fwrite(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fout);
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        
//        header.data_size = buffer4[0] |
//                       (buffer4[1] << 8) |
//                       (buffer4[2] << 16) | 
//                       (buffer4[3] << 24 );
//        printf("(41-44) Size of data chunk: %u B \n", header.data_size);
//        
//        // calculate no.of samples
//        long num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);
//        printf("Number of samples: %lu \n", num_samples);
//        // first samples are of the right-side hanning window, i.e., 1st frame,
//        // the remaining samples are then divided by FRAME_SHIFT to get the remaining frames
//        long num_frame = 1 + (num_samples - RIGHT_SAMPLES) / FRAME_SHIFT;
//        // if theere exist remainder samples of the 2nd-Nth frame,
//        // use reflected samples of the last buffer for the missing amount of samples to process one last frame
//        long num_reflected_right_edge_samples = (num_samples - RIGHT_SAMPLES) % FRAME_SHIFT;
//        if (num_reflected_right_edge_samples > 0) {
//            num_reflected_right_edge_samples = FRAME_SHIFT - num_reflected_right_edge_samples;
//            num_frame += 1; //use additional reflected samples for the trailing remainder samples on the right-edge
//        }
//        long num_samples_frame = num_frame * FRAME_SHIFT;
//        printf("Number of samples in processed output wav: %lu \n", num_samples_frame);
//        header.data_size = (num_samples_frame * header.channels * header.bits_per_sample) / 8;
//        printf("(41-44) Size of data chunk to be written: %u B \n", header.data_size);
//        fwrite(&header.data_size, sizeof(header.data_size), 1, fout);
//        
//        long size_of_each_sample = (header.channels * header.bits_per_sample) / 8;
//        printf("Size of each sample: %ld B\n", size_of_each_sample);
//        
//        // calculate duration of file
//        float duration_in_seconds = (float) header.overall_size / header.byterate;
//        printf("Approx.Duration in seconds= %f\n", duration_in_seconds);
//        
//        // read each sample from data chunk if PCM
//        if (header.format_type == 1 && header.channels == 1) { // PCM and mono
//            clock_t t = clock();
//            DSPState *dsp;
//            MWDLP8NetState *net;
//    
//            float features[FEATURES_DIM];
//            short pcm[MAX_N_OUTPUT]; //output is in short 2-byte (16-bit) format [-32768,32767]
//            int first_buffer_flag = 0;
//            int waveform_buffer_flag = 0;
//            int n_output = 0;
//            int samples = 0;
//            float data_in_channel = 0;
//            int i, j;
//            int k;
//            char data_buffer[size_of_each_sample];
//            float x_buffer[FRAME_SHIFT];
//    
//            // initialize waveform-->features processing struct
//            dsp = dspstate_create();
//    
//            // initialize mwdlp+cyclevae struct
//            net = mwdlp8net_create();
//
//            //FILE *tmp_file5;
//            //tmp_file5 = fopen("melsp.txt", "wt");
//            for (i = 0, j = 0, k = 0; i < num_samples; i++) {
//                //printf("==========Sample %d / %ld=============\n", i+1, num_samples);
//                read = fread(data_buffer, sizeof(data_buffer), 1, fin);
//                if (read == 1) {
//                
//                    /* Receives only mono 16-bit PCM, convert to float [-1,0.999969482421875] */
//                    data_in_channel = ((float) ((data_buffer[0] & 0x00ff) | (data_buffer[1] << 8))) / 32768;
//                    //printf("%f\n", data_in_channel);
//    
//                    // high-pass filter to remove DC component of recording device
//                    shift_apply_hpassfilt(dsp, &data_in_channel);
//            
//                    //check waveform buffer here
//                    if (first_buffer_flag) { //first frame has been processed, now taking every FRAME_SHIFT amount of samples
//                    //    printf("after first frame\n");
//                        x_buffer[j] = data_in_channel;
//                        j += 1;
//                        if (j >= FRAME_SHIFT) {
//                            shift_apply_window(dsp, x_buffer); //shift old FRAME_SHIFT amount for new FRAME_SHIFT amount and window
//                            waveform_buffer_flag = 1;
//                            j = 0;
//                            k += 1;
//                            printf(" [%d]", k);
//                        }
//                    } else { //take RIGHT_SAMPLES amount of samples as the first samples
//                        //put only LEFT_SAMPLES and RIGHT_SAMPLES amount in window buffer, because zero-padding until FFT_LENGTH with centered window position
//                        //(i//FRAME_SHIFT)th frame = i*FRAME_SHIFT [i: 0->(n_samples-1)]
//                    //    printf("first frame\n");
//                        dsp->samples_win[LEFT_SAMPLES_1+i] = data_in_channel;
//                        if (i <= LEFT_SAMPLES_2) //reflect only LEFT_SAMPLES-1 amount because 0 value for the 1st coeff. of window
//                            dsp->samples_win[LEFT_SAMPLES_2-i] = data_in_channel; 
//                        if (i >= RIGHT_SAMPLES_1) { //process current buffer, and next, take for every FRAME_SHIFT amount samples
//                            apply_window(dsp); //hanning window
//                            first_buffer_flag = 1;
//                            waveform_buffer_flag = 1;
//                            k += 1;
//                            printf("frame: [%d]", k);
//                        }
//                    }
//                    //printf("wav buffer\n");
//    
//                    if (waveform_buffer_flag) {
//                        //extract melspectrogram here
//                        mel_spec_extract(dsp, features);
//                        //printf("melsp extract, nth-frame: %d\n", k);
//                        if (print_melsp_flag) fwrite((void*)features, sizeof(features[0]), FEATURES_DIM, fout_msp);
//    
//                        //if (k > FEATURE_CONV_DELAY) {
//                        //    for (l=0;l<FEATURES_DIM;l++)
//                        //        if (l < (FEATURES_DIM-1))
//                        //            fprintf(tmp_file5, "%f ", features[l]);
//                        //        else
//                        //            fprintf(tmp_file5, "%f\n", features[l]);
//                        //}
//                        mwdlp8net_synthesize(net, features, pcm, &n_output, 0);
//            
//                        if (n_output > 0)  { //delay is reached, samples are generated
//                            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
//                            samples += n_output;
//                        //    printf("write %d\n", samples);
//                        }
//    
//                        waveform_buffer_flag = 0;
//                    }
//                    //printf("buffer processed\n");
//                } else {
//                    printf("\nError reading file. %d bytes\n", read);
//                    fclose(fin);
//                    fclose(fout);
//                    if (print_melsp_flag) fclose(fout_msp);
//                    dspstate_destroy(dsp);
//                    mwdlp8net_destroy(net);
//                    exit(1);
//                }
//            }
//        
//            if (!waveform_buffer_flag && j > 0) {
//                //set additional reflected samples for trailing remainder samples on the right edge here
//                int k;
//                for (i = 0, k=j-1; i < num_reflected_right_edge_samples; i++, j++)
//                    x_buffer[j] = x_buffer[k-i];
//    
//                if (j != FRAME_SHIFT) {
//                    printf("\nError remainder right-edge samples calculation %d %d %ld\n", j, FRAME_SHIFT, num_reflected_right_edge_samples);
//                    fclose(fin);
//                    fclose(fout);
//                    if (print_melsp_flag) fclose(fout_msp);
//                    dspstate_destroy(dsp);
//                    mwdlp8net_destroy(net);
//                    exit(1);
//                } else {
//                    printf(" [last frame]\n");
//                }
//    
//                mel_spec_extract(dsp, features);
//                if (print_melsp_flag) fwrite((void*)features, sizeof(features[0]), FEATURES_DIM, fout_msp);
//    
//                //    for (l=0;l<FEATURES_DIM;l++)
//                //    if (l < (FEATURES_DIM-1))
//                //        fprintf(tmp_file5, "%f ", features[l]);
//                //    else
//                //        fprintf(tmp_file5, "%f\n", features[l]);
//                mwdlp8net_synthesize(net, features, pcm, &n_output, 0); //last_frame, synth pad_right
//    
//                if (n_output > 0)  {
//                    fwrite(pcm, sizeof(pcm[0]), n_output, fout);
//                    samples += n_output;
//                //    printf("write %d\n", samples);
//                }
//                mwdlp8net_synthesize(net, features, pcm, &n_output, 1); //synth pad_right
//                if (n_output > 0)  {
//                    fwrite(pcm, sizeof(pcm[0]), n_output, fout);
//                    samples += n_output;
//                //    printf("write pad_right %d\n", samples);
//                }
//            }
//        
//            t = clock() - t;
//            double time_taken = ((double)t)/CLOCKS_PER_SEC;
//            //printf("%d [frames], %d [samples] synthesis in %f seconds \n", idx, samples, time_taken);
//            printf("%d [frames] %d [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
//                "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
//                (int)((double)samples/FRAME_SHIFT), samples, (double)samples/SAMPLING_FREQUENCY, time_taken,
//                    ((double)samples/SAMPLING_FREQUENCY)/time_taken, time_taken/((double)samples/SAMPLING_FREQUENCY),
//                        N_SAMPLE_BANDS*((double)samples/SAMPLING_FREQUENCY)/time_taken);
//    
//            fclose(fin);
//            fclose(fout);
//            if (print_melsp_flag) fclose(fout_msp);
//            dspstate_destroy(dsp);
//            mwdlp8net_destroy(net);
//        } else {
//            printf("Error not mono and 16-bit pcm\n");
//            fclose(fin);
//            fclose(fout);
//            if (print_melsp_flag) fclose(fout_msp);
//            exit(1);
//        }
//    } else { //input melsp
//        //fseek(fin, 0, SEEK_END);
//        //int size = ftell(fin);
//        //fseek(fin, 0, SEEK_SET);
//        //int n_frames = (int) round((float) size / (FEATURES_DIM * sizeof(float)));
//        //size_t result, result_test = FEATURES_DIM;
//
//        float readfloat(FILE *f) {
//            float v;
//            fread((void*)(&v), sizeof(v), 1, f);
//            return v;
//        }
//
//        unsigned char buffer4[4];
//        unsigned char buffer2[2];
//    
//        // WAVE header structure
//        struct HEADER header;
//    
//        // write header parts [output wav, following a wav file format]
//        header.riff[0] = 'R';
//        header.riff[1] = 'I';
//        header.riff[2] = 'F';
//        header.riff[3] = 'F';
//        printf("(1-4): %s \n", header.riff); 
//        fwrite(header.riff, sizeof(header.riff), 1, fout);
//        
//        //read = fread(buffer4, sizeof(buffer4), 1, fin);
//        //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        //fwrite(buffer4, sizeof(buffer4), 1, fout);
//        //
//        //// convert little endian to big endian 4 byte int
//        //header.overall_size  = buffer4[0] | 
//        //                       (buffer4[1]<<8) | 
//        //                       (buffer4[2]<<16) | 
//        //                       (buffer4[3]<<24);
//        //
//        //printf("(5-8) Overall size: bytes:%u, Kb:%u \n", header.overall_size, header.overall_size/1024);
//        
//        header.wave[0] = 'W';
//        header.wave[1] = 'A';
//        header.wave[2] = 'V';
//        header.wave[3] = 'E';
//        printf("(9-12) Wave marker: %s\n", header.wave);
//        fwrite(header.wave, sizeof(header.wave), 1, fout);
//        
//        header.fmt_chunk_marker[0] = 'f';
//        header.fmt_chunk_marker[1] = 'm';
//        header.fmt_chunk_marker[2] = 't';
//        header.fmt_chunk_marker[3] = '\0';
//        printf("(13-16) Fmt marker: %s\n", header.fmt_chunk_marker);
//        fwrite(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fout);
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        fwrite(buffer4, sizeof(buffer4), 1, fout);
//        
//        // convert little endian to big endian 4 byte integer
//        header.length_of_fmt = buffer4[0] |
//                                   (buffer4[1] << 8) |
//                                   (buffer4[2] << 16) |
//                                   (buffer4[3] << 24);
//        printf("(17-20) Length of Fmt header: %u \n", header.length_of_fmt);
//        
//        read = fread(buffer2, sizeof(buffer2), 1, fin);
//        printf("%u %u \n", buffer2[0], buffer2[1]);
//        fwrite(buffer2, sizeof(buffer2), 1, fout);
//        
//        header.format_type = buffer2[0] | (buffer2[1] << 8);
//        char format_name[10] = "";
//        if (header.format_type == 1)
//            strcpy(format_name,"PCM"); 
//        else if (header.format_type == 6)
//            strcpy(format_name, "A-law");
//        else if (header.format_type == 7)
//            strcpy(format_name, "Mu-law");
//        
//        printf("(21-22) Format type: %u %s \n", header.format_type, format_name);
//        if (header.format_type != 1)
//        {
//            printf("Format is not PCM.\n");
//            fclose(fin);
//            fclose(fout);
//            exit(1);
//        }
//        
//        read = fread(buffer2, sizeof(buffer2), 1, fin);
//        printf("%u %u \n", buffer2[0], buffer2[1]);
//        fwrite(buffer2, sizeof(buffer2), 1, fout);
//        
//        header.channels = buffer2[0] | (buffer2[1] << 8);
//        printf("(23-24) Channels: %u \n", header.channels);
//        if (header.channels != 1)
//        {
//            printf("Channels are more than 1.\n");
//            fclose(fin);
//            fclose(fout);
//            exit(1);
//        }
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        fwrite(buffer4, sizeof(buffer4), 1, fout);
//        
//        header.sample_rate = buffer4[0] |
//                               (buffer4[1] << 8) |
//                               (buffer4[2] << 16) |
//                               (buffer4[3] << 24);
//        
//        printf("(25-28) Sample rate: %u Hz\n", header.sample_rate);
//        if (header.sample_rate != SAMPLING_FREQUENCY)
//        {
//            printf("Sampling rate is not %d Hz\n", SAMPLING_FREQUENCY);
//            fclose(fin);
//            fclose(fout);
//            exit(1);
//        }
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        fwrite(buffer4, sizeof(buffer4), 1, fout);
//        
//        header.byterate  = buffer4[0] |
//                               (buffer4[1] << 8) |
//                               (buffer4[2] << 16) |
//                               (buffer4[3] << 24);
//        printf("(29-32) Byte Rate: %u B/s , Bit Rate:%u b/s\n", header.byterate, header.byterate*8);
//        
//        read = fread(buffer2, sizeof(buffer2), 1, fin);
//        printf("%u %u \n", buffer2[0], buffer2[1]);
//        fwrite(buffer2, sizeof(buffer2), 1, fout);
//        
//        header.block_align = buffer2[0] |
//                           (buffer2[1] << 8);
//        printf("(33-34) Block Alignment: %u \n", header.block_align);
//        
//        read = fread(buffer2, sizeof(buffer2), 1, fin);
//        printf("%u %u \n", buffer2[0], buffer2[1]);
//        fwrite(buffer2, sizeof(buffer2), 1, fout);
//        
//        header.bits_per_sample = buffer2[0] |
//                           (buffer2[1] << 8);
//        printf("(35-36) Bits per sample: %u \n", header.bits_per_sample);
//        
//        read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fin);
//        printf("(37-40) Data Marker: %s \n", header.data_chunk_header);
//        fwrite(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fout);
//        
//        read = fread(buffer4, sizeof(buffer4), 1, fin);
//        printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
//        
//        header.data_size = buffer4[0] |
//                       (buffer4[1] << 8) |
//                       (buffer4[2] << 16) | 
//                       (buffer4[3] << 24 );
//        printf("(41-44) Size of data chunk: %u B \n", header.data_size);
//        
//        // calculate no.of samples
//        long num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);
//        printf("Number of samples: %lu \n", num_samples);
//        // first samples are of the right-side hanning window, i.e., 1st frame,
//        // the remaining samples are then divided by FRAME_SHIFT to get the remaining frames
//        long num_frame = 1 + (num_samples - RIGHT_SAMPLES) / FRAME_SHIFT;
//        // if theere exist remainder samples of the 2nd-Nth frame,
//        // use reflected samples of the last buffer for the missing amount of samples to process one last frame
//        long num_reflected_right_edge_samples = (num_samples - RIGHT_SAMPLES) % FRAME_SHIFT;
//        if (num_reflected_right_edge_samples > 0) {
//            num_reflected_right_edge_samples = FRAME_SHIFT - num_reflected_right_edge_samples;
//            num_frame += 1; //use additional reflected samples for the trailing remainder samples on the right-edge
//        }
//        long num_samples_frame = num_frame * FRAME_SHIFT;
//        printf("Number of samples in processed output wav: %lu \n", num_samples_frame);
//        header.data_size = (num_samples_frame * header.channels * header.bits_per_sample) / 8;
//        printf("(41-44) Size of data chunk to be written: %u B \n", header.data_size);
//        fwrite(&header.data_size, sizeof(header.data_size), 1, fout);
//        
//        long size_of_each_sample = (header.channels * header.bits_per_sample) / 8;
//        printf("Size of each sample: %ld B\n", size_of_each_sample);
//        
//        // calculate duration of file
//        float duration_in_seconds = (float) header.overall_size / header.byterate;
//        printf("Approx.Duration in seconds= %f\n", duration_in_seconds);
//        
//        // read each sample from data chunk if PCM
//        if (header.format_type == 1 && header.channels == 1) { // PCM and mono
//            clock_t t = clock();
//            DSPState *dsp;
//            MWDLP8NetState *net;
//    
//            float features[FEATURES_DIM];
//            short pcm[MAX_N_OUTPUT]; //output is in short 2-byte (16-bit) format [-32768,32767]
//            int first_buffer_flag = 0;
//            int waveform_buffer_flag = 0;
//            int n_output = 0;
//            int samples = 0;
//            float data_in_channel = 0;
//            int i, j;
//            int k;
//            char data_buffer[size_of_each_sample];
//            float x_buffer[FRAME_SHIFT];
//    
//            // initialize waveform-->features processing struct
//            dsp = dspstate_create();
//    
//            // initialize mwdlp+cyclevae struct
//            net = mwdlp8net_create();
//
//            //FILE *tmp_file5;
//            //tmp_file5 = fopen("melsp.txt", "wt");
//            for (i = 0, j = 0, k = 0; i < num_samples; i++) {
//                //printf("==========Sample %d / %ld=============\n", i+1, num_samples);
//                read = fread(data_buffer, sizeof(data_buffer), 1, fin);
//                if (read == 1) {
//                
//                    /* Receives only mono 16-bit PCM, convert to float [-1,0.999969482421875] */
//                    data_in_channel = ((float) ((data_buffer[0] & 0x00ff) | (data_buffer[1] << 8))) / 32768;
//                    //printf("%f\n", data_in_channel);
//    
//                    // high-pass filter to remove DC component of recording device
//                    shift_apply_hpassfilt(dsp, &data_in_channel);
//            
//                    //check waveform buffer here
//                    if (first_buffer_flag) { //first frame has been processed, now taking every FRAME_SHIFT amount of samples
//                    //    printf("after first frame\n");
//                        x_buffer[j] = data_in_channel;
//                        j += 1;
//                        if (j >= FRAME_SHIFT) {
//                            shift_apply_window(dsp, x_buffer); //shift old FRAME_SHIFT amount for new FRAME_SHIFT amount and window
//                            waveform_buffer_flag = 1;
//                            j = 0;
//                            k += 1;
//                            printf(" [%d]", k);
//                        }
//                    } else { //take RIGHT_SAMPLES amount of samples as the first samples
//                        //put only LEFT_SAMPLES and RIGHT_SAMPLES amount in window buffer, because zero-padding until FFT_LENGTH with centered window position
//                        //(i//FRAME_SHIFT)th frame = i*FRAME_SHIFT [i: 0->(n_samples-1)]
//                    //    printf("first frame\n");
//                        dsp->samples_win[LEFT_SAMPLES_1+i] = data_in_channel;
//                        if (i <= LEFT_SAMPLES_2) //reflect only LEFT_SAMPLES-1 amount because 0 value for the 1st coeff. of window
//                            dsp->samples_win[LEFT_SAMPLES_2-i] = data_in_channel; 
//                        if (i >= RIGHT_SAMPLES_1) { //process current buffer, and next, take for every FRAME_SHIFT amount samples
//                            apply_window(dsp); //hanning window
//                            first_buffer_flag = 1;
//                            waveform_buffer_flag = 1;
//                            k += 1;
//                            printf("frame: [%d]", k);
//                        }
//                    }
//                    //printf("wav buffer\n");
//    
//                    if (waveform_buffer_flag) {
//                        //extract melspectrogram here
//                        mel_spec_extract(dsp, features);
//                        //printf("melsp extract, nth-frame: %d\n", k);
//                        if (print_melsp_flag) fwrite((void*)features, sizeof(features[0]), FEATURES_DIM, fout_msp);
//    
//                        //if (k > FEATURE_CONV_DELAY) {
//                        //    for (l=0;l<FEATURES_DIM;l++)
//                        //        if (l < (FEATURES_DIM-1))
//                        //            fprintf(tmp_file5, "%f ", features[l]);
//                        //        else
//                        //            fprintf(tmp_file5, "%f\n", features[l]);
//                        //}
//                        mwdlp8net_synthesize(net, features, pcm, &n_output, 0);
//            
//                        if (n_output > 0)  { //delay is reached, samples are generated
//                            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
//                            samples += n_output;
//                        //    printf("write %d\n", samples);
//                        }
//    
//                        waveform_buffer_flag = 0;
//                    }
//                    //printf("buffer processed\n");
//                } else {
//                    printf("\nError reading file. %d bytes\n", read);
//                    fclose(fin);
//                    fclose(fout);
//                    if (print_melsp_flag) fclose(fout_msp);
//                    dspstate_destroy(dsp);
//                    mwdlp8net_destroy(net);
//                    exit(1);
//                }
//            }
//        
//            if (!waveform_buffer_flag && j > 0) {
//                //set additional reflected samples for trailing remainder samples on the right edge here
//                int k;
//                for (i = 0, k=j-1; i < num_reflected_right_edge_samples; i++, j++)
//                    x_buffer[j] = x_buffer[k-i];
//    
//                if (j != FRAME_SHIFT) {
//                    printf("\nError remainder right-edge samples calculation %d %d %ld\n", j, FRAME_SHIFT, num_reflected_right_edge_samples);
//                    fclose(fin);
//                    fclose(fout);
//                    if (print_melsp_flag) fclose(fout_msp);
//                    dspstate_destroy(dsp);
//                    mwdlp8net_destroy(net);
//                    exit(1);
//                } else {
//                    printf(" [last frame]\n");
//                }
//    
//                mel_spec_extract(dsp, features);
//                if (print_melsp_flag) fwrite((void*)features, sizeof(features[0]), FEATURES_DIM, fout_msp);
//    
//                //    for (l=0;l<FEATURES_DIM;l++)
//                //    if (l < (FEATURES_DIM-1))
//                //        fprintf(tmp_file5, "%f ", features[l]);
//                //    else
//                //        fprintf(tmp_file5, "%f\n", features[l]);
//                mwdlp8net_synthesize(net, features, pcm, &n_output, 0); //last_frame, synth pad_right
//    
//                if (n_output > 0)  {
//                    fwrite(pcm, sizeof(pcm[0]), n_output, fout);
//                    samples += n_output;
//                //    printf("write %d\n", samples);
//                }
//                mwdlp8net_synthesize(net, features, pcm, &n_output, 1); //synth pad_right
//                if (n_output > 0)  {
//                    fwrite(pcm, sizeof(pcm[0]), n_output, fout);
//                    samples += n_output;
//                //    printf("write pad_right %d\n", samples);
//                }
//            }
//        
//            t = clock() - t;
//            double time_taken = ((double)t)/CLOCKS_PER_SEC;
//            //printf("%d [frames], %d [samples] synthesis in %f seconds \n", idx, samples, time_taken);
//            printf("%d [frames] %d [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
//                "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
//                (int)((double)samples/FRAME_SHIFT), samples, (double)samples/SAMPLING_FREQUENCY, time_taken,
//                    ((double)samples/SAMPLING_FREQUENCY)/time_taken, time_taken/((double)samples/SAMPLING_FREQUENCY),
//                        N_SAMPLE_BANDS*((double)samples/SAMPLING_FREQUENCY)/time_taken);
//    
//            fclose(fin);
//            fclose(fout);
//            if (print_melsp_flag) fclose(fout_msp);
//            dspstate_destroy(dsp);
//            mwdlp8net_destroy(net);
//        } else {
//            printf("Error not mono and 16-bit pcm\n");
//            fclose(fin);
//            fclose(fout);
//            if (print_melsp_flag) fclose(fout_msp);
//            exit(1);
//        }
//
//    }
//
//    return 0;
//}
