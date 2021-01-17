/*
   Copyright 2021 Patrick Lumban Tobing (Nagoya University)
   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

   WAV file read/write is based on http://truelogic.org/wordpress/2015/09/04/parsing-a-wav-file-in-c

   Argument parser is based on SPTK-3.11 [https://sourceforge.net/projects/sp-tk/files/SPTK/]
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_STRING_H
#include <string.h>
#else
#include <strings.h>
#ifndef HAVE_STRRCHR
#define strrchr rindex
#endif
#endif

#include <time.h>
#include <unistd.h>
#include "mwdlp10net_cycvae.h"
#include "freq.h"
#include "nnet.h"
#include "nnet_data.h"
#include "nnet_cv_data.h"
#include "mwdlp10net_cycvae_private.h"
#include "wave.h"


/* Command Name  */
char *cmnd;

void usage(int status)
{
   fprintf(stderr, "\n");
   fprintf(stderr, " %s - Multiband WaveRNN with data-driven Linear Prediction\n", cmnd);
   fprintf(stderr, "                using 10-bit mu-law coarse-fine output architecture\n");
   fprintf(stderr, "                with Voice Conversion function using sparse CycleVAE\n");
   fprintf(stderr, "\n");
   fprintf(stderr, "  usage:\n");
   fprintf(stderr, "       %s [ options ] [ infile ] [ outfile ]\n", cmnd);
   fprintf(stderr, "  options:\n");
   fprintf(stderr, "       -i [speaker-index] : mel-spec converted to speaker-index's voice [ignored if -c used]\n");
   fprintf(stderr, "       -c [speaker-x-coord] [speaker-y-coord] : mel-spec converted to speaker-coord's voice [interpolated voice]\n");
   fprintf(stderr, "       -o [outfile binary mel-spec] [outfile text mel-spec] : print mel-spec [ignored if -b/-t used and -i/-c not used]\n");
   fprintf(stderr, "       -b    : flag to decode using input binary mel-spec [ignored if -t used]\n");
   fprintf(stderr, "       -t    : flag to decode using input text mel-spec\n");
   fprintf(stderr, "       -h    : print this message\n");
   fprintf(stderr, "  infile:\n");
   fprintf(stderr, "       input wav-file/mel-spec       [default wav-file / binary mel-spec / text mel-spec]\n");
   fprintf(stderr, "  outfile:\n");
   fprintf(stderr, "       output wav-file\n");
   fprintf(stderr, "\n");
   exit(status);
}


int main(int argc, char **argv) {
    short wav_in_flag = 1;
    short melsp_bin_in_flag = 0;
    short melsp_txt_in_flag = 0;
    short cv_point_flag = 0;
    short cv_interp_flag = 0;
    short print_melsp_flag = 0;
    short spk_idx;
    float x_coord, y_coord;
    FILE *fin, *fout_msp_bin, *fout_msp_txt;

    if ((cmnd = strrchr(argv[0], '/')) == NULL) {
        cmnd = argv[0];
    } else {
        cmnd++;
    }

    while (--argc) {
        if (**++argv == '-') {
            switch (*(*argv + 1)) {
                case 'i':
                    cv_point_flag = 1;
                    spk_idx = atoi(*++argv, "rb");
                    if (spk_idx > FEATURE_N_SPK) {
	                    fprintf(stderr, "Speaker id %d is more than n_spk %d\n", spk_idx, FEATURE_N_SPK);
                        if (print_melsp_flag) {
                            fclose(fout_msp_bin);
                            fclose(fout_msp_txt);
                        }
	                    exit(1);
                    }
                    --argc;
                    break;
                case 'c':
                    cv_interp_flag = 1;
                    x_coord = atof(*++argv, "rb");
                    --argc;
                    y_coord = atof(*++argv, "rb");
                    --argc;
                    break;
                case 'e':
                    print_melsp_flag = 1;
                    //file handler melsp binary output
                    fout_msp_bin = fopen(*++argv, "wb");
                    if (fout_msp_bin == NULL) {
                        fprintf(stderr, "Can't open %s\n", *argv);
                        exit(1);
                    }
                    --argc;
                    //file handler melsp text output
                    fout_msp_txt = fopen(*++argv, "w");
                    if (fout_msp_txt == NULL) {
                        fprintf(stderr, "Can't open %s\n", *argv);
                        fclose(fout_msp_bin);
                        exit(1);
                    }
                    --argc;
                    break;
                case 'b':
                    melsp_bin_in_flag = 1;
                    break;
                case 't':
                    melsp_txt_in_flag = 1;
                    break;
                case 'h':
                    usage(0);
                    break;
                default:
                    fprintf(stderr, "%s : Invalid option '%c'!\n", cmnd, *(*argv + 1));
                    usage(1);
            }
        } else {
            //file handler wav/binary-melsp/text-melsp input
            if (melsp_txt_in_flag) {
                fin = fopen(*++argv, "r");
                melsp_bin_in_flag = 0;
            } else fin = fopen(*++argv, "rb");
            if (fin == NULL) {
	            fprintf(stderr, "Can't open %s\n", *argv);
                if (print_melsp_flag) {
                    fclose(fout_msp_bin);
                    fclose(fout_msp_txt);
                }
	            exit(1);
            }
        }
    }

    //use input melsp
    if (melsp_txt_in_flag || melsp_bin_in_flag) wav_in_flag = 0;

    //use input melsp, only synthesis without conversion
    if (print_melsp_flag && !wav_in_flag && !cv_point_flag && !cv_interp_flag) {
        print_melsp_flag = 0;
        fclose(fout_msp_bin);
        fclose(fout_msp_txt);
    }

    // ignore spk-index option if interpolated is used
    if (cv_interp_flag) cv_point_flag = 0;

    //file handler wav output
    FILE *fout = fopen(*argv, "wb");
    if (fout == NULL) {
        fprintf(stderr, "Can't open %s\n", *argv);
        if (print_melsp_flag) {
            fclose(fout_msp_bin);
            fclose(fout_msp_txt);
        }
        fclose(fin);
        exit(1);
    }

    if (print_melsp_flag) {
        short l;
        short features_dim_1 = FEATURES_DIM - 1;
    }

    srand (time(NULL));
    clock_t t = clock();

    if (wav_in_flag) { //waveform input
        short num_reflected_right_edge_samples;
        long num_samples, size_of_each_sample;

        //read wave header input and initialize wave output header
        if (read_write_wav(fin, fout, &num_reflected_right_edge_samples, &num_samples, &size_of_each_sample)) {

            float features[FEATURES_DIM];
            short pcm[MAX_N_OUTPUT]; //output is in short 2-byte (16-bit) format [-32768,32767]
            short first_buffer_flag = 0;
            short waveform_buffer_flag = 0;
            int n_output = 0;
            long samples = 0;
            float data_in_channel = 0;
            short i, j, k;
            char data_buffer[size_of_each_sample];
            float x_buffer[FRAME_SHIFT];
            int read = 0;

            // initialize waveform-->features processing struct
            DSPState *dsp;
            dsp = dspstate_create();

            if (!cv_point_flag && !cv_interp_flag) { //analysis-synthesis
                // initialize mwdlp struct
                MWDLP10NetState *net;
                net = mwdlp10net_create();

                for (i = 0, j = 0, k = 0; i < num_samples; i++) {
                    if ((read = fread(data_buffer, sizeof(data_buffer), 1, fin)) == 1) {
                    
                        /* Receives only mono 16-bit PCM, convert to float [-1,0.999969482421875] */
                        data_in_channel = ((float) ((data_buffer[0] & 0x00ff) | (data_buffer[1] << 8))) / 32768;

                        // high-pass filter to remove DC component of recording device
                        shift_apply_hpassfilt(dsp, &data_in_channel);
                
                        //check waveform buffer here
                        if (first_buffer_flag) { //first frame has been processed, now taking every FRAME_SHIFT amount of samples
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
                            // put only LEFT_SAMPLES and RIGHT_SAMPLES amount in window buffer,
                            // because zero-padding until FFT_LENGTH with centered window position
                            // (i//FRAME_SHIFT)th frame = i*FRAME_SHIFT [i: 0->(n_samples-1)]
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

                        if (waveform_buffer_flag) {
                            //extract melspectrogram here
                            mel_spec_extract(dsp, features);

                            if (print_melsp_flag) {
                                fwrite(features, sizeof(features), 1, fout_msp_bin);
                                for (l=0;l<FEATURES_DIM;l++)
                                    if (l < features_dim_1)
                                        fprintf(fout_msp_txt, "%f ", features[l]);
                                    else
                                        fprintf(fout_msp_txt, "%f\n", features[l]);
                            }

                            mwdlp10net_synthesize(net, features, pcm, &n_output, 0);
                
                            if (n_output > 0)  { //delay is reached, samples are generated
                                fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                                samples += n_output;
                            }

                            waveform_buffer_flag = 0;
                        }
                    } else {
                        fprintf(stderr, "\nError reading file. %d bytes\n", read);
                        fclose(fin);
                        fclose(fout);
                        if (print_melsp_flag) {
                            fclose(fout_msp_bin);
                            fclose(fout_msp_txt);
                        }
                        dspstate_destroy(dsp);
                        mwdlp10net_destroy(net);
                        exit(1);
                    }
                }
        
                if (!waveform_buffer_flag && j > 0) {
                    //set additional reflected samples for trailing remainder samples on the right edge here
                    for (i = 0, k=j-1; i < num_reflected_right_edge_samples; i++, j++)
                        x_buffer[j] = x_buffer[k-i];

                    if (j == FRAME_SHIFT) printf(" [last frame]\n");
                    else {
                        fprintf(stderr, "\nError remainder right-edge samples calculation %d %d %ld\n", j, FRAME_SHIFT, num_reflected_right_edge_samples);
                        fclose(fin);
                        fclose(fout);
                        if (print_melsp_flag) {
                            fclose(fout_msp_bin);
                            fclose(fout_msp_txt);
                        }
                        dspstate_destroy(dsp);
                        mwdlp10net_destroy(net);
                        exit(1);
                    }

                    mel_spec_extract(dsp, features);
                    if (print_melsp_flag) {
                        fwrite(features, sizeof(features), 1, fout_msp_bin);
                        for (l=0;l<FEATURES_DIM;l++)
                            if (l < features_dim_1)
                                fprintf(fout_msp_txt, "%f ", features[l]);
                            else
                                fprintf(fout_msp_txt, "%f\n", features[l]);
                    }

                    mwdlp10net_synthesize(net, features, pcm, &n_output, 0); //last_frame, synth pad_right

                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    }

                    mwdlp10net_synthesize(net, features, pcm, &n_output, 1); //synth pad_right

                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    }
                }
    
                fclose(fin);
                fclose(fout);
                if (print_melsp_flag) {
                    fclose(fout_msp_bin);
                    fclose(fout_msp_txt);
                }
                dspstate_destroy(dsp);
                mwdlp10net_destroy(net);
            } else { //analysis-conversion-synthesis
                // initialize mwdlp+cyclevae struct
                MWDLP10CycleVAEMelspExcitSpkNetState *net;
                net = mwdlp10cyclevaenet_create();

                // set spk-conditioning here
                float spk_code_aux[FEATURE_N_SPK_2];
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
                    }
                    printf("\n");
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

                for (i = 0, j = 0, k = 0; i < num_samples; i++) {
                    read = fread(data_buffer, sizeof(data_buffer), 1, fin);
                    if (read == 1) {
                    
                        /* Receives only mono 16-bit PCM, convert to float [-1,0.999969482421875] */
                        data_in_channel = ((float) ((data_buffer[0] & 0x00ff) | (data_buffer[1] << 8))) / 32768;

                        // high-pass filter to remove DC component of recording device
                        shift_apply_hpassfilt(dsp, &data_in_channel);
                
                        //check waveform buffer here
                        if (first_buffer_flag) { //first frame has been processed, now taking every FRAME_SHIFT amount of samples
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
                            // put only LEFT_SAMPLES and RIGHT_SAMPLES amount in window buffer,
                            // because zero-padding until FFT_LENGTH with centered window position
                            // (i//FRAME_SHIFT)th frame = i*FRAME_SHIFT [i: 0->(n_samples-1)]
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

                        if (waveform_buffer_flag) {
                            mel_spec_extract(dsp, features);
                            if (print_melsp_flag) {
                                fwrite(features, sizeof(features), 1, fout_msp_bin);
                                for (l=0;l<FEATURES_DIM;l++)
                                    if (l < features_dim_1)
                                        fprintf(fout_msp_txt, "%f ", features[l]);
                                    else
                                        fprintf(fout_msp_txt, "%f\n", features[l]);
                            }
                
                            cyclevae_melsp_excit_spk_convert_mwdlp10net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0);
                
                            if (n_output > 0)  { //delay is reached, samples are generated
                                fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                                samples += n_output;
                            }

                            waveform_buffer_flag = 0;
                        }
                    } else {
                        fprintf(stderr, "\nError reading file. %d bytes\n", read);
                        fclose(fin);
                        fclose(fout);
                        if (print_melsp_flag) {
                            fclose(fout_msp_bin);
                            fclose(fout_msp_txt);
                        }
                        dspstate_destroy(dsp);
                        mwdlp10cyclevaenet_destroy(net);
                        exit(1);
                    }
                }
        
                if (!waveform_buffer_flag && j > 0) {
                    //set additional reflected samples for trailing remainder samples on the right edge here
                    int k;
                    for (i = 0, k=j-1; i < num_reflected_right_edge_samples; i++, j++)
                        x_buffer[j] = x_buffer[k-i];

                    if (j == FRAME_SHIFT) printf(" [last frame]\n");
                    else {
                        fprintf(stderr, "\nError remainder right-edge samples calculation %d %d %ld\n", j, FRAME_SHIFT, num_reflected_right_edge_samples);
                        fclose(fin);
                        fclose(fout);
                        dspstate_destroy(dsp);
                        mwdlp10cyclevaenet_destroy(net);
                        exit(1);
                    }

                    mel_spec_extract(dsp, features);
                    if (print_melsp_flag) {
                        fwrite(features, sizeof(features), 1, fout_msp_bin);
                        for (l=0;l<FEATURES_DIM;l++)
                            if (l < features_dim_1)
                                fprintf(fout_msp_txt, "%f ", features[l]);
                            else
                                fprintf(fout_msp_txt, "%f\n", features[l]);
                    }

                    cyclevae_melsp_excit_spk_convert_mwdlp10net_synthesize(net, features, spk_code_aux, pcm, &n_output, 0); //last_frame, synth pad_right

                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    }

                    cyclevae_melsp_excit_spk_convert_mwdlp10net_synthesize(net, features, spk_code_aux, pcm, &n_output, 1); //synth pad_right

                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    }
                }
    
                fclose(fin);
                fclose(fout);
                if (print_melsp_flag) {
                    fclose(fout_msp_bin);
                    fclose(fout_msp_txt);
                }
                dspstate_destroy(dsp);
                mwdlp10cyclevaenet_destroy(net);
            }
        } else {
            if (print_melsp_flag) {
                fclose(fout_msp_bin);
                fclose(fout_msp_txt);
            }
            exit(1);
        }
    } else { //melsp input
        long num_frame;

        //read input features and initialize wave output header
        if ((num_frame = read_feat_write_wav(fin, fout, melsp_bin_in_flasg) > 0) {

            float features[FEATURES_DIM];
            short pcm[MAX_N_OUTPUT]; //output is in short 2-byte (16-bit) format [-32768,32767]
            int n_output = 0;
            long samples = 0;
            short i, j, k;

            if (!cv_point_flag && !cv_interp_flag) { //analysis-synthesis
                // initialize mwdlp struct
                MWDLP10NetState *net;
                net = mwdlp10net_create();

                char c;
                char *buffer = (char*) calloc(128, sizeof(char));
                short flag = 0; //mark current read is in a column
                short frame = 0; //mark process rame
                i = 0; //character counter
                j = 0; //column counter
                k = 0; //frame counter
                while ((c = getc(fin)) != EOF) { //read per character
                    if (flag) { //within a column
                        if (c == ' ') { //add column
                            features[j] = atof(buffer);
                            memset(buffer,'\0',i);
                            j++;
                            i = 0;
                            flag = 0;
                        } else if (c == '\n') { //found end-of-line
                            features[j] = atof(buffer);
                            memset(buffer,'\0',i);
                            j++;
                            if (j == FEATURES_DIM) { //add row, process frame
                                k++;
                                i = 0;
                                j = 0;
                                flag = 0;
                                frame = 1;
                            } else { //columns not appropriate
                                fprintf(stderr, "Error input text format %d %d\n", j, FEATURES_DIM);
                                fclose(fin);
                                fclose(fout);
                                mwdlp10net_destroy(net);
                                exit(1);
                            }
                        } else {
                            buffer[i] = c;
                            i++;
                        }
                    } else { //finding new column
                        if (c != ' ' && c != '\n') { //add starting column character
                            flag = 1;
                            buffer[i] = c;
                            i++;
                        } else if (c == '\n') { //found end-of-line
                            if (j == FEATURES_DIM) { //add row, process frame
                                k++;
                                i = 0;
                                j = 0;
                                frame = 1;
                            } else { //columns not appropriate
                                fprintf(stderr, "Error input text format  %d %d\n", j, FEATURES_DIM);
                                fclose(fin);
                                fclose(fout);
                                mwdlp10net_destroy(net);
                                exit(1);
                            }
                        }
                    }
                    if (frame) {
                        if (k < num_frame) printf("frame: [%d]", k);
                        else printf(" [last frame]\n");

                        mwdlp10net_synthesize(net, features, pcm, &n_output, 0);
                
                        if (n_output > 0)  { //delay is reached, samples are generated
                            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                            samples += n_output;
                        }

                        frame = 0;
                    }
                }

                if (k == num_frame) {
                    mwdlp10net_synthesize(net, features, pcm, &n_output, 1); //synth pad_right

                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    }
                } else {
                    fprintf(stderr, "\nError input frames  %d -- %d\n", k, num_frame);
                    fclose(fin);
                    fclose(fout);
                    mwdlp10net_destroy(net);
                    exit(1);
                }
    
                fclose(fin);
                fclose(fout);
                mwdlp10net_destroy(net);
            } else { //analysis-conversion-synthesis
                // initialize mwdlp+cyclevae struct
                MWDLP10CycleVAEMelspExcitSpkNetState *net;
                net = mwdlp10cyclevaenet_create();

                // set spk-conditioning here
                float spk_code_aux[FEATURE_N_SPK_2];
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
                    }
                    printf("\n");
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

                char c;
                char *buffer = (char*) calloc(128, sizeof(char));
                short flag = 0; //mark current read is in a column
                short frame = 0; //mark process rame
                i = 0; //character counter
                j = 0; //column counter
                k = 0; //frame counter
                while ((c = getc(fin)) != EOF) { //read per character
                    if (flag) { //within a column
                        if (c == ' ') { //add column
                            features[j] = atof(buffer);
                            memset(buffer,'\0',i);
                            j++;
                            i = 0;
                            flag = 0;
                        } else if (c == '\n') { //found end-of-line
                            features[j] = atof(buffer);
                            memset(buffer,'\0',i);
                            j++;
                            if (j == FEATURES_DIM) { //add row, process frame
                                k++;
                                i = 0;
                                j = 0;
                                flag = 0;
                                frame = 1;
                            } else { //columns not appropriate
                                fprintf(stderr, "Error input text format %d %d\n", j, FEATURES_DIM);
                                fclose(fin);
                                fclose(fout);
                                if (print_melsp_flag) {
                                    fclose(fout_msp_bin);
                                    fclose(fout_msp_txt);
                                }
                                mwdlp10cyclevaenet_destroy(net);
                                exit(1);
                            }
                        } else {
                            buffer[i] = c;
                            i++;
                        }
                    } else { //finding new column
                        if (c != ' ' && c != '\n') { //add starting column character
                            flag = 1;
                            buffer[i] = c;
                            i++;
                        } else if (c == '\n') { //found end-of-line
                            if (j == FEATURES_DIM) { //add row, process frame
                                k++;
                                i = 0;
                                j = 0;
                                frame = 1;
                            } else { //columns not appropriate
                                fprintf(stderr, "Error input text format  %d %d\n", j, FEATURES_DIM);
                                fclose(fin);
                                fclose(fout);
                                if (print_melsp_flag) {
                                    fclose(fout_msp_bin);
                                    fclose(fout_msp_txt);
                                }
                                mwdlp10cyclevaenet_destroy(net);
                                exit(1);
                            }
                        }
                    }
                    if (frame) {
                        if (k < num_frame) printf("frame: [%d]", k);
                        else printf(" [last frame]\n");

                        cyclevae_melsp_excit_spk_convert_mwdlp10net_synthesize_feat_in(net, features, spk_code_aux, pcm, &n_output, 0);
                
                        if (print_melsp_flag) {
                            fwrite(features, sizeof(features), 1, fout_msp_bin);
                            for (l=0;l<FEATURES_DIM;l++)
                                if (l < features_dim_1)
                                    fprintf(fout_msp_txt, "%f ", features[l]);
                                else
                                    fprintf(fout_msp_txt, "%f\n", features[l]);
                        }

                        if (n_output > 0)  { //delay is reached, samples are generated
                            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                            samples += n_output;
                        }

                        frame = 0;
                    }
                }

                if (k == num_frame) {
                    cyclevae_melsp_excit_spk_convert_mwdlp10net_synthesize(net, features, spk_code_aux, pcm, &n_output, 1); //synth pad_right

                    if (n_output > 0)  {
                        fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                        samples += n_output;
                    }
                } else {
                    fprintf(stderr, "\nError input frames  %d -- %d\n", k, num_frame);
                    fclose(fin);
                    fclose(fout);
                    if (print_melsp_flag) {
                        fclose(fout_msp_bin);
                        fclose(fout_msp_txt);
                    }
                    mwdlp10cyclevaenet_destroy(net);
                    exit(1);
                }
    
                fclose(fin);
                fclose(fout);
                if (print_melsp_flag) {
                    fclose(fout_msp_bin);
                    fclose(fout_msp_txt);
                }
                mwdlp10cyclevaenet_destroy(net);
            }
        } else {
            if (print_melsp_flag) {
                fclose(fout_msp_bin);
                fclose(fout_msp_txt);
            }
            exit(1);
        }
    }

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("%d [frames] %d [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
        "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
        (int)((double)samples/FRAME_SHIFT), samples, (double)samples/SAMPLING_FREQUENCY, time_taken,
            ((double)samples/SAMPLING_FREQUENCY)/time_taken, time_taken/((double)samples/SAMPLING_FREQUENCY),
                N_SAMPLE_BANDS*((double)samples/SAMPLING_FREQUENCY)/time_taken);

    return 0;
}
