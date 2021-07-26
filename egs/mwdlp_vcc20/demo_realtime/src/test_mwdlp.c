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
#include "mwdlp10net.h"
#include "freq.h"
#include "nnet.h"
#include "nnet_data.h"
#include "mwdlp10net_private.h"
#include "wave.h"


/* Command Name  */
char *cmnd;

void usage(int status)
{
   fprintf(stderr, "\n");
   fprintf(stderr, " %s - Multiband WaveRNN with data-driven Linear Prediction\n", cmnd);
   fprintf(stderr, "                using 10-bit mu-law coarse-fine output architecture\n");
   fprintf(stderr, "\n");
   fprintf(stderr, "  usage:\n");
   fprintf(stderr, "       %s [ options ] [ infile ] [ outfile ]\n", cmnd);
   fprintf(stderr, "  options:\n");
   fprintf(stderr, "       -o [outfile binary mel-spec] [outfile text mel-spec] : print mel-spec [ignored if -b/-t used]\n");
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
    short melsp_bin_in_flag = 0, melsp_txt_in_flag = 0;
    short print_melsp_flag = 0;
    int n_argc = argc;
    /*
    int m; 
    short idx; 
    FILE* fout_ddlpc_coarse[N_MBANDS];
    FILE* fout_ddlpc_fine[N_MBANDS];
    FILE* fout_band[N_MBANDS];
    */
    FILE *fin = NULL, *fout = NULL, *fout_msp_bin = NULL, *fout_msp_txt = NULL;

    if ((cmnd = strrchr(argv[0], '/')) == NULL) {
        cmnd = argv[0];
    } else {
        cmnd++;
    }

    while (--argc) {
        if (**++argv == '-') {
            switch (*(*argv + 1)) {
                case 'o':
                    print_melsp_flag = 1;
                    //file handler melsp binary output
                    fout_msp_bin = fopen(*++argv, "wb");
                    if (fout_msp_bin == NULL) {
                        fprintf(stderr, "Can't open fout_bin %s\n", *argv);
                        exit(1);
                    }
                    --argc;
                    //file handler melsp text output
                    fout_msp_txt = fopen(*++argv, "w");
                    if (fout_msp_txt == NULL) {
                        fprintf(stderr, "Can't open fout_txt %s\n", *argv);
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
                    if (print_melsp_flag) {
                        fclose(fout_msp_bin);
                        fclose(fout_msp_txt);
                    }
                    usage(0);
                    break;
                default:
                    fprintf(stderr, "%s : Invalid option '%c'!\n", cmnd, *(*argv + 1));
                    usage(1);
            }
        } else {
            if (n_argc < 3) {
                fprintf(stderr, "Invalid usage n_arg\n");
                usage(1);
            }

            if (argc == 2) {
                //file handler wav/binary-melsp/text-melsp input
                if (melsp_txt_in_flag) {
                    fin = fopen(*argv, "r");
                    melsp_bin_in_flag = 0;
                } else fin = fopen(*argv, "rb");
                if (fin == NULL) {
	                fprintf(stderr, "Can't open fin %s\n", *argv);
                    if (print_melsp_flag) {
                        fclose(fout_msp_bin);
                        fclose(fout_msp_txt);
                    }
	                exit(1);
                }
           } else if (argc == 1) {
                //file handler wav output
                fout = fopen(*argv, "wb");
                /*
                size_t len_argv = strlen(*argv)-4;
                //printf("%ld\n", len_argv);
                size_t len_name_coarse = len_argv+16+1+4+1;
                size_t len_name_fine = len_argv+14+1+4+1;
                size_t len_name_band = len_argv+3+1+4+1;
                char fout_name_coarse[N_MBANDS][len_name_coarse]; //remove .wav ext + _ddlpc_coarse_B-<>.txt
                char fout_name_fine[N_MBANDS][len_name_fine]; //remove .wav ext + _ddlpc_fine_B-<>.txt
                char fout_name_band[N_MBANDS][len_name_band]; //remove .wav ext + _B-<>.txt
                char tmp_name[len_argv+1];
                //strncpy(tmp_name, *argv, len_argv);
                strcpy(tmp_name, *argv);
                tmp_name[len_argv] = '\0';
                //printf("%s\n", tmp_name);
                for (idx=0; idx<N_MBANDS; idx++) {
                    sprintf(fout_name_coarse[idx], "%s%s%d%s", tmp_name, "_ddlpc_coarse_B-", idx+1, ".txt");
                    sprintf(fout_name_fine[idx], "%s%s%d%s", tmp_name, "_ddlpc_fine_B-", idx+1, ".txt");
                    sprintf(fout_name_band[idx], "%s%s%d%s", tmp_name, "_B-", idx+1, ".wav");
                    fout_name_coarse[idx][len_name_coarse] = '\0';
                    fout_name_fine[idx][len_name_fine] = '\0';
                    fout_name_band[idx][len_name_band] = '\0';
                //    printf("%d %s %s %s %s\n", idx, tmp_name, fout_name_coarse[idx], fout_name_fine[idx], fout_name_band[idx]);
                    fout_ddlpc_coarse[idx] = fopen(fout_name_coarse[idx], "w");
                    if (fout_ddlpc_coarse[idx] == NULL) {
                        fprintf(stderr, "Can't open fout_ddlpc_coarse %s\n", fout_name_coarse[idx]);
                        exit(1);
                    }
                    fout_ddlpc_fine[idx] = fopen(fout_name_fine[idx], "w");
                    if (fout_ddlpc_fine[idx] == NULL) {
                        fprintf(stderr, "Can't open fout_ddlpc_fine %s\n", fout_name_fine[idx]);
                        exit(1);
                    }
                    fout_band[idx] = fopen(fout_name_band[idx], "wb");
                    if (fout_band[idx] == NULL) {
                        fprintf(stderr, "Can't open fout_band %s\n", fout_name_band[idx]);
                        exit(1);
                    }
                }
                //exit(1);
                */
                if (fout == NULL) {
                    fprintf(stderr, "Can't open fout %s\n", *argv);
                    if (print_melsp_flag) {
                        fclose(fout_msp_bin);
                        fclose(fout_msp_txt);
                    }
                    fclose(fin);
                    exit(1);
                }
            } else {
                fprintf(stderr, "Invalid usage in/out\n");
                if (print_melsp_flag) {
                    fclose(fout_msp_bin);
                    fclose(fout_msp_txt);
                }
                usage(1);
            }
        }
    }

    if (fin == NULL || fout == NULL) {
        fprintf(stderr, "Invalid usage\n");
        if (print_melsp_flag) {
            fclose(fout_msp_bin);
            fclose(fout_msp_txt);
        }
        if (fin != NULL) {
            fclose(fin);
        }
        usage(1);
    }

    //use input melsp
    if (melsp_txt_in_flag || melsp_bin_in_flag) wav_in_flag = 0;
    if (print_melsp_flag && !wav_in_flag) {
        print_melsp_flag = 0;
        fclose(fout_msp_bin);
        fclose(fout_msp_txt);
    }

    srand (time(NULL));
    clock_t t = clock();

    short l, features_dim_1;
    if (print_melsp_flag) features_dim_1 = FEATURES_DIM - 1;
    long samples = 0;

    if (wav_in_flag) { //waveform input
        short num_reflected_right_edge_samples;
        long num_samples, size_of_each_sample;

        //read wave header input and initialize wave output header
        /*
        for (idx=0; idx<N_MBANDS; idx++) {
            read_write_wav_band(fin, fout_band[idx], N_MBANDS);
        }
        */
        if (read_write_wav(fin, fout, &num_reflected_right_edge_samples, &num_samples, &size_of_each_sample)) {

            float features[FEATURES_DIM];
            short pcm[MAX_N_OUTPUT]; //output is in short 2-byte (16-bit) format [-32768,32767]
            //short pcm_band[N_MBANDS*N_SAMPLE_BANDS];
            short first_buffer_flag = 0;
            short waveform_buffer_flag = 0;
            int n_output = 0;
            float data_in_channel = 0;
            long i, j, k;
            char data_buffer[size_of_each_sample];
            float x_buffer[FRAME_SHIFT];
            int read = 0;
            //float out_ddlpc_coarse[N_SAMPLE_BANDS*LPC_ORDER_MBANDS];
            //float out_ddlpc_fine[N_SAMPLE_BANDS*LPC_ORDER_MBANDS];

            // initialize waveform-->features processing struct
            DSPState *dsp;
            dsp = dspstate_create();

            // initialize mwdlp struct
            MWDLP10NetState *net;
            net = mwdlp10net_create();

            for (i = 0, j = 0, k = 0; i < num_samples; i++) {
                if ((read = fread(data_buffer, sizeof(data_buffer), 1, fin))) {
                
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
                            printf(" [%ld]", k);
                        }
                    } else { //take RIGHT_SAMPLES amount of samples as the first samples
                        // put only LEFT_SAMPLES and RIGHT_SAMPLES amount in window buffer,
                        // because zero-padding until FFT_LENGTH with centered window position
                        // (i//FRAME_SHIFT)th frame = i*FRAME_SHIFT [i: 0->(n_samples-1)]
                        dsp->samples_win[LEFT_SAMPLES_1+i] = data_in_channel;
                        if (i <= LEFT_SAMPLES_2) { //reflect only LEFT_SAMPLES-1 amount because 0 value for the 1st coeff. of window
                            dsp->samples_win[LEFT_SAMPLES_2-i] = data_in_channel; 
                        }
                        if (i >= RIGHT_SAMPLES_1) { //process current buffer, and next, take for every FRAME_SHIFT amount samples
                            apply_window(dsp); //hanning window
                            first_buffer_flag = 1;
                            waveform_buffer_flag = 1;
                            k += 1;
                            printf("frame: [%ld]", k);
                        }
                    }

                    if (waveform_buffer_flag) {
                        //extract melspectrogram here
                        mel_spec_extract(dsp, features);

                        if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 0);
                        else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 0);
                        //if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 0, out_ddlpc_coarse, out_ddlpc_fine, pcm_band);
                        //else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 0, pcm_band);

                        if (print_melsp_flag) {
                            for (l=0;l<FEATURES_DIM;l++) {
                                features[l] = (exp(features[l])-1)/10000;
                            }
                            fwrite(features, sizeof(features), 1, fout_msp_bin);
                            for (l=0;l<FEATURES_DIM;l++) {
                                if (l < features_dim_1) {
                                    fprintf(fout_msp_txt, "%f ", features[l]);
                                } else {
                                    fprintf(fout_msp_txt, "%f\n", features[l]);
                                }
                            }
                        }
            
                        if (n_output > 0)  { //delay is reached, samples are generated
                            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                            samples += n_output;
                            /*
                            for (idx=0, m=0; idx<N_MBANDS; idx++) {
                                for (l=0; l<DLPC_ORDER; l++, m++) {
                                    if (l < DLPC_ORDER-1) {
                                        fprintf(fout_ddlpc_coarse[idx], "%f ", out_ddlpc_coarse[m]);
                                        fprintf(fout_ddlpc_fine[idx], "%f ", out_ddlpc_fine[m]);
                                    } else {
                                        fprintf(fout_ddlpc_coarse[idx], "%f\n", out_ddlpc_coarse[m]);
                                        fprintf(fout_ddlpc_fine[idx], "%f\n", out_ddlpc_fine[m]);
                                    }
                                }
                                fwrite(&pcm_band[idx*N_SAMPLE_BANDS], sizeof(pcm_band[0]), N_SAMPLE_BANDS, fout_band[idx]);
                            }
                            */
                        }

                        waveform_buffer_flag = 0;
                    }
                } else {
                    fprintf(stderr, "\nError reading file. %d bytes -- %ld -- %ld\n", read, i+1, num_samples);
                    fclose(fin);
                    fclose(fout);
                    /*
                    for (idx=0; idx<N_MBANDS; idx++) {
                        fclose(fout_ddlpc_coarse[idx]);
                        fclose(fout_ddlpc_fine[idx]);
                        fclose(fout_band[idx]);
                    }
                    */
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
                for (i = 0, k=j-1; i < num_reflected_right_edge_samples; i++, j++) {
                    x_buffer[j] = x_buffer[k-i];
                }

                if (j == FRAME_SHIFT) {
                    printf(" [last frame]\n");
                } else {
                    fprintf(stderr, "\nError remainder right-edge samples calculation %ld %d %d\n", j, FRAME_SHIFT, num_reflected_right_edge_samples);
                    fclose(fin);
                    fclose(fout);
                    /*
                    for (idx=0; idx<N_MBANDS; idx++) {
                        fclose(fout_ddlpc_coarse[idx]);
                        fclose(fout_ddlpc_fine[idx]);
                        fclose(fout_band[idx]);
                    }
                    */
                    if (print_melsp_flag) {
                        fclose(fout_msp_bin);
                        fclose(fout_msp_txt);
                    }
                    dspstate_destroy(dsp);
                    mwdlp10net_destroy(net);
                    exit(1);
                }

                mel_spec_extract(dsp, features);

                if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 0);
                else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 0);
                //if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 0, out_ddlpc_coarse, out_ddlpc_fine, pcm_band);
                //else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 0, pcm_band);

                if (n_output > 0)  {
                    fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                    samples += n_output;
                    /*
                    for (idx=0, m=0; idx<N_MBANDS; idx++) {
                        for (l=0; l<DLPC_ORDER; l++, m++) {
                            if (l < DLPC_ORDER-1) {
                                fprintf(fout_ddlpc_coarse[idx], "%f ", out_ddlpc_coarse[m]);
                                fprintf(fout_ddlpc_fine[idx], "%f ", out_ddlpc_fine[m]);
                            } else {
                                fprintf(fout_ddlpc_coarse[idx], "%f\n", out_ddlpc_coarse[m]);
                                fprintf(fout_ddlpc_fine[idx], "%f\n", out_ddlpc_fine[m]);
                            }
                        }
                        fwrite(&pcm_band[idx*N_SAMPLE_BANDS], sizeof(pcm_band[0]), N_SAMPLE_BANDS, fout_band[idx]);
                    }
                    */
                }

                if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 1); //last_frame_flag, synth pad_right
                else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 1);
                //if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 1, out_ddlpc_coarse, out_ddlpc_fine, pcm_band); //last_frame_flag, synth pad_right
                //else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 1, pcm_band);

                if (print_melsp_flag) {
                    for (l=0;l<FEATURES_DIM;l++) {
                        features[l] = (exp(features[l])-1)/10000;
                    }
                    fwrite(features, sizeof(features), 1, fout_msp_bin);
                    for (l=0;l<FEATURES_DIM;l++) {
                        if (l < features_dim_1) {
                            fprintf(fout_msp_txt, "%f ", features[l]);
                        } else {
                            fprintf(fout_msp_txt, "%f\n", features[l]);
                        }
                    }
                }

                if (n_output > 0)  {
                    fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                    samples += n_output;
                    /*
                    for (idx=0, m=0; idx<N_MBANDS; idx++) {
                        for (l=0; l<DLPC_ORDER; l++, m++) {
                            if (l < DLPC_ORDER-1) {
                                fprintf(fout_ddlpc_coarse[idx], "%f ", out_ddlpc_coarse[m]);
                                fprintf(fout_ddlpc_fine[idx], "%f ", out_ddlpc_fine[m]);
                            } else {
                                fprintf(fout_ddlpc_coarse[idx], "%f\n", out_ddlpc_coarse[m]);
                                fprintf(fout_ddlpc_fine[idx], "%f\n", out_ddlpc_fine[m]);
                            }
                        }
                        fwrite(&pcm_band[idx*N_SAMPLE_BANDS], sizeof(pcm_band[0]), N_SAMPLE_BANDS, fout_band[idx]);
                    }
                    */
                }
            }
    
            t = clock() - t;
            double time_taken = ((double)t)/CLOCKS_PER_SEC;
            printf("%d [frames] %ld [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
                "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
                (int)((double)samples/FRAME_SHIFT), samples, (double)samples/SAMPLING_FREQUENCY, time_taken,
                    ((double)samples/SAMPLING_FREQUENCY)/time_taken, time_taken/((double)samples/SAMPLING_FREQUENCY),
                        N_SAMPLE_BANDS*((double)samples/SAMPLING_FREQUENCY)/time_taken);

            fclose(fin);
            fclose(fout);
            /*
            for (idx=0; idx<N_MBANDS; idx++) {
                fclose(fout_ddlpc_coarse[idx]);
                fclose(fout_ddlpc_fine[idx]);
                fclose(fout_band[idx]);
            }
            */
            if (print_melsp_flag) {
                fclose(fout_msp_bin);
                fclose(fout_msp_txt);
            }
            dspstate_destroy(dsp);
            mwdlp10net_destroy(net);
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
        /*
        for (idx=0; idx<N_MBANDS; idx++) {
            read_feat_write_wav_band(fin, fout_band[idx], melsp_bin_in_flag, N_MBANDS);
        }
        */
        if ((num_frame = read_feat_write_wav(fin, fout, melsp_bin_in_flag)) > 0) {

            float features[FEATURES_DIM];
            short pcm[MAX_N_OUTPUT]; //output is in short 2-byte (16-bit) format [-32768,32767]
            //short pcm_band[N_MBANDS*N_SAMPLE_BANDS];
            int n_output = 0;
            short i, j;
            long k;
            //float out_ddlpc_coarse[N_SAMPLE_BANDS*LPC_ORDER_MBANDS];
            //float out_ddlpc_fine[N_SAMPLE_BANDS*LPC_ORDER_MBANDS];

            // initialize mwdlp struct
            MWDLP10NetState *net;
            net = mwdlp10net_create();

            if (melsp_txt_in_flag) {
                char c;
                //char *buffer = (char*) calloc(128, sizeof(char));
                char buffer[128];
                memset(buffer,'\0',128);
                short flag = 0; //mark current read is in a column
                short frame = 0; //mark process rame
                i = 0; //character counter
                j = 0; //column counter
                k = 0; //frame counter
                while ((c = getc(fin)) != EOF) { //read per character
                    if (flag) { //within a column
                        if (c == ' ') { //add column
                            features[j] = log(1+10000*atof(buffer));
                            memset(buffer,'\0',i);
                            j++;
                            i = 0;
                            flag = 0;
                        } else if (c == '\n') { //found end-of-line
                            features[j] = log(1+10000*atof(buffer));
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
                                //free(buffer);
                                fclose(fin);
                                fclose(fout);
                                //if (print_melsp_flag) {
                                //    fclose(fout_msp_bin);
                                //    fclose(fout_msp_txt);
                                //}
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
                                //free(buffer);
                                fclose(fin);
                                fclose(fout);
                                //if (print_melsp_flag) {
                                //    fclose(fout_msp_bin);
                                //    fclose(fout_msp_txt);
                                //}
                                mwdlp10net_destroy(net);
                                exit(1);
                            }
                        }
                    }
                    if (frame) {
                        if (k < num_frame && k > 1) printf(" [%ld]", k);
                        else if (k < num_frame) printf("frame: [%ld]", k);
                        else printf(" [last frame]\n");

                        if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 0);
                        else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 0);
                        //if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 0, out_ddlpc_coarse, out_ddlpc_fine, pcm_band);
                        //else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 0, pcm_band);
                
                        //if (print_melsp_flag) {
                        //    for (l=0;l<FEATURES_DIM;l++)
                        //        features[l] = (exp(features[l])-1)/10000;
                        //    fwrite(features, sizeof(features), 1, fout_msp_bin);
                        //    for (l=0;l<FEATURES_DIM;l++)
                        //        if (l < features_dim_1)
                        //            fprintf(fout_msp_txt, "%f ", features[l]);
                        //        else
                        //            fprintf(fout_msp_txt, "%f\n", features[l]);
                        //}

                        if (n_output > 0)  { //delay is reached, samples are generated
                            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                            samples += n_output;
                            /*
                            for (idx=0, m=0; idx<N_MBANDS; idx++) {
                                for (l=0; l<DLPC_ORDER; l++, m++) {
                                    if (l < DLPC_ORDER-1) {
                                        fprintf(fout_ddlpc_coarse[idx], "%f ", out_ddlpc_coarse[m]);
                                        fprintf(fout_ddlpc_fine[idx], "%f ", out_ddlpc_fine[m]);
                                    } else {
                                        fprintf(fout_ddlpc_coarse[idx], "%f\n", out_ddlpc_coarse[m]);
                                        fprintf(fout_ddlpc_fine[idx], "%f\n", out_ddlpc_fine[m]);
                                    }
                                }
                                fwrite(&pcm_band[idx*N_SAMPLE_BANDS], sizeof(pcm_band[0]), N_SAMPLE_BANDS, fout_band[idx]);
                            }
                            */
                        }

                        frame = 0;
                    }
                }
            } else if (melsp_bin_in_flag) {
                //float *buffer = (float*) calloc(FEATURES_DIM, sizeof(float));
                float buffer[FEATURES_DIM];
                int read = 0;

                for (k = 0; k < num_frame; k++) {
                    if ((read = fread(buffer, sizeof(buffer), 1, fin))) {
                        for (j = 0; j < FEATURES_DIM; j++)
                            features[j] = log(1+10000*buffer[j]);

                        if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 0);
                        else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 0);
                        //if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 0, out_ddlpc_coarse, out_ddlpc_fine, pcm_band);
                        //else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 0, pcm_band);

                        //if (print_melsp_flag) {
                        //    for (l=0;l<FEATURES_DIM;l++)
                        //        features[l] = (exp(features[l])-1)/10000;
                        //    fwrite(features, sizeof(features), 1, fout_msp_bin);
                        //    for (l=0;l<FEATURES_DIM;l++)
                        //        if (l < features_dim_1)
                        //            fprintf(fout_msp_txt, "%f ", features[l]);
                        //        else
                        //            fprintf(fout_msp_txt, "%f\n", features[l]);
                        //}

                        if (n_output > 0)  { //delay is reached, samples are generated
                            fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                            samples += n_output;
                            /*
                            for (idx=0, m=0; idx<N_MBANDS; idx++) {
                                for (l=0; l<DLPC_ORDER; l++, m++) {
                                    if (l < DLPC_ORDER-1) {
                                        fprintf(fout_ddlpc_coarse[idx], "%f ", out_ddlpc_coarse[m]);
                                        fprintf(fout_ddlpc_fine[idx], "%f ", out_ddlpc_fine[m]);
                                    } else {
                                        fprintf(fout_ddlpc_coarse[idx], "%f\n", out_ddlpc_coarse[m]);
                                        fprintf(fout_ddlpc_fine[idx], "%f\n", out_ddlpc_fine[m]);
                                    }
                                }
                                fwrite(&pcm_band[idx*N_SAMPLE_BANDS], sizeof(pcm_band[0]), N_SAMPLE_BANDS, fout_band[idx]);
                            }
                            */
                        }
                    } else {
                        fprintf(stderr, "\nError reading input. %d %ld -- %ld\n", read, num_frame, k+1);
                        //free(buffer);
                        fclose(fin);
                        fclose(fout);
                        /*
                        for (idx=0; idx<N_MBANDS; idx++) {
                            fclose(fout_ddlpc_coarse[idx]);
                            fclose(fout_ddlpc_fine[idx]);
                            fclose(fout_band[idx]);
                        }
                        */
                        //if (print_melsp_flag) {
                        //    fclose(fout_msp_bin);
                        //    fclose(fout_msp_txt);
                        //}
                        mwdlp10net_destroy(net);
                        exit(1);
                    }
                }
            } else {
                fprintf(stderr, "\nError input option %d -- %d -- %d\n", wav_in_flag, melsp_bin_in_flag, melsp_txt_in_flag);
                fclose(fin);
                fclose(fout);
                /*
                for (idx=0; idx<N_MBANDS; idx++) {
                    fclose(fout_ddlpc_coarse[idx]);
                    fclose(fout_ddlpc_fine[idx]);
                    fclose(fout_band[idx]);
                }
                */
                //if (print_melsp_flag) {
                //    fclose(fout_msp_bin);
                //    fclose(fout_msp_txt);
                //}
                mwdlp10net_destroy(net);
                exit(1);
            }

            if (k == num_frame) {
                if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 1); //last_frame_flag, synth pad_right
                else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 1);
                //if (!NO_DLPC) mwdlp10net_synthesize(net, features, pcm, &n_output, 1, out_ddlpc_coarse, out_ddlpc_fine, pcm_band); //last_frame_flag, synth pad_right
                //else mwdlp10net_synthesize_nodlpc(net, features, pcm, &n_output, 1, pcm_band);

                if (n_output > 0)  {
                    fwrite(pcm, sizeof(pcm[0]), n_output, fout);
                    samples += n_output;
                    /*
                    for (idx=0, m=0; idx<N_MBANDS; idx++) {
                        for (l=0; l<DLPC_ORDER; l++, m++) {
                            if (l < DLPC_ORDER-1) {
                                fprintf(fout_ddlpc_coarse[idx], "%f ", out_ddlpc_coarse[m]);
                                fprintf(fout_ddlpc_fine[idx], "%f ", out_ddlpc_fine[m]);
                            } else {
                                fprintf(fout_ddlpc_coarse[idx], "%f\n", out_ddlpc_coarse[m]);
                                fprintf(fout_ddlpc_fine[idx], "%f\n", out_ddlpc_fine[m]);
                            }
                        }
                        fwrite(&pcm_band[idx*N_SAMPLE_BANDS], sizeof(pcm_band[0]), N_SAMPLE_BANDS, fout_band[idx]);
                    }
                    */
                }
            } else {
                fprintf(stderr, "\nError input frames  %ld -- %ld\n", k, num_frame);
                //free(buffer);
                fclose(fin);
                fclose(fout);
                /*
                for (idx=0; idx<N_MBANDS; idx++) {
                    fclose(fout_ddlpc_coarse[idx]);
                    fclose(fout_ddlpc_fine[idx]);
                    fclose(fout_band[idx]);
                }
                */
                //if (print_melsp_flag) {
                //    fclose(fout_msp_bin);
                //    fclose(fout_msp_txt);
                //}
                mwdlp10net_destroy(net);
                exit(1);
            }
    
            t = clock() - t;
            double time_taken = ((double)t)/CLOCKS_PER_SEC;
            printf("%d [frames] %ld [samples] %.2f [sec.] synthesis in %.2f seconds \n"\
                "[%.2f x faster than real-time] [%.2f RTF] [%.2f kHz/sec]\n",
                (int)((double)samples/FRAME_SHIFT), samples, (double)samples/SAMPLING_FREQUENCY, time_taken,
                    ((double)samples/SAMPLING_FREQUENCY)/time_taken, time_taken/((double)samples/SAMPLING_FREQUENCY),
                        N_SAMPLE_BANDS*((double)samples/SAMPLING_FREQUENCY)/time_taken);

            //free(buffer);
            fclose(fin);
            fclose(fout);
            /*
            for (idx=0; idx<N_MBANDS; idx++) {
                fclose(fout_ddlpc_coarse[idx]);
                fclose(fout_ddlpc_fine[idx]);
                fclose(fout_band[idx]);
            }
            */
            //if (print_melsp_flag) {
            //    fclose(fout_msp_bin);
            //    fclose(fout_msp_txt);
            //}
            mwdlp10net_destroy(net);
        } else {
            if (print_melsp_flag) {
                fclose(fout_msp_bin);
                fclose(fout_msp_txt);
            }
            exit(1);
        }
    }

    return 0;
}
