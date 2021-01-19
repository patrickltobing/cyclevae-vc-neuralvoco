#include <stdio.h>
#include <stdlib.h>

#include "freq.h"
#include "freq_conf.h"
#include "wave.h"


short read_write_wav(FILE *fin, FILE *fout, short *num_reflected_right_edge_samples, long *num_samples, long *size_of_each_sample) {
    unsigned char buffer4[4];
    unsigned char buffer2[2];
    
    // WAVE header structure
    struct HEADER header;
   
    size_t read;
 
    // read header parts [input wav]
    if (!(read = fread(header.riff, sizeof(header.riff), 1, fin))) {
        fprintf(stderr, "\nError reading file. riff %lu bytes * 4 %s\n", read, header.riff);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    //printf("(1-4): %u %u %u %u \n", header.riff[0], header.riff[1], header.riff[2], header.riff[3]); 
    printf("(1-4): %s \n", header.riff); 
    // write header parts [output wav, following input reading]
    fwrite(header.riff, sizeof(header.riff), 1, fout);
    
    if (!(read = fread(buffer4, sizeof(buffer4), 1, fin))) {
        fprintf(stderr, "\nError reading file. size %lu bytes * 4 %s\n", read, buffer4);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
    fwrite(buffer4, sizeof(buffer4), 1, fout);
    
    // convert little endian to big endian 4 byte int
    header.overall_size  = buffer4[0] | 
                           (buffer4[1]<<8) | 
                           (buffer4[2]<<16) | 
                           (buffer4[3]<<24);
    
    printf("(5-8) Overall size: bytes:%u, Kb:%u \n", header.overall_size, header.overall_size/1024);
    
    if (!(read = fread(header.wave, sizeof(header.wave), 1, fin))) {
        fprintf(stderr, "\nError reading file. wave %lu bytes * 4 %s\n", read, header.wave);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    //printf("(9-12): %u %u %u %u \n", header.wave[0], header.wave[1], header.wave[2], header.wave[3]); 
    printf("(9-12) Wave marker: %s\n", header.wave);
    fwrite(header.wave, sizeof(header.wave), 1, fout);
    
    if (!(read = fread(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fin))) {
        fprintf(stderr, "\nError reading file. fmt %lu bytes * 4 %s\n", read, header.fmt_chunk_marker);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    //printf("(13-16): %u %u %u %u \n", header.fmt_chunk_marker[0], header.fmt_chunk_marker[1], header.fmt_chunk_marker[2], header.fmt_chunk_marker[3]); 
    printf("(13-16) Fmt marker: %s\n", header.fmt_chunk_marker);
    fwrite(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fout);
    
    if (!(read = fread(buffer4, sizeof(buffer4), 1, fin))) {
        fprintf(stderr, "\nError reading file. fmt_length %lu bytes * 4 %s\n", read, buffer4);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
    fwrite(buffer4, sizeof(buffer4), 1, fout);
    
    // convert little endian to big endian 4 byte integer
    header.length_of_fmt = buffer4[0] |
                               (buffer4[1] << 8) |
                               (buffer4[2] << 16) |
                               (buffer4[3] << 24);
    printf("(17-20) Length of Fmt header: %u \n", header.length_of_fmt);
    
    if (!(read = fread(buffer2, sizeof(buffer2), 1, fin))) {
        fprintf(stderr, "\nError reading file. type %lu bytes * 2 %s\n", read, buffer2);
        fclose(fin);
        fclose(fout);
        return -1;
    }
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
    if (header.format_type != 1) {
        printf("Format is not PCM.\n");
        fclose(fin);
        fclose(fout);
        return -1;
    }
    
    if (!(read = fread(buffer2, sizeof(buffer2), 1, fin))) {
        fprintf(stderr, "\nError reading file. channels %lu bytes * 2 %s\n", read, buffer2);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    printf("%u %u \n", buffer2[0], buffer2[1]);
    fwrite(buffer2, sizeof(buffer2), 1, fout);
    
    header.channels = buffer2[0] | (buffer2[1] << 8);
    printf("(23-24) Channels: %u \n", header.channels);
    if (header.channels != 1) {
        printf("Channels are more than 1.\n");
        fclose(fin);
        fclose(fout);
        return -1;
    }
    
    if (!(read = fread(buffer4, sizeof(buffer4), 1, fin))) {
        fprintf(stderr, "\nError reading file. sr %lu bytes * 4 %s\n", read, buffer4);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
    fwrite(buffer4, sizeof(buffer4), 1, fout);
    
    header.sample_rate = buffer4[0] |
                           (buffer4[1] << 8) |
                           (buffer4[2] << 16) |
                           (buffer4[3] << 24);
    
    printf("(25-28) Sample rate: %u Hz\n", header.sample_rate);
    if (header.sample_rate != SAMPLING_FREQUENCY) {
        printf("Sampling rate is not %d Hz\n", SAMPLING_FREQUENCY);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    
    if (!(read = fread(buffer4, sizeof(buffer4), 1, fin))) {
        fprintf(stderr, "\nError reading file. byterate %lu bytes * 4 %s\n", read, buffer4);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
    fwrite(buffer4, sizeof(buffer4), 1, fout);
    
    header.byterate  = buffer4[0] |
                           (buffer4[1] << 8) |
                           (buffer4[2] << 16) |
                           (buffer4[3] << 24);
    printf("(29-32) Byte Rate: %u B/s , Bit Rate:%u b/s\n", header.byterate, header.byterate*8);
    
    if (!(read = fread(buffer2, sizeof(buffer2), 1, fin))) {
        fprintf(stderr, "\nError reading file. block %lu bytes * 2 %s\n", read, buffer2);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    printf("%u %u \n", buffer2[0], buffer2[1]);
    fwrite(buffer2, sizeof(buffer2), 1, fout);
    
    header.block_align = buffer2[0] |
                       (buffer2[1] << 8);
    printf("(33-34) Block Alignment: %u \n", header.block_align);
    
    if (!(read = fread(buffer2, sizeof(buffer2), 1, fin))) {
        fprintf(stderr, "\nError reading file. bits %lu bytes * 2 %s\n", read, buffer2);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    printf("%u %u \n", buffer2[0], buffer2[1]);
    fwrite(buffer2, sizeof(buffer2), 1, fout);
    
    header.bits_per_sample = buffer2[0] |
                       (buffer2[1] << 8);
    printf("(35-36) Bits per sample: %u \n", header.bits_per_sample);
    
    if (!(read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fin))) {
        fprintf(stderr, "\nError reading file. data_mark %lu bytes * 4 %s\n", read, header.data_chunk_header);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    //printf("(37-40): %u %u %u %u \n", header.data_chunk_header[0], header.data_chunk_header[1], header.data_chunk_header[2], header.data_chunk_header[3]); 
    printf("(37-40) Data Marker: %s \n", header.data_chunk_header);
    fwrite(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fout);
    
    if (!(read = fread(buffer4, sizeof(buffer4), 1, fin))) {
        fprintf(stderr, "\nError reading file. data_size %lu bytes * 4 %s\n", read, buffer4);
        fclose(fin);
        fclose(fout);
        return -1;
    }
    printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
    
    header.data_size = buffer4[0] |
                   (buffer4[1] << 8) |
                   (buffer4[2] << 16) | 
                   (buffer4[3] << 24 );
    printf("(41-44) Size of data chunk: %u B \n", header.data_size);
    
    // calculate no.of samples
    *num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);
    printf("Number of samples: %lu \n", *num_samples);
    // first samples are of the right-side hanning window, i.e., 1st frame,
    // the remaining samples are then divided by FRAME_SHIFT to get the remaining frames
    long num_frame = 1 + (*num_samples - RIGHT_SAMPLES) / FRAME_SHIFT;
    // if theere exist remainder samples of the 2nd-Nth frame,
    // use reflected samples of the last buffer for the missing amount of samples to process one last frame
    *num_reflected_right_edge_samples = (*num_samples - RIGHT_SAMPLES) % FRAME_SHIFT;
    if (*num_reflected_right_edge_samples > 0) {
        *num_reflected_right_edge_samples = FRAME_SHIFT - *num_reflected_right_edge_samples;
        num_frame += 1; //use additional reflected samples for the trailing remainder samples on the right-edge
    }
    long num_samples_frame = num_frame * FRAME_SHIFT;
    printf("Number of samples in processed output wav: %lu \n", num_samples_frame);
    header.data_size = (num_samples_frame * header.channels * header.bits_per_sample) / 8;
    printf("(41-44) Size of data chunk to be written: %u B \n", header.data_size);
    fwrite(&header.data_size, sizeof(header.data_size), 1, fout);
    
    *size_of_each_sample = (header.channels * header.bits_per_sample) / 8;
    printf("Size of each sample: %ld B\n", *size_of_each_sample);
    
    // calculate duration of file
    float duration_in_seconds = (float) header.overall_size / header.byterate;
    printf("Approx.Duration in seconds= %f\n", duration_in_seconds);

    return 1;
}


long read_feat_write_wav(FILE* fin, FILE* fout, int bin_flag) {
    // WAVE header structure
    struct HEADER header;

    // calculate no.of frames -- samples
    long num_frame;
    if (bin_flag) {
        fseek(fin, 0, SEEK_END);
        long size = ftell(fin);
        num_frame = (long) round((float) size / (MEL_DIM * sizeof(float)));
    } else {
        char c;
        short count = 0;
        short flag = 0; //mark current read is in a column
        num_frame = 0;
        while ((c = getc(fin)) != EOF) { //read per character
            if (flag) { //within a column
                if (c == ' ') { //add column
                    count++;
                    flag = 0;
                } else if (c == '\n') { //found end-of-line
                    count++;
                    if (count == MEL_DIM) { //add row
                        num_frame++;
                        count = 0;
                        flag = 0;
                    } else { //columns not appropriate
                        fprintf(stderr, "Error input text format %d %d\n", count, MEL_DIM);
                        fclose(fin);
                        fclose(fout);
                        return -1;
                    }
                }
            } else { //finding new column
                if (c != ' ' && c != '\n') { //add starting column character
                    flag = 1;
                } else if (c == '\n') { //found end-of-line
                    if (count == MEL_DIM) { //add row
                        num_frame++;
                        count = 0;
                    } else { //columns not appropriate
                        fprintf(stderr, "Error input text format  %d %d\n", count, MEL_DIM);
                        fclose(fin);
                        fclose(fout);
                        return -1;
                    }
                }
            }
        }
    }
    fseek(fin, 0, SEEK_SET);
    long num_samples_frame = num_frame * FRAME_SHIFT;
    printf("number of frames -- samples: %ld -- %ld \n", num_frame, num_samples_frame);
    
    // write header parts [output wav, following a wave format]
    header.riff[0] = 'R';
    header.riff[1] = 'I';
    header.riff[2] = 'F';
    header.riff[3] = 'F';
    printf("(1-4): %s \n", header.riff); 
    fwrite(header.riff, sizeof(header.riff), 1, fout);
    
    // header.overall_size = data_size [samples_size] + 44 [header_size] - 8 [RIFF_size + overall_size]
    header.channels = 1;
    header.bits_per_sample = 16;
    header.data_size = (num_samples_frame * header.channels * header.bits_per_sample) / 8;
    header.overall_size = header.data_size + 36;
    fwrite(&header.overall_size, sizeof(header.overall_size), 1, fout);
    printf("(5-8) Overall size: bytes:%u, Kb:%u \n", header.overall_size, header.overall_size/1024);
    
    header.wave[0] = 'W';
    header.wave[1] = 'A';
    header.wave[2] = 'V';
    header.wave[3] = 'E';
    printf("(9-12) Wave marker: %s\n", header.wave);
    fwrite(header.wave, sizeof(header.wave), 1, fout);
    
    header.fmt_chunk_marker[0] = 'f';
    header.fmt_chunk_marker[1] = 'm';
    header.fmt_chunk_marker[2] = 't';
    header.fmt_chunk_marker[3] = ' ';
    printf("(13-16) Fmt marker: %s\n", header.fmt_chunk_marker);
    fwrite(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, fout);
    
    header.length_of_fmt = 16;
    printf("(17-20) Length of Fmt header: %u \n", header.length_of_fmt);
    fwrite(&header.length_of_fmt, sizeof(header.length_of_fmt), 1, fout);
    
    char format_name[10] = "PCM";
    header.format_type = 1;
    fwrite(&header.format_type, sizeof(header.format_type), 1, fout);
    printf("(21-22) Format type: %u %s \n", header.format_type, format_name);
    
    fwrite(&header.channels, sizeof(header.channels), 1, fout);
    printf("(23-24) Channels: %u \n", header.channels);
    
    header.sample_rate = SAMPLING_FREQUENCY;
    fwrite(&header.sample_rate, sizeof(header.sample_rate), 1, fout);
    printf("(25-28) Sample rate: %u Hz\n", header.sample_rate);
    
    header.byterate = (header.sample_rate * header.bits_per_sample * header.channels) / 8;
    fwrite(&header.byterate, sizeof(header.byterate), 1, fout);
    printf("(29-32) Byte Rate: %u B/s , Bit Rate:%u b/s\n", header.byterate, header.byterate*8);
    
    header.block_align = (header.channels * header.bits_per_sample) / 8;
    fwrite(&header.block_align, sizeof(header.block_align), 1, fout);
    printf("(33-34) Block Alignment: %u \n", header.block_align);
    
    fwrite(&header.bits_per_sample, sizeof(header.bits_per_sample), 1, fout);
    printf("(35-36) Bits per sample: %u \n", header.bits_per_sample);
    
    header.data_chunk_header[0] = 'd';
    header.data_chunk_header[1] = 'a';
    header.data_chunk_header[2] = 't';
    header.data_chunk_header[3] = 'a';
    printf("(37-40) Data Marker: %s \n", header.data_chunk_header);
    fwrite(header.data_chunk_header, sizeof(header.data_chunk_header), 1, fout);
    
    //header.data_size = data_size [samples_size]
    printf("Number of samples in processed output wav: %lu \n", num_samples_frame);
    printf("(41-44) Size of data chunk to be written: %u B \n", header.data_size);
    fwrite(&header.data_size, sizeof(header.data_size), 1, fout);
    
    long size_of_each_sample = (header.channels * header.bits_per_sample) / 8;
    printf("Size of each sample: %ld B\n", size_of_each_sample);
    
    // calculate duration of file
    float duration_in_seconds = (float) header.overall_size / header.byterate;
    printf("Approx.Duration in seconds= %f\n", duration_in_seconds);

    return num_frame;
}
