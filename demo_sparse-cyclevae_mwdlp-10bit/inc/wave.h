/*
   Copyright 2021 Patrick Lumban Tobing (Nagoya University)
   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

   WAV file read/write is based on http://truelogic.org/wordpress/2015/09/04/parsing-a-wav-file-in-c
*/


// WAVE file header format
struct HEADER {
    unsigned char riff[4];                      // RIFF string
    unsigned int overall_size;               // overall size of file in bytes
    unsigned char wave[4];                      // WAVE string
    unsigned char fmt_chunk_marker[4];          // fmt string with trailing null char
    unsigned int length_of_fmt;                 // length of the format data
    unsigned short format_type;                   // format type. 1-PCM, 3- IEEE float, 6 - 8bit A law, 7 - 8bit mu law
    unsigned short channels;                      // no.of channels
    unsigned int sample_rate;                   // sampling rate (blocks per second)
    unsigned int byterate;                      // SampleRate * NumChannels * BitsPerSample/8
    unsigned short block_align;                   // NumChannels * BitsPerSample/8
    unsigned short bits_per_sample;               // bits per sample, 8- 8bits, 16- 16 bits etc
    unsigned char data_chunk_header [4];        // DATA string or FLLR string
    unsigned int data_size;                     // NumSamples * NumChannels * BitsPerSample/8 - size of the next chunk that will be read
};

/*
    Positions   Sample Value    Description
    1 – 4   “RIFF”  Marks the file as a riff file. Characters are each 1 byte long.
    5 – 8   File size (integer)     Size of the overall file – 8 bytes, in bytes (32-bit integer). Typically, you’d fill this in after creation.
    9 -12   “WAVE”  File Type Header. For our purposes, it always equals “WAVE”.
    13-16   “fmt “  Format chunk marker. Includes trailing null
    17-20   16  Length of format data as listed above
    21-22   1   Type of format (1 is PCM) – 2 byte integer
    23-24   2   Number of Channels – 2 byte integer
    25-28   44100   Sample Rate – 32 byte integer. Common values are 44100 (CD), 48000 (DAT). Sample Rate = Number of Samples per second, or Hertz.
    29-32   176400  (Sample Rate * BitsPerSample * Channels) / 8.
    33-34   4   (BitsPerSample * Channels) / 8.1 – 8 bit mono2 – 8 bit stereo/16 bit mono4 – 16 bit stereo
    35-36   16  Bits per sample
    37-40   “data”  “data” chunk header. Marks the beginning of the data section.
    41-44   File size (data)    Size of the data section.
    Sample values are given above for a 16-bit stereo source.

    It is important to note that the WAV format uses little-endian [LSB in smallest address] format to store bytes,
    so you need to convert the bytes to big-endian [MSB in smallest address] in code for the values to make sense.
*/

short read_write_wav(FILE *fin, FILE *fout, short *num_reflected_right_edge_samples, long *num_samples, long *size_of_each_sample);
long read_feat_write_wav(FILE* fin, FILE* fout, int bin_flag);
