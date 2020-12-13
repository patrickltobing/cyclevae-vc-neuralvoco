#!/bin/sh

#gcc -Wall -W -O3 -g -I../include dump_data.c freq.c kiss_fft.c pitch.c celt_lpc.c -o dump_data -lm
#gcc -o test_lpcnet -mavx2 -mfma -g -O3 -Wall -W -Wextra test_lpcnet.c lpcnet.c nnet.c nnet_data.c freq.c kiss_fft.c pitch.c celt_lpc.c -lm

gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic -Iinclude/ -Isrc/ -c src/lpcnet.c src/nnet.c src/nnet_data.c -lm
ar rsv libmwdlp16.a lpcnet.o nnet.o nnet_data.o
gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic src/test_mwdlp.c -Iinclude/ -Isrc/ -L. -lmwdlp16 -lm -o test_mwdlp

