#!/bin/sh

#gcc -Wall -W -O3 -g -I../include dump_data.c freq.c kiss_fft.c pitch.c celt_lpc.c -o dump_data -lm
#gcc -o test_lpcnet -mavx2 -mfma -g -O3 -Wall -W -Wextra test_lpcnet.c lpcnet.c nnet.c nnet_data.c freq.c kiss_fft.c pitch.c celt_lpc.c -lm

#gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic -Isrc/ -c src/mwdlp10net_cycvae.c src/freq.c src/kiss_fft.c src/nnet.c src/nnet_data.c src/nnet_cv_data.c -lm
#ar rsv libmwdlp10cycvae.a mwdlp10net_cycvae.o freq.o kiss_fft.o nnet.o nnet_data.o nnet_cv_data.o
#gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic src/test_mwdlp.c -Isrc/ -L. -lmwdlp10cycvae -lm -o test_mwdlp

gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic -I. -c mwdlp10net_cycvae.c freq.c kiss_fft.c nnet.c nnet_data.c nnet_cv_data.c -lm
#ar rsv libmwdlp10cycvae.a mwdlp10net_cycvae.o freq.o kiss_fft.o nnet.o nnet_data.o nnet_cv_data.o
#gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic -I. -c freq.c -lm
#gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic -I. -c mwdlp10net_cycvae.c -lm
#gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic -I. -c mwdlp10net_cycvae.c nnet.c -lm
ar rsv libmwdlp10cycvae.a mwdlp10net_cycvae.o freq.o kiss_fft.o nnet.o nnet_data.o nnet_cv_data.o
gcc -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic test_mwdlp.c -I. -L. -lmwdlp10cycvae -lm -o test_mwdlp

