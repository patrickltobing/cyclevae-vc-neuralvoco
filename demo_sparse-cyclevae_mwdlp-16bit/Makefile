LDIR = lib
$(shell mkdir -p $(LDIR))
LIBNAME = mwdlp16cycvae
OUT = ${LDIR}/lib${LIBNAME}.a

CC = gcc
CFLAGS = -mavx2 -mfma -g -O3 -Wall -W -Wextra -fpic
LFLAGS = -lm

ODIR = obj
$(shell mkdir -p $(ODIR))

SDIR = src
IDIR = inc
BDIR = bin
$(shell mkdir -p $(BDIR))

INC = -I${IDIR}
LIB = -L${LDIR} -l${LIBNAME}
TARGET = test_cycvae_mwdlp

_OBJS = mwdlp16net_cycvae.o freq.o kiss_fft.o nnet.o nnet_data.o nnet_cv_data.o
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))


all: ${OUT}
	$(CC) $(CFLAGS) ${SDIR}/${TARGET}.c $(INC) ${LIB} ${LFLAGS} -o ${BDIR}/${TARGET}

$(OUT): $(OBJS) 
	ar rvs $(OUT) $^

$(ODIR)/%.o: $(SDIR)/%.c
	$(CC) $(CFLAGS) $(INC) -c -o $@ $< ${LFLAGS}

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(OUT) ${BDIR}/${TARGET}
