# Real-time low-latency multispeaker voice conversion (VC) with cyclic variational autoencoder (CycleVAE) and multiband WaveRNN using data-driven linear prediction (MWDLP)


## Requirements:
- UNIX
- 3.6 >= python <= 3.8
- CUDA 10.1
- virtualenv
- jq
- make
- gcc
- g++


## Installation
```
$ cd tools
$ make
$ cd ..
```

## Steps to build the models:
1. Data preparation and preprocessing
2. VC and neural vocoder models training [~ 2.5 and 4 days each, respectively]
3. VC fine-tuning with fixed neural vocoder [~ 2.5 days]
4. VC decoder fine-tuning with fixed encoder and neural vocoder [~ 1.5 days]


## Steps for real-time low-latency decoding with CPU:
1. Dump and compile models
2. Decode


## Details

Please see **egs/cycvae_mwdlp_vcc20/README.md** for more details on VC + neural vocoder

or

**egs/mwdlp_vcc20/README.md** for more details on neural vocoder only.


## Contact

Patrick Lumban Tobing

patrickltobing@gmail.com

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp

