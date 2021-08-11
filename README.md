# Low-latency real-time multispeaker voice conversion (VC) with cyclic variational autoencoder (CycleVAE) and multiband WaveRNN using data-driven linear prediction (MWDLP)

- VC Paper can be found here: 

[Low-latency real-time non-parallel voice conversion based on cyclic variational autoencoder and multiband WaveRNN with data-driven linear prediction](https://arxiv.org/pdf/2105.09858.pdf)

- Neural vocoder paper can be found here:

[High-Fidelity and Low-Latency Universal Neural Vocoder based on Multiband WaveRNN with Data-Driven Linear Prediction for Discrete Waveform Modeling](https://arxiv.org/abs/2105.09856.pdf)


## Requirements:
- UNIX
- 3.6 >= python <= 3.9
- CUDA 11.1
- virtualenv
- jq
- make
- gcc


## Installation
```
$ cd tools
$ make
$ cd ..
```


## Latest version
- 3.1
    - Finalize VC and MWDLP Python implementations (impl.)
    - Bug fixes on C impl. to match the output of Python impl.
    - Fix input segmental convolution to output 128 dimension to reduce parameters load
    - Add one more FC layer after segmental convolution for VC module as in original MWDLP


## Steps to build the models:
1. Data preparation and preprocessing
2. VC and neural vocoder models training [~ 2.5 and 4 days each, respectively]
3. VC fine-tuning with fixed neural vocoder [~ 2.5 days]
4. VC decoder fine-tuning with fixed encoder and neural vocoder [~ 2.5 days]


## Steps for real-time low-latency decoding with CPU:
1. Dump and compile models
2. Decode

Real-time implementation is based on [LPCNet](https://github.com/mozilla/LPCNet/).


## Details

Please see **egs/cycvae_mwdlp_vcc20/README.md** for more details on VC + neural vocoder

or

**egs/mwdlp_vcc20/README.md** for more details on neural vocoder only.


## Contact

Patrick Lumban Tobing

patrickltobing@gmail.com

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
