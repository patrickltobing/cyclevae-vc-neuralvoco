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
    - Slight fix on MWDLP max. step from prev. commit [1ef01de](https://github.com/patrickltobing/cyclevae-vc-neuralvoco/commit/1ef01de18f5810022aa90bbd3ec8d6b0096ade08).
    - Slight fix on MWDLP dev. acc. check criterion.
    - Update samples and demo


## Samples and real-time compilable demo with CPU [Updated with version 3.1]
* [Samples (dev. set)](https://drive.google.com/drive/folders/1uRZNczzD_jVmwVQghITVT0y9EQX7nmFn?usp=sharing)
* [Samples (test. set)](https://drive.google.com/drive/folders/1T6MYe-Kg37_2aDtyUWU7eB7px4-y3lVW?usp=sharing)
* [Real-time compilable demo with CPU](https://drive.google.com/file/d/1wt-QL5x4PUGNM8QL7YpL9NbWLdKrCMRj/view?usp=sharing)


## Steps to build the models:
1. Data preparation and preprocessing
2. VC and neural vocoder models training [~ 2.5 and 4 days each, respectively]
3. VC fine-tuning with fixed neural vocoder [~ 2.5 days]
4. VC decoder fine-tuning with fixed encoder and neural vocoder [~ 1.5 days]


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
