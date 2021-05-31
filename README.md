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


## Latest version and current status
- 3.1:
    - Revised handling of speaker-code with factorized and controllable trainable basis vectors
    - Revised MWDLP architecture with trainable basis vectors of previous logits and an additional FC layer for data-driven LP
    - Revised C implementation for faster FC layer computation in MWDLP
- On going to update samples and demo


## Samples and real-time compilable demo with CPU [from version 3.0]
* [Samples](https://drive.google.com/drive/folders/14pJSpYsoPpLR6Ah-EbENSsN6ABcSvB0w?usp=sharing)
* [Real-time compilable demo with CPU](https://drive.google.com/file/d/1j7ddvltaWwie0wEp79W6VL2EV-SSAW-g/view?usp=sharing)


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
