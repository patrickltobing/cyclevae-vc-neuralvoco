# Low-latency real-time multispeaker voice conversion (VC) with cyclic variational autoencoder (CycleVAE) and multiband WaveRNN using data-driven linear prediction (MWDLP)


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
- 3.1 (2021/09/25)
    - Finalize VC and MWDLP Python implementations (impl.)
    - Bug fixes on C impl. to match the output of Python impl.
    - Fix input segmental convolution impl. as in original papers while allowing usage in real-time demo
    - Update MWDLP demo and samples with VCC20 dataset
    - Update VC demo and samples with VCC20 dataset


## Compilable demo

- MWDLP: [demo_mwdlp-10bit_emb-v2_vcc20](https://drive.google.com/file/d/1hR7N-iCSUMNx9P-pDVxftGIIKLLyXsnt/view?usp=sharing)

- VC: [demo_sparse-cyclevae-weightembv2-smpl_jnt_mwdlp-10bit_emb_vcc20](https://drive.google.com/file/d/1LtuQmnUP45iWoREbPK0vBTdu2tDZKYeT/view?usp=sharing)


## Samples from compilable demo

- MWDLP: [samples_demo_mwdlp-10bit_emb-v2_vcc20](https://drive.google.com/drive/folders/1by_BO-fkeouDgTZBWEeu6EnzaX8UgHL8?usp=sharing)

- VC: [samples_demo_sparse-cyclevae-weightembv2-smpl_jnt_mwdlp-10bit_emb_vcc20](https://drive.google.com/drive/folders/1PanNaqsOccCImHECywzsaX6mFwausznz?usp=sharing)


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


## References

[1] [High-Fidelity and Low-Latency Universal Neural Vocoder based on Multiband WaveRNN with Data-Driven Linear Prediction for Discrete Waveform Modeling](https://arxiv.org/abs/2105.09856.pdf)

[2] [Low-latency real-time non-parallel voice conversion based on cyclic variational autoencoder and multiband WaveRNN with data-driven linear prediction](https://arxiv.org/pdf/2105.09858.pdf)


## Contact

Patrick Lumban Tobing

patrickltobing@gmail.com

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
