# Multispeaker VC with Nonparallel Cyclic VAE/VQVAE and/or Neural Vocoder with Multiband WaveRNN using Data-Driven Linear Prediction / Shallow-WaveNet

* Previous version is in v1.0 branch
* Scripts overhauled starting on v2.0 branch
    * Added CycleVAE with mel-spectrogram
    * Added multiband WaveRNN with data-driven linear prediction for low-latency universal neural vocoder with high-quality output (submitted for ICASSP 2021) [demo_page](https://demo-icassp2021.audioeval.net/)
* Created experiment branch (exp) for agile update version
    * Added Laplace fine-sampling for fine-structure recovery of mel-spectrogram
    * Added the use of mel-filterbank to encode WORLD spectrum instead of mel-cepstrum
    * Development of mel-cepstrum based (WORLD) is suspended to allocate resources for mel-spectrogram systems
    * Contents will be merged with v2.1 and master when confirmed
* In progress
    * Publish code of Multiband WaveRNN real-time in C (AVX/FMA/Neon) ~ 0.3 - 0.5 RT
    * VC real-time in C following WaveRNN code
    * Pretrained models
    * CycleVQVAE


## Requirements

* Python 3.6/3.7/3.8
* CUDA 10.1
* Linux 64-bit [for Windows 10 user, please use Windows Subsystem for Linux]


## Installation

```
$ cd tools
$ make
$ cd ..
```

## Usage

```
$ cd egs/cyclevae_melsp_wavernn
```

README file is given in the experiment directory.


## Contact

Patrick Lumban Tobing

Nagoya University

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
