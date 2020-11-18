# Multispeaker VC with Nonparallel Cyclic VAE/VQVAE and/or Neural Vocoder with Multiband WaveRNN using Data-Driven Linear Prediction / Shallow-WaveNet

* Previous version is in v1.0 branch
* Scripts overhauled starting on v2.0 branch
    * Added CycleVAE with mel-spectrogram
    * Added multiband WaveRNN with data-driven linear prediction for low-latency universal neural vocoder with high-quality output (submitted for ICASSP 2021) [demo_page](https://demo-icassp2021.audioeval.net/)
* Created experiment branch (exp) for agile update version
    * Added Laplace fine-sampling for fine-structure recovery of mel-spectrogram
    * Added the use of mel-filterbank to encode WORLD spectrum instead of mel-cepstrum
    * Contents will be merged with v2.[] and master when confirmed
* Lots of work to be done...
    * CycleVAE with mcep+excit
    * CycleVQVAE (mcep,mcep+excit,melsp)
    * Reduced speaker-dimension for speaker-interpolation
    * Pretrained models
    * Multiband WaveRNN real-time in C (AVX/FMA/Neon)
    * VC real-time in C following WaveRNN code


## Requirements

* Python 3.6/3.7/3.8
* CUDA 10.1
* Linux 64-bit


## Installation

```
$ cd tools
$ make
$ cd ..
```

## Usage

```
$ cd egs/cyclevae_mcep_wavenet
```
or
```
$ cd egs/cyclevae_melsp_wavernn
```

README files are given in each experiment directory.


## Contact

Patrick Lumban Tobing

Nagoya University

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
