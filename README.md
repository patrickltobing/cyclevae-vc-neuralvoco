# Multispeaker VC with Nonparallel Cyclic VAE/VQVAE and/or Neural Vocoder with Multiband WaveRNN using Data-Driven Linear Prediction

* Previous version is in v1.0 branch
* Scripts overhauled starting on v2.0 branch
    * Added CycleVAE with mel-spectrogram
    * Added MWDLP (Multiband WaveRNN with data-driven linear prediction for low-latency universal neural vocoder with high-quality output (submitted for ICASSP 2021)) [demo_page](https://demo-icassp2021.audioeval.net/)
* Created experiment branch (exp) for agile update version
    * Added the use of mel-filterbank to encode WORLD spectrum instead of mel-cepstrum
    * Development of mel-cepstrum based (WORLD) is suspended to focus on mel-spectrogram systems
    * Sparse CycleVAE + Fine-tuned decoder-melspec
    * MWDLP + Fine-tuned MWDLP with reconst. melspec from fine-tuned decoder-melspec CycleVAE
    * Contents will be merged with v2.2 and master
* In progress
    * Multiband WaveRNN w/ data-driven LPC real-time in C (based on [LPCNet](https://github.com/mozilla/LPCNet)) [AVX2/FMA/Neon] ~0.3--0.5 RT
    * VC real-time in C following WaveRNN code [currently 0.95 - 1.15 RT including waveform I/O and mel-spec extract]
    * Pretrained models
    * CycleVQVAE


## Requirements

* Python 3.6/3.7/3.8
* `virtualenv` and `jq`
* CUDA 10.1
* Linux 64-bit [for Windows 10 user, please use Windows Subsystem for Linux (WSL)]
    * For WSL:
        - Requires gcc, g++, and Python [please don't use WSL-based Python]
        - Use a separate dedicated Python environment package, such as from Miniconda
        ```
        $ sudo apt update && sudo apt upgrade
        $ sudo apt-get install virtualenv
        $ sudo apt-get install jq
        $ sudo apt-get install gcc
        $ sudo apt-get install g++
        $ cd <home_dir>
        $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        $ bash Miniconda3-latest-Linux-x86_64.sh
        ```
        - Follow and complete the installation by pressing enter / answering yes
        - Exit and restart the WSL terminal


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

