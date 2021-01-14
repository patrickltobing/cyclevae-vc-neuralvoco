# Multispeaker VC with Nonparallel Cyclic Variational Autoencoder (CycleVAE) and Multiband WaveRNN with Data-Driven Linear Prediction (MWDLP)


This is a C implementation of [cyclevae-vc-neuralvoco](https://github.com/patrickltobing/cyclevae-vc-neuralvoco).


* The neural vocoder model (MWDLP) uses 10-bit mu-law coarse-fine output architecture with sparse 800-GRU and 10% sparsification.
* The VC model (CycleVAE) uses sparse 512-GRU for 2 encoders and 640-GRU for 1 decoder-melsp with 40 % sparsification.

* The MWDLP model is still being trained, current demo uses a model that has been trained for only 1.4 days. (Minimum is about 3 days to guarantee stability)
* The CycleVAE model is still being fine-tuned, current demo uses a fine-tuned decoder-melsp that has been trained for only 1.8 days.


* The quality will be significantly improved after fine-tuning the MWDLP with the reconstructed mel-spec from the CycleVAE with fine-tuned decoder-melsp.


* With a single core of 2.6 GHz CPU, the total time required (including waveform read-write and features extraction) to process 1 sec. of 16 kHz audio file is about 1.00 sec,
  i.e., just about real-time (1.00 RT). [Only MWDLP + waveform I/O + feature extract is about 0.50 RT, i.e., 2.00 faster than real-time, hence 50/50 cost for CycleVAE+MWDLP]


* Speaker list details and 2-dimensional speaker-space are located in the folder speaker_info.
* Some example of input, analysis-synthesis waveforms, converted waveforms with interpolated-points and speaker-points are located in the folders wav, wav_anasyn, wav_cv_interp, and wav_cv_point, respectively.


A brief overview of the process:
* read each waveform sample --> check frame buffer windowing condition (length/shift) [if sufficient, next, if not, go to beginning]
* --> hanning window, STFT, and extract mel-spectrogram (using mel-filterbank with respect to magnitude spectrogram) --> convert mel-spectrogram with CycleVAE
* --> synthesize converted band-samples with MWDLP (using PQMF [pseudo-quadratic-mirror filterbank] to recover full-band signal) --> write waveform samples
* --> go to beginning if input samples still exist, else end.


## Requirements

* gcc
* Intel CPU [It is possible to also use ARM CPU, but has not been tested yet on such a machine]
* make


## Installation

```
$ make
```


## Usage

```
$ bash <demo_script>
```
**demo_script: demo.sh, demo_point.sh, demo_interp.sh, demo_anasyn.sh

or
```
$ ./bin/test_cycvae_mwdlp <trg_spk_id> <input_wav> <output_cv_wav>
```
or
```
$ ./bin/test_cycvae_mwdlp <x_coord> <y_coord> <input_wav> <output_cv_wav>
```
or
```
$ ./bin/test_cycvae_mwdlp <input_wav> <output_anasyn_wav>
```


## Contact

Patrick Lumban Tobing

Nagoya University

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp

