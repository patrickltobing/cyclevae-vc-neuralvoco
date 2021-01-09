# Multispeaker VC with Nonparallel Cyclic Variational Autoencoder (CycleVAE) and Multiband WaveRNN with Data-Driven Linear Prediction (MWDLP)


This is a C implementation of [cyclevae-vc-neuralvoco](https://github.com/patrickltobing/cyclevae-vc-neuralvoco).


* The post-network of VC model (CycleVAE+PostNet) is still being trained.
* PostNet is used to generate finer structure of mel-spectrogram for alleviating spectral oversmoothnes through Laplace sampling,
  which is trained after CycleVAE is fixed to preserve the lower-bound conversion.


* The neural vocoder model (MWDLP) uses 10-bit mu-law coarse-fine output architecture with 704-GRU and 10% sparsification.
* Currently, an investigation is ongoing for higher MWDLP performance with a larger 768/1024-GRU and a lower density of 5%/2% sparsification,
  which will also provide much lower cost and complexity.


* The quality will be significantly improved after fine-tuning the MWDLP with the reconstructed mel-spec from the PostNet output using the cyclic flow from CycleVAE.


* With a single core of 2.7 GHz CPU, the total time required (including waveform read-write and features extraction)
  to process 1 sec. of 16 kHz audio file is about 3.64 sec.,
  i.e., 0.27 slower than real-time (3.64 RT). [The bottleneck is on CycleVAE; MWDLP + waveform I/O + feature extract is about 0.46 RT, i.e., 2.17 faster than real-time.]
* The total VC computation time will be real-time with the use of sparsification on larger GRUs for CycleVAE (ongoing investigation).


* Speaker list details and 2-dimensional speaker-space are located in the folder speaker_info.
* Some example of input, converted waveforms with interpolated-points and speaker-points are located in the folders wav, wav_cv_interp, and wav_cv_point, respectively.
* Some ppt slides including diagram and audio samples are located in the folder slides


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
* The size of nnet_cv_data.c is a bit large due to a more complex model of CycleVAE.
* The use of sparse GRUs with larger hidden units (currently ongoing) [of 2 encoders and 1 melspec-decoder] will greatly reduces its size and complexity.


## Usage

```
$ bash demo_interp.sh
```
or
```
$ bash demo_point.sh
```
or
```
$ ./bin/test_cycvae_mwdlp.exe <trg_spk_id> <input_wav> <output_wav>
```
or
```
$ ./bin/test_cycvae_mwdlp.exe <x_coord> <y_coord> <input_wav> <output_wav>
```


## Contact

Patrick Lumban Tobing

Nagoya University

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
