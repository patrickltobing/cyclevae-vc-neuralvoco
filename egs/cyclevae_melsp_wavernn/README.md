# A package of voice conversion with CycleVAE-based using mel-spectrogram and neural vocoder with multiband WaveRNN using data-driven linear prediction


## Requirements

* Python 3.6/3.7/3.8
* CUDA 10.1
* Linux 64-bit [for Windows 10 user, please use Windows Subsystem for Linux]


## Usage

With respect to the parent-root folder,

```
$ cd tools
$ make
$ cd ../egs/cyclevae_melsp_wavernn
```

Set your experiment directory at `egs/` as the `egs/cyclevae_melsp_wavernn` example

Put wav files in `wav_<sampling_rate_in_kHz>kHz` directory

Set `stage` and other variables on `run.sh` and `conf/config.yml` accordingly to run the feature-extraction/training/decoding

Set `spks` and `data_name` variables in `run.sh` accordingly

Set `STAGE 0` to accomodate your training/development/testing dataset list

Set F0 and power threshold configuration files for each speaker in `conf/*.f0` or `conf/*.pow` (calculate speaker histogram with `STAGE init`)

Please check the optimum model index from training log file on exp/tr_<model_folder>/log/train.log by searching with the phrase "sme" or "min_idx" without quotes, e.g., `vim log/train.log`, press backslash "/", then type "sme" or "min_idx", and enter.


## STAGE description

`stage=0123` for data list preparation, feature extraction, statistics calculation, and pre-emphasis/multiband-analysis, respectively

`stage=init` for histogram calculation of F0 and power statistics (for speaker F0 and power threshold configurations)

`stage=4` for CycleVAE-based VC with mel-spectrogram training

`stage=post` for post-network training for mel-spectrogram refinement after CycleVAE is trained

`stage=5` for reconstruction/cyclic-reconstruction generation (for further tuning of trained neural vocoder)

`stage=6` for conversion features generation and synthesis with Griffin-Lim

`stage=7` for multiband WaveRNN training with natural features

`stage=89` for multiband WaveRNN decoding with natural features and de-emphasis, respectively

`stage=ab` for multiband WaveRNN decoding with converted features and de-emphasis, respectively


## Some variable descriptions on `conf/config.yml`

`fs` waveform sampling rate

`pad_len` padding for frame number, set to the maximum frame length in your data (get it with the provided `get_max_frame.sh`)

`batch_size_utt` number of batch size in VC training, set it so that the `number of utterances / batch_size_utt` is at least `100`

`mdl_name_wave` use 9-bit mu-law/16-bit output multiband WaveRNN (16-bit has better output)

`spkidtr_dim` speaker dimension reduction for speaker-space interpolation, e.g., from N- to 2-dim space in the case of N number of speakers


## Some variable descriptions on `run.sh`

`spks` speaker list in your dataset

`data_name` dataset name that will be used throughout the script

`GPU_device` index of GPU device used in training

`GPU_device_str` indices of GPU devices used in decoding

`n_gpus` number of GPU devices used in decoding (make it synchronized with GPU_device_str)

`mdl_name_post` model name for post-net to refine mel-spec after cyclevae trained

`idx_resume_cycvae` checkpoint index for resume training of cyclevae model

`min_idx_cycvae` checkpoint index for cyclevae model

`idx_resume` checkpoint index for resume training of post-net model

`min_idx` checkpoint index for post-net model

`idx_resume_wave` checkpoint index for resume training of shallow WaveNEt model

`min_idx_wave` checkpoint index for WaveRNN model

`spks_trg_rec` speaker list in reconstruction/cyclic-reconstruction for WaveRNN fine-tuning

`spks_src_dec` source speaker in conversion

`spks_trg_dec` target speaker in conversion

`spks_dec` speaker in neural vocoder decoding with natural features

`decode_batch_size` number of batch sequence per GPU in WaveNet decoding


## References

[1] P. L. Tobing, Y.-C. Wu, T. Hayashi, K. Kobayashi, and T. Toda, "Non-parallel voice conversion with cyclic variational autoencoder," in Proc. INTERSPEECH, Graz, Austria, Sep. 2019, pp. 674--678.

[2] P. L. Tobing, K. Kobayashi, T. Hayashi, and T. Toda, "High-fidelity multiband WaveRNN using data-driven linear prediction for low-latency universal neural vocoder with high-quality output," submitted for ICASSP 2021.


## Contact


Patrick Lumban Tobing

Nagoya University

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
