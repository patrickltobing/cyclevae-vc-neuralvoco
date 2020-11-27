# A package of voice conversion with CycleVAE-based using mel-cepstrum and neural vocoder with shallow WaveNet vocoder


## Requirements

* Python 3.6/3.7/3.8
* CUDA 10.1
* Linux 64-bit


## Usage

With respect to the parent-root folder,

```
$ cd tools
$ cd make
$ cd ../egs/cyclevae_mcep_wavenet
```

Set your experiment directory at `egs/` as the `egs/cyclevae_mcep_wavenet` example

Put wav files in `wav_<sampling_rate_in_kHz>kHz` directory

Set `stage` and other variables on `run.sh` and `conf/config.yml` accordingly to run the feature-extraction/training/decoding

Set `spks` and `data_name` variables in `run.sh` accordingly

Set `STAGE 0` to accomodate your training/development/testing dataset list

Set F0 and power threshold configuration files for each speaker in `conf/*.f0` or `conf/*.pow` (calculate speaker histogram with `STAGE init`)


## STAGE description

`stage=0123` for data list preparation, feature extraction, statistics calculation, and pre-emphasis, respectively

`stage=init` for histogram calculation of F0 and power statistics (for speaker F0 and power threshold configurations)

`stage=4` for CycleVAE-based VC with mel-cepstrum training

`stage=post` for post-network training for mel-cepstrum refinement after CycleVAE is trained

`stage=5` for reconstruction/cyclic-reconstruction generation and GV stats calculation

`stage=6` for conversion features generation and synthesis with WORLD

`stage=7` for shallow WaveNet training with natural features

`stage=89` for shallow WaveNet decoding with natural features and de-emphasis, respectively

`stage=ab` for shallow WaveNet decoding with converted features and de-emphasis, respectively


## Some variable descriptions on `conf/config.yml`

`fs` waveform sampling rate

`pad_len` padding for frame number, set to the maximum frame length in your data (get it with the provided `get_max_frame.sh`)

`batch_size_utt` number of batch size in VC training, set it so that the `number of utterances / batch_size_utt` is at least `100`


## Some variable descriptions on `run.sh`

`spks` speaker list in your dataset

`data_name` dataset name that will be used throughout the script

`GPU_device` index of GPU device used in training

`GPU_device_str` indices of GPU devices used in decoding

`n_gpus` number of GPU devices used in decoding (make it synchronized with GPU_device_str)

`mdl_name_post` model name for post-net to refine mel-ceps after cyclevae trained

`idx_resume_cycvae` checkpoint index for resume training of cyclevae model

`min_idx_cycvae` checkpoint index for cyclevae model

`idx_resume` checkpoint index for resume training of post-net model

`min_idx` checkpoint index for post-net model

`idx_resume_wave` checkpoint index for resume training of shallow WaveNEt model

`min_idx_wave` checkpoint index for shallow WaveNEt model

`spks_trg_rec` speaker list in reconstruction/cyclic-reconstruction for shallow WaveNet fine-tuning

`spks_src_dec` source speaker in conversion

`spks_trg_dec` target speaker in conversion

`spks_dec` speaker in neural vocoder decoding with natural features

`decode_batch_size` number of batch sequence per GPU in WaveNet decoding


## References

[1] P. L. Tobing, Y.-C. Wu, T. Hayashi, K. Kobayashi, and T. Toda, "Non-parallel voice conversion with cyclic variational autoencoder," in Proc. INTERSPEECH, Graz, Austria, Sep. 2019, pp. 674--678.

[2] P. L. Tobing, T. Hayashi, and T. Toda, "Investigation of shallow WaveNet vocoder with Laplacian distribution output," in Proc. IEEE ASRU, Sentosa, Singapore, Dec. 2019, pp. 176--183.

[3] P. L. Tobing, Y.-C. Wu, T. Hayashi, K. Kobayashi, and T. Toda, “Efficient shallow WaveNet vocoder using multiple samples output based on Laplacian distribution and linear prediction,” in Proc. ICASSP, Barcelona, Spain, May 2020, pp. 7204–-7208.


## Contact


Patrick Lumban Tobing

Nagoya University

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
