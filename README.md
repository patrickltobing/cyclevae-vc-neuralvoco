# Multispeaker VC with Nonparallel Cyclic VAE/VQVAE and Neural Vocoder with WaveRNN/WaveNet

* This is a comprehensive repository for multispeaker voice conversion (VC) with nonparallel modeling based on cyclic variational autoencoder (CycleVAE) [1,2] / vector-quantized VAE (CycleVQVAE) [2].
* It also provides neural vocoder implementations based on shallow WaveNet architecture [3,4] / compact WaveRNN (LPCNet-like [5]) with data-driven linear prediction (LPC) [4] for high-quality waveform synthesis (conversion / copy-synthesis).
* The VC modeling further allows speaker interpolation of either spectral / excitation characteristics on a 2-dimensional space.
* Default speech features and waveform synthesis are based on WORLD [6] (mel-cepstrum from spectral envelope, F0, aperiodicity).
* Modeling and synthesis with mel-spectrogram are under construction.
* [Our group](https://www.toda.is.i.nagoya-u.ac.jp/) is located in Nagoya University, Japan

## Requirements

* Python 3.7 or 3.6
* CUDA 10.0

## Installation

```
$ cd tools
$ make
```

## Configs/pretrained models

* Available
    * VCC 2018 [7] [12 speakers] (VC) `egs/vcc18`
        * [download\_wavs](https://drive.google.com/file/d/1n_uRWMuXcXpwf1Mjuu-Eqm9Evea2Jy8S/view?usp=sharing) --> put in `egs/vcc18/wav_22.05kHz`
        * [download\_mdls](https://drive.google.com/file/d/15erWSja13MZj0UJZpfomwYAZfkc3AsfN/view?usp=sharing) --> put in `egs/vcc18/exp`
* Under-construction
    * VCC 2018 (Neural vocoder)
    * VCC 2020 [14 speakers]
    * VCC 2018 + ARCTIC [2 speakers]
    * VCC 2020 + VCTK [24 speakers]
    * VCC 2018 + VCC 2020 + ARCTIC + VCTK

## Type of models

### VC

* Spectral CycleVAE `[mdl_name: cycmcepvae-laplace]`
* Spectral + excitation CycleVAE `[mdl_name: cycmceplf0capvae-laplace]`
* Spectral + excitation CycleVAE with 2-dim speaker space `[spkidtr_dim: 2]`
* Spectral CyleVQVAE `[mdl_name: cycmcepvae-vq]`
* Spectral + excitation CycleVQVAE `[mdl_name: cycmceplf0capvae-vq]`
* Spectral + excitation CycleVQVAE with 2-dim speaker space `[spkidtr_dim: 2]`

### Neural vocoder

* Shallow WaveNet `[mdl_name_wave: wavenet]`
* Compact WaveRNN with data driven LPC `[mdl_name_wave: wavernn_dualgru_compact_lpc]`
* Compact WaveRNN with data driven LPC and multiple samples output `[mdl_name_wave: wavernn_dualgru_compact_lpcseg]`

## Configurations

Located in `egs/<dataset>/conf/config.yml`

* `fs:` sampling rate
* `shiftms:` frame-shift in ms
* `right_size:` limit of lookup frame on input convolution of encoder/neural-vocoder [for low-latency app.] (if 0, use a balanced two-sided conv.; else it is a skewed conv.)
* `pad_len`: number of maximum frame length in padding for batch processing
* `epoch_count:` number of maximum epoch

### VC

* `n_half_cyc:` number of half-cycles, e.g., for 1 cycle, it is 2, and for 2 cycles, it is 4.
    * default
        * CycleVAE: `2`
        * CycleVQVAE: `4`
* `lat_dim:` number of latent-dimension for spectral latent
    * default 
        * Spectral / spectral+excitation CycleVAE: `32`
        * Spectral CycleVQVAE: `40`
        * Spectral+excitation CycleVQVAE: `50`
* `lat_dim_e:` number of latent-dimension for excitation latent
    * default 
        * Spectral+excitation CycleVAE: `32`
        * Spectral+excitation CycleVQVAE: `50`
* `causal_conv_dec:` flag to use causal input conv. on spectral decoder (for low-latency)
* `causal_conv_lf0:` flag to use causal input conv. on excitation decoder 
* `ctr_size:` size of VQ-codebook for CycleVQVAE
* `ar_enc:` flag to use autoregressive (AR) flow (feedback output) in encoder
    * default: `false`
* `ar_dec:` flag to use AR-flow in decoder
    * default
        * CycleVAE: `true`
        * CycleVQVAE: `false`
* `diff:` flag to use differential spectrum estimation for spectral decoder (only for decoder with AR-flow)
    * default CycleVAE: `true`
* `detach:` flag to detach conversion/cyclic flow in backpropagation graph
    * default: `true`
* `spkidtr_dim`: number of the reduced dimensionality of N-dimensional one-hot speaker-code for speaker interpolation (if 0, keep using one-hot speaker-code)
* `batch_size:` number of frames per batch
* `batch_size_utt:` number of utterances per batch in optimization
* `batch_size_utt_eval:` number of utterances per batch in validation
* `mdl_name:` type of VC model

### Neural vocoder

* `batch_size_wave:` numer of frames per batch
* `batch_size_utt_wave:` number of utterances per batch in optimization
* `batch_size_utt_eval_wave:` number of utterances per batch in validation
* `hidden_units_wave_2:` number of hidden units of 2nd GRU in compact WaveRNN
* `t_start:` starting step for sparsification in compact WaveRNN
* `t_end:` ending step for sparsification
* `interval:` interval step in sparsification
* `densities:` target densities of reset, update, and new gates of 1st GRU in compact WaveRNN
* `n_stage:` number of stages in sparsification
```
# at each sparsification step, this is the function of target density
r = 1 - (iter_idx-t_start)/(t_end - t_start)
density = density_stage[k-1] - (density_stage[k-1]-density_stage[k])*(1 - r)**5
# k is the index of stage [0,..,n_stage-1]
# density_stage contains the target density on each stage, for k=0, it is set 1
# number of steps per stage is set to:
## [0.2, 0.3, 0.5]*t_delta for n_stage=3
## [0.1, 0.1, 0.3, 0.5]*t_delta for n_stage=4
## [0.1, 0.1, 0.15, 0.15, 0.5]*t_delta for n_stage=5
## where t_delta = t_end - t_start + 1
```
* `lpc:` number of data-driven LPC (modeled/estimated by network) in compact WaveRNN
* `seg:` number of multiple samples output
* `mdl_name_wave:` type of neural vocoder model

## Executions

Located in `egs/<dataset>/conf/run.sh`

* `stage=0` prepare lists of training/validation/testing sets (`egs/<dataset>/data/<tr/dv/ts>_<dataset>_<sampling-rate>`) and speaker configs (`egs/<dataset>/conf/spkr.yml`)
* `stage=init` compute speaker statistics [histograms of F0 and normalized-power] with initial F0 configurations (`egs/<dataset>/init_spk_stat/tr_<dataset>_<sampling-rate>`). For new speakers, run `stage=0` and `stage=init`, and change the initial F0 and pow values on `conf/spkr.yml` by following the procedure given in this [slide](https://www.slideshare.net/NU_I_TODALAB/hands-on-voice-conversion).
* `stage=1` feature extraction
* `stage=2` calculate feature statistics
* `stage=3` pre-emphasis (noise-shaping) on waveform data for neural vocoder development
* `stage=4` training of VC model
* `stage=5` decoding of reconstruction and cyclic-reconstruction features to compute global variance (GV) [8] statistics and to be used for neural vocoder training with data augmentation
* `stage=6` decoding of converted features/waveform from VC model using conventional vocoder
* `stage=7` training of neural vocoder model
* `stage=8` decoding of converted/copy-synthesis waveform from neural vocoder
* `stage=9` de-emphasis (restored noise-shaping) on synthesized waveform

Run as `$ bash run.sh`

### More settings

* `n_jobs=` number of jobs/threads in feature extraction, statistics calculation, and pre-emphasis processing
* `spks=` list of speakers
* `data_name=` name of dataset
* `GPU_device=` index of GPU device used during training
* `idx_resume=` resume VC model training from this checkpoint `[set the arguments in running call on STAGE 4]`
* `idx_resume_wave=` resume neural vocoder model training from this checkpoint `[set the arguments in running call on STAGE 7]`
* `min_idx=` decode VC model using this checkpoint
* `n_interp=` decode spectral-excitation VC conversion with this number of interpolated speaker points (if 0, it is just a source-to-target conversion)
* `min_idx_wave=` decode neural vocoder model using this checkpoint
* `GPU_device_str=` indices of GPUs used during decoding
* `n_gpus=` number of GPUs used during decoding
* `spks_trg_rec=` list of speakers considered during reconstruction/cyclic-reconstruction in `stage=5`
* `spks_src_dec=` list of source speakers considered during conversion in `stage=6`
* `spks_trg_dec=` list of target speakers considered during conversion in `stage=6`
* `decode_batch_size=` number of concurrent utterances when decoding with neural vocoder

## Summarizations

Located in `egs/<dataset>/local`

* `proc_loss_log_vae-spec.awk` for spectral model
* `proc_loss_log_vae-spec-excit.awk` for spectral-excitation model
* `loss_summary.sh` to run the `awk` scripts for summarizing model accuracies during training
* `summary_acc.awk` for decoding accuracy in development/testing sets
* `summary_acc.sh` to extract desired accuracy statistics in development/testing sets
* `get_max_frame.sh` to get maximum frame number statistics for each speaker for `pad_len` config

Tensorboard graph statistics are also simultaneously calculated/provided during training in the corresponding model folder `egs/<dataset>/exp/<model_expdir>`.

## Examples

2-dimensional speaker space on VCC 2018 dataset with spectral-excitation CycleVAE 

Spectral space
![](https://i.imgur.com/8QQM765.png)

Excitation space
![](https://i.imgur.com/gB2y4z8.png)

## References

[1] P. L. Tobing, Y.-C. Wu, T. Hayashi, K. Kobayashi, and T. Toda, "Non-parallel
voice conversion with cyclic variational autoencoder," in Proc. INTERSPEECH,
Graz, Austria, Sep. 2019, pp. 674--678.

[2] P. L. Tobing, T. Hayashi, Y.-C. Wu, K. Kobayashi, and T. Toda, "Cyclic spectral modeling for unsupervised unit discovery into voice
conversion with excitation and waveform modeling," Accepted for INTERSPEECH 2020.

[3] P. L. Tobing, T. Hayashi, and T. Toda, "Investigation of shallow WaveNet vocoder
with Laplacian distribution output," in Proc. IEEE ASRU, Sentosa, Singapore,
Dec. 2019, pp. 176--183.

[4] P. L. Tobing, Y.-C. Wu, T. Hayashi, K. Kobayashi, and T. Toda, “Efficient shallow WaveNet vocoder using multiple samples output
based on Laplacian distribution and linear prediction,” in Proc. ICASSP, Barcelona, Spain, May 2020, pp. 7204–-7208.

[5] J.-M. Valin, J. Skoglund, A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet, in Proc. INTERSPEECH, Graz, Austria, Sep. 2019, pp. 3406--3410.

[6] M. Morise, F. Yokomori, and K. Ozawa, “WORLD: A vocoder based high-quality speech synthesis system for real-time applications,” IEICE Trans. Inf. Syst., vol. 99, no. 7, pp. 1877--1884, 2016.

[7] J. Lorenzo-Trueba, J. Yamagishi, T. Toda, D. Saito, F. Villavicencio, T. Kinnunen, and Z. Ling, “The Voice Conversion Challenge 2018:
Promoting development of parallel and nonparallel methods,” in Proc. Speaker Odyssey, Les Sables d’Olonne, France, Jun. 2018, pp. 195--202.

[8] T. Toda, A. W. Black, and K. Tokuda, “Voice conversion based on maximum-likelihood estimation of spectral parameter trajectory,” IEEE Trans. Audio Speech Lang. Process., vol. 15, no. 8, pp.
2222-–2235, 2007.

## Acknowledgements

- [@kan-bayashi](https://github.com/kan-bayashi)
- [@r9y9](https://github.com/r9y9)
- [@JeremyCCHsu](https://github.com/JeremyCCHsu)
- [@k2kobayashi](https://github.com/k2kobayashi)
- [@bigpon](https://github.com/bigpon)


## To-do-list

- Complete VC pretrained models
- Complete neural vocoder decoding scripts
- Complete neural vocoder models
- Mel-spectrogram modeling
