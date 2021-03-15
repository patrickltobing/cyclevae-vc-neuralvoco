# Real-time low-latency multispeaker VC with cyclic variational autoencoder (CycleVAE) and multiband WaveRNN using data-driven linear prediction (MWDLP)


This package uses Voice Conversion Challenge 2020 dataset [VCC20](http://vc-challenge.org/).

Real-time implementation is based on [LPCNet](https://github.com/mozilla/LPCNet/).


## Data preparation
```
$ bash download_vcc20.sh
```

## Data preprocessing
1. Open **run.sh**
2. Set `stage=0init123`
3. Set a value of `n_jobs=` for number of parallel threads in preprocessing
4. `$ bash run.sh`


## VC model training [~ 2.5 days]
1. Open **run.sh**
2. Set `stage=4`
3. Set a value of `GPU_device=` for GPU device selection
4. `$ bash run.sh`


## Neural vocoder training [~ 4 days]
1. Open **run.sh**
2. Set `stage=5`
3. Set a value of `GPU_device=` for GPU device selection
4. `$ bash run.sh`


## VC fine-tuning with fixed neural vocoder [~ 2.5 days]
1. Open **run.sh**
2. Set `stage=6`
3. Set a value of `GPU_device=` for GPU device selection
4. `$ bash run.sh`


## VC decoder fine-tuning with fixed encoder and neural vocoder [~ 1.5 days]
1. Open **run.sh**
2. Set `stage=6`
3. Set a value of `GPU_device=` for GPU device selection
4. `$ bash run.sh`


## Compile CPU real-time program
1. Open **run_realtime.sh**
2. Set `stage=0`
3. `$ bash run_realtime.sh`


## Decode with target speaker points
1. Open **run_realtime.sh**
2. Set `stage=3`
3. Set values in `spks_src_dec=` for source speakers
4. Set values in `spks_trg_dec=` for target speakers
5. `$ bash run_realtime.sh`


## Decode with interpolated speaker points
1. Open **run_realtime.sh**
2. Set `stage=4`
3. Set values in `spks_src_dec=` for source speakers
4. Set a value in `n_interp=` for number of interpolated points
5. `$ bash run_realtime.sh`


## Contact

Patrick Lumbantobing

patrickltobing@gmail.com

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
