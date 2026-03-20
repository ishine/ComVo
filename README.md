# [ICLR 2026] Toward Complex-Valued Neural Networks for Waveform Generation
#### Hyung-Seok Oh, Deok-Hyeon Cho, Seung-Bin Kim and Seong-Whan Lee

This repository contains the official implementation of ComVo,
a complex-valued neural vocoder for waveform generation based on iSTFT.

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=U4GXPqm3Va)
[![Demo](https://img.shields.io/badge/Demo-Audio_Samples-green)](https://hs-oh-prml.github.io/ComVo/)
[![GitHub Stars](https://img.shields.io/github/stars/hs-oh-prml/ComVo?style=social)](https://github.com/hs-oh-prml/ComVo)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-hsoh%2FComVo-yellow)](https://huggingface.co/hsoh/ComVo)
[![Pretrained](https://img.shields.io/badge/Pretrained-Checkpoint-orange)](https://works.do/xM2ttS4)

<p align="center">
  <img src="assets/architecture.png" alt="">
  <br>
  <em>Overall architecture of ComVo</em>
</p>


## Abstract

Neural vocoders have recently advanced waveform generation, yielding natural and expressive audio.
Among these approaches, iSTFT-based vocoders have gained attention.
They predict a complex-valued spectrogram and then synthesize the waveform via iSTFT, thereby avoiding redundant, computationally expensive upsampling.
However, current approaches use real-valued networks that process the real and imaginary parts independently.
This separation limits their ability to capture the inherent structure of complex spectrograms.
We present ComVo, a complex-valued neural vocoder whose generator and discriminator use native complex arithmetic.
This enables an adversarial training framework that provides structured feedback directly in the complex domain.
To guide phase transformations in a structured manner, we introduce phase quantization, which discretizes phase values and regularizes the training process.
Finally, we propose a block-matrix computation scheme to improve training efficiency by reducing redundant operations.
Experiments demonstrate that ComVo achieves higher synthesis quality than comparable real-valued baselines, and that its block-matrix scheme reduces training time by 25%.
Audio samples and code are available at [https://hs-oh-prml.github.io/ComVo/](https://hs-oh-prml.github.io/ComVo/).


### Installation

```bash
pip install -r requirements
```

#### Recommended environment

- Python >= 3.8
- PyTorch >= 2.0
- CUDA-enabled GPU

## Pretrained checkpoint

We provide a pretrained ComVo checkpoint for quick inference:

- Download: [Checkpoint link](https://works.do/xM2ttS4)
- Config: `configs/configs.yaml`
- Target sampling rate: `24 kHz`

Run inference with:

```bash
python infer.py -c configs/configs.yaml --ckpt /path/to/comvo.ckpt --wavfile /path/to/input.wav --out_dir ./results
```

## Train

```bash
python train.py -c configs/configs.yaml
```

Hyperparameters are specified in `configs/configs.yaml`.

## Inference

```bash
python infer.py -c $CONFIG --ckpt=$CKPT --wavfile=$FILE_NAME --out_dir $OUTPUT_DIR
```

## Citation

```bibtex
@inproceedings{
  oh2026toward,
  title={Toward Complex-Valued Neural Networks for Waveform Generation},
  author={Hyung-Seok Oh and Deok-Hyeon Cho and Seung-Bin Kim and Seong-Whan Lee},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=U4GXPqm3Va}
}
```
