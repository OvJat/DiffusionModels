# Diffusion Models Tutorials

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description
This is a PyTorch-based tutorial for Diffusion Models.

## setup environment

### setup environment (step by step)

```bash
# step1. create anaconda environment 
conda create -n DiffusionModels python=3.8

# step2. then activate this environment
conda activate DiffusionModels

# step3. install pytorch
# if on MacOSX
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
# if on Linux/Windows, CUDA 11.6
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# if on Linux/Windows, CUDA 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# if on Linux/Windows, CPU Only
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch 

# step4. install other packages
pip install diffusers

```

### setup environment (easy way)
```shell
# step1. create anaconda environment 
conda create -n DiffusionModels python=3.8

# step2. then activate this environment
conda activate DiffusionModels

# step3. using requirements.txt
pip install -r requirements.txt
```

## Files

* `models.py` is Neural Networks.
* `train.py` 
    * function `train_vae` shows how to train AutoEncoderKL or AutoEncoderVQ.
    * function `make_conditions` shows how to make timesteps and condition for Diffusion.
    * function `train_diffusion` shows how to train an Unet for Diffusion.
    * function `sampling_diffusion` shows how to sample using a pretrained U-Net.