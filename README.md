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
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
# if on Linux/Windows, CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# if on Linux/Windows, CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# if on Linux/Windows, CPU Only
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# step4. install other packages
pip install diffusers

```

### setup environment (on Linux/Windows, CUDA 11.7)
```shell
# step1. create anaconda environment 
conda create -n DiffusionModels python=3.8

# step2. then activate this environment
conda activate DiffusionModels

# step3. using requirements.txt
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

```

## Files

* `models.py` is Neural Networks.
* `train.py` 
    * function `train_vae` shows how to train AutoEncoderKL or AutoEncoderVQ.
    * function `make_conditions` shows how to make timesteps and condition for Diffusion.
    * function `train_diffusion` shows how to train an Unet for Diffusion.
    * function `sampling_diffusion` shows how to sample using a pretrained U-Net.
