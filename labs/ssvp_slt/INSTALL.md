## Installation

### Requirements
- Linux, CUDA >= 11.7, Python >= 3.8, PyTorch >= 2.0.0 (our setup below is based on CUDA 11.8, Python 3.10, PyTorch 2.0.1; more recent versions should work too, but no guarantees)
- Conda (anaconda / miniconda work well)

### Setup

#### 1. Create conda environment
```bash
conda create --name ssvp_slt python=3.10
conda activate ssvp_slt
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install av -c conda-forge
```

#### 2. (Optional) Video dataloader GPU decoding backend support
If you want to use our video dataloader with GPU decoding backend, you need to reinstall torchvision by building it from scratch following the steps at [https://github.com/pytorch/vision/tree/main/torchvision/csrc/io/decoder/gpu](https://github.com/pytorch/vision/tree/main/torchvision/csrc/io/decoder/gpu). We found that this does not work with ffmpeg 6.1, so we recommend running `conda install 'ffmpeg<6.0'`. If you get a warning that torchvision is built without GPU decoding support due to `bsf.h` missing, we recommend manually downloading `bsf.h` from the `ffmpeg` source code ([https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/bsf.h](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/bsf.h), **make sure it matches your ffmpeg version!**) and placing it under `$(path-to-your-conda)/ssvp_slt/include/libavcodec`.

#### 3. Pip Installs
Install the remaining dependencies and an egg of our ssvp-slt package:
```bash
pip install git+https://github.com/facebookresearch/stopes.git
pip install -r requirements.txt

# Move into fairseq folder and install egg
cd fairseq-sl
pip install -e .
cd ..

# Install ssvp_slt egg from the repo's root
pip install -e .
```

#### 4. BLEURT 
If you want to compute BLEURT scores as part of the translation evals (via `common.compute_bleurt=true`), you need to install BLEURT:
```bash
pip install git+https://github.com/google-research/bleurt.git
```

You will also likely need to set some environment variables before running the translation code: 
```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(path-to-your-cuda-11.8)
export LD_LIBRARY_PATH=$(path-to-your-cuda-11.8)/lib64:${LD_LIBRARY_PATH}
```

#### 5. Weights and Biases
If you want to use [Weights and Biases](https://wandb.ai) to track your training runs (via `cfg.wandb.enabled=true`), you need to ensure your `WAND_API_KEY` environment variable is set correctly.