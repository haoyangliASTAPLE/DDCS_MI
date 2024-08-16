# Model Inversion
This is the code for the manuscript "A Sample-Level Evaluation and Generative Framework for Model Inversion Attacks". All experiments are conducted on 4 GPUs of NVIDIA GeForce RTX 3090 server with Linus OS, CUDA 12.2, and PyTorch 1.7.1. Our code utilizes *DistributedDataParallel* from PyTorch to train the model in parallel manner.

## Setup
1. Install StyleGAN Docker environment at https://github.com/NVlabs/stylegan2-ada-pytorch.
2. Download relevant datasets used in the manuscript and preprocess these datasets with the scripts in the `./datasets` folder. For preparing the CelebA-HQ dataset, you need to follow the instruction in https://github.com/nperraud/download-celebA-HQ.
3. Download pretrained StyleGAN weights from https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/. Create a `./pretrained` directory and place these weights in this directory.

## Run
Run the scripts in the `./scripts` folder. You can change the parameters in the scripts for different settings of MI attacks. All experimental results will be saved to a automatically created folder `./.logger`.
