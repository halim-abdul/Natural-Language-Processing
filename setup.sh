#!/bin/bash -i
set -e
conda create -n dnlp python=3.10
conda activate dnlp
conda install --yes "numpy<2.0" scikit-learn -c conda-forge
if command -v nvidia-smi &>/dev/null; then
  conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
else
  conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
fi
conda install tqdm==4.66.2 requests==2.31.0 transformers==4.38.2 tensorboard==2.16.2 tokenizers==0.15.1 -c conda-forge -c huggingface
pip install explainaboard-client==0.1.4 sacrebleu==2.4.0
