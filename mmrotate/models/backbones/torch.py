# -*- coding: utf-8 -*-
import torch

print(f"PyTorch version: {torch.__version__}")

print(f"CUDA version: {torch.version.cuda}")

print(f"cuDNN version: {torch.backends.cudnn.version()}")

print(f"Is CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
