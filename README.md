# MELRNet

MELRNet is a lightweight network project for rotated object detection in low-quality remote sensing images. This repository provides ready-to-use configuration files for training and evaluation, making it suitable for research on robust object detection under degraded remote sensing image conditions.

## Project Overview

As remote sensing images suffer from quality degradation, increased noise, and blurred edges, traditional detection methods often struggle in complex scenes. MELRNet aims to improve the detection of rotated objects in low-quality remote sensing images through lightweight structural design and edge-/Gaussian-related feature modeling.


## Key Features

- Lightweight design: balances accuracy and computational cost
- Robustness to low-quality images: optimized for blurry edges, noise, and loss of fine details
- Rotated object detection: suitable for the tilted targets commonly found in remote sensing imagery
- Rich configuration set: includes several stage-based and variant settings for quick trials

## Repository Structure

```text
MELRNet/
  configs/
    MELRNet/
      melr.py
      melr_tiny.py
      melr-hrsc.py
      mamba_stage1_s.py
      mamba_stage2_s.py
      mamba_stage3_s.py
      mamba_stage4_s.py
      melr_DTN_s.py
  mmrotate/
    models/
      backbones/
        legnet.py
  tools/
    train.py
    test.py
```

## Environment Requirements

The following environment is recommended for experiments:

- Python 3.8+
- PyTorch 1.10+
- MMCV / MMEngine
- MMDetection / MMRotate

You can install the project in editable mode as follows:

```bash
pip install -U pip
pip install -e .
```

## Quick Start

### 1. Prepare the Dataset

This project follows the dataset organization convention used by MMRotate / MMDetection. If you are using DOTA or another remote sensing rotated-object detection dataset, prepare the image files and annotation files according to the corresponding format.

### 2. Train the Model

From the project root, run:

```bash
cd MELRNet
python tools/train.py configs/MELRNet/melr.py --work-dir work_dirs/melr
```

To try a lighter variant, use:

```bash
python tools/train.py configs/MELRNet/melr_tiny.py --work-dir work_dirs/melr_tiny
```

### 3. Test and Evaluate

After training, you can evaluate the model with:

```bash
python tools/test.py configs/MELRNet/melr.py work_dirs/melr/epoch_36.pth --eval mAP
```

## License

This project follows the license statement already included in the repository. Please refer to the repository files for the exact terms.

## Contact

If you have any questions, suggestions, or collaboration ideas, please feel free to open an Issue.
