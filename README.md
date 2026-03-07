# WaveComm

**[ICRA2026] WaveComm: Lightweight Communication for Collaborative Perception via Wavelet Feature Distillation**

> This is the official implementation of WaveComm, a wavelet-based communication framework for collaborative perception.

## Overview

WaveComm is a novel collaborative perception framework that achieves communication efficiency by exploiting the frequency structure of feature maps. The core idea is to decompose BEV feature maps using Discrete Wavelet Transform (DWT), transmit only compact low-frequency components, and reconstruct the full features at the receiver using a lightweight generator.

<div align="center">
<img src="assets/framework.png" width="80%">
</div>

## Video

<div align="center">
<img src="assets/ICRA2026_WaveComm.gif" width="80%">
</div>

## Key Features

- **Wavelet Feature Distillation**: Uses DWT to decompose features into low-frequency and high-frequency components. Only low-frequency components are transmitted to reduce bandwidth.
- **Multi-Scale Distillation (MSD) Loss**: A hybrid loss function combining:
  - Reconstruction Loss (pixel-level)
  - SSIM Loss (structural-level)
  - Perceptual Loss (semantic-level)
  - Adversarial Loss (distributional-level)
- **Lightweight Generator**: A lightweight neural network that reconstructs high-frequency details from transmitted low-frequency components.
- **Communication efficiency**: Reduces communication volume to **86.3%** (OPV2V) and **87.0%** (DAIR-V2X) of the original while maintaining state-of-the-art performance.

## Method

<div align="center">
<img src="assets/pipeline.png" width="80%">
</div>

### Wavelet Feature Distillation Module

The Wavelet Feature Distillation Module consists of two parts:

1. **Feature Decomposition**: Using DWT to decompose BEV features into low-frequency and high-frequency components. Low-frequency components retain most semantic information and global structure.

2. **Feature Reconstruction**: A lightweight generator reconstructs the full feature map from the transmitted low-frequency components. The generator is optimized using MSD Loss to ensure high-fidelity reconstruction.

## Performance

### OPV2V Dataset

<table>
    <tr>
        <th rowspan="2" style="text-align: center;">Method</th>
        <th colspan="3" style="text-align: center;">Camera-based</th>
        <th colspan="3" style="text-align: center;">LiDAR-based</th>
    </tr>
    <tr>
        <th style="text-align: center;">AP50</th>
        <th style="text-align: center;">AP70</th>
        <th style="text-align: center;">Comm</th>
        <th style="text-align: center;">AP50</th>
        <th style="text-align: center;">AP70</th>
        <th style="text-align: center;">Comm</th>
    </tr>
    <tr>
        <td>No Collaboration</td>
        <td>0.405</td><td>0.216</td><td>0.0</td>
        <td>0.782</td><td>0.634</td><td>0.0</td>
    </tr>
    <tr>
        <td>F-Cooper</td>
        <td>0.469</td><td>0.219</td><td>22.0</td>
        <td>0.763</td><td>0.481</td><td>24.0</td>
    </tr>
    <tr>
        <td>DiscoNet</td>
        <td>0.517</td><td>0.234</td><td>22.0</td>
        <td>0.882</td><td>0.737</td><td>24.0</td>
    </tr>
    <tr>
        <td>AttFusion</td>
        <td>0.529</td><td>0.252</td><td>22.0</td>
        <td>0.878</td><td>0.751</td><td>24.0</td>
    </tr>
    <tr>
        <td>V2X-ViT</td>
        <td>0.603</td><td>0.289</td><td>22.0</td>
        <td>0.917</td><td>0.790</td><td>24.0</td>
    </tr>
    <tr>
        <td>CoBEVT</td>
        <td>0.571</td><td>0.261</td><td>22.0</td>
        <td>0.935</td><td>0.821</td><td>24.0</td>
    </tr>
    <tr>
        <td>HM-ViT</td>
        <td>0.643</td><td>0.370</td><td>22.0</td>
        <td>0.950</td><td>0.873</td><td>24.0</td>
    </tr>
    <tr>
        <td><b>WaveComm (ours)</b></td>
        <td><b>0.681</b></td><td><b>0.451</b></td><td><b>19.0</b></td>
        <td><b>0.965</b></td><td><b>0.926</b></td><td><b>21.0</b></td>
    </tr>
</table>

### DAIR-V2X Dataset

<table>
    <tr>
        <th rowspan="2" style="text-align: center;">Method</th>
        <th colspan="3" style="text-align: center;">Camera-based</th>
        <th colspan="3" style="text-align: center;">LiDAR-based</th>
    </tr>
    <tr>
        <th style="text-align: center;">AP30</th>
        <th style="text-align: center;">AP50</th>
        <th style="text-align: center;">Comm</th>
        <th style="text-align: center;">AP30</th>
        <th style="text-align: center;">AP50</th>
        <th style="text-align: center;">Comm</th>
    </tr>
    <tr>
        <td>No Collaboration</td>
        <td>0.014</td><td>0.004</td><td>0.0</td>
        <td>0.421</td><td>0.405</td><td>0.0</td>
    </tr>
    <tr>
        <td>F-Cooper</td>
        <td>0.115</td><td>0.026</td><td>23.0</td>
        <td>0.723</td><td>0.620</td><td>23.0</td>
    </tr>
    <tr>
        <td>DiscoNet</td>
        <td>0.083</td><td>0.017</td><td>23.0</td>
        <td>0.746</td><td>0.685</td><td>23.0</td>
    </tr>
    <tr>
        <td>AttFusion</td>
        <td>0.094</td><td>0.021</td><td>23.0</td>
        <td>0.738</td><td>0.673</td><td>23.0</td>
    </tr>
    <tr>
        <td>V2X-ViT</td>
        <td>0.198</td><td>0.057</td><td>23.0</td>
        <td>0.785</td><td>0.521</td><td>23.0</td>
    </tr>
    <tr>
        <td>CoBEVT</td>
        <td>0.182</td><td>0.042</td><td>23.0</td>
        <td>0.787</td><td>0.692</td><td>23.0</td>
    </tr>
    <tr>
        <td>HM-ViT</td>
        <td>0.163</td><td>0.044</td><td>23.0</td>
        <td>0.818</td><td>0.761</td><td>23.0</td>
    </tr>
    <tr>
        <td><b>WaveComm (ours)</b></td>
        <td><b>0.274</b></td><td><b>0.123</b></td><td><b>20.0</b></td>
        <td><b>0.831</b></td><td><b>0.790</b></td><td><b>20.0</b></td>
    </tr>
</table>

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12.0+
- CUDA 11.6+

### Step 1: Create Environment

```bash
conda create -n wavecomm python=3.8
conda activate wavecomm
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Spconv

```bash
pip install spconv-cu116  # match your cudatoolkit version
```

### Step 4: Compile Bbx IoU CUDA Version

```bash
python opencood/utils/setup.py build_ext --inplace
```

### Step 5: Install This Project

```bash
python setup.py develop
```

## Data Preparation

### OPV2V Dataset

Please refer to [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). You also need to download `additional-001.zip` for camera modality.

### DAIR-V2X Dataset

Download from [DAIR-V2X](https://thudair.baai.ac.cn/index). Use complemented annotation.

Create a `dataset` folder and organize as follows:

```
WaveComm/
└── dataset/
    ├── my_dair_v2x/
    │   ├── v2x_c/
    │   ├── v2x_i/
    │   └── v2x_v/
    └── OPV2V/
        ├── additional/
        ├── test/
        ├── train/
        └── validate/
```

## Training

### Train the Model

```bash
python opencood/tools/train.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```

Arguments:
- `-y` or `--hypes_yaml`: Path to training configuration file
- `--model_dir` (optional): Path to checkpoint for fine-tuning or continuing training

### Train with DDP

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y ${CONFIG_FILE}
```

## Testing

```bash
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} [--fusion_method intermediate]
```

## Project Structure

```
WaveComm/
├── opencood/
│   ├── hypes_yaml/          # Configuration files
│   ├── models/              # Model architectures
│   │   └── comm_modules/    # Communication modules
│   ├── data_utils/          # Data loading and preprocessing
│   ├── fuse_modules/        # Feature fusion modules
│   ├── loss/                # Loss functions
│   ├── utils/               # Utility functions
│   └── tools/               # Training and testing scripts
├── dataset/                 # Dataset directory
├── assets/                  # assets for README
└── README.md
```

## Citation

```latex
@article{wavecomm2026,
  title={WaveComm: Lightweight Communication of BEV Feature Maps for Collaborative Perception via Wavelet Feature Distillation},
  author={},
  journal={},
  year={2026}
}
```

## Acknowledgments

This project is built upon the [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [HEAL](https://github.com/yifanlu0227/HEAL) frameworks. We thank the authors for their excellent work.
