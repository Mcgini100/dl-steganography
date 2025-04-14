# Deep Learning Based Steganography for Certificate Attestation

This project implements a deep learning approach (Encoder-Decoder network) using PyTorch to embed a unique hash into a digital certificate image and extract it robustly.

## Project Structure
dl-steganography/
├── models/ # Network definitions and noise simulation
│ ├── init.py
│ ├── common_blocks.py
│ ├── encoder.py
│ ├── decoder.py
│ └── noise_simulation.py
├── data/ # Data loading utilities
│ ├── init.py
│ ├── dataset.py
│ └── certificates/ # Your specific certificates (for testing/inference)
│ ├── original/
│ └── embedded_dl/
├── training/ # Training scripts and losses
│ ├── init.py
│ ├── train.py
│ └── losses.py
├── inference/ # Scripts for using trained models
│ ├── init.py
│ ├── embed_hash.py
│ └── extract_hash.py
├── saved_models/ # Directory for trained model weights (.pth)
├── requirements.txt # Python dependencies
└── README.md # This file

## Setup

1.  **Clone:** `git clone <my-repo-url>`
2.  **Environment:** Create a Python virtual environment (e.g., `python -m venv venv`).
3.  **Activate:** `source venv/bin/activate` (or `venv\Scripts\activate` on Windows).
4.  **Install Dependencies:** `pip install -r requirements.txt`. Ensure you have a compatible PyTorch version installed (check for CUDA support if you have an Nvidia GPU).
5.  **Training Data:** **Crucially, download a large image dataset** (e.g., COCO, DIV2K, subsets of ImageNet) and place it in a directory accessible by the training script. This is required for `train.py`. Training only on the certificates will not work well.

## Usage

### 1. Training

**(Requires a large dataset and GPU recommended)**

Run the main training script, pointing it to your large image dataset:

```bash
python training/train.py --dataset-dir /path/to/my/large/image/dataset --epochs 100 --batch-size 16 --save-dir saved_models/ --use-lpips