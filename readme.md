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
│ ├── original/ # Place original certificate images here
│ └── embedded_dl/ # Output directory for images with embedded hashes
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

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd dl-steganography
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Data:**
    *   Place your original certificate images (e.g., `.png`, `.jpg`) into the `data/certificates/original/` directory. You need a reasonable number of diverse images for effective training.

## Training

1.  **Run the training script:**
    ```bash
    python training/train.py \
        --data-dir data/certificates/original/ \
        --image-size 128 \
        --message-length 64 \
        --epochs 150 \
        --batch-size 32 \
        --learning-rate 0.0005 \
        --image-loss-weight 0.8 \
        --message-loss-weight 1.0 \
        --noise-level 0.05 \
        --jpeg-quality 60 95 \
        --blur-kernel 3 5 \
        --dropout-prob 0.05 \
        --save-dir saved_models/ \
        --save-interval 25 \
        --num-workers 4 # Adjust based on your system
    ```
    *   Adjust hyperparameters (`image-size`, `message-length`, `epochs`, `batch-size`, `learning-rate`, loss weights, noise parameters) as needed based on your dataset and desired performance (imperceptibility vs. robustness).
    *   Training requires a CUDA-enabled GPU for reasonable speed. It will automatically use the GPU if available.
    *   Model checkpoints (`encoder_epoch_*.pth`, `decoder_epoch_*.pth`) will be saved in the `saved_models/` directory.

## Inference

### Embedding a Hash

1.  Use the `embed_hash.py` script:
    ```bash
    python inference/embed_hash.py \
        --input-image data/certificates/original/my_certificate.png \
        --output-image data/certificates/embedded_dl/my_certificate_embedded.png \
        --model-path saved_models/encoder_epoch_150.pth \
        --hash "a1b2c3d4e5f6a7b8" \
        --image-size 128 \
        --message-length 64 \
        --hidden-dim 64 # Ensure this matches the trained model config
    ```
    *   Replace placeholders with your actual file paths, desired output path, the specific trained encoder model, and the hex hash you want to embed.
    *   `image-size`, `message-length`, and `hidden-dim` **must match** the parameters used during training for the loaded model.

### Extracting a Hash

1.  Use the `extract_hash.py` script:
    ```bash
    python inference/extract_hash.py \
        --input-image data/certificates/embedded_dl/my_certificate_embedded.png \
        --model-path saved_models/decoder_epoch_150.pth \
        --image-size 128 \
        --message-length 64 \
        --hidden-dim 64 # Ensure this matches the trained model config
    ```
    *   Replace placeholders with the path to the image containing the embedded hash and the specific trained decoder model.
    *   `image-size`, `message-length`, and `hidden-dim` **must match** the parameters used during training for the loaded model.
    *   The script will print the extracted hexadecimal hash to the console.

## Notes

*   **Imperceptibility vs. Robustness:** There's a trade-off. Higher `image_loss_weight` generally leads to less visible changes but potentially lower robustness to noise/attacks. Higher `message_loss_weight` prioritizes accurate message recovery, potentially at the cost of more noticeable image alterations. The `NoiseLayer` during training is crucial for robustness.
*   **Hash Length:** The `message_length` parameter determines the number of *bits* that can be embedded. A 64-bit message length can store a 16-character hexadecimal hash (64 bits / 4 bits per hex char = 16 chars). Adjust accordingly.
*   **Model Architecture:** The provided `Encoder` and `Decoder` are examples. You might need to adjust their complexity (number of layers, hidden dimensions) based on the `image_size`, `message_length`, and dataset complexity.
*   **GPU Usage:** Training is computationally intensive and benefits significantly from a GPU. Inference is generally faster. The scripts will attempt to use CUDA if available. Google Colab's free GPU tier is a good option for training.
*   **Dependencies:** Ensure all packages in `requirements.txt` are installed, especially `kornia` if you want to use JPEG compression simulation during training.