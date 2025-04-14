import torch
import torch.optim as optim
import argparse
import os
import json
from tqdm import tqdm # Progress bar

# Import local modules
from models.encoder import Encoder
from models.decoder import Decoder
from models.noise_simulation import apply_random_noise # Or specific noise layers
from data.dataset import create_dataloader
from training.losses import SteganographyLoss
from utils import HASH_LENGTH_BITS # Use the constant

def calculate_ber(secret_true, secret_pred_probs):
    """Calculates the Bit Error Rate."""
    pred_bits = (secret_pred_probs >= 0.5).float()
    incorrect_bits = torch.sum(torch.abs(secret_true - pred_bits))
    ber = incorrect_bits / (secret_true.size(0) * secret_true.size(1)) # Per bit
    return ber.item()

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Prepare DataLoaders (Using a generic dataset loader)
    # IMPORTANT: Replace args.dataset_dir with path to a LARGE dataset (e.g., COCO, ImageNet subset)
    print("Loading data...")
    if not os.path.isdir(args.dataset_dir):
         print(f"Error: Dataset directory not found: {args.dataset_dir}")
         print("Please provide path to a large image dataset for training.")
         return

    train_loader = create_dataloader(args.dataset_dir, args.image_size, args.batch_size, num_workers=args.workers)
    # Add validation loader if you have a separate validation set
    # val_loader = create_dataloader(args.val_dataset_dir, args.image_size, args.batch_size, num_workers=args.workers)
    print("Data loaded.")

    # Initialize Models
    encoder = Encoder(secret_len=HASH_LENGTH_BITS).to(device)
    decoder = Decoder(secret_len=HASH_LENGTH_BITS).to(device)

    # Initialize Optimizer (Optimize parameters of both models)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    # Initialize Loss function
    criterion = SteganographyLoss(secret_len=HASH_LENGTH_BITS, beta=args.beta, use_lpips=args.use_lpips, device=device).to(device)

    # Training Loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

        total_loss_epoch = 0.0
        enc_loss_epoch = 0.0
        dec_loss_epoch = 0.0
        ber_epoch = 0.0

        for batch_idx, cover_images in pbar:
            cover_images = cover_images.to(device)
            batch_size = cover_images.size(0)

            # Generate random secret bits for this batch
            # Training with random bits usually helps generalization
            secret_true = torch.randint(0, 2, (batch_size, HASH_LENGTH_BITS), dtype=torch.float32).to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass - Encoder
            stego_images = encoder(cover_images, secret_true)

            # Apply Noise Simulation Layer/Function
            distorted_stego = apply_random_noise(stego_images) # Use the noise simulation

            # Forward pass - Decoder
            secret_pred_probs = decoder(distorted_stego)

            # Calculate Loss
            total_loss, encoder_loss, decoder_loss = criterion(cover_images, secret_true, stego_images, secret_pred_probs)

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            # Log metrics for progress bar
            batch_ber = calculate_ber(secret_true, secret_pred_probs)
            total_loss_epoch += total_loss.item()
            enc_loss_epoch += encoder_loss.item()
            dec_loss_epoch += decoder_loss.item()
            ber_epoch += batch_ber

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'L_enc': f"{encoder_loss.item():.4f}",
                'L_dec': f"{decoder_loss.item():.4f}",
                'BER': f"{batch_ber:.4f}"
            })

        # Print epoch summary
        avg_loss = total_loss_epoch / len(train_loader)
        avg_enc_loss = enc_loss_epoch / len(train_loader)
        avg_dec_loss = dec_loss_epoch / len(train_loader)
        avg_ber = ber_epoch / len(train_loader)
        print(f"Epoch {epoch} Summary: Avg Loss: {avg_loss:.4f}, Avg L_enc: {avg_enc_loss:.4f}, Avg L_dec: {avg_dec_loss:.4f}, Avg BER: {avg_ber:.4f}")

        # Validation Loop (Optional but Recommended)
        # if val_loader:
        #     encoder.eval()
        #     decoder.eval()
        #     val_loss = 0.0
        #     val_ber = 0.0
        #     with torch.no_grad():
        #         for val_images in val_loader:
        #             # ... similar process as training loop but without backprop ...
        #     print(f"Epoch {epoch} Validation: Avg Loss: ..., Avg BER: ...")

        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_path_encoder = os.path.join(args.save_dir, f"encoder_epoch_{epoch}.pth")
            save_path_decoder = os.path.join(args.save_dir, f"decoder_epoch_{epoch}.pth")
            torch.save(encoder.state_dict(), save_path_encoder)
            torch.save(decoder.state_dict(), save_path_decoder)
            print(f"Saved models to {args.save_dir}")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep Steganography Model")
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to the LARGE training image dataset directory.')
    # parser.add_argument('--val-dataset-dir', type=str, default=None, help='Path to the validation image dataset directory.')
    parser.add_argument('--image-size', type=int, default=256, help='Image size to train on (e.g., 256)')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.75, help='Weight for decoder loss (L_total = L_enc + beta*L_dec)')
    parser.add_argument('--use-lpips', action='store_true', help='Use LPIPS perceptual loss for encoder')
    parser.add_argument('--save-dir', type=str, default='saved_models', help='Directory to save model checkpoints')
    parser.add_argument('--save-interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available')

    args = parser.parse_args()

    # Simple validation
    if not os.path.isdir(args.dataset_dir):
         print(f"Error: Dataset directory '{args.dataset_dir}' not found.")
    else:
        main(args)