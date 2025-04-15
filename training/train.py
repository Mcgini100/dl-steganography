import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm # Progress bar

# Import project modules
from data.dataset import CertificateDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.noise_simulation import NoiseLayer
from training.losses import calculate_image_loss, calculate_message_loss, calculate_combined_loss, calculate_bit_error_rate

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --- Data ---
    print("Loading dataset...")
    train_dataset = CertificateDataset(
        image_dir=args.data_dir,
        image_size=args.image_size,
        message_length=args.message_length,
        training=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True # Helps speed up CPU-GPU transfer
    )
    print(f"Dataset loaded with {len(train_dataset)} images.")

    # --- Models ---
    print("Initializing models...")
    encoder = Encoder(message_length=args.message_length, image_size=args.image_size, hidden_dim=args.hidden_dim).to(device)
    decoder = Decoder(message_length=args.message_length, image_size=args.image_size, hidden_dim=args.hidden_dim).to(device)
    noise_layer = NoiseLayer(
        noise_level=args.noise_level,
        jpeg_quality_range=args.jpeg_quality,
        blur_kernel_range=args.blur_kernel,
        dropout_prob=args.dropout_prob
    ).to(device)
    print("Models initialized.")

    # --- Optimizer ---
    # Combine parameters from both encoder and decoder
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_params, lr=args.learning_rate)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        noise_layer.train() # Make sure noise layer is in training mode

        total_loss_epoch = 0.0
        img_loss_epoch = 0.0
        msg_loss_epoch = 0.0
        ber_epoch = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        for cover_images, messages in progress_bar:
            cover_images = cover_images.to(device)
            messages = messages.to(device) # Shape: (batch_size, message_length)

            optimizer.zero_grad()

            # --- Forward Pass ---
            embedded_images = encoder(cover_images, messages)
            noisy_images = noise_layer(embedded_images) # Apply noise only during training
            decoded_message_logits = decoder(noisy_images)

            # --- Loss Calculation ---
            loss_img = calculate_image_loss(embedded_images, cover_images, loss_type='mse')
            # Ensure messages are in the format expected by the loss (e.g., float 0/1 for BCEWithLogits)
            loss_msg = calculate_message_loss(decoded_message_logits, messages, loss_type='bce')
            total_loss = calculate_combined_loss(loss_img, loss_msg, args.image_loss_weight, args.message_loss_weight)

            # --- Backward Pass & Optimize ---
            total_loss.backward()
            optimizer.step()

            # --- Logging & Metrics ---
            batch_ber = calculate_bit_error_rate(decoded_message_logits.detach(), messages)

            total_loss_epoch += total_loss.item()
            img_loss_epoch += loss_img.item()
            msg_loss_epoch += loss_msg.item()
            ber_epoch += batch_ber

            # Update progress bar description
            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Img Loss": f"{loss_img.item():.4f}",
                "Msg Loss": f"{loss_msg.item():.4f}",
                "BER": f"{batch_ber:.4f}"
            })

        # --- End of Epoch ---
        avg_total_loss = total_loss_epoch / len(train_loader)
        avg_img_loss = img_loss_epoch / len(train_loader)
        avg_msg_loss = msg_loss_epoch / len(train_loader)
        avg_ber = ber_epoch / len(train_loader)

        print(f"Epoch {epoch+1} Summary: Avg Total Loss: {avg_total_loss:.4f}, Avg Img Loss: {avg_img_loss:.4f}, "
              f"Avg Msg Loss: {avg_msg_loss:.4f}, Avg BER: {avg_ber:.4f}")

        # --- Save Model Checkpoint ---
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            encoder_path = os.path.join(args.save_dir, f"encoder_epoch_{epoch+1}.pth")
            decoder_path = os.path.join(args.save_dir, f"decoder_epoch_{epoch+1}.pth")
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
            print(f"Saved models at epoch {epoch+1} to {args.save_dir}")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Steganography Encoder-Decoder Network")

    # Data args
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing training images (data/certificates/original)')
    parser.add_argument('--image-size', type=int, default=128, help='Image size (resized square image)')
    parser.add_argument('--message-length', type=int, default=32, help='Length of the secret message in bits')

    # Model args
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size for networks')

    # Training args
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader')

    # Loss args
    parser.add_argument('--image-loss-weight', type=float, default=0.7, help='Weight for image reconstruction loss')
    parser.add_argument('--message-loss-weight', type=float, default=1.0, help='Weight for message decoding loss')

    # Noise args
    parser.add_argument('--noise-level', type=float, default=0.05, help='Std deviation for Gaussian noise')
    parser.add_argument('--jpeg-quality', type=int, nargs=2, default=[50, 95], help='Range for JPEG quality (min max)')
    parser.add_argument('--blur-kernel', type=int, nargs=2, default=[3, 7], help='Range for Gaussian blur kernel size (min max, odd numbers)')
    parser.add_argument('--dropout-prob', type=float, default=0.1, help='Probability for dropout noise')

    # Saving args
    parser.add_argument('--save-dir', type=str, default='saved_models', help='Directory to save trained models')
    parser.add_argument('--save-interval', type=int, default=10, help='Save model checkpoint every N epochs')

    args = parser.parse_args()

    # Basic validation
    if args.message_loss_weight <= 0:
        print("Warning: Message loss weight should be positive to learn message embedding.")
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        exit(1)

    main(args)