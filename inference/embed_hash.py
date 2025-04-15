import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import numpy as np

# Import project modules
from models.encoder import Encoder # Make sure Encoder class is importable

# Helper function to convert hex hash to binary tensor
def hex_to_binary_tensor(hex_hash, message_length):
    """Converts a hex hash string to a binary tensor of specified length."""
    try:
        scale = 16 ## equals to hexadecimal
        num_of_bits = message_length
        binary_string = bin(int(hex_hash, scale))[2:].zfill(num_of_bits)
        if len(binary_string) > message_length:
             # If hash is too long, truncate (or handle error differently)
             print(f"Warning: Provided hash '{hex_hash}' is longer than message length {message_length}. Truncating.")
             binary_string = binary_string[:message_length]
        elif len(binary_string) < message_length:
             # If hash is too short, pad (or handle error)
             print(f"Warning: Provided hash '{hex_hash}' is shorter than message length {message_length}. Padding with zeros.")
             binary_string = binary_string.zfill(message_length)

        # Convert binary string to tensor of floats (0.0 or 1.0)
        message_tensor = torch.tensor([float(bit) for bit in binary_string], dtype=torch.float32)
        return message_tensor
    except ValueError:
        raise ValueError(f"Invalid hex hash string provided: {hex_hash}")
    except Exception as e:
        raise RuntimeError(f"Error converting hash to tensor: {e}")


# Helper function to convert tensor image back to PIL Image
def tensor_to_pil(image_tensor):
    """Converts a PyTorch tensor (CxHxW, range [-1, 1]) back to a PIL Image."""
    # Ensure tensor is on CPU and remove batch dimension if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.cpu().detach()

    # Denormalize from [-1, 1] to [0, 1]
    image_tensor = (image_tensor + 1.0) / 2.0

    # Clamp values to [0, 1] to avoid artifacts
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Convert to PIL Image format (HxWxC, uint8)
    image_pil = transforms.ToPILImage()(image_tensor)
    return image_pil

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isfile(args.input_image):
        print(f"Error: Input image not found: {args.input_image}")
        return
    if not os.path.isfile(args.model_path):
        print(f"Error: Model weights not found: {args.model_path}")
        return

    output_dir = os.path.dirname(args.output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Load Model ---
    print("Loading Encoder model...")
    # Infer image size and message length from args (or potentially from model state_dict if saved)
    # We need image_size used during training for the transforms and model init.
    # Assume it's passed via args for now. Message length is crucial.
    encoder = Encoder(message_length=args.message_length, image_size=args.image_size, hidden_dim=args.hidden_dim).to(device)
    try:
        encoder.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure --message-length, --image-size, and --hidden-dim match the trained model.")
        return

    encoder.eval() # Set model to evaluation mode
    print("Model loaded.")

    # --- Prepare Inputs ---
    print("Preparing input image and hash...")
    # 1. Load Image
    try:
        cover_image_pil = Image.open(args.input_image).convert('RGB')
    except Exception as e:
        print(f"Error opening image file {args.input_image}: {e}")
        return

    # 2. Define Transformations (MUST match training)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
    ])
    cover_image_tensor = transform(cover_image_pil).unsqueeze(0).to(device) # Add batch dim and move to device

    # 3. Convert Hash to Message Tensor
    try:
        message_tensor = hex_to_binary_tensor(args.hash, args.message_length).unsqueeze(0).to(device) # Add batch dim
    except (ValueError, RuntimeError) as e:
        print(f"Error processing hash: {e}")
        return

    # --- Inference ---
    print("Embedding hash into image...")
    with torch.no_grad(): # Disable gradient calculations
        embedded_image_tensor = encoder(cover_image_tensor, message_tensor)

    # --- Save Output ---
    print("Saving embedded image...")
    try:
        embedded_image_pil = tensor_to_pil(embedded_image_tensor)
        embedded_image_pil.save(args.output_image)
        print(f"Embedded image saved successfully to: {args.output_image}")
    except Exception as e:
        print(f"Error saving output image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a hash into an image using a trained Encoder model.")

    # Required Args
    parser.add_argument('--input-image', type=str, required=True, help='Path to the original cover image.')
    parser.add_argument('--output-image', type=str, required=True, help='Path to save the embedded (steganographic) image.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained Encoder model weights (.pth file).')
    parser.add_argument('--hash', type=str, required=True, help='The unique hash string (hexadecimal) to embed.')

    # Model/Data Config (MUST match training)
    parser.add_argument('--image-size', type=int, default=128, help='Image size used during training.')
    parser.add_argument('--message-length', type=int, default=32, help='Message length (in bits) used during training.')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size of the model (must match trained model).')


    args = parser.parse_args()
    main(args)