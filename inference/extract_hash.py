import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import numpy as np

# Import project modules
from models.decoder import Decoder # Make sure Decoder class is importable

# Helper function to convert binary tensor back to hex hash
def binary_tensor_to_hex(binary_tensor, threshold=0.0):
    """Converts a binary tensor (logits or probabilities) back to a hex hash string."""
    # Ensure tensor is on CPU and remove batch dimension if present
    if binary_tensor.dim() == 2: # If batch dim exists
        binary_tensor = binary_tensor.squeeze(0)
    binary_tensor = binary_tensor.cpu().detach()

    # Apply threshold to get binary values (0s and 1s)
    # Use 0.0 threshold for logits (output of BCEWithLogitsLoss)
    # Use 0.5 threshold for probabilities (output of Sigmoid)
    bits = (binary_tensor > threshold).int().numpy()

    # Convert binary array to string
    binary_string = "".join(map(str, bits))

    # Convert binary string to integer, then to hex
    try:
        if not binary_string:
             return "Error: No bits extracted."
        num = int(binary_string, 2)
        # Format as hex string, removing the '0x' prefix
        # Calculate required hex length (each hex char is 4 bits)
        hex_len = (len(binary_string) + 3) // 4
        hex_hash = hex(num)[2:].zfill(hex_len) # Pad with leading zeros if needed
        return hex_hash
    except ValueError:
        # This might happen if binary_string is empty or invalid
         return "Error: Could not convert extracted bits to hex."
    except Exception as e:
         return f"Error during hex conversion: {e}"

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

    # --- Load Model ---
    print("Loading Decoder model...")
    # We need image_size and message_length used during training.
    decoder = Decoder(message_length=args.message_length, image_size=args.image_size, hidden_dim=args.hidden_dim).to(device)
    try:
        decoder.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure --message-length, --image-size, and --hidden-dim match the trained model.")
        return

    decoder.eval() # Set model to evaluation mode
    print("Model loaded.")

    # --- Prepare Input ---
    print("Preparing input image...")
    # 1. Load Image
    try:
        embedded_image_pil = Image.open(args.input_image).convert('RGB')
    except Exception as e:
        print(f"Error opening image file {args.input_image}: {e}")
        return

    # 2. Define Transformations (MUST match training)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
    ])
    embedded_image_tensor = transform(embedded_image_pil).unsqueeze(0).to(device) # Add batch dim

    # --- Inference ---
    print("Extracting hash from image...")
    with torch.no_grad(): # Disable gradient calculations
        decoded_message_logits = decoder(embedded_image_tensor)

    # --- Process Output ---
    print("Converting extracted data to hash...")
    # Convert the logits tensor to a hex string
    # Assuming the model outputs logits and was trained with BCEWithLogitsLoss, use threshold 0.0
    extracted_hash = binary_tensor_to_hex(decoded_message_logits, threshold=0.0)

    print("-" * 30)
    print(f"Extracted Hex Hash: {extracted_hash}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a hash from an image using a trained Decoder model.")

    # Required Args
    parser.add_argument('--input-image', type=str, required=True, help='Path to the embedded (steganographic) image.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained Decoder model weights (.pth file).')

    # Model/Data Config (MUST match training)
    parser.add_argument('--image-size', type=int, default=128, help='Image size used during training.')
    parser.add_argument('--message-length', type=int, default=32, help='Message length (in bits) used during training.')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size of the model (must match trained model).')

    args = parser.parse_args()
    main(args)