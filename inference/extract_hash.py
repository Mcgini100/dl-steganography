import torch
import argparse
import os
from models.decoder import Decoder
from utils import load_image_tensor, bits_tensor_to_string_hash, HASH_LENGTH_BITS

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Decoder model
    print(f"Loading decoder model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print("Error: Model file not found.")
        return
    decoder = Decoder(secret_len=HASH_LENGTH_BITS).to(device)
    try:
        decoder.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return
    decoder.eval() # Set to evaluation mode
    print("Decoder model loaded.")

    # Prepare input image
    try:
        print(f"Loading stego image: {args.stego_image}")
        # Assume decoder was trained on a certain size, use that size here
        stego_tensor = load_image_tensor(args.stego_image, size=args.image_size, device=device).unsqueeze(0) # Add batch dim
    except Exception as e:
        print(f"Error loading stego image: {e}")
        return

    # Perform extraction
    print("Extracting hash from image...")
    with torch.no_grad(): # No need to track gradients
        secret_pred_probs = decoder(stego_tensor)
    print("Extraction finished.")

    # Convert output tensor to hash string
    try:
        extracted_hash = bits_tensor_to_string_hash(secret_pred_probs.squeeze(0)) # Remove batch dim
        print(f"\nExtracted Hash: {extracted_hash}")
    except Exception as e:
        print(f"\nError converting extracted bits to hash: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Hash using Trained DL Steganography Decoder")
    parser.add_argument('-s', '--stego-image', type=str, required=True, help='Path to the stego image containing the hidden hash.')
    parser.add_argument('-m', '--model-path', type=str, required=True, help='Path to the trained decoder model (.pth file).')
    parser.add_argument('--image-size', type=int, default=256, help='Image size the model was trained on.')

    args = parser.parse_args()
    main(args)