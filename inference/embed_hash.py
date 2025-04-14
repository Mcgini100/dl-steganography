import torch
import argparse
import os
import json
from models.encoder import Encoder
from utils import load_image_tensor, save_image_tensor, string_hash_to_bits_tensor, generate_certificate_hash, HASH_LENGTH_BITS

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Encoder model
    print(f"Loading encoder model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print("Error: Model file not found.")
        return
    encoder = Encoder(secret_len=HASH_LENGTH_BITS).to(device)
    try:
        encoder.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return
    encoder.eval() # Set to evaluation mode
    print("Encoder model loaded.")

    # Get hash string (either directly or from JSON)
    hash_string = None
    if args.hash:
        hash_string = args.hash
        if len(hash_string) * 4 != HASH_LENGTH_BITS:
             print(f"Error: Provided hash length is incorrect ({len(hash_string)} chars != {HASH_LENGTH_BITS//4} expected)")
             return
    elif args.json_data:
        print(f"Loading degree data from: {args.json_data}")
        try:
            with open(args.json_data, 'r') as f:
                degree_data = json.load(f)
            hash_string = generate_certificate_hash(degree_data)
            print(f"Generated hash: {hash_string}")
        except Exception as e:
            print(f"Error reading JSON or generating hash: {e}")
            return
    else:
        print("Error: Either --hash or --json-data must be provided.")
        return

    # Prepare input image and secret tensor
    try:
        print(f"Loading cover image: {args.cover_image}")
        # Assume encoder was trained on a certain size, use that size here
        cover_tensor = load_image_tensor(args.cover_image, size=args.image_size, device=device).unsqueeze(0) # Add batch dim
        secret_tensor = string_hash_to_bits_tensor(hash_string, device=device).unsqueeze(0) # Add batch dim
    except Exception as e:
        print(f"Error preparing inputs: {e}")
        return

    # Perform embedding
    print("Embedding hash into image...")
    with torch.no_grad(): # No need to track gradients during inference
        stego_tensor = encoder(cover_tensor, secret_tensor)
    print("Embedding finished.")

    # Save output
    os.makedirs(os.path.dirname(args.output_image), exist_ok=True)
    print(f"Saving stego image to: {args.output_image}")
    try:
        save_image_tensor(stego_tensor, args.output_image)
        print("Stego image saved successfully.")
    except Exception as e:
        print(f"Error saving output image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed Hash using Trained DL Steganography Encoder")
    parser.add_argument('-c', '--cover-image', type=str, required=True, help='Path to the original cover image (certificate).')
    parser.add_argument('-o', '--output-image', type=str, required=True, help='Path to save the output stego image.')
    parser.add_argument('-m', '--model-path', type=str, required=True, help='Path to the trained encoder model (.pth file).')
    parser.add_argument('--image-size', type=int, default=256, help='Image size the model was trained on.')

    hash_group = parser.add_mutually_exclusive_group(required=True)
    hash_group.add_argument('--hash', type=str, help='The exact SHA-256 hash string to embed.')
    hash_group.add_argument('--json-data', type=str, help='Path to JSON file with certificate data to generate hash from.')

    args = parser.parse_args()
    main(args)