import torch
import torchvision.transforms.functional as TF
from PIL import Image
import io
import random
import numpy as np

# --- Simple Noise Functions (Applied OUTSIDE model graph during training loop) ---

def apply_jpeg_bytes(img_tensor_batch: torch.Tensor, quality: int) -> torch.Tensor:
    """Applies JPEG compression by saving/loading via bytes buffer."""
    device = img_tensor_batch.device
    batch_size, _, h, w = img_tensor_batch.shape
    result_batch = torch.zeros_like(img_tensor_batch)

    for i in range(batch_size):
        img_tensor = img_tensor_batch[i] # Get single image C, H, W
        try:
            # Convert tensor [0,1] to PIL Image
            img_np = img_tensor.cpu().detach().numpy()
            img_np = np.transpose(img_np, (1, 2, 0)) # C, H, W -> H, W, C
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # Save to bytes buffer as JPEG
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)

            # Load back from buffer
            img_pil_jpeg = Image.open(buffer).convert('RGB')

            # Convert back to tensor [0,1]
            img_tensor_jpeg = TF.to_tensor(img_pil_jpeg) # Scales to [0.0, 1.0]
            result_batch[i] = img_tensor_jpeg

        except Exception as e:
            print(f"Warning: Error during JPEG simulation (Q={quality}): {e}. Returning original.")
            result_batch[i] = img_tensor # Return original if error

    return result_batch.to(device)


def add_gaussian_noise(img_tensor_batch: torch.Tensor, std_dev: float) -> torch.Tensor:
    """Adds Gaussian noise to image tensor batch [0, 1]."""
    noise = torch.randn_like(img_tensor_batch) * std_dev
    noisy_img = img_tensor_batch + noise
    return torch.clamp(noisy_img, 0.0, 1.0)

def apply_random_noise(stego_image_batch: torch.Tensor) -> torch.Tensor:
    """Applies a random selection of noise layers."""
    device = stego_image_batch.device
    processed_batch = stego_image_batch.clone() # Work on a copy

    # Apply noise layers with certain probabilities
    if random.random() < 0.8: # High chance of JPEG
        quality = random.randint(50, 95)
        processed_batch = apply_jpeg_bytes(processed_batch, quality)

    if random.random() < 0.3: # Lower chance of Gaussian noise
        std = random.uniform(0, 0.1) # Noise level up to 10%
        processed_batch = add_gaussian_noise(processed_batch, std)

    # Add other distortions like blur, dropout, etc. here if needed

    return processed_batch.to(device)