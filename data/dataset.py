import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random

class CertificateDataset(Dataset):
    """
    Dataset for loading certificate images and generating random messages.
    """
    def __init__(self, image_dir, image_size=128, message_length=32, training=True):
        """
        Args:
            image_dir (str): Directory containing the original certificate images.
            image_size (int): The size to resize images to (square image assumed).
            message_length (int): The length of the binary message to embed.
            training (bool): If True, generates random messages. If False, behavior might differ (e.g., for fixed message testing).
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.message_length = message_length
        self.training = training

        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Define image transformations
        # Normalize to [-1, 1] to match Encoder's Tanh output
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # Converts to [0, 1] range, CxHxW
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Converts to [-1, 1] range
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB') # Ensure 3 channels
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder or skip? For now, load the next one recursively.
            # Be careful with recursion depth. Better to handle this in dataloader collation or filter list initially.
            return self.__getitem__((idx + 1) % len(self))


        # Apply transformations
        image_tensor = self.transform(image)

        # Generate random binary message (as floats 0.0 or 1.0)
        # Using 0/1 is common with BCEWithLogitsLoss. Using -1/1 might be used with Tanh + MSE loss.
        # Let's stick to 0/1 for BCEWithLogitsLoss.
        if self.training:
            message = torch.randint(0, 2, (self.message_length,), dtype=torch.float32)
        else:
            # For validation/testing, maybe use a fixed message or sequence?
            # Here, we'll just generate random ones too, assuming validation still tests general capability.
            message = torch.randint(0, 2, (self.message_length,), dtype=torch.float32)

        return image_tensor, message