import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import glob

class SteganographyDataset(Dataset):
    """Dataset for loading cover images for training."""
    def __init__(self, image_dir, image_size, limit=None):
        """
        Args:
            image_dir (str): Directory containing training cover images.
                             IMPORTANT: Use a large dataset like COCO, ImageNet subset, etc.
            image_size (int): Target size to resize images (e.g., 256).
            limit (int, optional): Limit the number of images used. Defaults to None.
        """
        super().__init__()
        self.image_size = image_size
        # Find all common image files
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper()))) # Include uppercase extensions

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in directory: {image_dir}")

        if limit:
            self.image_paths = self.image_paths[:limit]
        print(f"Dataset: Found {len(self.image_paths)} images in {image_dir}")

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize([image_size, image_size], interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(), # Converts PIL image [0, 255] to tensor [0.0, 1.0]
            # Add normalization if desired (e.g., transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            # If normalizing, ensure inverse transform when saving/displaying images
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            # Apply transformations
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Skipping.")
            # Return a dummy tensor or the next valid item
            return self.__getitem__((idx + 1) % len(self))

def create_dataloader(image_dir, image_size, batch_size, limit=None, num_workers=4):
    """Creates a DataLoader for the steganography dataset."""
    dataset = SteganographyDataset(image_dir, image_size, limit=limit)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle data for training
        num_workers=num_workers,
        pin_memory=True, # Speeds up data transfer to GPU
        drop_last=True # Drop last incomplete batch
    )
    return dataloader