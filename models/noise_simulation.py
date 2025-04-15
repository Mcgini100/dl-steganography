import torch
import torch.nn as nn
import random
import torchvision.transforms.functional as TF
import kornia # Use kornia for more advanced augmentations like JPEG

# Ensure kornia is installed: pip install kornia

class NoiseLayer(nn.Module):
    """
    Applies random noise/distortions to images during training.
    Simulates real-world conditions the embedded image might face.
    """
    def __init__(self, noise_level=0.1, jpeg_quality_range=(50, 95), blur_kernel_range=(3, 7), dropout_prob=0.1):
        super(NoiseLayer, self).__init__()
        self.noise_level = noise_level
        self.jpeg_quality_min, self.jpeg_quality_max = jpeg_quality_range
        self.blur_kernel_min, self.blur_kernel_max = blur_kernel_range
        self.dropout_prob = dropout_prob

        # Kornia requires input tensor in [0, 1] range for JPEG compression
        # If your network uses [-1, 1], you need to rescale before and after JPEG.

    def apply_gaussian_noise(self, image):
        noise = torch.randn_like(image) * self.noise_level
        return image + noise

    def apply_jpeg_compression(self, image):
        # Kornia JPEG works on [0, 1] range, BxCxHxW float tensor
        # Assuming input `image` is in [-1, 1] range from Encoder (Tanh output)
        image_01 = (image + 1.0) / 2.0 # Rescale to [0, 1]

        quality = torch.tensor(random.randint(self.jpeg_quality_min, self.jpeg_quality_max)).float().to(image.device)
        jpeg_compressor = kornia.augmentation.RandomJPEG(p=1.0, quality=(self.jpeg_quality_min, self.jpeg_quality_max)) # Apply JPEG to all images in batch with random quality
        image_jpeg_01 = jpeg_compressor(image_01)

        # Rescale back to [-1, 1]
        image_jpeg = (image_jpeg_01 * 2.0) - 1.0
        return image_jpeg

    def apply_gaussian_blur(self, image):
        kernel_size = random.choice(list(range(self.blur_kernel_min, self.blur_kernel_max + 1, 2))) # Odd kernel sizes
        # Use torchvision functional transform for simplicity here
        # Note: Requires PIL images or tensors depending on version/method
        # Kornia provides tensor-based blurring too: kornia.filters.gaussian_blur2d
        blurred_image = kornia.filters.gaussian_blur2d(image, (kernel_size, kernel_size), sigma=(1.5, 1.5))
        return blurred_image

    def apply_dropout(self, image):
        # Simulate information loss by randomly zeroing out patches or pixels
        # Using spatial dropout
        dropout_layer = nn.Dropout2d(p=self.dropout_prob)
        return dropout_layer(image)


    def forward(self, x):
        if not self.training: # Only apply noise during training
            return x

        batch_size = x.size(0)
        device = x.device
        noisy_images = []

        for i in range(batch_size):
            image = x[i:i+1] # Process one image at a time (keep batch dim)

            # Randomly choose which noise(s) to apply
            choice = random.choice(['identity', 'noise', 'jpeg', 'blur', 'dropout', 'all']) # Can add more combinations

            if choice == 'noise':
                image = self.apply_gaussian_noise(image)
            elif choice == 'jpeg':
                 # Check if kornia is available before trying to use it
                if 'kornia' in globals():
                    image = self.apply_jpeg_compression(image)
                else:
                     print("Warning: Kornia not available. Skipping JPEG compression.")
                     image = self.apply_gaussian_noise(image) # Fallback perhaps
            elif choice == 'blur':
                 image = self.apply_gaussian_blur(image)
            elif choice == 'dropout':
                 image = self.apply_dropout(image)
            elif choice == 'all': # Apply a sequence
                 image = self.apply_gaussian_noise(image)
                 if 'kornia' in globals():
                     image = self.apply_jpeg_compression(image)
                 image = self.apply_gaussian_blur(image)
                 image = self.apply_dropout(image)
            # 'identity' means no noise applied for this image

            # Clamp image values back to the expected range after adding noise/distortions
            # Important if network expects specific range (e.g., [-1, 1] from Tanh)
            image = torch.clamp(image, -1.0, 1.0) # Adjust range if necessary (e.g., [0, 1])

            noisy_images.append(image)

        return torch.cat(noisy_images, dim=0)