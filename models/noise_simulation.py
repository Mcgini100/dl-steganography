import torch
import torch.nn as nn
import random
import torchvision.transforms.functional as TF

try:
    import kornia
    # Kornia specific imports if needed later
    # from kornia.augmentation import RandomJPEG
    # from kornia.filters import gaussian_blur2d
    KORNIA_AVAILABLE = True
except ImportError:
    print("Warning: Kornia library not found. JPEG compression and advanced blurring will be unavailable.")
    KORNIA_AVAILABLE = False


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

        # --- Initialize Augmentations Here ---
        self.jpeg_augment = None
        if KORNIA_AVAILABLE:
            try:
                # Ensure quality values are floats for Kornia API consistency
                q_min = float(self.jpeg_quality_min)
                q_max = float(self.jpeg_quality_max)
                # Check if min > max, swap if necessary (robustness)
                if q_min > q_max:
                    q_min, q_max = q_max, q_min
                # Ensure quality is within valid JPEG range [0, 100] - kornia might handle this, but safe to check
                q_min = max(0.0, q_min)
                q_max = min(100.0, q_max)

                # Use the quality parameter directly if available and correct
                self.jpeg_augment = kornia.augmentation.RandomJPEG(
                    quality=(q_min, q_max), # Pass the range tuple
                    p=1.0 # We control application probability externally in forward
                )
                print(f"Initialized Kornia RandomJPEG with quality range: ({q_min}, {q_max})")
            except TypeError as e:
                 print(f"Warning: Could not initialize kornia.augmentation.RandomJPEG with 'quality' parameter: {e}")
                 print("This might indicate an older Kornia version or API change.")
                 print("JPEG compression noise will be skipped.")
                 self.jpeg_augment = None # Fallback: disable JPEG if init fails
            except Exception as e: # Catch other potential kornia errors
                print(f"Warning: Unexpected error initializing Kornia RandomJPEG: {e}")
                self.jpeg_augment = None


        # Initialize dropout layer (stateless, can be defined here)
        self.dropout_layer = nn.Dropout2d(p=self.dropout_prob)

    def apply_gaussian_noise(self, image):
        noise = torch.randn_like(image) * self.noise_level
        return image + noise

    def apply_jpeg_compression(self, image):
        if not KORNIA_AVAILABLE or self.jpeg_augment is None:
            # print("Skipping JPEG: Kornia unavailable or augmentation failed to initialize.")
            return image # Skip if Kornia not installed or init failed

        # Kornia JPEG works on [0, 1] range, BxCxHxW float tensor
        # Assuming input `image` is in [-1, 1] range from Encoder (Tanh output)
        image_01 = (image + 1.0) / 2.0 # Rescale to [0, 1]

        # --- Call the initialized augmentation instance ---
        image_jpeg_01 = self.jpeg_augment(image_01)

        # Rescale back to [-1, 1]
        image_jpeg = (image_jpeg_01 * 2.0) - 1.0
        return image_jpeg

    def apply_gaussian_blur(self, image):
        if not KORNIA_AVAILABLE:
            # print("Skipping Blur: Kornia unavailable.")
            return image # Skip if Kornia not installed

        kernel_size = random.choice(list(range(self.blur_kernel_min, self.blur_kernel_max + 1, 2))) # Odd kernel sizes
        # Ensure sigma is appropriate for the kernel size (e.g., kornia default or a calculated value)
        # Using fixed sigma for simplicity here, adjust if needed
        sigma = (1.5, 1.5)
        blurred_image = kornia.filters.gaussian_blur2d(image, (kernel_size, kernel_size), sigma=sigma)
        return blurred_image

    def apply_dropout(self, image):
        # Simulate information loss by randomly zeroing out patches or pixels
        return self.dropout_layer(image) # Use the initialized layer


    def forward(self, x):
        if not self.training: # Only apply noise during training
            return x

        batch_size = x.size(0)
        device = x.device
        noisy_images = []

        # Apply noise simulation image by image (can be slow, consider batch ops if performance critical)
        # This loop allows different random noises per image in the batch
        for i in range(batch_size):
            image = x[i:i+1] # Process one image at a time (keep batch dim)

            # Randomly choose which noise(s) to apply
            choice = random.choice(['identity', 'noise', 'jpeg', 'blur', 'dropout', 'all']) # Can add more combinations

            original_image_for_all = image # Keep original if 'all' is chosen

            if choice == 'noise':
                image = self.apply_gaussian_noise(image)
            elif choice == 'jpeg':
                 image = self.apply_jpeg_compression(image)
            elif choice == 'blur':
                 image = self.apply_gaussian_blur(image)
            elif choice == 'dropout':
                 image = self.apply_dropout(image)
            elif choice == 'all': # Apply a sequence
                 image = self.apply_gaussian_noise(image)
                 image = self.apply_jpeg_compression(image) # Apply even if Kornia not avail (will be skipped internally)
                 image = self.apply_gaussian_blur(image)    # Apply even if Kornia not avail (will be skipped internally)
                 image = self.apply_dropout(image)
            # 'identity' means no noise applied for this image

            # Clamp image values back to the expected range after adding noise/distortions
            image = torch.clamp(image, -1.0, 1.0) # Adjust range if necessary (e.g., [0, 1])

            noisy_images.append(image)

        return torch.cat(noisy_images, dim=0)