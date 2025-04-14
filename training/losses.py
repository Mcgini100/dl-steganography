import torch
import torch.nn as nn
import torch.nn.functional as F
# Optional: Import LPIPS if using perceptual loss
try:
    import lpips
except ImportError:
    lpips = None

class SteganographyLoss(nn.Module):
    """Calculates the combined loss for the steganography model."""
    def __init__(self, secret_len=256, beta=0.75, use_lpips=False, device='cpu'):
        """
        Args:
            secret_len (int): Length of the secret message in bits.
            beta (float): Weight factor balancing encoder and decoder losses.
                          L_total = L_encoder + beta * L_decoder
            use_lpips (bool): Whether to use LPIPS for perceptual encoder loss.
            device (str): Device ('cpu' or 'cuda') for LPIPS model.
        """
        super().__init__()
        self.secret_len = secret_len
        self.beta = beta

        # Encoder Loss (Imperceptibility)
        self.encoder_loss_fn = nn.MSELoss()
        self.lpips_loss_fn = None
        if use_lpips:
            if lpips is not None:
                print("Using LPIPS perceptual loss for encoder.")
                self.lpips_loss_fn = lpips.LPIPS(net='alex').to(device) # Or 'vgg'
                # Ensure requires_grad=False for LPIPS model parameters
                for param in self.lpips_loss_fn.parameters():
                    param.requires_grad = False
            else:
                print("Warning: lpips library not found. Falling back to MSE for encoder loss.")

        # Decoder Loss (Message Recovery)
        # Use BCEWithLogitsLoss if the decoder outputs raw logits
        # Use BCELoss if the decoder outputs probabilities (after sigmoid)
        self.decoder_loss_fn = nn.BCELoss() # Assumes decoder outputs sigmoid probabilities

    def calculate_encoder_loss(self, cover_img, stego_img):
        """Calculates the imperceptibility loss."""
        if self.lpips_loss_fn:
            # LPIPS expects images in range [-1, 1]
            cover_norm = cover_img * 2.0 - 1.0
            stego_norm = stego_img * 2.0 - 1.0
            loss_lpips = self.lpips_loss_fn(cover_norm, stego_norm).mean()
            # Combine with MSE? Optional.
            loss_mse = self.encoder_loss_fn(cover_img, stego_img)
            return loss_lpips + loss_mse * 0.1 # Example weighting
        else:
            return self.encoder_loss_fn(cover_img, stego_img)

    def calculate_decoder_loss(self, secret_true, secret_pred_probs):
        """Calculates the message recovery loss."""
        return self.decoder_loss_fn(secret_pred_probs, secret_true)

    def forward(self, cover_img, secret_true, stego_img, secret_pred_probs):
        """
        Calculates the combined loss.

        Args:
            cover_img (torch.Tensor): Original cover images (B, C, H, W), [0, 1].
            secret_true (torch.Tensor): Original secret bits (B, SecretLen), {0., 1.}.
            stego_img (torch.Tensor): Generated stego images (B, C, H, W), [0, 1].
            secret_pred_probs (torch.Tensor): Predicted secret probabilities (B, SecretLen), [0, 1].

        Returns:
            tuple: (total_loss, encoder_loss, decoder_loss)
        """
        encoder_loss = self.calculate_encoder_loss(cover_img, stego_img)
        decoder_loss = self.calculate_decoder_loss(secret_true, secret_pred_probs)
        total_loss = encoder_loss + self.beta * decoder_loss

        return total_loss, encoder_loss, decoder_loss