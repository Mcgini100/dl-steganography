import torch
import torch.nn as nn

def calculate_image_loss(embedded_image, cover_image, loss_type='mse'):
    """ Calculates the loss between the cover image and the embedded image. """
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss()
    else:
        raise ValueError(f"Unknown image loss type: {loss_type}")
    return loss_fn(embedded_image, cover_image)

def calculate_message_loss(decoded_message_logits, original_message, loss_type='bce'):
    """ Calculates the loss between the original message and the decoded message logits. """
    if loss_type == 'bce':
        # Assumes original_message contains 0s and 1s
        # Assumes decoded_message_logits are raw scores (before sigmoid)
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_type == 'mse':
         # Assumes original_message contains -1s and 1s, and decoded_message has Tanh activation
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown message loss type: {loss_type}")
    return loss_fn(decoded_message_logits, original_message)

def calculate_combined_loss(image_loss, message_loss, image_weight, message_weight):
    """ Calculates the weighted combined loss. """
    return (image_weight * image_loss) + (message_weight * message_loss)

def calculate_message_accuracy(decoded_message_logits, original_message, threshold=0.0):
    """ Calculates the accuracy of decoded bits (using logits). """
    # Apply threshold (0.0 for logits from BCEWithLogitsLoss) to get predicted bits
    predicted_bits = (decoded_message_logits > threshold).float()
    # Compare with original message (assuming 0/1)
    correct_bits = (predicted_bits == original_message).sum().item()
    total_bits = original_message.numel() # Total number of bits in the batch
    accuracy = correct_bits / total_bits
    return accuracy

def calculate_bit_error_rate(decoded_message_logits, original_message, threshold=0.0):
    """ Calculates the Bit Error Rate (BER). """
    predicted_bits = (decoded_message_logits > threshold).float()
    errors = (predicted_bits != original_message).sum().item()
    total_bits = original_message.numel()
    ber = errors / total_bits if total_bits > 0 else 0.0
    return ber