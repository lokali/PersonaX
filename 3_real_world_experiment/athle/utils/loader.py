import os
import numpy as np
import torch
import torch
import torch.nn.functional as F
from PIL import Image


#################################
# MNIST Data Processing Utilities
#################################
def load_image(data_source, image_path):
    """ Load an image and convert it to a NumPy array. """
    return np.array(Image.open(os.path.join(data_source, image_path)))

def downsample(data, is_rgb):
    """
    Downsample image data to 14x14.
    
    Args:
        data (np.array): Input image array.
        is_rgb (bool): Whether the image is RGB (True) or Grayscale (False).
    
    Returns:
        np.array: Downsampled image array.
    """
    tensor = torch.tensor(data).float()
    if is_rgb:
        tensor = tensor.permute(0, 3, 1, 2)  # Convert (BS, H, W, C) -> (BS, C, H, W)
    else:
        tensor = tensor.unsqueeze(1)  # Convert (BS, H, W) -> (BS, 1, H, W)

    # tensor = F.interpolate(tensor, size=(150, 150), mode='bilinear', align_corners=False)

    if is_rgb:
        return tensor.permute(0, 2, 3, 1).numpy()  # Convert back to (BS, 14, 14, C)
    else:
        return tensor.squeeze(1).numpy()  # Convert back to (BS, 14, 14)

def flatten_data(data_dict):
    """
    Flatten image tensors and tabular data.
    
    Args:
        data_dict (dict): Dictionary containing training and validation data.
    
    Returns:
        dict: Updated dictionary with flattened tensors.
    """
    for key in data_dict:
        if len(data_dict[key].shape) == 4:  # RGB (BS, H, W, C)
            data_dict[key] = data_dict[key].reshape(data_dict[key].shape[0], -1)
        elif len(data_dict[key].shape) == 3:  # Grayscale (BS, H, W)
            data_dict[key] = data_dict[key].reshape(data_dict[key].shape[0], -1)
    return data_dict

def stack_data(train_val_split, split_type, case_id):
    """
    Stack the features for training and validation sets.
    
    Args:
        split_type (str): "train" or "val".
    
    Returns:
        np.array: Combined feature matrix.
    """
    if case_id == "athlete_1m":
        X_flat = [train_val_split[f"X1_{split_type}"], train_val_split[f"X21_{split_type}"], train_val_split[f"X22_{split_type}"], train_val_split[f"X23_{split_type}"]]
    elif case_id == "athlete_3m":
        X_flat = [train_val_split[f"X1_{split_type}"], train_val_split[f"X21_{split_type}"], train_val_split[f"X22_{split_type}"], train_val_split[f"X23_{split_type}"]]

    z_layers = [train_val_split[f"Z_all_{split_type}"]]
    return np.hstack([*X_flat, *z_layers])