import numpy as np


def numpy_to_torch(img):
    """
    Convert numpy array to torch tensor
    Args:
        img: numpy array
    """
    import torch

    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_numpy(img):
    """
    Convert torch tensor to numpy array
    Args:
        img: torch tensor
    """

    img = img.cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = img.squeeze()
    return img


def create_dir(_dir):
    """
    Create directory if it doesn't exist
    Args:
        _dir: str
    """
    import os

    if not os.path.exists(_dir):
        os.makedirs(_dir)
