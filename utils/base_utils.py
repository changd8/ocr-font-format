from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

BICUBIC = Image.BICUBIC


def set_seed(args):
    """ Set seed for experiment."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def to_device(inputs: Union[List, Dict], device):
    """ Attach each input to specified device"""
    if isinstance(inputs, list):
        return [to_device(d, device) for d in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        return inputs.to(device)


def resize_transform(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=BICUBIC),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

