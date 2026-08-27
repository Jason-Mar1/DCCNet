"""Small, dependency-light helpers shared by training and inference."""

from collections.abc import Mapping
from pathlib import Path
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class GaussianNoise(nn.Module):
    """Add pixel-wise Gaussian noise to a ``[0, 1]`` image tensor.

    This is deliberately a noise transform, rather than a Gaussian blur: the
    second AIFE view should vary in appearance without changing its geometry.
    """

    def __init__(self, std=0.05, mean=0.0, clamp=True):
        super().__init__()
        if std < 0:
            raise ValueError("Gaussian noise standard deviation must be non-negative.")
        self.std = float(std)
        self.mean = float(mean)
        self.clamp = bool(clamp)

    def forward(self, image):
        if not torch.is_floating_point(image):
            raise TypeError("GaussianNoise expects a floating-point image tensor.")
        noisy = image + torch.randn_like(image) * self.std + self.mean
        return noisy.clamp_(0.0, 1.0) if self.clamp else noisy


def _extract_state_dict(checkpoint):
    """Return a model state dict from raw or trainer-style checkpoints."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a state dict or a mapping containing one; "
            f"received {type(checkpoint).__name__}."
        )

    for key in ("state_dict", "model_state_dict", "model", "net"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break

    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise ValueError("Checkpoint does not contain a valid string-keyed state dict.")

    state_dict = dict(checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_model_checkpoint(model, checkpoint_path, device, allow_partial=False):
    """Load a checkpoint with explicit compatibility diagnostics.

    The historical trainer wrote a raw state dict, while the release trainer
    stores it under ``state_dict``.  Both formats are accepted.  Partial loads
    are rejected by default so a depth/AIFE/AGSA mismatch cannot go unnoticed.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    state_dict = _extract_state_dict(torch.load(path, map_location=device))
    incompatible = model.load_state_dict(state_dict, strict=False)
    report = {
        "path": str(path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if report["missing_keys"] or report["unexpected_keys"]:
        summary = (
            f"Checkpoint compatibility report for {path}: "
            f"missing={report['missing_keys']}, unexpected={report['unexpected_keys']}"
        )
        if not allow_partial:
            raise RuntimeError(
                summary
                + ". Refusing a partial load. Use the matching DCCNet config "
                "or explicitly enable partial loading."
            )
        print("WARNING: " + summary)
    else:
        print(f"Loaded checkpoint with an exact key match: {path}")
    return report

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j

def get_padding(h, w, new_h, new_w):
    if h >= new_h:
        top = 0
        bottom = 0
    else:
        dh = new_h - h
        top = dh // 2
        bottom = dh // 2 + dh % 2
        h = new_h
    if w >= new_w:
        left = 0
        right = 0
    else:
        dw = new_w - w
        left = dw // 2
        right = dw // 2 + dw % 2
        w = new_w

    return (left, top, right, bottom), h, w

def cal_inner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

# 那这里的切patch应该是没有问题
def divide_img_into_patches(img, patch_size):
    h, w = img.shape[-2:]

    img_patches = []
    h_stride = int(np.ceil(1.0 * h / patch_size))
    w_stride = int(np.ceil(1.0 * w / patch_size))
    for i in range(h_stride):
        for j in range(w_stride):
            h_start = i * patch_size
            if i != h_stride - 1:
                h_end = (i + 1) * patch_size
            else:
                h_end = h
            w_start = j * patch_size
            if j != w_stride - 1:
                w_end = (j + 1) * patch_size
            else:
                w_end = w
            img_patches.append(img[..., h_start:h_end, w_start:w_end])

    return img_patches, h_stride, w_stride

# def divide_img_into_patches(img, patch_size):
#     h, w = img.shape[-2:]
#     h_stride = int(np.ceil(h / patch_size))
#     w_stride = int(np.ceil(w / patch_size))
#     for i in range(h_stride):
#         for j in range(w_stride):
#             h_start = i * patch_size
#             h_end = (i + 1) * patch_size if i != h_stride - 1 else h
#             w_start = j * patch_size
#             w_end = (j + 1) * patch_size if j != w_stride - 1 else w
#             yield img[..., h_start:h_end, w_start:w_end]



def denormalize(img_tensor):
    # denormalize a image tensor
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
        img_tensor = img_tensor + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
    elif len(img_tensor.shape) == 4:
        img_tensor = img_tensor * torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img_tensor.device)
        img_tensor = img_tensor + torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img_tensor.device)
    # img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    # img_tensor = img_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    return img_tensor

def denormalize2(img_tensor):
    # denormalize a image tensor
    img_tensor = (img_tensor - img_tensor.min() / (img_tensor.max() - img_tensor.min()))
    return img_tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DictAvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val, n=1):
        for k, v in val.items():
            if k not in self.val:
                self.val[k] = 0
                self.sum[k] = 0
                self.count[k] = 0
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_seeded_generator(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def get_current_datetime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def easy_track(iterable, description=None):
    return tqdm(iterable, desc=description, total=len(iterable), leave=False)
