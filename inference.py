"""Run DCCNet inference on one image or a directory of images."""

import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from PIL import Image

from main import DEFAULT_CONFIG, _validate_runtime_device, build_model, read_config
from utils.misc import denormalize, divide_img_into_patches, get_padding, load_model_checkpoint


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@torch.no_grad()
def predict(model, image, patch_size, log_para):
    """Predict a density map and count, splitting only oversized images."""

    height, width = image.shape[-2:]
    if height >= patch_size or width >= patch_size:
        density_map = torch.zeros((1, 1, height, width), device=image.device, dtype=image.dtype)
        image_patches, rows, columns = divide_img_into_patches(image, patch_size)
        for row in range(rows):
            for column in range(columns):
                patch = image_patches[row * columns + column]
                density_patch = model(patch)[0]
                expected_shape = density_map[:, :, row * patch_size:(row + 1) * patch_size,
                                             column * patch_size:(column + 1) * patch_size].shape
                if density_patch.shape != expected_shape:
                    raise RuntimeError(
                        "DCCNet returned patch shape {} for input patch shape {}; expected {}."
                        .format(tuple(density_patch.shape), tuple(patch.shape), tuple(expected_shape))
                    )
                density_map[:, :, row * patch_size:(row + 1) * patch_size,
                            column * patch_size:(column + 1) * patch_size] = density_patch
    else:
        density_map = model(image)[0]
    return density_map.squeeze().cpu().numpy(), density_map.sum().item() / log_para


def load_images(image_path, unit_size, device):
    image_path = Path(image_path).expanduser()
    if image_path.is_dir():
        paths = sorted(path for path in image_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise FileNotFoundError("No .jpg, .jpeg, or .png images found in {}.".format(image_path))
    elif image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
        paths = [image_path]
    elif not image_path.exists():
        raise FileNotFoundError("Image path does not exist: {}.".format(image_path))
    else:
        raise ValueError("Only .jpg, .jpeg, and .png inputs are supported: {}.".format(image_path))

    images = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if unit_size > 0:
            width, height = image.size
            padded_width = ((width + unit_size - 1) // unit_size) * unit_size
            padded_height = ((height + unit_size - 1) // unit_size) * unit_size
            padding, _, _ = get_padding(height, width, padded_height, padded_width)
            image = F.pad(image, padding)
        image = F.normalize(F.to_tensor(image), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        images.append(image.unsqueeze(0).to(device))
    return images, [path.name for path in paths]


def load_model(model_path, device, config_path=DEFAULT_CONFIG, depth_checkpoint=None):
    """Build a full DCCNet (including DepthAnything) and load exact weights."""

    config, _ = read_config(config_path)
    # Checkpoint loading overwrites the VGG encoder, so do not request an
    # unnecessary ImageNet VGG download during inference.
    model = build_model(
        config,
        depth_checkpoint=depth_checkpoint,
        pretrained_backbone=False,
    ).to(device)
    report = load_model_checkpoint(model, model_path, device, allow_partial=False)
    print(
        "Checkpoint report: missing_keys={}, unexpected_keys={}".format(
            report["missing_keys"], report["unexpected_keys"]
        )
    )
    model.eval()
    return model, config


def run(args):
    config, _ = read_config(args.config)
    device = args.device or config["device"]
    _validate_runtime_device(device)
    model, config = load_model(
        args.model_path,
        device,
        config_path=args.config,
        depth_checkpoint=args.depth_checkpoint,
    )
    patch_size = args.patch_size or config["patch_size"]
    log_para = args.log_para or config["log_para"]
    images, image_names = load_images(args.img_path, args.unit_size, device)

    output_path = Path(args.save_path).expanduser() if args.save_path else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.vis_dir:
        Path(args.vis_dir).expanduser().mkdir(parents=True, exist_ok=True)

    start_time = time()
    for image, image_name in zip(images, image_names):
        density_map, count = predict(model, image, patch_size, log_para)
        print("{}: {:.4f}".format(image_name, count))

        if output_path:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write("{}: {:.4f}\n".format(image_name, count))
        if args.vis_dir:
            denormalized = denormalize(image)[0].cpu().permute(1, 2, 0).numpy()
            figure, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(denormalized)
            axes[0].set_title(image_name)
            axes[0].axis("off")
            axes[1].imshow(density_map)
            axes[1].set_title("Predicted count: {:.4f}".format(count))
            axes[1].axis("off")
            figure.tight_layout()
            figure.savefig(Path(args.vis_dir).expanduser() / (Path(image_name).stem + ".png"))
            plt.close(figure)
    print("Total time: {:.2f}s".format(time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(description="DCCNet single-image/directory inference")
    parser.add_argument("--img-path", required=True, help="Input image or directory.")
    parser.add_argument("--model-path", required=True, help="DCCNet checkpoint (raw state_dict or trainer checkpoint).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="DCCNet release config used to build the model.")
    parser.add_argument("--depth-checkpoint", default=None, help="Optional local DepthAnything pytorch_model.bin.")
    parser.add_argument("--save-path", default=None, help="Optional text file for predicted counts.")
    parser.add_argument("--vis-dir", default=None, help="Optional density-map visualization directory.")
    parser.add_argument("--unit-size", type=int, default=16, help="Pad image dimensions to this multiple.")
    parser.add_argument("--patch-size", type=int, default=None, help="Override the config patch size.")
    parser.add_argument("--log-para", type=int, default=None, help="Override the config density-map scale factor.")
    parser.add_argument("--device", default=None, help="Override config device (default config is cuda:0).")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit("DCCNet inference error: {}".format(exc)) from exc
