"""Training, evaluation, and visualization entry point for DCCNet.

The release configs mirror the paper's 200-epoch Adam protocol.  DepthAnything
weights are explicit and pinned: the helper downloads them only when an actual
training/evaluation run constructs the model, never while parsing a config.
"""

import argparse
from collections.abc import Mapping
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import yaml
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from datasets.den_cls_dataset import DenClsDataset
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset
from depth_anything.dpt import DepthAnything
from models.models import DCCNet
from trainers.dgtrainer import DGTrainer
from utils.misc import get_seeded_generator, seed_everything, seed_worker


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "shanghaitech_a_to_b.yaml"
REQUIRED_CONFIG_KEYS = (
    "seed",
    "version",
    "device",
    "log_para",
    "patch_size",
    "mode",
    "num_epochs",
    "checkpoint",
    "resume_checkpoint",
    "foreground_threshold",
    "loss_weights",
    "depth_anything",
    "model",
    "train_dataset",
    "val_dataset",
    "test_dataset",
    "train_loader",
    "val_loader",
    "test_loader",
    "optimizer",
    "scheduler",
)


class ConfigError(ValueError):
    """Raised when a release YAML is incomplete or internally inconsistent."""


def _as_mapping(value, path):
    if not isinstance(value, Mapping):
        raise ConfigError("{} must be a mapping, got {}.".format(path, type(value).__name__))
    return value


def _config_path(path):
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.is_file():
        return project_candidate.resolve()
    raise FileNotFoundError(
        "Configuration file not found: {}. Use --config <path>; the release default is {}."
        .format(candidate, DEFAULT_CONFIG)
    )


def validate_config(config, source="configuration"):
    """Validate the schema before importing data or downloading model weights."""

    config = _as_mapping(config, source)
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ConfigError("{} is missing required keys: {}.".format(source, ", ".join(missing)))

    if config["mode"] != "final":
        raise ConfigError("{} must set mode: final for the published DCCNet model.".format(source))
    if not isinstance(config["num_epochs"], int) or config["num_epochs"] <= 0:
        raise ConfigError("{}.num_epochs must be a positive integer.".format(source))
    if not isinstance(config["device"], str) or not config["device"]:
        raise ConfigError("{}.device must be a non-empty device string.".format(source))

    model = _as_mapping(config["model"], source + ".model")
    params = _as_mapping(model.get("params"), source + ".model.params")
    if str(model.get("name", "")).lower() not in {"dccnet", "gpd", "final", "dgmodel_final"}:
        raise ConfigError("{}.model.name must identify DCCNet.".format(source))
    if "cls_thrs" not in params:
        raise ConfigError("{}.model.params.cls_thrs is required.".format(source))
    if float(params["cls_thrs"]) != float(config["foreground_threshold"]):
        raise ConfigError(
            "{}.foreground_threshold and model.params.cls_thrs must match.".format(source)
        )

    loss_weights = _as_mapping(config["loss_weights"], source + ".loss_weights")
    for name in ("lambda_aux", "lambda_con"):
        if name not in loss_weights or float(loss_weights[name]) < 0:
            raise ConfigError("{}.loss_weights.{} must be a non-negative number.".format(source, name))

    depth = _as_mapping(config["depth_anything"], source + ".depth_anything")
    for name in ("repo_id", "revision", "encoder", "features", "out_channels"):
        if name not in depth:
            raise ConfigError("{}.depth_anything.{} is required.".format(source, name))
    if depth["encoder"] != "vits":
        raise ConfigError("{}.depth_anything.encoder must be vits for the released vits14 prior.".format(source))

    optimizer = _as_mapping(config["optimizer"], source + ".optimizer")
    if str(optimizer.get("name", "")).lower() != "adam":
        raise ConfigError("{} uses the paper's Adam optimizer; set optimizer.name: adam.".format(source))
    optimizer_params = _as_mapping(optimizer.get("params"), source + ".optimizer.params")
    if "lr" not in optimizer_params or float(optimizer_params["lr"]) <= 0:
        raise ConfigError("{}.optimizer.params.lr must be positive.".format(source))

    for name in ("train_dataset", "val_dataset", "test_dataset"):
        dataset = _as_mapping(config[name], source + "." + name)
        dataset_params = _as_mapping(dataset.get("params"), source + "." + name + ".params")
        if dataset.get("name") not in {"den_cls", "jhu_domain_cls"}:
            raise ConfigError("{}.{}.name must be den_cls or jhu_domain_cls.".format(source, name))
        if name in {"train_dataset", "val_dataset"}:
            if "foreground_threshold" not in dataset_params:
                raise ConfigError("{}.{}.params.foreground_threshold is required.".format(source, name))
            if float(dataset_params["foreground_threshold"]) != float(config["foreground_threshold"]):
                raise ConfigError(
                    "{}.{}.params.foreground_threshold must match foreground_threshold."
                    .format(source, name)
                )
    for name in ("train_loader", "val_loader", "test_loader"):
        _as_mapping(config[name], source + "." + name)
    if int(config["train_loader"].get("batch_size", 0)) <= 0:
        raise ConfigError("{}.train_loader.batch_size must be positive.".format(source))
    return config


def read_config(path=DEFAULT_CONFIG):
    """Read a release YAML without side effects such as model downloads."""

    config_path = _config_path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError("Invalid YAML in {}: {}".format(config_path, exc)) from exc
    if config is None:
        raise ConfigError("Configuration file is empty: {}".format(config_path))
    return validate_config(config, str(config_path)), config_path


def _resolve_local_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_file():
        return path.resolve()
    project_path = PROJECT_ROOT / path
    if project_path.is_file():
        return project_path.resolve()
    raise FileNotFoundError("Local DepthAnything checkpoint does not exist: {}".format(path_value))


def build_depth_encoder(depth_config, checkpoint_override=None):
    """Build the pinned, frozen DepthAnything vits14 geometry prior."""

    depth_config = _as_mapping(depth_config, "depth_anything")
    local_checkpoint = checkpoint_override or depth_config.get("depth_checkpoint")
    if local_checkpoint:
        checkpoint_path = _resolve_local_path(local_checkpoint)
    else:
        try:
            checkpoint_path = Path(
                hf_hub_download(
                    repo_id=depth_config["repo_id"],
                    filename="pytorch_model.bin",
                    revision=depth_config["revision"],
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not obtain the pinned DepthAnything checkpoint. Set "
                "depth_anything.depth_checkpoint (or --depth-checkpoint) to a local file, "
                "or check Hugging Face network access."
            ) from exc

    architecture = {
        "encoder": depth_config["encoder"],
        "features": int(depth_config["features"]),
        "out_channels": tuple(depth_config["out_channels"]),
    }
    depth_encoder = DepthAnything.from_pretrained(
        checkpoint_path=str(checkpoint_path),
        config=architecture,
    )
    depth_encoder.eval()
    for parameter in depth_encoder.parameters():
        parameter.requires_grad_(False)
    return depth_encoder


def build_model(config, depth_checkpoint=None, pretrained_backbone=None):
    """Construct DCCNet with its frozen DepthAnything prior."""

    model_config = _as_mapping(config["model"], "model")
    params = dict(_as_mapping(model_config["params"], "model.params"))
    if pretrained_backbone is not None:
        params["pretrained"] = bool(pretrained_backbone)
    params["depth_anything"] = build_depth_encoder(config["depth_anything"], depth_checkpoint)
    return DCCNet(**params)


def get_dataset(name, params, method):
    datasets = {
        "den_cls": DenClsDataset,
        "jhu_domain_cls": JHUDomainClsDataset,
    }
    try:
        dataset_class = datasets[name]
    except KeyError as exc:
        raise ConfigError("Unknown release dataset: {}.".format(name)) from exc
    dataset = dataset_class(method=method, **dict(params))
    return dataset, dataset_class.collate


def get_optimizer(config, model):
    optimizer = _as_mapping(config, "optimizer")
    if str(optimizer["name"]).lower() != "adam":
        raise ConfigError("Only Adam is supported by the published release config.")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("DCCNet has no trainable parameters after freezing the depth prior.")
    return torch.optim.Adam(trainable_parameters, **dict(optimizer["params"]))


def get_scheduler(config, optimizer):
    if config is None:
        return None
    scheduler = _as_mapping(config, "scheduler")
    name = str(scheduler.get("name", "")).lower()
    params = dict(_as_mapping(scheduler.get("params"), "scheduler.params"))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    raise ConfigError("Unknown scheduler: {}.".format(name))


def _checkpoint_for_task(config, task, checkpoint_override):
    if checkpoint_override:
        return checkpoint_override
    if task == "train":
        return config["resume_checkpoint"]
    return config["checkpoint"]


def build_task(config, task, checkpoint_override=None, depth_checkpoint=None):
    """Instantiate all runtime objects after a config has passed validation."""

    seed_everything(config["seed"])
    # ImageNet VGG initialization is useful only when starting training.  An
    # evaluation checkpoint overwrites the backbone, so test/vis must not make
    # a redundant network request on a cold cache.
    pretrained_backbone = None if task == "train" else False
    model = build_model(
        config,
        depth_checkpoint=depth_checkpoint,
        pretrained_backbone=pretrained_backbone,
    )
    generator = get_seeded_generator(config["seed"])
    init_params = {
        "seed": config["seed"],
        "version": config["version"],
        "device": config["device"],
        "log_para": config["log_para"],
        "patch_size": config["patch_size"],
        "mode": config["mode"],
        "lambda_aux": config["loss_weights"]["lambda_aux"],
        "lambda_con": config["loss_weights"]["lambda_con"],
    }
    task_params = {"model": model, "checkpoint": _checkpoint_for_task(config, task, checkpoint_override)}

    if task == "train":
        task_params["loss"] = nn.MSELoss()
        train_dataset, collate = get_dataset(
            config["train_dataset"]["name"], config["train_dataset"]["params"], method="train"
        )
        task_params["train_dataloader"] = DataLoader(
            train_dataset,
            collate_fn=collate,
            worker_init_fn=seed_worker,
            generator=generator,
            **dict(config["train_loader"])
        )
        val_dataset, _ = get_dataset(config["val_dataset"]["name"], config["val_dataset"]["params"], method="val")
        task_params["val_dataloader"] = DataLoader(val_dataset, **dict(config["val_loader"]))
        task_params["optimizer"] = get_optimizer(config["optimizer"], model)
        task_params["scheduler"] = get_scheduler(config["scheduler"], task_params["optimizer"])
        task_params["num_epochs"] = config["num_epochs"]
    else:
        test_dataset, _ = get_dataset(config["test_dataset"]["name"], config["test_dataset"]["params"], method="test")
        task_params["test_dataloader"] = DataLoader(test_dataset, **dict(config["test_loader"]))
    return init_params, task_params


def _validate_runtime_device(device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "The config requests {} but CUDA is unavailable. Use --config with device: cpu for a CPU smoke test."
            .format(device)
        )


def parse_args():
    parser = argparse.ArgumentParser(description="DCCNet training and evaluation")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a release YAML configuration.")
    parser.add_argument("--task", default="train", choices=("train", "test", "vis"), help="Action to run.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the evaluation checkpoint (or the resume checkpoint for --task train).",
    )
    parser.add_argument(
        "--depth-checkpoint",
        default=None,
        help="Use a local pinned DepthAnything pytorch_model.bin instead of downloading it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config, config_path = read_config(args.config)
        _validate_runtime_device(config["device"])
        init_params, task_params = build_task(
            config,
            args.task,
            checkpoint_override=args.checkpoint,
            depth_checkpoint=args.depth_checkpoint,
        )
    except (ConfigError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit("DCCNet setup error: {}".format(exc)) from exc

    trainer = DGTrainer(**init_params)
    # copy2 preserves provenance metadata and is portable across POSIX/Windows.
    shutil.copy2(config_path, Path(trainer.log_dir) / "config.yaml")
    if args.task == "train":
        trainer.train(**task_params)
    elif args.task == "test":
        trainer.test(**task_params)
    else:
        trainer.vis(**task_params)


if __name__ == "__main__":
    main()
