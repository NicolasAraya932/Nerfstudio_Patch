"""Serialize dataparser outputs as a frozen dataset coordinate contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class DataparserContractConfig:
    """Small, tyro-friendly config for saving a Nerfstudio dataparser contract."""

    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """Evaluation split rule used by the Nerfstudio dataparser."""
    train_split_fraction: float = 0.9
    """Fraction of images used for training when eval_mode is fraction."""
    eval_interval: int = 8
    """Evaluation interval when eval_mode is interval."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """Pose orientation method used by the Nerfstudio dataparser."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """Pose centering method used by the Nerfstudio dataparser."""
    auto_scale_poses: bool = True
    """Whether the dataparser auto-scales poses into the normalized scene box."""
    scene_scale: float = 1.0
    """Scene-box scale used by the Nerfstudio dataparser."""
    scale_factor: float = 1.0
    """Additional scale factor used by the Nerfstudio dataparser."""
    downscale_factor: Optional[int] = None
    """Image downscale factor to force. If unset, the dataparser auto-selects it."""


def _jsonable(value: Any) -> Any:
    """Convert common Nerfstudio/PyTorch values into JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


def _relative_paths(paths: Optional[list[Optional[Path]]], root: Path) -> Optional[list[Optional[str]]]:
    if paths is None:
        return None
    result: list[Optional[str]] = []
    for path in paths:
        if path is None:
            result.append(None)
            continue
        try:
            result.append(path.relative_to(root).as_posix())
        except ValueError:
            result.append(path.as_posix())
    return result


def _metadata_payload(outputs: DataparserOutputs, dataset_dir: Path) -> Dict[str, Any]:
    """Serialize metadata while keeping auxiliary image modality paths dataset-relative."""
    metadata = dict(outputs.metadata)
    image_modalities = metadata.get("image_modalities")
    if isinstance(image_modalities, dict):
        metadata["image_modalities"] = {
            str(key): _relative_paths(paths, dataset_dir)
            for key, paths in image_modalities.items()
        }
    return _jsonable(metadata)


def _camera_payload(outputs: DataparserOutputs) -> Dict[str, Any]:
    cameras = outputs.cameras
    return {
        "camera_to_worlds": _jsonable(cameras.camera_to_worlds),
        "fx": _jsonable(cameras.fx),
        "fy": _jsonable(cameras.fy),
        "cx": _jsonable(cameras.cx),
        "cy": _jsonable(cameras.cy),
        "height": _jsonable(cameras.height),
        "width": _jsonable(cameras.width),
        "distortion_params": _jsonable(cameras.distortion_params),
        "camera_type": _jsonable(cameras.camera_type),
        "times": _jsonable(cameras.times),
        "metadata": _jsonable(cameras.metadata),
    }


def _split_payload(outputs: DataparserOutputs, dataset_dir: Path) -> Dict[str, Any]:
    return {
        "image_filenames": _relative_paths(outputs.image_filenames, dataset_dir),
        "mask_filenames": _relative_paths(outputs.mask_filenames, dataset_dir),
        "metadata": _metadata_payload(outputs, dataset_dir),
        "cameras": _camera_payload(outputs),
    }


def _build_nerfstudio_dataparser_config(dataset_dir: Path, contract: DataparserContractConfig) -> NerfstudioDataParserConfig:
    return NerfstudioDataParserConfig(
        data=dataset_dir,
        scale_factor=contract.scale_factor,
        downscale_factor=contract.downscale_factor,
        scene_scale=contract.scene_scale,
        orientation_method=contract.orientation_method,
        center_method=contract.center_method,
        auto_scale_poses=contract.auto_scale_poses,
        eval_mode=contract.eval_mode,
        train_split_fraction=contract.train_split_fraction,
        eval_interval=contract.eval_interval,
        load_frozen_contract=False,
    )


def serialize_dataparser_contract(
    dataset_dir: Path,
    contract: Optional[DataparserContractConfig] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Run the Nerfstudio dataparser and serialize train/test outputs.

    Args:
        dataset_dir: Processed Nerfstudio dataset directory containing transforms.json.
        contract: Dataparser contract options used to instantiate NerfstudioDataParserConfig.
        output_dir: Optional output directory. Defaults to dataset_dir / metadata / dataparser.

    Returns:
        Path to the written contract JSON.
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    if not (dataset_dir / "transforms.json").exists() and dataset_dir.suffix != ".json":
        raise FileNotFoundError(f"Could not find transforms.json in {dataset_dir}")

    contract = contract or DataparserContractConfig()
    output_dir = (output_dir or dataset_dir / "metadata" / "dataparser").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_nerfstudio_dataparser_config(dataset_dir, contract)
    dataparser = config.setup()
    train_outputs = dataparser.get_dataparser_outputs(split="train")
    test_outputs = dataparser.get_dataparser_outputs(split="test")

    payload = {
        "schema_version": 1,
        "dataparser_class": type(dataparser).__name__,
        "dataparser_config": _jsonable(asdict(config)),
        "contract_config": _jsonable(asdict(contract)),
        "dataset_dir": dataset_dir.as_posix(),
        "shared": {
            "dataparser_transform": _jsonable(train_outputs.dataparser_transform),
            "dataparser_scale": float(train_outputs.dataparser_scale),
            "scene_box": {"aabb": _jsonable(train_outputs.scene_box.aabb)},
        },
        "splits": {
            "train": _split_payload(train_outputs, dataset_dir),
            "test": _split_payload(test_outputs, dataset_dir),
        },
    }

    contract_path = output_dir / "contract.json"
    contract_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "scene_box.json").write_text(json.dumps(payload["shared"]["scene_box"], indent=2), encoding="utf-8")
    (output_dir / "splits.json").write_text(
        json.dumps(
            {
                "train": payload["splits"]["train"]["image_filenames"],
                "test": payload["splits"]["test"]["image_filenames"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    CONSOLE.log(f"[bold green]Saved dataparser contract to {contract_path}")
    return contract_path
