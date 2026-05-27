"""Shared reader for frozen Nerfstudio dataparser contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

SUPPORTED_SCHEMA_VERSION = 1


def get_dataparser_contract_path(data: Path) -> Optional[Path]:
    """Return the frozen dataparser contract path for a dataset when it exists."""
    dataset_dir = data.parent if data.suffix == ".json" else data
    contract_path = dataset_dir / "metadata" / "dataparser" / "contract.json"
    return contract_path if contract_path.exists() else None


def _resolve_contract_paths(paths: Optional[list[Optional[str]]], dataset_dir: Path) -> Optional[list[Optional[Path]]]:
    """Resolve dataset-relative contract paths while preserving missing optional paths."""
    if paths is None:
        return None
    restored: list[Optional[Path]] = []
    for path in paths:
        if path is None:
            restored.append(None)
            continue
        candidate = Path(path)
        restored.append(candidate if candidate.is_absolute() else dataset_dir / candidate)
    return restored


def restore_contract_metadata(metadata: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    """Restore JSON metadata values that should be Path objects after deserialization."""
    restored = dict(metadata)

    depth_filenames = restored.get("depth_filenames")
    if depth_filenames is not None:
        restored["depth_filenames"] = _resolve_contract_paths(depth_filenames, dataset_dir)

    image_modalities = restored.get("image_modalities")
    if isinstance(image_modalities, dict):
        restored["image_modalities"] = {
            str(key): _resolve_contract_paths(paths, dataset_dir) for key, paths in image_modalities.items()
        }

    return restored


def _tensor_from_contract(value: Any, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Convert a serialized contract value into a tensor."""
    if value is None:
        return None
    return torch.tensor(value, dtype=dtype)


def _validate_payload(payload: dict[str, Any], contract_path: Path) -> None:
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported dataparser contract schema {schema_version} in {contract_path}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}."
        )
    for key in ("dataset_dir", "shared", "splits"):
        if key not in payload:
            raise ValueError(f"Malformed dataparser contract {contract_path}: missing '{key}'.")


def _split_key(split: str) -> str:
    return "test" if split == "val" else split


def load_dataparser_outputs_from_contract(contract_path: Path, split: str) -> DataparserOutputs:
    """Deserialize a frozen dataparser contract split into DataparserOutputs."""
    contract_path = contract_path.expanduser().resolve()
    payload = load_from_json(contract_path)
    _validate_payload(payload, contract_path)

    split_key = _split_key(split)
    splits_payload = payload["splits"]
    if split_key not in splits_payload:
        available = ", ".join(sorted(splits_payload))
        raise ValueError(f"Unknown dataparser contract split '{split}' in {contract_path}; available: {available}.")

    dataset_dir = Path(payload["dataset_dir"])
    split_payload = splits_payload[split_key]
    shared_payload = payload["shared"]
    cameras_payload = split_payload["cameras"]

    cameras = Cameras(
        fx=_tensor_from_contract(cameras_payload["fx"], torch.float32),
        fy=_tensor_from_contract(cameras_payload["fy"], torch.float32),
        cx=_tensor_from_contract(cameras_payload["cx"], torch.float32),
        cy=_tensor_from_contract(cameras_payload["cy"], torch.float32),
        distortion_params=_tensor_from_contract(cameras_payload["distortion_params"], torch.float32),
        height=_tensor_from_contract(cameras_payload["height"], torch.int32),
        width=_tensor_from_contract(cameras_payload["width"], torch.int32),
        camera_to_worlds=_tensor_from_contract(cameras_payload["camera_to_worlds"], torch.float32),
        camera_type=_tensor_from_contract(cameras_payload["camera_type"], torch.int64),
        times=_tensor_from_contract(cameras_payload.get("times"), torch.float32),
        metadata=cameras_payload.get("metadata", {}),
    )

    image_filenames = _resolve_contract_paths(split_payload.get("image_filenames"), dataset_dir)
    if image_filenames is None:
        raise ValueError(f"Malformed dataparser contract {contract_path}: split '{split_key}' has no image_filenames.")

    outputs = DataparserOutputs(
        image_filenames=image_filenames,
        cameras=cameras,
        scene_box=SceneBox(aabb=torch.tensor(shared_payload["scene_box"]["aabb"], dtype=torch.float32)),
        mask_filenames=_resolve_contract_paths(split_payload.get("mask_filenames"), dataset_dir),
        dataparser_scale=float(shared_payload["dataparser_scale"]),
        dataparser_transform=torch.tensor(shared_payload["dataparser_transform"], dtype=torch.float32),
        metadata=restore_contract_metadata(split_payload.get("metadata", {}), dataset_dir),
    )
    CONSOLE.log(f"[green]Loaded frozen dataparser contract from {contract_path} ({split_key} split)")
    return outputs
