#!/usr/bin/env python3
"""Train Nerfacto and InvNeRF with the Nerfstudio_Patch frozen dataparser contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _contract_path(data: Path) -> Path:
    return data / "metadata" / "dataparser" / "contract.json"


def _read_contract(data: Path) -> dict:
    path = _contract_path(data)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen dataparser contract: {path}\n"
            "Create it with nerfstudio.process_data.dataparser_contract.serialize_dataparser_contract first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _require_contract_valid(data: Path) -> tuple[int | None, int, int]:
    contract = _read_contract(data)
    downscale = contract.get("contract_config", {}).get("downscale_factor")
    splits = contract.get("splits", {})
    train = splits.get("train", {})
    test = splits.get("test", {})
    train_images = train.get("image_filenames", [])
    test_images = test.get("image_filenames", [])
    for split_name, split in (("train", train), ("test", test)):
        modalities = split.get("metadata", {}).get("image_modalities", {})
        binary = modalities.get("binary_img")
        if binary is None:
            raise ValueError(f"Contract {data} split {split_name} is missing image_modalities.binary_img")
        if len(binary) != len(split.get("image_filenames", [])):
            raise ValueError(f"Contract {data} split {split_name} has mismatched RGB/binary_img counts")
        if any(path is None for path in binary):
            raise ValueError(f"Contract {data} split {split_name} has null binary_img paths")
    return downscale, len(train_images), len(test_images)


def _run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def _add_common_train_args(cmd: list[str], args: argparse.Namespace, max_iters: int) -> list[str]:
    cmd.extend(
        [
            "--vis",
            args.vis,
            "--max-num-iterations",
            str(max_iters),
            "--steps-per-save",
            str(args.steps_per_save),
            "--steps-per-eval-batch",
            str(args.steps_per_eval_batch),
            "--steps-per-eval-image",
            str(args.steps_per_eval_image),
            "--steps-per-eval-all-images",
            str(args.steps_per_eval_all_images),
        ]
    )
    if args.cpu:
        cmd.extend(["--mixed-precision", "False", "--machine.device-type", "cpu"])
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared dataset directory containing metadata/dataparser/contract.json.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root where nerfacto/ and invnerf/ training outputs are written.")
    parser.add_argument("--invnerf-seg-root", type=Path, default=Path("/workspace/Desktop/Repos/InvNeRF-Seg"))
    parser.add_argument("--experiment-prefix", type=str, default=None, help="Experiment prefix. Defaults to dataset directory name.")
    parser.add_argument("--timestamp", type=str, default="run_000")
    parser.add_argument("--vis", type=str, default="wandb")
    parser.add_argument("--nerfacto-project-name", type=str, default=None)
    parser.add_argument("--invnerf-project-name", type=str, default=None)
    parser.add_argument("--nerfacto-iters", type=int, default=30000)
    parser.add_argument("--invnerf-iters", type=int, default=40000)
    parser.add_argument("--steps-per-save", type=int, default=5000)
    parser.add_argument("--steps-per-eval-batch", type=int, default=500)
    parser.add_argument("--steps-per-eval-image", type=int, default=500)
    parser.add_argument("--steps-per-eval-all-images", type=int, default=5000)
    parser.add_argument("--run-nerfacto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-invnerf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-from-disk", action=argparse.BooleanOptionalAction, default=True, help="Use Nerfacto load-from-disk image loading.")
    parser.add_argument("--cpu", action="store_true", help="CPU smoke/debug mode. Adds torch model implementation flags.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = args.data.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    invnerf_root = args.invnerf_seg_root.expanduser().resolve()
    patch_root = Path(__file__).resolve().parents[1]

    if not data.exists():
        raise FileNotFoundError(f"Dataset does not exist: {data}")
    if not invnerf_root.exists():
        raise FileNotFoundError(f"InvNeRF-Seg root does not exist: {invnerf_root}")

    downscale, train_count, test_count = _require_contract_valid(data)
    ds_suffix = f"ds{downscale}" if downscale is not None else "dsauto"
    prefix = args.experiment_prefix or data.name
    nerfacto_exp = f"{prefix}_nerfacto_{ds_suffix}"
    invnerf_exp = f"{prefix}_invnerf_{ds_suffix}"

    env = dict(os.environ)
    pythonpath_parts = [str(patch_root), str(invnerf_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    print(f"dataset={data}")
    print(f"contract={_contract_path(data)}")
    print(f"downscale_factor={downscale} train={train_count} test={test_count}")
    print(f"output_root={output_root}")
    print(f"PYTHONPATH={env['PYTHONPATH']}")

    if args.run_nerfacto:
        cmd = [
            "ns-train",
            "nerfacto",
            "--data",
            str(data),
            "--output-dir",
            str(output_root / "nerfacto"),
            "--experiment-name",
            nerfacto_exp,
            "--timestamp",
            args.timestamp,
        ]
        if args.nerfacto_project_name:
            cmd.extend(["--project-name", args.nerfacto_project_name])
        _add_common_train_args(cmd, args, args.nerfacto_iters)
        if args.cpu:
            cmd.extend(["--pipeline.model.implementation", "torch"])
        if args.load_from_disk:
            cmd.extend(["--pipeline.datamanager.load-from-disk", "True"])
        cmd.append("nerfstudio-data")
        _run(cmd, env, args.dry_run)

    if args.run_invnerf:
        cmd = [
            "ns-train",
            "inv-nerf",
            "--data",
            str(data),
            "--output-dir",
            str(output_root / "invnerf"),
            "--experiment-name",
            invnerf_exp,
            "--timestamp",
            args.timestamp,
        ]
        if args.invnerf_project_name:
            cmd.extend(["--project-name", args.invnerf_project_name])
        _add_common_train_args(cmd, args, args.invnerf_iters)
        if args.cpu:
            cmd.extend(["--pipeline.model.implementation", "torch"])
        _run(cmd, env, args.dry_run)


if __name__ == "__main__":
    main()
