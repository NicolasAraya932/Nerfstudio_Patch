# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processes an image sequence to a nerfstudio compatible dataset."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ImagesToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process images into a nerfstudio dataset.

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""

    def _find_binary_image_dir(self, image_dir: Path) -> Optional[Path]:
        """Find a semantic binary image folder associated with an input image directory."""
        candidates = [image_dir / "binary_imgs", image_dir.parent / "binary_imgs"]
        for candidate in candidates:
            if candidate.exists() and process_data_utils.list_images(candidate):
                return candidate
        return None

    def _find_binary_image(self, binary_dir: Path, image_path: Path) -> Optional[Path]:
        """Match a binary image by stem, preferring the PNG masks used by the project datasets."""
        suffixes = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        relative_candidates = []
        try:
            relative_candidates.append(image_path.relative_to(image_path.parents[0]).with_suffix(".png"))
        except ValueError:
            pass
        relative_candidates.append(Path(f"{image_path.stem}.png"))

        for relative_candidate in relative_candidates:
            candidate = binary_dir / relative_candidate
            if candidate.exists():
                return candidate
        for suffix in suffixes:
            candidate = binary_dir / f"{image_path.stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def _copy_binary_images_for_transforms(
        self,
        image_path_map: dict[Path, Path],
    ) -> dict[str, str]:
        """Copy paired binary images and return a map from output RGB filename to binary path."""
        if not image_path_map:
            return {}

        binary_map: dict[str, str] = {}

        for source_image, output_image in image_path_map.items():
            binary_dir = self._find_binary_image_dir(source_image.parent)
            if binary_dir is None:
                continue
            binary_image = self._find_binary_image(binary_dir, source_image)
            if binary_image is None:
                continue

            binary_output_dir = self.output_dir / "binary_imgs"
            binary_output_dir.mkdir(parents=True, exist_ok=True)
            output_binary = binary_output_dir / f"{output_image.stem}{binary_image.suffix.lower()}"
            shutil.copy2(binary_image, output_binary)
            binary_map[output_image.name] = output_binary.relative_to(self.output_dir).as_posix()

        return binary_map

    def _add_binary_images_to_transforms(self, binary_map: dict[str, str]) -> str:
        """Append binary_img entries to transforms.json when paired masks were discovered."""
        if not binary_map:
            return "No binary_imgs folder found; transforms.json was left unchanged."

        transforms_path = self.output_dir / "transforms.json"
        if not transforms_path.exists():
            return "No transforms.json found; binary_img entries were not written."

        payload = json.loads(transforms_path.read_text(encoding="utf-8"))
        matched = 0
        for frame in payload.get("frames", []):
            image_name = Path(frame["file_path"]).name
            binary_path = binary_map.get(image_name)
            if binary_path is None:
                continue
            frame["binary_img"] = binary_path
            matched += 1

        transforms_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
        return f"Added binary_img entries for {matched} frames."

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        require_cameras_exist = False
        if self.colmap_model_path != ColmapConverterToNerfstudioDataset.default_colmap_path():
            if not self.skip_colmap:
                raise RuntimeError("The --colmap-model-path can only be used when --skip-colmap is not set.")
            if not (self.output_dir / self.colmap_model_path).exists():
                raise RuntimeError(f"The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist.")
            require_cameras_exist = True

        image_rename_map: Optional[dict[str, str]] = None
        binary_image_map: dict[str, str] = {}

        # Generate planar projections if equirectangular
        if self.camera_type == "equirectangular":
            if self.eval_data is not None:
                raise ValueError("Cannot use eval_data with camera_type equirectangular.")

            pers_size = equirect_utils.compute_resolution_from_equirect(self.data, self.images_per_equirect)
            CONSOLE.log(f"Generating {self.images_per_equirect} {pers_size} sized images per equirectangular image")
            self.data = equirect_utils.generate_planar_projections_from_equirectangular(
                self.data, pers_size, self.images_per_equirect, crop_factor=self.crop_factor
            )

            self.camera_type = "perspective"

        summary_log = []

        # Copy and downscale images
        if not self.skip_image_processing:
            # Copy images to output directory
            image_rename_map_paths = process_data_utils.copy_images(
                self.data,
                image_dir=self.image_dir,
                crop_factor=self.crop_factor,
                image_prefix="frame_train_" if self.eval_data is not None else "frame_",
                verbose=self.verbose,
                num_downscales=self.num_downscales,
                same_dimensions=self.same_dimensions,
                keep_image_dir=False,
            )
            image_rename_map = dict(
                (a.relative_to(self.data).as_posix(), b.name) for a, b in image_rename_map_paths.items()
            )
            binary_image_map.update(self._copy_binary_images_for_transforms(dict(image_rename_map_paths)))
            if self.eval_data is not None:
                eval_image_rename_map_paths = process_data_utils.copy_images(
                    self.eval_data,
                    image_dir=self.image_dir,
                    crop_factor=self.crop_factor,
                    image_prefix="frame_eval_",
                    verbose=self.verbose,
                    num_downscales=self.num_downscales,
                    same_dimensions=self.same_dimensions,
                    keep_image_dir=True,
                )
                eval_image_rename_map = dict(
                    (a.relative_to(self.eval_data).as_posix(), b.name) for a, b in eval_image_rename_map_paths.items()
                )
                image_rename_map.update(eval_image_rename_map)
                binary_image_map.update(self._copy_binary_images_for_transforms(dict(eval_image_rename_map_paths)))

            num_frames = len(image_rename_map)
            summary_log.append(f"Starting with {num_frames} images")

            # # Create mask
            mask_path = process_data_utils.save_mask(
                image_dir=self.image_dir,
                num_downscales=self.num_downscales,
                crop_factor=(0.0, 0.0, 0.0, 0.0),
                percent_radius=self.percent_radius_crop,
            )
            if mask_path is not None:
                summary_log.append("Saved mask(s)")
        else:
            source_images = process_data_utils.list_images(self.data)
            num_frames = len(source_images)
            if num_frames == 0:
                raise RuntimeError("No usable images in the data folder.")
            binary_image_map.update(self._copy_binary_images_for_transforms({path: path for path in source_images}))
            summary_log.append(f"Starting with {num_frames} images")

        # Run COLMAP
        if not self.skip_colmap:
            require_cameras_exist = True
            self._run_colmap()
            # Colmap uses renamed images
            image_rename_map = None

        # Export depth maps
        image_id_to_depth_path, log_tmp = self._export_depth()
        summary_log += log_tmp

        if require_cameras_exist and not (self.absolute_colmap_model_path / "cameras.bin").exists():
            raise RuntimeError(f"Could not find existing COLMAP results ({self.colmap_model_path / 'cameras.bin'}).")

        summary_log += self._save_transforms(
            num_frames,
            image_id_to_depth_path,
            None,
            image_rename_map,
        )
        summary_log.append(self._add_binary_images_to_transforms(binary_image_map))
        summary_log.append(self._save_dataparser_contract())

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)
