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

"""
Base class to process images or video into a nerfstudio dataset
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nerfstudio.process_data.dataparser_contract import DataparserContractConfig, serialize_dataparser_contract


@dataclass
class BaseConverterToNerfstudioDataset(ABC):
    """Base class to process images or video into a nerfstudio dataset."""

    data: Path
    """Path the data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    eval_data: Optional[Path] = None
    """Path the eval data, either a video file or a directory of images. If set to None, the first will be used both for training and eval"""
    verbose: bool = False
    """If True, print extra logging."""
    dataparser_contract: DataparserContractConfig = field(default_factory=DataparserContractConfig)
    """Dataparser contract options serialized after transforms.json is written."""

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    @property
    def image_dir(self) -> Path:
        return self.output_dir / "images"

    def _save_dataparser_contract(self) -> str:
        """Serialize the frozen dataparser contract for the processed dataset."""
        contract_path = serialize_dataparser_contract(self.output_dir, contract=self.dataparser_contract)
        return f"Saved dataparser contract to {contract_path}"

    @abstractmethod
    def main(self) -> None:
        """This method implements the conversion logic for each type of data"""
        raise NotImplementedError("the main method for conversion needs to be implemented")
