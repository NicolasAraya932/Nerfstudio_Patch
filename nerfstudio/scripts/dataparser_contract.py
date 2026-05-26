"""CLI for serializing a frozen dataparser contract after ns-process-data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tyro

from nerfstudio.process_data.dataparser_contract import DataparserContractConfig, serialize_dataparser_contract
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SerializeDataparserContract:
    """Serialize Nerfstudio dataparser outputs for an already processed dataset."""

    data: Path
    """Processed dataset directory containing transforms.json."""
    output_dir: Optional[Path] = None
    """Output directory. Defaults to <data>/metadata/dataparser."""
    contract: DataparserContractConfig = field(default_factory=lambda: DataparserContractConfig(enabled=True))
    """Dataparser contract options passed to the Nerfstudio dataparser."""

    def main(self) -> None:
        path = serialize_dataparser_contract(self.data, contract=self.contract, output_dir=self.output_dir)
        CONSOLE.log(f"[bold green]Dataparser contract ready: {path}")


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(SerializeDataparserContract).main()


if __name__ == "__main__":
    entrypoint()
