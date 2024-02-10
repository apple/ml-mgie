from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ml_mgie.mgie import MGIE, MGIEParams
from PIL import Image
from simple_parsing import ArgumentParser


@dataclass
class Command:
    def run(self):
        raise NotImplementedError()


@dataclass
class MGIECommand:
    """"""

    input_path: Path
    instruction: str
    output_path: Path
    save_thumbnail: bool = False
    params: MGIEParams = MGIEParams()

    @cached_property
    def mgie(self) -> MGIE:
        return MGIE(params=self.params)

    def run(self):
        print(
            f"Running MGIE command with instruction: {self.instruction} onto image: {self.input_path}"
        )
        image = Image.open(self.input_path).convert("RGB")
        if self.params.max_size:
            image.thumbnail((self.params.max_size, self.params.max_size))
            if self.save_thumbnail:
                image.save(self.output_path.with_suffix(".thumb.jpg"))

        result_image, inner_thought = self.mgie.edit(
            image=image,
            instruction=self.instruction,
        )
        print(f"Inner thought: {inner_thought}")
        result_image.save(self.output_path)
        print(f"Saved result image to: {self.output_path}")


@dataclass
class Program:
    command: MGIECommand

    def run(self) -> Any:
        return self.command.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="ml-mgie.main",
        description="""
        ML-MGIE: Guiding Instruction-based Image Editing via Multimodal Large Language Models.
        """,
    )
    parser.add_arguments(Program, dest="program")
    args = parser.parse_args()
    program: Program = args.program
    program.run()
