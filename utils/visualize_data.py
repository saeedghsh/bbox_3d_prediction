"""Testing data handler"""

import os
import sys
from typing import Sequence

from data.dataset_handler import DatasetHandler
from visualization.visualization import Visualizer


def _draw_frame(dataset_handler: DatasetHandler, visualizer: Visualizer) -> None:
    while True:
        user_input = input("Enter index (or 'q' to quit): ").strip()
        if user_input.lower() == "q":
            return
        if not user_input.isdigit():
            print("Warning: enter a number")
            continue
        idx = int(user_input)
        if not 0 <= idx < len(dataset_handler):
            print(f"Warning: out of index (has to be between 0 and {len(dataset_handler) - 1})")
            continue
        frame = dataset_handler[idx]
        visualizer.visualize_frame(frame)


def main(_: Sequence[str]) -> int:
    """Main entry point for splitting and caching the dataset."""
    data_dir = "dataset/dl_challenge"

    dataset_handler = DatasetHandler(data_dir)
    visualizer = Visualizer(config={"visualize_2d": True, "visualize_3d": True})
    _draw_frame(dataset_handler, visualizer)

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
