"""Testing data handler"""

import os
import sys
from typing import Sequence

from data.dataset_handler import DatasetHandler
from visualization.visualization import Visualizer

DATA_DIR = "dataset/dl_challenge"


def _draw_frame(dataset_handler: DatasetHandler, visualizer: Visualizer) -> None:
    idx = -1
    while True:
        user_input = input("Enter index, or empty string for next (or 'q' to quit): ").strip()
        if not user_input.isdigit() and not user_input.lower() in ["q", ""]:
            print("Warning: enter a number, empty string, or 'q'")
            continue

        if user_input.lower() == "q":
            return
        if user_input.lower() == "":
            idx += 1
        else:
            idx = int(user_input)

        if not 0 <= idx < len(dataset_handler):
            print(f"Warning: out of index (has to be between 0 and {len(dataset_handler) - 1})")
            continue

        print(f"Visualizing frame {idx}")
        frame = dataset_handler[idx]
        visualizer.visualize_frame(frame)


def main(_: Sequence[str]) -> int:
    """Main function: loads the dataset and visualizes the data."""
    dataset_handler = DatasetHandler(DATA_DIR)
    visualizer = Visualizer(config={"visualize_2d": True, "visualize_3d": True})
    _draw_frame(dataset_handler, visualizer)

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
