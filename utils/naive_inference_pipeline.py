"""A naive pipeline for 3D bounding box prediction."""

import os
import sys
import time
from typing import Sequence, Tuple, cast

import torch
from torch import Tensor

from data.data_structure import Frame
from data.dataset_handler import DatasetHandler
from logger_wrapper.logger_wrapper import setup_logger
from models.model_fusion import FusionModel

logger = setup_logger(name_appendix="Naive Pipeline")

DATA_DIR = "dataset/dl_challenge"


def _mount_google_drive() -> None:
    """Mount Google Drive if running on Colab."""
    # pylint: disable=import-outside-toplevel, import-error, no-name-in-module
    # pylint: disable=global-statement
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        logger.info("Google Drive mounted successfully.")

        global DATA_DIR
        DATA_DIR = "/content/drive/My Drive/dataset/dl_challenge/"
        logger.info("Dataset directory updated to: %s", DATA_DIR)

    except Exception:  # pylint: disable=broad-except
        logger.info("Not running in Colab or drive mount not needed.")


def _dataset_exists(dataset_path: str) -> bool:
    if not os.path.exists(dataset_path):
        logger.error("Dataset path %s does not exist!", dataset_path)
        return False
    return True


def _device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


def prepare_sample(frame: Frame, device: torch.device) -> Tuple[Tensor, Tensor]:
    """
    Convert a Frame to tensors and reshape inputs appropriately.

    - Converts the frame using Frame.as_tensor().
    - Unsqueezes the RGB image to add a batch dimension.
    - Reshapes the point cloud from (3, H, W) to (1, H*W, 3).
    """
    frame_tensor = Frame.as_tensor(frame, device)
    # Process image: from (3, H, W) -> (1, 3, H, W)
    image_tensor = cast(Tensor, frame_tensor.rgb)
    image_tensor = image_tensor.unsqueeze(0)
    # Process point cloud: from (3, H, W) -> (H*W, 3) and add batch dimension: (1, N, 3)
    pc_tensor = cast(Tensor, frame_tensor.pc)  # shape: (3, H, W)
    pc_tensor = pc_tensor.view(3, -1).permute(1, 0)  # shape: (H*W, 3)
    pc_tensor = pc_tensor.unsqueeze(0)  # shape: (1, H*W, 3)
    return image_tensor, pc_tensor


def main(_: Sequence[str]) -> int:
    """Main function: loads the dataset and runs a forward pass on a sample frame."""
    _mount_google_drive()  # Mount Google Drive if running on Colab

    dataset_path = DATA_DIR
    if not _dataset_exists(dataset_path):
        return os.EX_DATAERR

    device = _device()

    dataset = DatasetHandler(dataset_path)
    logger.info("Dataset size: %d frames", len(dataset))

    fusion_model = FusionModel().to(device)
    fusion_model.eval()  # set model to evaluation mode

    logger.info("Running forward pass on one sample...")
    sample_frame = dataset[0]
    image_tensor, pc_tensor = prepare_sample(sample_frame, device)
    start_time = time.time()
    with torch.no_grad():
        output = fusion_model(image_tensor, pc_tensor)
    elapsed = time.time() - start_time
    logger.info("Forward pass output type: %s", type(output))
    logger.info("Forward pass output shape: %s", output.shape)
    logger.info("Forward pass took %.4f seconds", elapsed)

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
