import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 31


def make_xingchen_example() -> dict:
    """Creates a random input example for the Xingchen policy."""
    return {
        "observation/image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_high": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_left_wrist": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_right_wrist": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/state": np.random.rand(ACTION_DIM).astype(np.float32),
        "prompt": "fold clothes",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class XingchenInputs(transforms.DataTransformFn):
    """Maps Xingchen data fields to the format expected by pi0.5."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if "observation/cam_high" in data:
            base_image = _parse_image(data["observation/cam_high"])
        else:
            base_image = _parse_image(data["observation/image"])

        if "observation/cam_left_wrist" in data:
            left_wrist_image = _parse_image(data["observation/cam_left_wrist"])
            left_wrist_mask = np.True_
        else:
            left_wrist_image = np.zeros_like(base_image)
            left_wrist_mask = np.False_

        if "observation/cam_right_wrist" in data:
            right_wrist_image = _parse_image(data["observation/cam_right_wrist"])
            right_wrist_mask = np.True_
        else:
            right_wrist_image = np.zeros_like(base_image)
            right_wrist_mask = np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_wrist_mask,
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XingchenOutputs(transforms.DataTransformFn):
    """Extracts the first 31 action dimensions from model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
