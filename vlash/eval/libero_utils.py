# Copyright 2025 VLASH team. All rights reserved.
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
"""Utils for evaluating policies in LIBERO simulation environments."""

import json
import math
import os
from logging import getLogger

import imageio
import numpy as np
from PIL import Image

try:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError as exc:
    if exc.name != "libero":
        raise
    raise ModuleNotFoundError(
        "LIBERO is not installed in the current Python environment. "
        "Install LIBERO with `python3 -m pip install /path/to/LIBERO`, then install "
        "VLASH's extra LIBERO dependencies with "
        "`python3 -m pip install -r /path/to/vlash/examples/eval/libero_requirements.txt`. "
        "Verify from a neutral directory such as `/tmp`, not from inside the LIBERO source tree."
    ) from exc


logger = getLogger(__name__)


def get_libero_env(task, render_resolution=256, seed=7):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": render_resolution,
        "camera_widths": render_resolution,
        # We explicitly request both agentview and eye-in-hand cameras for PI0.5
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert floating-point images back to uint8, matching openpi's client helper."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(img: np.ndarray, target_height: int, target_width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicate openpi/openpi-client resize_with_pad semantics for a single HWC image."""
    img = np.asarray(img)
    if img.shape[:2] == (target_height, target_width):
        return np.ascontiguousarray(convert_to_uint8(img))

    pil_img = Image.fromarray(convert_to_uint8(img))
    cur_height, cur_width = img.shape[:2]
    ratio = max(cur_width / target_width, cur_height / target_height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = pil_img.resize((resized_width, resized_height), resample=method)

    padded = Image.new(resized_img.mode, (target_width, target_height), 0)
    pad_height = max(0, int((target_height - resized_height) / 2))
    pad_width = max(0, int((target_width - resized_width) / 2))
    padded.paste(resized_img, (pad_width, pad_height))
    return np.ascontiguousarray(np.asarray(padded))


def _extract_rotated_libero_images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract and 180-rotate LIBERO agentview and wrist images to match training preprocessing."""
    img_agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    if "robot0_eye_in_hand_image" in obs:
        img_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    else:
        img_wrist = np.zeros_like(img_agent)
    return img_agent, img_wrist


def get_libero_images(obs, resize_size):
    """Extracts images from observations and preprocesses them.
    
    Returns a dict containing both the agentview (third-person) and
    eye_in_hand (wrist) images required for pi0.5.
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img_agent, img_wrist = _extract_rotated_libero_images(obs)
    img_agent = resize_with_pad(img_agent, resize_size[0], resize_size[1])
    img_wrist = resize_with_pad(img_wrist, resize_size[0], resize_size[1])

    return {
        "observation.image": convert_to_uint8(img_agent),
        "observation.wrist_image": convert_to_uint8(img_wrist),
    }


def get_libero_debug_images(obs, resize_size):
    """Return raw rotated and final resized LIBERO images for alignment debugging."""
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    raw_agent, raw_wrist = _extract_rotated_libero_images(obs)
    processed = get_libero_images(obs, resize_size)
    return {
        "raw_agentview": raw_agent,
        "raw_wrist": raw_wrist,
        "processed_agentview": processed["observation.image"],
        "processed_wrist": processed["observation.wrist_image"],
    }


def _to_policy_image(image: np.ndarray) -> np.ndarray:
    """Convert LIBERO image to CHW float32 in [0, 1] for VLASH policies."""
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")

    if image.shape[-1] == 3:
        image = np.transpose(image, (2, 0, 1))
    elif image.shape[0] != 3:
        raise ValueError(f"Expected image in HWC or CHW RGB format, got shape {image.shape}")

    image = image.astype(np.float32, copy=False)
    if image.max() > 1.0 or image.min() < 0.0:
        image = np.clip(image / 255.0, 0.0, 1.0)

    return np.ascontiguousarray(image)


def build_libero_model_obs(img_dict: dict, robot_state: np.ndarray, policy_config) -> dict:
    """Map LIBERO observations to the image/state keys expected by a policy checkpoint."""
    agent_img = _to_policy_image(img_dict["observation.image"])
    wrist_img = _to_policy_image(img_dict["observation.wrist_image"])

    image_features = getattr(policy_config, "image_features", {}) or {}
    image_feature_keys = list(image_features.keys())

    model_obs = {
        "observation.state": np.asarray(robot_state, dtype=np.float32),
    }

    if not image_feature_keys:
        model_obs["observation.image"] = agent_img
        model_obs["observation.wrist_image"] = wrist_img
        return model_obs

    fallback_img = np.zeros_like(agent_img)
    assigned_agent = False
    assigned_wrist = False

    for index, feature_key in enumerate(image_feature_keys):
        lower_key = feature_key.lower()

        if "empty_camera_" in lower_key:
            continue
        if any(token in lower_key for token in ("wrist", "image2", "left", "right", "hand")):
            model_obs[feature_key] = wrist_img
            assigned_wrist = True
            continue
        if not assigned_agent:
            model_obs[feature_key] = agent_img
            assigned_agent = True
            continue
        if not assigned_wrist:
            model_obs[feature_key] = wrist_img
            assigned_wrist = True
            continue

        # Some checkpoints may declare additional cameras that LIBERO does not expose.
        model_obs[feature_key] = fallback_img
        logger.warning(
            "Policy expects extra image feature '%s' beyond LIBERO's available cameras; "
            "using a zero image placeholder.",
            feature_key,
        )

    return model_obs


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, log_dir="./experiments/logs"):
    """Saves an MP4 replay of an episode."""
    rollout_dir = os.path.join(log_dir, "rollouts")
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/episode_{idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_alignment_debug_artifacts(
    debug_root: str,
    task_id: int,
    trial_idx: int,
    task_description: str,
    debug_images: dict,
    robot_state: np.ndarray,
    raw_action_chunk: np.ndarray,
    clipped_action_chunk: np.ndarray,
    clip_count: int,
    max_abs_before_clip: float,
) -> str:
    """Persist a compact debug bundle for one trial's first alignment sample."""
    safe_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    trial_dir = os.path.join(debug_root, f"task_{task_id:02d}_trial_{trial_idx + 1:03d}_{safe_task}")
    os.makedirs(trial_dir, exist_ok=True)

    imageio.imwrite(os.path.join(trial_dir, "raw_agentview.png"), debug_images["raw_agentview"])
    imageio.imwrite(os.path.join(trial_dir, "raw_wrist.png"), debug_images["raw_wrist"])
    imageio.imwrite(os.path.join(trial_dir, "processed_agentview.png"), debug_images["processed_agentview"])
    imageio.imwrite(os.path.join(trial_dir, "processed_wrist.png"), debug_images["processed_wrist"])

    np.save(os.path.join(trial_dir, "observation_state.npy"), np.asarray(robot_state, dtype=np.float32))
    np.save(os.path.join(trial_dir, "first_action_chunk_raw.npy"), np.asarray(raw_action_chunk, dtype=np.float32))
    np.save(
        os.path.join(trial_dir, "first_action_chunk_clipped.npy"),
        np.asarray(clipped_action_chunk, dtype=np.float32),
    )

    stats = {
        "clip_count": int(clip_count),
        "max_abs_before_clip": float(max_abs_before_clip),
        "raw_chunk_shape": list(np.asarray(raw_action_chunk).shape),
        "clipped_chunk_shape": list(np.asarray(clipped_action_chunk).shape),
    }
    with open(os.path.join(trial_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return trial_dir


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
