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
"""Utils for evaluating policies in RoboFactory simulation environments."""

import os
import logging

import cv2
import imageio
import numpy as np

logger = logging.getLogger(__name__)


def get_robofactory_images(raw_obs: dict, agent_id: int, resolution: int = 224) -> dict:
    """Extract and preprocess images from RoboFactory observations for a specific robot.
    
    Converts sensor_data camera images to the format expected by vlash pi0.5 model.
    Uses zero image for wrist camera since RoboFactory does not have wrist cameras.
    
    Args:
        raw_obs: Raw observation dict from env.step() or env.reset().
                 Expected keys: raw_obs['sensor_data']['head_camera_agent{id}']['rgb']
        agent_id: Robot agent index (0, 1, ...).
        resolution: Target image resolution.
        
    Returns:
        Dict with keys:
            - "observation.image": [H, W, 3] uint8 numpy array (head camera)
            - "observation.wrist_image": [H, W, 3] uint8 numpy array (zero image)
    """
    camera_name = f"head_camera_agent{agent_id}"
    
    # Extract head camera RGB [1, H, W, 3] tensor -> [H, W, 3] numpy
    head_img = raw_obs["sensor_data"][camera_name]["rgb"].squeeze(0).numpy()
    
    # Ensure uint8
    if head_img.dtype != np.uint8:
        head_img = (head_img * 255).astype(np.uint8)
    
    # Resize if needed
    if head_img.shape[0] != resolution or head_img.shape[1] != resolution:
        head_img = cv2.resize(head_img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    
    # Zero image for wrist camera (RoboFactory has no wrist cameras)
    wrist_img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    
    return {
        "observation.image": head_img,
        "observation.wrist_image": wrist_img,
    }


def get_robofactory_state(raw_obs: dict, agent_id: int) -> np.ndarray:
    """Extract robot state for a specific agent from RoboFactory observations.
    
    Args:
        raw_obs: Raw observation dict from env.
        agent_id: Robot agent index.
        
    Returns:
        State vector as float32 numpy array: joint positions (7) + gripper (2) = 9D.
    """
    qpos = raw_obs["agent"][f"panda-{agent_id}"]["qpos"].squeeze(0).numpy()
    return qpos.astype(np.float32)


def build_model_observation(raw_obs: dict, agent_id: int, resolution: int = 224) -> dict:
    """Build the complete observation dict for pi0.5 model inference.
    
    Combines image and state observations into the format expected by
    the vlash pi0.5 model.
    
    Args:
        raw_obs: Raw observation dict from RoboFactory env.
        agent_id: Robot agent index.
        resolution: Target image resolution.
        
    Returns:
        Dict ready for model inference with keys:
            - "observation.image": [H, W, 3] uint8
            - "observation.wrist_image": [H, W, 3] uint8
            - "observation.state": [state_dim] float32
    """
    img_dict = get_robofactory_images(raw_obs, agent_id, resolution)
    state = get_robofactory_state(raw_obs, agent_id)
    
    return {
        "observation.image": img_dict["observation.image"],
        "observation.wrist_image": img_dict["observation.wrist_image"],
        "observation.state": state,
    }


def save_robofactory_video(
    frames: list[np.ndarray],
    episode_idx: int,
    success: bool,
    task_name: str,
    log_dir: str = "./experiments/logs",
) -> str:
    """Save an episode replay as MP4 video.
    
    Args:
        frames: List of image frames (np.ndarray [H, W, 3]).
        episode_idx: Episode number for filename.
        success: Whether the episode was successful.
        task_name: Task name for filename.
        log_dir: Directory to save videos.
        
    Returns:
        Path to the saved video file.
    """
    rollout_dir = os.path.join(log_dir, "rollouts")
    os.makedirs(rollout_dir, exist_ok=True)
    
    clean_name = task_name.lower().replace(" ", "_").replace("\n", "_")[:50]
    mp4_path = os.path.join(
        rollout_dir,
        f"episode_{episode_idx}--success={success}--task={clean_name}.mp4"
    )
    
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for frame in frames:
        video_writer.append_data(frame)
    video_writer.close()
    
    logger.info(f"Saved rollout video: {mp4_path}")
    return mp4_path
