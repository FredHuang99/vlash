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
"""
AsyncEngine: per-GPU FCFS inference engine for pi0.5 policy.

Each AsyncEngine instance loads one pi0.5 model on a specific GPU device
and processes inference requests one-at-a-time in FIFO order (no batching,
since pi0.5 does not currently support batch inference).
"""

import asyncio
import logging
import pickle
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from lerobot.configs import parser as lerobot_parser
from lerobot.utils.utils import get_safe_torch_device

from vlash.configs import RunConfig
from vlash.policies.factory import get_policy_class

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A single inference request queued in the AsyncEngine."""
    request_id: str
    observation: dict  # numpy dict: observation.image, observation.wrist_image, observation.state
    task_description: str
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class AsyncEngine:
    """Per-GPU asynchronous inference engine with FCFS scheduling.
    
    Loads a pi0.5 model on the specified device and runs an inference loop
    that processes requests one at a time from a FIFO queue.
    
    Args:
        config_path: Path to vlash RunConfig YAML.
        device: Target GPU device string (e.g. "cuda:0", "cuda:1").
    """

    def __init__(self, config_path: str, device: str):
        self.config_path = config_path
        self.device_str = device
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._policy = None
        self._device = None
        self._prev_action_chunks: dict[str, torch.Tensor] = {}  # keyed by request_id prefix (robot session)
        self._running = False
        self._inference_task: Optional[asyncio.Task] = None

    def load_model(self):
        """Load the pi0.5 model onto the target GPU. Must be called before start()."""
        cfg = lerobot_parser.parse(RunConfig, ["--config", self.config_path])
        self._device = get_safe_torch_device(self.device_str)

        policy_class = get_policy_class(cfg.policy.type)
        self._policy = policy_class.from_pretrained(
            pretrained_name_or_path=cfg.policy.pretrained_path,
            config=cfg.policy,
        )
        self._policy.to(self._device)
        self._policy.eval()
        logger.info(f"AsyncEngine: Model loaded on {self._device}")

    async def start(self):
        """Start the background inference loop."""
        self._running = True
        self._inference_task = asyncio.create_task(self._inference_loop())
        logger.info(f"AsyncEngine: Inference loop started on {self.device_str}")

    async def stop(self):
        """Stop the inference loop gracefully."""
        self._running = False
        if self._inference_task:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
        logger.info(f"AsyncEngine: Stopped on {self.device_str}")

    async def submit(self, request_id: str, observation: dict, task_description: str) -> np.ndarray:
        """Submit an inference request and await the result.
        
        Args:
            request_id: Unique ID for tracking. Format: "{robot_id}_{counter}" 
                        to enable per-robot prev_action_chunk statefulness.
            observation: Dict of numpy arrays with keys:
                - "observation.image": [H, W, 3] uint8
                - "observation.wrist_image": [H, W, 3] uint8  
                - "observation.state": [state_dim] float32
            task_description: Natural language task description.
            
        Returns:
            Action chunk as numpy array [n_action_steps, action_dim].
        """
        loop = asyncio.get_event_loop()
        request = InferenceRequest(
            request_id=request_id,
            observation=observation,
            task_description=task_description,
            future=loop.create_future(),
        )
        await self.queue.put(request)
        return await request.future

    async def _inference_loop(self):
        """Main loop: dequeue requests one at a time and run inference."""
        while self._running:
            try:
                request = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                action_chunk = self._run_inference(request)
                request.future.set_result(action_chunk)
            except Exception as e:
                logger.error(f"AsyncEngine inference error: {e}\n{traceback.format_exc()}")
                request.future.set_exception(e)

    def _run_inference(self, request: InferenceRequest) -> np.ndarray:
        """Run a single inference on the loaded model.
        
        Args:
            request: The inference request to process.
            
        Returns:
            Action chunk as numpy array [n_action_steps, action_dim].
        """
        obs = request.observation

        # Prepare observation dict for the model
        model_obs = {}
        for k, v in obs.items():
            # Add batch dimension and move to device
            model_obs[k] = torch.from_numpy(v).unsqueeze(0).to(self._device)

        # Task description as list (batch of 1)
        model_obs["task"] = [request.task_description]

        # Extract robot session ID for prev_action_chunk tracking
        # request_id format: "{robot_id}_{counter}" -> session key is robot_id
        session_key = request.request_id.rsplit("_", 1)[0] if "_" in request.request_id else request.request_id

        with torch.inference_mode():
            kwargs = {}
            if (hasattr(self._policy, "config") 
                    and getattr(self._policy.config, "inference_prefix_mask_steps", 0) > 0):
                kwargs["prev_action_chunk"] = self._prev_action_chunks.get(session_key)

            action_chunk = self._policy.predict_action_chunk(model_obs, **kwargs)

            if "prev_action_chunk" in kwargs:
                self._prev_action_chunks[session_key] = action_chunk.clone()

            # Remove batch dim -> [n_action_steps, action_dim], move to CPU as numpy
            action_chunk_np = action_chunk.squeeze(0).cpu().numpy()

        return action_chunk_np
