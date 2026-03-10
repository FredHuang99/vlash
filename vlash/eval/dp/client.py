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
HTTP client for the DP inference server.

Provides both synchronous and asynchronous interfaces for sending
inference requests to the server. Designed for use in the RoboFactory
evaluation loop where multiple robots independently request predictions.
"""

import base64
import logging
import pickle
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for the DP inference server.
    
    Supports non-blocking inference via submit_async() + poll_result(),
    designed for the async evaluation loop where robots fire off requests
    and continue executing actions while waiting.
    
    Args:
        server_address: URL of the server (e.g. "http://localhost:50051").
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, server_address: str, timeout: float = 120.0):
        if not server_address.startswith("http"):
            server_address = f"http://{server_address}"
        self.server_address = server_address
        self.timeout = timeout
        self._session = requests.Session()
        self._executor = ThreadPoolExecutor(max_workers=16)
        self._pending: dict[str, Future] = {}

    def health_check(self) -> bool:
        """Check if the server is ready."""
        try:
            resp = self._session.get(
                f"{self.server_address}/health", timeout=5.0
            )
            return resp.status_code == 200
        except Exception:
            return False

    def wait_for_server(self, max_wait: float = 300.0, poll_interval: float = 2.0):
        """Block until the server is ready or timeout."""
        start = time.time()
        while time.time() - start < max_wait:
            if self.health_check():
                logger.info("Server is ready.")
                return True
            time.sleep(poll_interval)
        raise TimeoutError(f"Server not ready after {max_wait}s")

    def predict_sync(self, request_id: str, observation: dict, task_description: str) -> np.ndarray:
        """Synchronous (blocking) prediction.
        
        Args:
            request_id: Unique request identifier.
            observation: Dict of numpy arrays.
            task_description: Task description string.
            
        Returns:
            Action chunk as numpy array [n_action_steps, action_dim].
        """
        obs_bytes = pickle.dumps(observation)
        obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")

        payload = {
            "request_id": request_id,
            "observation_b64": obs_b64,
            "task_description": task_description,
        }

        resp = self._session.post(
            f"{self.server_address}/predict",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data["success"]:
            raise RuntimeError(f"Inference failed: {data['error_message']}")

        action_bytes = base64.b64decode(data["action_chunk_b64"])
        action_chunk = pickle.loads(action_bytes)
        return action_chunk

    def submit_async(self, request_id: str, observation: dict, task_description: str):
        """Submit a non-blocking inference request.
        
        The result can be polled via poll_result() or retrieved with get_result().
        
        Args:
            request_id: Unique request identifier.
            observation: Dict of numpy arrays.
            task_description: Task description string.
        """
        future = self._executor.submit(
            self.predict_sync, request_id, observation, task_description
        )
        self._pending[request_id] = future

    def poll_result(self, request_id: str) -> Optional[np.ndarray]:
        """Check if an async result is ready (non-blocking).
        
        Args:
            request_id: The request ID submitted via submit_async().
            
        Returns:
            Action chunk numpy array if ready, None otherwise.
            
        Raises:
            RuntimeError: If the inference failed.
            KeyError: If request_id is unknown.
        """
        if request_id not in self._pending:
            raise KeyError(f"Unknown request_id: {request_id}")

        future = self._pending[request_id]
        if not future.done():
            return None

        # Retrieve result and clean up
        del self._pending[request_id]
        return future.result()  # raises if exception occurred

    def get_result(self, request_id: str, timeout: Optional[float] = None) -> np.ndarray:
        """Block until the async result is available.
        
        Args:
            request_id: The request ID submitted via submit_async().
            timeout: Max seconds to wait (None = wait indefinitely).
            
        Returns:
            Action chunk numpy array.
        """
        if request_id not in self._pending:
            raise KeyError(f"Unknown request_id: {request_id}")

        future = self._pending.pop(request_id)
        return future.result(timeout=timeout)

    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self._session.close()
