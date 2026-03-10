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
HTTP Inference Server for DP (DataPoint) mode.

Hosts a FastAPI server that receives inference requests and dispatches them
to k AsyncEngine instances (one per GPU) using round-robin scheduling.

Usage:
    python -m vlash.eval.dp.server --config <path_to_config.yaml> --gpus 0,1 --port 50051
"""

import argparse
import asyncio
import base64
import logging
import pickle
import sys
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from vlash.eval.dp.async_engine import AsyncEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Request / Response Models
# ============================================================================

class PredictRequest(BaseModel):
    """HTTP request body for inference."""
    request_id: str
    observation_b64: str  # base64-encoded pickle of observation dict
    task_description: str


class PredictResponse(BaseModel):
    """HTTP response body for inference."""
    request_id: str
    action_chunk_b64: str  # base64-encoded pickle of numpy action chunk
    success: bool
    error_message: str = ""


# ============================================================================
# Server State
# ============================================================================

class ServerState:
    """Shared state for the inference server."""
    
    def __init__(self):
        self.engines: list[AsyncEngine] = []
        self.rr_counter: int = 0
        self._lock = asyncio.Lock()

    async def get_next_engine(self) -> AsyncEngine:
        """Round-robin select the next engine."""
        async with self._lock:
            engine = self.engines[self.rr_counter % len(self.engines)]
            self.rr_counter += 1
            return engine


server_state = ServerState()


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start all engines on app startup, stop them on shutdown."""
    for engine in server_state.engines:
        await engine.start()
    logger.info(f"All {len(server_state.engines)} engines started.")
    yield
    for engine in server_state.engines:
        await engine.stop()
    logger.info("All engines stopped.")


app = FastAPI(title="VLASH DP Inference Server", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "num_engines": len(server_state.engines),
        "rr_counter": server_state.rr_counter,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Handle an inference request.
    
    Deserializes the observation, dispatches to the next engine via
    round-robin, waits for the result, and returns the serialized action chunk.
    """
    try:
        # Deserialize observation
        obs_bytes = base64.b64decode(request.observation_b64)
        observation = pickle.loads(obs_bytes)

        # Round-robin dispatch
        engine = await server_state.get_next_engine()

        # Submit and await result
        action_chunk = await engine.submit(
            request_id=request.request_id,
            observation=observation,
            task_description=request.task_description,
        )

        # Serialize result
        action_bytes = pickle.dumps(action_chunk)
        action_b64 = base64.b64encode(action_bytes).decode("utf-8")

        return PredictResponse(
            request_id=request.request_id,
            action_chunk_b64=action_b64,
            success=True,
        )

    except Exception as e:
        logger.error(f"Predict error for {request.request_id}: {e}")
        return PredictResponse(
            request_id=request.request_id,
            action_chunk_b64="",
            success=False,
            error_message=str(e),
        )


# ============================================================================
# Server Launch
# ============================================================================

def create_and_load_engines(config_path: str, gpu_ids: list[int]):
    """Create AsyncEngine instances and load models onto each GPU.
    
    This is called before the event loop starts, so model loading
    happens synchronously during server startup.
    """
    for gpu_id in gpu_ids:
        device = f"cuda:{gpu_id}"
        engine = AsyncEngine(config_path=config_path, device=device)
        logger.info(f"Loading model on {device}...")
        engine.load_model()
        server_state.engines.append(engine)
    logger.info(f"All {len(gpu_ids)} models loaded.")


def main():
    parser = argparse.ArgumentParser(description="VLASH DP Inference Server")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to vlash RunConfig YAML")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g. '0,1,2')")
    parser.add_argument("--port", type=int, default=50051,
                        help="HTTP port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    logger.info(f"Starting DP server with GPUs: {gpu_ids}, port: {args.port}")

    # Load models synchronously before starting the async server
    create_and_load_engines(args.config, gpu_ids)

    logger.info(f"Server ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
