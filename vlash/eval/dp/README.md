# VLASH DP (DataPoint) Mode

Multi-GPU inference serving for evaluating pi0.5 policies on RoboFactory multi-robot tasks.

## Architecture

```
RoboFactory Eval Script (per-robot async requests)
        │
        │  HTTP (submit_async / poll_result)
        ▼
  FastAPI Server (round-robin dispatch)
        │
        ▼
  AsyncEngine_0 (GPU:0)    AsyncEngine_1 (GPU:1)    ...
  (FCFS queue, 1 request    (FCFS queue, 1 request
   at a time)                at a time)
```

## Dependencies

```bash
pip install fastapi uvicorn requests
# Plus existing vlash + robofactory dependencies
```

## Quick Start

### 1. Start the inference server

```bash
python -m vlash.eval.dp.server \
    --config configs/pi05_robofactory.yaml \
    --gpus 0,1 \
    --port 50051
```

### 2. Run the evaluation (in another terminal)

```bash
python -m vlash.eval.run_robofactory_dp_eval \
    --server_address localhost:50051 \
    --rf_config /path/to/robofactory/configs/table/two_robots_stack_cube.yaml \
    --num_trials 20 \
    --async_threshold 20 \
    --async_wait 5 \
    --max_steps 250 \
    --verbose
```

### 3. Or use the launch script (does both)

```bash
bash scripts/launch_dp_eval.sh \
    configs/pi05_robofactory.yaml \
    /path/to/robofactory/configs/table/two_robots_stack_cube.yaml \
    0,1 \
    50051
```

## Smoke Test Commands

### Health check (server must be running)
```bash
curl http://localhost:50051/health
```

### Import verification
```bash
python -c "from vlash.eval.dp.async_engine import AsyncEngine; print('OK')"
python -c "from vlash.eval.dp.server import app; print('OK')"
python -c "from vlash.eval.dp.client import InferenceClient; print('OK')"
python -c "from vlash.eval.robofactory_utils import build_model_observation; print('OK')"
```

### Client-server roundtrip test (server must be running)
```python
from vlash.eval.dp.client import InferenceClient
import numpy as np

client = InferenceClient("localhost:50051")
client.wait_for_server()

# Send dummy observation
obs = {
    "observation.image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation.wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation.state": np.zeros(9, dtype=np.float32),
}
result = client.predict_sync("test_0", obs, "pick up the cube")
print(f"Action chunk shape: {result.shape}")  # e.g. [50, 8]
client.close()
```

## CLI Reference

### Server (`vlash.eval.dp.server`)

| Arg | Default | Description |
|-----|---------|-------------|
| `--config` | required | Path to vlash RunConfig YAML |
| `--gpus` | `0` | Comma-separated GPU IDs |
| `--port` | `50051` | HTTP port |
| `--host` | `0.0.0.0` | Bind address |

### Eval (`vlash.eval.run_robofactory_dp_eval`)

| Arg | Default | Description |
|-----|---------|-------------|
| `--server_address` | `localhost:50051` | DP server address |
| `--rf_config` | required | RoboFactory task config YAML |
| `--num_trials` | `20` | Number of episodes |
| `--async_threshold` | `20` | Remaining actions before new inference |
| `--async_wait` | `5` | Chunk switching wait offset |
| `--max_steps` | `250` | Max iterations per episode |
| `--topp_dt` | `0.05` | TOPP interpolation timestep |
| `--seed` | `10000` | Random seed |
| `--sim_backend` | `cpu` | ManiSkill backend |
| `--verbose` | off | Enable verbose logging |

## File Structure

```
vlash/eval/
├── dp/
│   ├── __init__.py           # Package init
│   ├── async_engine.py       # Per-GPU FCFS inference engine
│   ├── server.py             # FastAPI HTTP server (round-robin)
│   └── client.py             # HTTP client (sync + async)
├── robofactory_utils.py      # Observation conversion utils
├── run_robofactory_dp_eval.py  # Main DP eval script
├── run_libero_eval.py        # (existing) LIBERO eval
└── libero_utils.py           # (existing) LIBERO utils

scripts/
└── launch_dp_eval.sh         # Launch server + eval
```
