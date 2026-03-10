#!/bin/bash
# Launch script for VLASH DP evaluation on RoboFactory tasks.
#
# Usage:
#   bash scripts/launch_dp_eval.sh <vlash_config> <robofactory_config> [gpu_ids] [port]
#
# Example:
#   bash scripts/launch_dp_eval.sh configs/pi05.yaml /path/to/two_robots_stack_cube.yaml 0,1 50051

set -e

VLASH_CONFIG=${1:?"Usage: $0 <vlash_config> <robofactory_config> [gpu_ids] [port]"}
RF_CONFIG=${2:?"Usage: $0 <vlash_config> <robofactory_config> [gpu_ids] [port]"}
GPU_IDS=${3:-"0"}
PORT=${4:-50051}

echo "=========================================="
echo "VLASH DP Evaluation Launcher"
echo "  VLASH Config:  $VLASH_CONFIG"
echo "  RF Config:     $RF_CONFIG"
echo "  GPUs:          $GPU_IDS"
echo "  Port:          $PORT"
echo "=========================================="

# 1. Start the DP inference server in the background
echo "[1/2] Starting DP inference server..."
python -m vlash.eval.dp.server \
    --config "$VLASH_CONFIG" \
    --gpus "$GPU_IDS" \
    --port "$PORT" &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to load models..."
sleep 5
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died during startup."
        exit 1
    fi
    sleep 5
done

# 2. Run the evaluation
echo "[2/2] Running DP evaluation..."
python -m vlash.eval.run_robofactory_dp_eval \
    --server_address "localhost:${PORT}" \
    --rf_config "$RF_CONFIG" \
    --num_trials 20 \
    --async_threshold 20 \
    --async_wait 5 \
    --max_steps 250 \
    --verbose

# Cleanup
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Done."
