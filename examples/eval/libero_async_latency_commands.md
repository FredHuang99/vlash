# LIBERO Async-vs-NonAsync Latency Benchmark

This benchmark reuses VLASH's existing LIBERO forward path and compares three scheduling modes:

- `case1_blocking_full_chunk`
- `case2_async_after_10`
- `case3_async_after_20`

The benchmark:

- runs real LIBERO simulation
- uses `task_id=0` and `init_state_idx=0` by default
- executes `4` runs per scenario
- treats run `1` as warmup
- averages runs `2-4`
- measures the wall-clock time from the first executed policy action to the completed `250`th policy action
- uses `async_wait=10` to simulate communication delay
- prints the per-run results and final averages directly to stdout

## Smoke Test

Use a short run first to make sure the benchmark script and worker path are healthy:

```bash
cd /workspace/vlash
python3 -m vlash.eval.benchmark_libero_async_latency \
  --config /workspace/vlash/examples/eval/libero_sim.yaml \
  --benchmark_steps 20 \
  --num_repeats 1 \
  --async_wait 10 \
  --task_id 0 \
  --init_state_idx 0
```

## Full Benchmark

Run the requested benchmark configuration:

```bash
cd /workspace/vlash
python3 -m vlash.eval.benchmark_libero_async_latency \
  --config /workspace/vlash/examples/eval/libero_sim.yaml \
  --benchmark_steps 250 \
  --num_repeats 4 \
  --async_wait 10 \
  --task_id 0 \
  --init_state_idx 0 \
  --case2_trigger_steps 10 \
  --case3_trigger_steps 20
```
