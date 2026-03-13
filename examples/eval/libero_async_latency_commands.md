# LIBERO Async-vs-NonAsync Latency Benchmark

This benchmark reuses VLASH's existing LIBERO forward path and compares three scheduling modes:

- `case1_blocking_full_chunk`
- `case2_async_after_10`
- `case3_async_after_20`

For `case2` and `case3`, the async scheduling now mirrors `run_libero_eval()`:

- `case2_async_after_10` means async trigger when `remaining_actions <= 10`
- `case3_async_after_20` means async trigger when `remaining_actions <= 20`

The benchmark:

- runs real LIBERO simulation
- uses `task_id=0` and `init_state_idx=0` by default
- executes `4` runs per scenario
- treats run `1` as warmup
- averages runs `2-4`
- measures the wall-clock time from the first executed policy action to the completed `250`th policy action
- if one LIBERO trial ends before reaching the requested action budget, keeps the single-trial result and imputes the remaining time from the observed per-step average
- uses `async_wait=10` to simulate communication delay
- prints the per-run results and final averages directly to stdout
- supports a safer single-scenario mode via `--scenario`
- can write/read a `case1` baseline JSON so `case2` and `case3` can compute speedup in separate commands

## Recommended Stable Flow

If `all` mode is unstable on your machine, run each scenario in a separate command.
This gives every scenario a fresh worker process and a fresh benchmark lifecycle.

Use a shared baseline file path such as:

```bash
/workspace/outputs/eval_outputs/libero_async_latency_case1_baseline.json
```

### 1. Run Case 1 And Write Baseline

```bash
cd /workspace/vlash
python3 -m vlash.eval.benchmark_libero_async_latency \
  --config /workspace/vlash/examples/eval/libero_sim.yaml \
  --scenario case1_blocking_full_chunk \
  --benchmark_steps 250 \
  --num_repeats 4 \
  --async_wait 10 \
  --task_id 0 \
  --init_state_idx 0 \
  --baseline_json /workspace/outputs/eval_outputs/libero_async_latency_case1_baseline.json
```

### 2. Run Case 2 And Compare Against Baseline

```bash
cd /workspace/vlash
python3 -m vlash.eval.benchmark_libero_async_latency \
  --config /workspace/vlash/examples/eval/libero_sim.yaml \
  --scenario case2_async_after_10 \
  --benchmark_steps 250 \
  --num_repeats 4 \
  --async_wait 10 \
  --task_id 0 \
  --init_state_idx 0 \
  --case2_trigger_steps 10 \
  --baseline_json /workspace/outputs/eval_outputs/libero_async_latency_case1_baseline.json
```

### 3. Run Case 3 And Compare Against Baseline

```bash
cd /workspace/vlash
python3 -m vlash.eval.benchmark_libero_async_latency \
  --config /workspace/vlash/examples/eval/libero_sim.yaml \
  --scenario case3_async_after_20 \
  --benchmark_steps 250 \
  --num_repeats 4 \
  --async_wait 10 \
  --task_id 0 \
  --init_state_idx 0 \
  --case3_trigger_steps 20 \
  --baseline_json /workspace/outputs/eval_outputs/libero_async_latency_case1_baseline.json
```

## Smoke Test

Use a short run first to make sure the benchmark script and worker path are healthy:

```bash
cd /workspace/vlash
python3 -m vlash.eval.benchmark_libero_async_latency \
  --config /workspace/vlash/examples/eval/libero_sim.yaml \
  --scenario case1_blocking_full_chunk \
  --benchmark_steps 20 \
  --num_repeats 1 \
  --async_wait 10 \
  --task_id 0 \
  --init_state_idx 0
```

## Full Benchmark In One Command

If you still want to try all three scenarios in one invocation:

```bash
cd /workspace/vlash
python3 -m vlash.eval.benchmark_libero_async_latency \
  --config /workspace/vlash/examples/eval/libero_sim.yaml \
  --scenario all \
  --benchmark_steps 250 \
  --num_repeats 4 \
  --async_wait 10 \
  --task_id 0 \
  --init_state_idx 0 \
  --case2_trigger_steps 10 \
  --case3_trigger_steps 20
```
