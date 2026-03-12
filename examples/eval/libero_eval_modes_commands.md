# LIBERO Four-Mode Evaluation Commands

Follow `C:/vlas/vlash/examples/eval/libero_setup.md` once before running these commands.

Set `policy.path` in `C:/vlas/vlash/examples/eval/libero_sim.yaml` to your checkpoint first.

## 1. No RTC + No VLASH
```bash
python -m vlash.eval.run_libero_eval --config C:/vlas/vlash/examples/eval/libero_sim.yaml --inference_mode none --rtc_prefix_k 0 --vlash_delay_k 1 --async_wait 5 --async_threshold 20
```

## 2. RTC Only
```bash
python -m vlash.eval.run_libero_eval --config C:/vlas/vlash/examples/eval/libero_sim.yaml --inference_mode rtc --rtc_prefix_k 4 --vlash_delay_k 1 --async_wait 5 --async_threshold 20
```

## 3. VLASH Only
```bash
python -m vlash.eval.run_libero_eval --config C:/vlas/vlash/examples/eval/libero_sim.yaml --inference_mode vlash --rtc_prefix_k 0 --vlash_delay_k 2 --async_wait 5 --async_threshold 20
```

## 4. RTC + VLASH
```bash
python -m vlash.eval.run_libero_eval --config C:/vlas/vlash/examples/eval/libero_sim.yaml --inference_mode rtc_vlash --rtc_prefix_k 4 --vlash_delay_k 2 --async_wait 5 --async_threshold 20
```

Notes:
- `k` is the unexecuted action index at inference trigger time.
- `async_wait` controls chunk handover: if `z < async_wait`, use `[async_wait:]`; if `z >= async_wait`, use `[z:]`.
- RTC prefix window uses old chunk range `[k, k + rtc_prefix_k - 1]`.
- VLASH state action index uses old chunk step `k + vlash_delay_k - 1` (clamped to valid range).
