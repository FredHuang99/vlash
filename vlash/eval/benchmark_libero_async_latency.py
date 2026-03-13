#!/usr/bin/env python

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
"""Benchmark async-vs-non-async LIBERO rollout latency without changing eval semantics."""

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from logging import getLogger

import numpy as np
import torch.multiprocessing as tmp

from vlash.eval.libero_utils import (
    build_libero_model_obs,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_images,
    quat2axisangle,
)
from vlash.eval.run_libero_eval import (
    _build_inference_meta,
    _env_check_success,
    _get_task_init_states,
    _load_eval_config,
    _sanitize_libero_action,
    _to_action_chunk,
    benchmark,
    run_inference_worker,
)

logger = getLogger(__name__)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    mode: str  # blocking | async
    trigger_steps: int | None = None


@dataclass
class RunMetrics:
    e2e_seconds: float
    steps_per_second: float
    executed_action_steps: int
    env_steps_total: int
    inference_requests: int
    ended_early: bool


SCENARIOS = (
    ScenarioSpec(
        name="case1_blocking_full_chunk",
        description=(
            "Execute the full current chunk, then request the next chunk; hold position with no-op while "
            "waiting for inference and 10-step communication delay."
        ),
        mode="blocking",
        trigger_steps=None,
    ),
    ScenarioSpec(
        name="case2_async_after_10",
        description=(
            "Launch async inference after executing 10 steps from the current chunk, keep executing the old "
            "chunk, then switch with async_wait handover semantics."
        ),
        mode="async",
        trigger_steps=10,
    ),
    ScenarioSpec(
        name="case3_async_after_20",
        description=(
            "Launch async inference after executing 20 steps from the current chunk, keep executing the old "
            "chunk, then switch with async_wait handover semantics."
        ),
        mode="async",
        trigger_steps=20,
    ),
)
CASE1_SCENARIO_NAME = "case1_blocking_full_chunk"


def _format_seconds(seconds: float) -> str:
    """Format a wall-clock duration for concise console output."""
    return f"{seconds:.3f}s"


def _build_scenarios(case2_trigger_steps: int, case3_trigger_steps: int) -> dict[str, ScenarioSpec]:
    """Build scenario map with CLI-overridden async trigger thresholds."""
    scenarios = (
        SCENARIOS[0],
        ScenarioSpec(
            name=SCENARIOS[1].name,
            description=SCENARIOS[1].description,
            mode=SCENARIOS[1].mode,
            trigger_steps=case2_trigger_steps,
        ),
        ScenarioSpec(
            name=SCENARIOS[2].name,
            description=SCENARIOS[2].description,
            mode=SCENARIOS[2].mode,
            trigger_steps=case3_trigger_steps,
        ),
    )
    return {scenario.name: scenario for scenario in scenarios}


def _resolve_baseline_json_path(cfg, baseline_json: str | None) -> str:
    """Resolve the optional case1 baseline file path."""
    if baseline_json:
        return baseline_json
    return os.path.join(cfg.local_log_dir, "libero_async_latency_case1_baseline.json")


def _write_case1_baseline(path: str, scenario_result: dict, *, cfg, task_id: int, init_state_idx: int) -> None:
    """Persist case1 summary for later case2/case3 speedup comparison."""
    baseline_dir = os.path.dirname(path)
    if baseline_dir:
        os.makedirs(baseline_dir, exist_ok=True)
    payload = {
        "scenario_name": scenario_result["name"],
        "avg_e2e_seconds": float(scenario_result["avg_e2e_seconds"]),
        "avg_steps_per_second": float(scenario_result["avg_steps_per_second"]),
        "benchmark_steps": int(scenario_result["benchmark_steps"]),
        "num_repeats": int(scenario_result["num_repeats"]),
        "async_wait": int(scenario_result["async_wait"]),
        "task_suite": cfg.task_suite,
        "task_id": int(task_id),
        "init_state_idx": int(init_state_idx),
        "policy_path": str(getattr(cfg.policy, "pretrained_path", "")),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {CASE1_SCENARIO_NAME} baseline to {path}")


def _load_case1_baseline(path: str) -> dict:
    """Load previously persisted case1 summary."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline file not found: {path}. Run --scenario {CASE1_SCENARIO_NAME} first to create it."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("scenario_name") != CASE1_SCENARIO_NAME:
        raise ValueError(
            f"Baseline file {path} does not contain {CASE1_SCENARIO_NAME}; "
            f"found {payload.get('scenario_name')!r} instead."
        )
    return payload


def _build_current_model_obs(obs: dict, cfg) -> dict:
    """Construct the exact LIBERO observation dict expected by the worker/policy."""
    img_dict = get_libero_images(obs, cfg.resolution)
    robot_state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return build_libero_model_obs(img_dict, robot_state, cfg.policy)


def _reset_env_to_init_state(env, initial_states, init_state_idx: int):
    """Reset the LIBERO environment to the benchmark's fixed initial state."""
    env.reset()
    return env.set_init_state(initial_states[init_state_idx])


def _make_benchmark_env(task, cfg):
    """Create a fresh LIBERO env for benchmark runs."""
    return get_libero_env(
        task,
        render_resolution=cfg.env_render_resolution,
        seed=cfg.env_seed,
    )


def _send_inference_request(parent_conn, cfg, current_chunk, current_k: int, current_model_obs: dict, task_desc: str):
    """Send one inference request to the shared worker process."""
    request_meta = _build_inference_meta(cfg, current_chunk, current_k)
    parent_conn.send((current_model_obs, task_desc, request_meta))


def _activate_chunk(chunk: np.ndarray, start_idx: int) -> tuple[np.ndarray, int]:
    """Activate a new chunk and clamp its starting index to a valid range."""
    chunk = _to_action_chunk(chunk)
    if chunk is None:
        raise RuntimeError("Worker returned an invalid action chunk shape during benchmark")
    start_idx = int(np.clip(start_idx, 0, chunk.shape[0]))
    return chunk, start_idx


def _receive_worker_chunk(parent_conn):
    """Receive and validate one worker response."""
    result = parent_conn.recv()
    if isinstance(result, Exception):
        raise result
    return _to_action_chunk(result)


def _drain_inflight_worker_result(parent_conn, wait_for_result: bool):
    """Ensure no stale worker result remains queued between benchmark runs."""
    if wait_for_result:
        result = _receive_worker_chunk(parent_conn)
        if result is None:
            raise RuntimeError("Worker returned an invalid action chunk shape while draining benchmark state")

    while parent_conn.poll():
        result = _receive_worker_chunk(parent_conn)
        if result is None:
            raise RuntimeError("Worker returned an invalid action chunk shape while draining benchmark state")


def _run_benchmark_once(
    cfg,
    parent_conn,
    task,
    task_description: str,
    initial_states,
    init_state_idx: int,
    benchmark_steps: int,
    async_wait: int,
    scenario: ScenarioSpec,
) -> RunMetrics:
    """Execute one benchmark run for a specific scheduling scenario.

    Timing starts right before the first policy action is executed and stops
    immediately after the 250th policy action completes. Communication-delay
    no-op steps still consume wall-clock time, but they do not count toward the
    250 executed policy actions.
    """
    env = None

    try:
        env, _ = _make_benchmark_env(task, cfg)
        obs = _reset_env_to_init_state(env, initial_states, init_state_idx)

        for _ in range(cfg.warmup_steps):
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.policy.type))
            if done or _env_check_success(env):
                logger.warning(
                    "Benchmark warmup ended the LIBERO trial early; stopping this run before timed rollout."
                )
                return RunMetrics(
                    e2e_seconds=0.0,
                    steps_per_second=0.0,
                    executed_action_steps=0,
                    env_steps_total=0,
                    inference_requests=0,
                    ended_early=True,
                )

        initial_model_obs = _build_current_model_obs(obs, cfg)
        _send_inference_request(parent_conn, cfg, None, 0, initial_model_obs, task_description)
        initial_chunk = _receive_worker_chunk(parent_conn)
        if initial_chunk is None:
            raise RuntimeError("Worker returned an invalid initial action chunk shape during benchmark")

        executed_action_steps = 0
        sim_step_index = 0
        inference_requests = 1

        current_chunk, current_k = _activate_chunk(initial_chunk, 0)
        current_chunk_activation_idx = current_k
        pending_new_chunk = None
        pending_new_start_idx = 0
        pending_activation_step = -1

        inference_pending = False
        request_sim_step = 0
        request_kind = ""
        request_sent_for_current_chunk = False
        done = False
        episode_success = False
        start_time = None

        while executed_action_steps < benchmark_steps and not done and not episode_success:
            current_model_obs = _build_current_model_obs(obs, cfg)
            chunk_len = 0 if current_chunk is None else int(current_chunk.shape[0])

            if (
                current_chunk is None
                and pending_new_chunk is None
                and not inference_pending
            ):
                _send_inference_request(parent_conn, cfg, current_chunk, current_k, current_model_obs, task_description)
                inference_requests += 1
                inference_pending = True
                request_sim_step = sim_step_index
                request_kind = "bootstrap"
            elif scenario.mode == "blocking":
                if (
                    current_chunk is not None
                    and current_k >= chunk_len
                    and pending_new_chunk is None
                    and not inference_pending
                ):
                    _send_inference_request(parent_conn, cfg, current_chunk, current_k, current_model_obs, task_description)
                    inference_requests += 1
                    inference_pending = True
                    request_sim_step = sim_step_index
                    request_kind = "blocking"
            else:
                executed_since_activation = max(0, int(current_k - current_chunk_activation_idx))
                if (
                    current_chunk is not None
                    and not request_sent_for_current_chunk
                    and pending_new_chunk is None
                    and not inference_pending
                    and executed_since_activation >= int(scenario.trigger_steps)
                ):
                    _send_inference_request(parent_conn, cfg, current_chunk, current_k, current_model_obs, task_description)
                    inference_requests += 1
                    inference_pending = True
                    request_sim_step = sim_step_index
                    request_kind = "async"
                    request_sent_for_current_chunk = True

            if inference_pending and parent_conn.poll():
                new_chunk = _receive_worker_chunk(parent_conn)
                if new_chunk is None:
                    raise RuntimeError("Worker returned an invalid action chunk shape during benchmark")

                inference_pending = False
                z = sim_step_index - request_sim_step
                if request_kind in {"bootstrap", "blocking"}:
                    if z >= async_wait:
                        current_chunk, current_k = _activate_chunk(new_chunk, 0)
                        current_chunk_activation_idx = current_k
                        request_sent_for_current_chunk = False
                    else:
                        pending_new_chunk = new_chunk
                        pending_new_start_idx = 0
                        pending_activation_step = request_sim_step + async_wait
                else:
                    if z >= async_wait:
                        current_chunk, current_k = _activate_chunk(new_chunk, z)
                        current_chunk_activation_idx = current_k
                        request_sent_for_current_chunk = False
                    else:
                        pending_new_chunk = new_chunk
                        pending_new_start_idx = min(async_wait, int(new_chunk.shape[0]))
                        pending_activation_step = request_sim_step + async_wait
                request_kind = ""

            if pending_new_chunk is not None and sim_step_index >= pending_activation_step:
                current_chunk, current_k = _activate_chunk(pending_new_chunk, pending_new_start_idx)
                current_chunk_activation_idx = current_k
                pending_new_chunk = None
                pending_new_start_idx = 0
                pending_activation_step = -1
                request_sent_for_current_chunk = False

            can_execute_chunk_action = current_chunk is not None and current_k < int(current_chunk.shape[0])
            if can_execute_chunk_action:
                action, _, _ = _sanitize_libero_action(current_chunk[current_k], cfg.policy.type)
                if start_time is None:
                    start_time = time.perf_counter()
                obs, _, done, _ = env.step(action)
                current_k += 1
                executed_action_steps += 1
            else:
                obs, _, done, _ = env.step(get_libero_dummy_action(cfg.policy.type))

            sim_step_index += 1
            episode_success = bool(done or _env_check_success(env))

        e2e_seconds = 0.0 if start_time is None else (time.perf_counter() - start_time)
        if inference_pending:
            _drain_inflight_worker_result(parent_conn, wait_for_result=True)
        else:
            _drain_inflight_worker_result(parent_conn, wait_for_result=False)

        ended_early = bool(executed_action_steps < benchmark_steps)
        if ended_early:
            logger.warning(
                "Benchmark run ended before reaching %d executed policy steps: got %d steps (done=%s, success=%s).",
                benchmark_steps,
                executed_action_steps,
                done,
                episode_success,
            )

        return RunMetrics(
            e2e_seconds=float(e2e_seconds),
            steps_per_second=float(executed_action_steps / e2e_seconds) if e2e_seconds > 0 else 0.0,
            executed_action_steps=int(executed_action_steps),
            env_steps_total=int(sim_step_index),
            inference_requests=int(inference_requests),
            ended_early=ended_early,
        )
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass


def _run_scenario(
    cfg,
    parent_conn,
    task,
    task_description: str,
    initial_states,
    init_state_idx: int,
    benchmark_steps: int,
    num_repeats: int,
    async_wait: int,
    scenario: ScenarioSpec,
) -> dict:
    """Run one benchmark scenario multiple times and summarize the results."""
    run_metrics: list[RunMetrics] = []
    print(f"\n[{scenario.name}] {scenario.description}")

    for run_idx in range(num_repeats):
        metrics = _run_benchmark_once(
            cfg=cfg,
            parent_conn=parent_conn,
            task=task,
            task_description=task_description,
            initial_states=initial_states,
            init_state_idx=init_state_idx,
            benchmark_steps=benchmark_steps,
            async_wait=async_wait,
            scenario=scenario,
        )
        run_metrics.append(metrics)
        warmup_marker = " (warmup)" if run_idx == 0 else ""
        print(
            f"  Run {run_idx + 1}/{num_repeats}{warmup_marker}: "
            f"e2e={_format_seconds(metrics.e2e_seconds)}, "
            f"steps/s={metrics.steps_per_second:.2f}, "
            f"executed_steps={metrics.executed_action_steps}/{benchmark_steps}, "
            f"env_steps={metrics.env_steps_total}, "
            f"requests={metrics.inference_requests}, ended_early={metrics.ended_early}"
        )

    measured_runs = run_metrics[1:] if len(run_metrics) > 1 else run_metrics
    avg_e2e_seconds = float(np.mean([m.e2e_seconds for m in measured_runs]))
    avg_steps_per_second = float(np.mean([m.steps_per_second for m in measured_runs]))

    print(
        f"  Avg over measured runs: e2e={_format_seconds(avg_e2e_seconds)}, "
        f"steps/s={avg_steps_per_second:.2f}"
    )

    return {
        "name": scenario.name,
        "description": scenario.description,
        "mode": scenario.mode,
        "trigger_steps": scenario.trigger_steps,
        "benchmark_steps": int(benchmark_steps),
        "num_repeats": int(num_repeats),
        "async_wait": int(async_wait),
        "runs": [asdict(m) for m in run_metrics],
        "warmup_run_index": 0,
        "measured_run_indices": list(range(1, len(run_metrics))),
        "avg_e2e_seconds": avg_e2e_seconds,
        "avg_steps_per_second": avg_steps_per_second,
    }


def _validate_task_selection(task_suite, task_id: int, init_state_idx: int, initial_states) -> None:
    """Validate benchmark task and init-state indices."""
    if task_id < 0 or task_id >= task_suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {task_suite.n_tasks - 1}], got {task_id}")
    if init_state_idx < 0 or init_state_idx >= len(initial_states):
        raise ValueError(
            f"init_state_idx must be in [0, {len(initial_states) - 1}], got {init_state_idx}"
        )


def benchmark_libero_async_latency(
    config_path: str,
    task_id: int,
    init_state_idx: int,
    benchmark_steps: int,
    num_repeats: int,
    async_wait: int,
    case2_trigger_steps: int,
    case3_trigger_steps: int,
    scenario_name: str,
    baseline_json: str | None,
):
    """Benchmark three LIBERO scheduling scenarios and compare e2e rollout latency."""
    if benchmark_steps <= 0:
        raise ValueError("benchmark_steps must be > 0")
    if num_repeats <= 0:
        raise ValueError("num_repeats must be > 0")
    if async_wait < 0:
        raise ValueError("async_wait must be >= 0")
    if case2_trigger_steps <= 0:
        raise ValueError("case2_trigger_steps must be > 0")
    if case3_trigger_steps <= 0:
        raise ValueError("case3_trigger_steps must be > 0")

    cfg = _load_eval_config(config_path)
    tmp.set_start_method("spawn", force=True)
    scenarios_by_name = _build_scenarios(case2_trigger_steps, case3_trigger_steps)
    valid_scenarios = {"all", *scenarios_by_name.keys()}
    if scenario_name not in valid_scenarios:
        raise ValueError(f"scenario must be one of {sorted(valid_scenarios)}, got {scenario_name!r}")
    baseline_path = _resolve_baseline_json_path(cfg, baseline_json)
    selected_scenarios = (
        tuple(scenarios_by_name.values())
        if scenario_name == "all"
        else (scenarios_by_name[scenario_name],)
    )

    parent_conn, child_conn = mp.Pipe()
    worker_process = mp.Process(target=run_inference_worker, args=(config_path, child_conn))
    worker_process.start()

    logger.info("Waiting for inference worker to load model for benchmark...")
    ready_msg = parent_conn.recv()
    if ready_msg != "READY":
        raise RuntimeError(f"Worker failed to initialize: {ready_msg}")
    logger.info("Benchmark worker ready.")

    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite]()
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise ValueError(f"task_id must be in [0, {task_suite.n_tasks - 1}], got {task_id}")
        task = task_suite.get_task(task_id)
        initial_states = _get_task_init_states(task_suite, task_id)
        _validate_task_selection(task_suite, task_id, init_state_idx, initial_states)

        env, task_description = _make_benchmark_env(
            task,
            cfg,
        )
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        env = None

        results = {
            "config_path": config_path,
            "task_suite": cfg.task_suite,
            "task_id": int(task_id),
            "init_state_idx": int(init_state_idx),
            "task_description": task_description,
            "benchmark_steps": int(benchmark_steps),
            "num_repeats": int(num_repeats),
            "async_wait": int(async_wait),
            "case2_trigger_steps": int(case2_trigger_steps),
            "case3_trigger_steps": int(case3_trigger_steps),
            "policy_path": str(getattr(cfg.policy, "pretrained_path", "")),
            "scenarios": {},
        }

        baseline_avg = None
        if scenario_name != "all" and scenario_name != CASE1_SCENARIO_NAME:
            baseline_payload = _load_case1_baseline(baseline_path)
            baseline_avg = float(baseline_payload["avg_e2e_seconds"])
            print(
                f"Loaded {CASE1_SCENARIO_NAME} baseline from {baseline_path}: "
                f"avg_e2e={_format_seconds(baseline_avg)}, "
                f"avg_steps/s={baseline_payload['avg_steps_per_second']:.2f}"
            )

        for scenario in selected_scenarios:
            scenario_result = _run_scenario(
                cfg=cfg,
                parent_conn=parent_conn,
                task=task,
                task_description=task_description,
                initial_states=initial_states,
                init_state_idx=init_state_idx,
                benchmark_steps=benchmark_steps,
                num_repeats=num_repeats,
                async_wait=async_wait,
                scenario=scenario,
            )
            results["scenarios"][scenario.name] = scenario_result

            if scenario.name == CASE1_SCENARIO_NAME:
                baseline_avg = scenario_result["avg_e2e_seconds"]
                if scenario_name != "all":
                    _write_case1_baseline(
                        baseline_path,
                        scenario_result,
                        cfg=cfg,
                        task_id=task_id,
                        init_state_idx=init_state_idx,
                    )
            elif baseline_avg is not None:
                speedup = float(baseline_avg / scenario_result["avg_e2e_seconds"])
                scenario_result["speedup_vs_case1_blocking_full_chunk"] = speedup
                print(
                    f"  Speedup vs case1_blocking_full_chunk: "
                    f"{speedup:.3f}x"
                )

        print("\nBenchmark summary:")
        print(f"  Task suite: {results['task_suite']}")
        print(f"  Task id: {results['task_id']}")
        print(f"  Init state idx: {results['init_state_idx']}")
        print(f"  Policy path: {results['policy_path']}")
        print(f"  Scenario mode: {scenario_name}")
        for scenario_name, scenario_result in results["scenarios"].items():
            print(
                f"  {scenario_name}: avg_e2e={_format_seconds(scenario_result['avg_e2e_seconds'])}, "
                f"avg_steps/s={scenario_result['avg_steps_per_second']:.2f}"
            )
            if "speedup_vs_case1_blocking_full_chunk" in scenario_result:
                print(
                    "    speedup_vs_case1_blocking_full_chunk="
                    f"{scenario_result['speedup_vs_case1_blocking_full_chunk']:.3f}x"
                )

    finally:
        try:
            parent_conn.send("TERMINATE")
        except Exception:
            pass
        worker_process.join(timeout=5)
        if worker_process.is_alive():
            worker_process.terminate()


def main():
    parser = argparse.ArgumentParser(description="Benchmark LIBERO async-vs-non-async rollout latency")
    parser.add_argument("--config", type=str, required=True, help="Path to LIBERO eval config YAML")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario to run: all, case1_blocking_full_chunk, case2_async_after_10, or case3_async_after_20",
    )
    parser.add_argument("--task_id", type=int, default=0, help="LIBERO task index to benchmark")
    parser.add_argument("--init_state_idx", type=int, default=0, help="Initial state index for the benchmark task")
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=250,
        help="Executed policy-action steps per run (timed from action 1 through action N)",
    )
    parser.add_argument("--num_repeats", type=int, default=4, help="Runs per scenario (first is warmup)")
    parser.add_argument("--async_wait", type=int, default=10, help="Communication-delay simulation in env steps")
    parser.add_argument(
        "--case2_trigger_steps",
        type=int,
        default=10,
        help="Steps executed before launching async inference in case2",
    )
    parser.add_argument(
        "--case3_trigger_steps",
        type=int,
        default=20,
        help="Steps executed before launching async inference in case3",
    )
    parser.add_argument(
        "--baseline_json",
        type=str,
        default=None,
        help=(
            "Optional case1 baseline file path. In single-scenario mode, case1 writes this file and "
            "case2/case3 read it to compute speedup."
        ),
    )
    args = parser.parse_args()

    benchmark_libero_async_latency(
        config_path=args.config,
        task_id=args.task_id,
        init_state_idx=args.init_state_idx,
        benchmark_steps=args.benchmark_steps,
        num_repeats=args.num_repeats,
        async_wait=args.async_wait,
        case2_trigger_steps=args.case2_trigger_steps,
        case3_trigger_steps=args.case3_trigger_steps,
        scenario_name=args.scenario,
        baseline_json=args.baseline_json,
    )


if __name__ == "__main__":
    main()
