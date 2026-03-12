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
LIBERO asynchronous evaluation script for VLASH policies.

Implements a multi-process architecture to bypass GIL and overlap simulation
execution with VLA policy inference, with dynamic chunk offset logic.
"""

import argparse
from contextlib import contextmanager
import inspect
import multiprocessing as mp
import os
import pickle
import time
from logging import getLogger

import numpy as np
import torch
import torch.multiprocessing as tmp
import tqdm
import yaml

try:
    from lerobot.policies.factory import make_pre_post_processors
except ImportError:
    make_pre_post_processors = None

try:
    from libero.libero import benchmark
except ModuleNotFoundError as exc:
    if exc.name != "libero":
        raise
    raise ModuleNotFoundError(
        "LIBERO is not installed in the current Python environment. "
        "Install LIBERO with `python3 -m pip install /path/to/LIBERO`, then install "
        "VLASH's extra LIBERO dependencies with "
        "`python3 -m pip install -r /path/to/vlash/examples/eval/libero_requirements.txt`. "
        "Verify from a neutral directory such as `/tmp`, not from inside the LIBERO source tree."
    ) from exc

from lerobot.utils.utils import get_safe_torch_device

from vlash.configs import LiberoEvalConfig
from vlash.policies.factory import get_policy_class
from vlash.eval.libero_utils import (
    build_libero_model_obs,
    get_libero_dummy_action,
    get_libero_debug_images,
    get_libero_env,
    get_libero_images,
    quat2axisangle,
    save_alignment_debug_artifacts,
    save_rollout_video,
)

logger = getLogger(__name__)
VALID_INFERENCE_MODES = {"none", "rtc", "vlash", "rtc_vlash"}
OFFICIAL_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _str_to_bool(value: str) -> bool:
    """Parse boolean CLI argument."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _to_action_chunk(action_chunk) -> np.ndarray | None:
    """Convert arbitrary action-chunk payload to [T, A] float32 array."""
    if action_chunk is None:
        return None

    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2:
        return None
    return chunk


def _remaining_actions(action_chunk: np.ndarray | None, k: int) -> int:
    """Return number of executable actions remaining in chunk from index k."""
    if action_chunk is None:
        return 0
    return max(0, int(action_chunk.shape[0] - k))


def _map_action_to_state(prev_action: np.ndarray, target_state_dim: int) -> np.ndarray:
    """Map action vector to state dimension by truncation or tail-zero-padding."""
    action = np.asarray(prev_action, dtype=np.float32).reshape(-1)
    if action.shape[0] >= target_state_dim:
        return action[:target_state_dim].copy()

    mapped = np.zeros(target_state_dim, dtype=np.float32)
    mapped[: action.shape[0]] = action
    return mapped


def _extract_rtc_prefix_actions(
    old_chunk: np.ndarray | None,
    k: int,
    rtc_prefix_k: int,
) -> np.ndarray | None:
    """Extract RTC prefix window [k, k + rtc_prefix_k - 1] from old chunk."""
    if old_chunk is None or rtc_prefix_k <= 0:
        return None

    start = max(int(k), 0)
    if start >= old_chunk.shape[0]:
        return None

    end = min(start + int(rtc_prefix_k), old_chunk.shape[0])
    if end <= start:
        return None
    return old_chunk[start:end].copy()


def _build_vlash_state(
    current_state: np.ndarray,
    old_chunk: np.ndarray | None,
    k: int,
    vlash_delay_k: int,
) -> np.ndarray:
    """Build VLASH state from old chunk action at index k + vlash_delay_k - 1."""
    if old_chunk is None or old_chunk.shape[0] == 0:
        return current_state

    raw_index = int(k) + int(vlash_delay_k) - 1
    action_index = int(np.clip(raw_index, 0, old_chunk.shape[0] - 1))
    return _map_action_to_state(old_chunk[action_index], target_state_dim=current_state.shape[0])


def _set_policy_prefix_steps(policy, prefix_steps: int) -> tuple[int | None, int | None]:
    """Temporarily override RTC prefix steps on policy/model config."""
    old_policy_steps = None
    old_model_steps = None

    if hasattr(policy, "config") and hasattr(policy.config, "inference_prefix_mask_steps"):
        old_policy_steps = int(policy.config.inference_prefix_mask_steps)
        policy.config.inference_prefix_mask_steps = int(prefix_steps)

    if hasattr(policy, "model") and hasattr(policy.model, "config"):
        model_cfg = policy.model.config
        if hasattr(model_cfg, "inference_prefix_mask_steps"):
            old_model_steps = int(model_cfg.inference_prefix_mask_steps)
            model_cfg.inference_prefix_mask_steps = int(prefix_steps)

    return old_policy_steps, old_model_steps


def _restore_policy_prefix_steps(policy, old_policy_steps: int | None, old_model_steps: int | None) -> None:
    """Restore RTC prefix steps after temporary override."""
    if old_policy_steps is not None and hasattr(policy, "config"):
        policy.config.inference_prefix_mask_steps = old_policy_steps

    if old_model_steps is not None and hasattr(policy, "model") and hasattr(policy.model, "config"):
        policy.model.config.inference_prefix_mask_steps = old_model_steps


def _validate_runtime_cfg(cfg: LiberoEvalConfig) -> None:
    """Validate config after applying CLI overrides."""
    if cfg.inference_mode not in VALID_INFERENCE_MODES:
        raise ValueError(
            f"inference_mode must be one of {sorted(VALID_INFERENCE_MODES)}, got '{cfg.inference_mode}'"
        )
    if cfg.async_threshold < 0:
        raise ValueError("async_threshold must be >= 0")
    if cfg.async_wait < 0:
        raise ValueError("async_wait must be >= 0")
    if cfg.rtc_prefix_k < 0:
        raise ValueError("rtc_prefix_k must be >= 0")
    if cfg.vlash_delay_k < 0:
        raise ValueError("vlash_delay_k must be >= 0")
    if cfg.max_steps is not None and cfg.max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if cfg.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be > 0")
    if cfg.resolution <= 0:
        raise ValueError("resolution must be > 0")
    if cfg.env_render_resolution <= 0:
        raise ValueError("env_render_resolution must be > 0")
    if cfg.replan_steps <= 0:
        raise ValueError("replan_steps must be > 0")
    if cfg.env_seed < 0:
        raise ValueError("env_seed must be >= 0")


def _build_inference_meta(cfg: LiberoEvalConfig, old_chunk: np.ndarray | None, k: int) -> dict:
    """Build per-request inference metadata sent from host to worker."""
    return {
        "inference_mode": cfg.inference_mode,
        "rtc_prefix_k": int(cfg.rtc_prefix_k),
        "vlash_delay_k": int(cfg.vlash_delay_k),
        "old_chunk": old_chunk.copy() if old_chunk is not None else None,
        "k": int(k),
    }


def _format_duration_s(duration_s: float) -> str:
    """Format wall-clock duration in seconds for logs."""
    return f"{duration_s:.2f}s"


def _env_check_success(env) -> bool:
    """Handle LIBERO env API differences across versions."""
    if hasattr(env, "check_success"):
        return bool(env.check_success())
    if hasattr(env, "_check_success"):
        return bool(env._check_success())

    logger.warning(
        "LIBERO env %s exposes neither check_success() nor _check_success(); "
        "treating the episode as not yet successful.",
        type(env).__name__,
    )
    return False


def _sanitize_libero_action(action, model_family: str) -> tuple[np.ndarray, bool, float]:
    """Keep actions finite and within robosuite's normalized control range."""
    expected_action = np.asarray(get_libero_dummy_action(model_family), dtype=np.float32)
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    if action.shape[0] != expected_action.shape[0]:
        logger.warning(
            "Policy emitted action_dim=%d but LIBERO expects %d; adapting action shape.",
            action.shape[0],
            expected_action.shape[0],
        )
        adapted = expected_action.copy()
        copy_dim = min(action.shape[0], adapted.shape[0])
        adapted[:copy_dim] = action[:copy_dim]
        action = adapted

    if not np.all(np.isfinite(action)):
        logger.warning(
            "Policy emitted non-finite LIBERO action %s; falling back to dummy no-op action.",
            action,
        )
        return expected_action.copy(), True, float("inf")

    max_abs_before_clip = float(np.max(np.abs(action)))
    clipped = np.clip(action, -1.0, 1.0)
    was_clipped = not np.allclose(action, clipped, atol=1e-6)
    # Tiny excursions like 1.0001 are usually just floating-point/unnormalization noise.
    # Only surface a warning when the action is meaningfully outside robosuite's range.
    if max_abs_before_clip > 1.01:
        logger.warning(
            "Clipped LIBERO action outside [-1, 1]; max_abs_before_clip=%.4f",
            max_abs_before_clip,
        )
    return clipped.astype(np.float32, copy=False), was_clipped, max_abs_before_clip


def _resolve_max_steps(task_suite: str, override_max_steps: int | None) -> int:
    """Use the official openpi suite budget unless the caller explicitly overrides it."""
    if override_max_steps is not None:
        return int(override_max_steps)
    if task_suite not in OFFICIAL_MAX_STEPS:
        raise ValueError(
            f"Unknown task suite '{task_suite}' with no official max_steps mapping. "
            "Set max_steps explicitly in the config or CLI."
        )
    return int(OFFICIAL_MAX_STEPS[task_suite])


def _truncate_sync_replan_window(action_chunk: np.ndarray, replan_steps: int) -> tuple[np.ndarray, int]:
    """Keep the full chunk for conditioning, but only execute the first replan window before replanning."""
    chunk = _to_action_chunk(action_chunk)
    if chunk is None:
        raise RuntimeError("Worker returned invalid action chunk shape")
    if chunk.shape[0] < replan_steps:
        raise RuntimeError(
            f"Policy predicted only {chunk.shape[0]} steps, but LIBERO sync eval wants at least {replan_steps}"
        )
    return chunk, int(replan_steps)


def _collect_chunk_clip_stats(action_chunk: np.ndarray, model_family: str) -> tuple[np.ndarray, int, float]:
    """Summarize how much clipping the first executed action chunk needs."""
    clipped_actions = []
    clip_count = 0
    max_abs_before_clip = 0.0
    for action in np.asarray(action_chunk, dtype=np.float32):
        clipped, was_clipped, max_abs = _sanitize_libero_action(action, model_family)
        clipped_actions.append(clipped)
        clip_count += int(was_clipped)
        max_abs_before_clip = max(max_abs_before_clip, max_abs)
    return np.stack(clipped_actions, axis=0), clip_count, max_abs_before_clip


def _should_dump_alignment_debug(cfg: LiberoEvalConfig, episode_idx: int) -> bool:
    """Keep debug output small: dump only the first trial per task when enabled."""
    return bool(cfg.debug_alignment and episode_idx == 0)


@contextmanager
def _torch_load_weights_only_disabled():
    """Temporarily force torch.load(..., weights_only=False) for legacy LIBERO assets."""
    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _get_task_init_states(task_suite, task_id: int):
    """Load LIBERO task init states with a PyTorch 2.6 compatibility fallback."""
    try:
        return task_suite.get_task_init_states(task_id)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise

        logger.warning(
            "LIBERO init states triggered torch.load(weights_only=True) on PyTorch >= 2.6; "
            "retrying task %d with weights_only=False. Use this only with trusted LIBERO assets.",
            task_id,
        )
        with _torch_load_weights_only_disabled():
            return task_suite.get_task_init_states(task_id)


def _load_eval_config(config_path: str, cli_overrides: dict | None = None) -> LiberoEvalConfig:
    """Load LIBERO eval config from YAML and apply CLI overrides."""
    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    if not isinstance(config_data, dict):
        raise ValueError(f"Expected mapping in config file, got {type(config_data).__name__}")

    cfg = LiberoEvalConfig(**config_data)
    _apply_cli_overrides(cfg, cli_overrides)
    return cfg


def _postprocess_action_chunk(action_chunk) -> np.ndarray:
    """Convert a policy/postprocessor action payload to a host-side numpy chunk."""
    if isinstance(action_chunk, dict):
        for key in ("action", "actions"):
            if key in action_chunk:
                action_chunk = action_chunk[key]
                break

    if isinstance(action_chunk, torch.Tensor):
        action_chunk = action_chunk.detach().cpu().numpy()
    else:
        action_chunk = np.asarray(action_chunk, dtype=np.float32)

    if action_chunk.ndim == 3 and action_chunk.shape[0] == 1:
        action_chunk = action_chunk[0]

    return np.asarray(action_chunk, dtype=np.float32)


def _make_lerobot_processors(policy_config, pretrained_path: str):
    """Best-effort creation of LeRobot pre/post processors across API variants."""
    if make_pre_post_processors is None:
        return None, None

    attempts = (
        lambda: make_pre_post_processors(policy_config, pretrained_path=pretrained_path),
        lambda: make_pre_post_processors(policy_config, pretrained_name_or_path=pretrained_path),
        lambda: make_pre_post_processors(policy_config, pretrained_path),
    )

    last_error = None
    for build in attempts:
        try:
            return build()
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    return None, None


def run_inference_worker(config_path, pipe_conn):
    """Worker process that holds the GPU model and performs inference."""
    try:
        cfg = _load_eval_config(config_path)
        device = get_safe_torch_device(cfg.policy.device)

        # Load policy
        policy_class = get_policy_class(cfg.policy.type)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=cfg.policy.pretrained_path,
            config=cfg.policy,
        )
        policy.to(device)
        policy.eval()
        supports_prev_action_chunk = "prev_action_chunk" in inspect.signature(policy.predict_action_chunk).parameters
        preprocessor = None
        postprocessor = None
        uses_official_processors = False

        if make_pre_post_processors is not None:
            try:
                preprocessor, postprocessor = _make_lerobot_processors(
                    policy.config,
                    cfg.policy.pretrained_path,
                )
                uses_official_processors = preprocessor is not None and postprocessor is not None
            except Exception as exc:
                logger.warning("Failed to initialize LeRobot pre/post processors: %s", exc)

        if getattr(policy, "_missing_internal_normalization_stats", None) and not uses_official_processors:
            raise RuntimeError(
                "Checkpoint is missing normalization statistics required for PI0.5 inference, "
                "and LeRobot pre/post processors could not be initialized."
            )

        logger.info(f"Worker: Model loaded on {device}")

        # Signal ready
        pipe_conn.send("READY")

        while True:
            # Wait for inference request
            if not pipe_conn.poll(0.01):
                time.sleep(0.005)
                continue

            msg = pipe_conn.recv()
            if msg == "TERMINATE":
                break

            # Backward-compatible unpacking: (obs, task) or (obs, task, meta)
            if isinstance(msg, tuple) and len(msg) == 3:
                obs, task_description, meta = msg
            elif isinstance(msg, tuple) and len(msg) == 2:
                obs, task_description = msg
                meta = {}
            else:
                raise ValueError(f"Invalid message payload in worker: {type(msg)}")

            inference_mode = str(meta.get("inference_mode", "none")).lower()
            if inference_mode not in VALID_INFERENCE_MODES:
                inference_mode = "none"
            use_rtc = inference_mode in {"rtc", "rtc_vlash"}
            use_vlash = inference_mode in {"vlash", "rtc_vlash"}

            k = int(meta.get("k", 0))
            rtc_prefix_k = int(meta.get("rtc_prefix_k", 0))
            vlash_delay_k = int(meta.get("vlash_delay_k", 1))
            old_chunk = _to_action_chunk(meta.get("old_chunk"))

            # Prepare observation dict exactly as VLA expects.
            obs_np = {name: np.asarray(value) for name, value in obs.items()}
            if use_vlash and "observation.state" in obs_np:
                state_np = np.asarray(obs_np["observation.state"], dtype=np.float32)
                obs_np["observation.state"] = _build_vlash_state(state_np, old_chunk, k, vlash_delay_k)

            if uses_official_processors:
                model_obs = {}
                for key, value in obs_np.items():
                    if isinstance(value, np.ndarray):
                        model_obs[key] = torch.from_numpy(value)
                    else:
                        model_obs[key] = value
                model_obs["task"] = task_description
                model_obs = dict(preprocessor(model_obs))
                model_obs["_already_preprocessed"] = True
            else:
                model_obs = {}
                for key, value in obs_np.items():
                    # Add batch dimension and move to device
                    model_obs[key] = torch.from_numpy(value).unsqueeze(0).to(device)

                # Fake language tasks/tools format
                model_obs["task"] = [task_description]

            kwargs = {}
            effective_prefix_steps = 0
            if use_rtc and supports_prev_action_chunk:
                if uses_official_processors and getattr(policy, "_missing_internal_normalization_stats", None):
                    raise RuntimeError(
                        "RTC prefix conditioning is not supported for this checkpoint when using external "
                        "LeRobot processors, because internal target normalizers are unavailable."
                    )
                rtc_prefix_actions = _extract_rtc_prefix_actions(old_chunk, k, rtc_prefix_k)
                if rtc_prefix_actions is not None and rtc_prefix_actions.shape[0] > 0:
                    rtc_prefix_tensor = torch.from_numpy(rtc_prefix_actions).unsqueeze(0).to(device)
                    if hasattr(policy, "normalize_targets"):
                        rtc_prefix_tensor = policy.normalize_targets({"action": rtc_prefix_tensor})["action"]
                    kwargs["prev_action_chunk"] = rtc_prefix_tensor
                    effective_prefix_steps = int(rtc_prefix_tensor.shape[1])

            old_policy_steps = None
            old_model_steps = None
            if effective_prefix_steps > 0:
                old_policy_steps, old_model_steps = _set_policy_prefix_steps(policy, effective_prefix_steps)

            with torch.inference_mode():
                try:
                    # Support basic unnormalized observation pass
                    action_chunk = policy.predict_action_chunk(model_obs, **kwargs)
                finally:
                    _restore_policy_prefix_steps(policy, old_policy_steps, old_model_steps)

            if uses_official_processors:
                action_chunk = postprocessor(action_chunk)

            # Remove batch dim and send back to host as numpy array
            action_chunk_np = _postprocess_action_chunk(action_chunk)
            pipe_conn.send(action_chunk_np)

    except Exception as e:
        logger.error(f"Inference worker failed: {e}")
        try:
            pipe_conn.send(e)
        except Exception:
            pass
        raise e


def _apply_cli_overrides(cfg: LiberoEvalConfig, cli_overrides: dict | None) -> None:
    """Apply command-line overrides to parsed config."""
    if not cli_overrides:
        return

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    _validate_runtime_cfg(cfg)


def eval_libero(config_path: str, cli_overrides: dict | None = None):
    """Main evaluation loop."""
    cfg = _load_eval_config(config_path, cli_overrides)
    resolved_max_steps = _resolve_max_steps(cfg.task_suite, cfg.max_steps)

    tmp.set_start_method("spawn", force=True)

    # Start inference worker
    parent_conn, child_conn = mp.Pipe()
    worker_process = mp.Process(target=run_inference_worker, args=(config_path, child_conn))
    worker_process.start()

    logger.info("Waiting for inference worker to load model...")
    ready_msg = parent_conn.recv()
    if ready_msg != "READY":
        raise RuntimeError(f"Worker failed to initialize: {ready_msg}")
    logger.info("Worker ready. Starting evaluation.")

    try:
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        log_file_path = os.path.join(cfg.local_log_dir, f"eval_{cfg.task_suite}.txt")
        log_file = open(log_file_path, "w")

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite]()
        num_tasks_in_suite = task_suite.n_tasks
        debug_root = os.path.join(cfg.local_log_dir, "alignment_debug")
        if cfg.debug_alignment:
            os.makedirs(debug_root, exist_ok=True)

        logger.info(f"Task suite: {cfg.task_suite} ({num_tasks_in_suite} tasks)")
        logger.info(
            "Inference mode=%s, rtc_prefix_k=%d, vlash_delay_k=%d, async_wait=%d, async_threshold=%d",
            cfg.inference_mode,
            cfg.rtc_prefix_k,
            cfg.vlash_delay_k,
            cfg.async_wait,
            cfg.async_threshold,
        )
        logger.info(
            "LIBERO eval settings: resolution=%d, env_render_resolution=%d, replan_steps=%d, warmup_steps=%d, "
            "max_steps=%d%s, env_seed=%d, async_mode=%s",
            cfg.resolution,
            cfg.env_render_resolution,
            cfg.replan_steps,
            cfg.warmup_steps,
            resolved_max_steps,
            " (official default)" if cfg.max_steps is None else " (explicit override)",
            cfg.env_seed,
            cfg.async_mode,
        )
        if cfg.debug_alignment:
            logger.info("Alignment debug bundles will be written to %s", debug_root)

        total_episodes = 0
        total_successes = 0

        all_inference_latencies = []
        all_idle_steps = []

        for task_id in range(num_tasks_in_suite):
            task = task_suite.get_task(task_id)
            initial_states = _get_task_init_states(task_suite, task_id)
            env, task_description = get_libero_env(
                task,
                render_resolution=cfg.env_render_resolution,
                seed=cfg.env_seed,
            )

            task_episodes = 0
            task_successes = 0
            task_trial_durations_s = []
            task_success_durations_s = []

            for episode_idx in range(cfg.num_trials_per_task):
                logger.info(
                    "Task %d: %s | Trial %d/%d",
                    task_id,
                    task_description,
                    episode_idx + 1,
                    cfg.num_trials_per_task,
                )
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                trial_start_time = time.perf_counter()

                t = 0
                replay_images = []

                # Host-side chunk state (old chunk + current unexecuted index k).
                current_chunk = None
                current_k = 0
                current_window_end = 0

                inference_pending = False
                inference_start_time = 0.0
                inference_request_frame = 0

                pending_new_chunk = None
                pending_new_start_idx = 0
                pending_activation_frame = -1

                episode_idle_steps = 0
                episode_clip_count = 0
                episode_replans = 0
                episode_max_abs_action = 0.0
                done = False
                debug_saved = False
                debug_request_snapshot = None

                pbar = tqdm.tqdm(total=resolved_max_steps)

                while t < resolved_max_steps + cfg.warmup_steps:
                    # 1. Warmup simulation
                    if t < cfg.warmup_steps:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.policy.type))
                        img_dict = get_libero_images(obs, cfg.resolution)
                        replay_images.append(img_dict["observation.image"])
                        t += 1
                        continue

                    effective_t = t - cfg.warmup_steps

                    # 2. Extract observations
                    img_dict = get_libero_images(obs, cfg.resolution)
                    replay_images.append(img_dict["observation.image"])

                    robot_state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ).astype(np.float32)

                    current_model_obs = build_libero_model_obs(img_dict, robot_state, cfg.policy)
                    remaining_actions = _remaining_actions(current_chunk, current_k)
                    sync_window_remaining = max(0, int(current_window_end - current_k))

                    # 3. Trigger async inference.
                    if cfg.async_mode and not inference_pending and remaining_actions <= cfg.async_threshold:
                        request_meta = _build_inference_meta(cfg, current_chunk, current_k)
                        parent_conn.send((current_model_obs, task_description, request_meta))
                        episode_replans += 1
                        inference_pending = True
                        inference_start_time = time.time()
                        inference_request_frame = effective_t
                        if _should_dump_alignment_debug(cfg, episode_idx) and not debug_saved:
                            debug_request_snapshot = {
                                "images": get_libero_debug_images(obs, cfg.resolution),
                                "state": robot_state.copy(),
                            }

                    # Sync fallback: request a fresh chunk every replan window, matching openpi's cadence.
                    if not cfg.async_mode and sync_window_remaining == 0:
                        request_meta = _build_inference_meta(cfg, current_chunk, current_k)
                        sync_request_start = time.time()
                        parent_conn.send((current_model_obs, task_description, request_meta))
                        episode_replans += 1
                        if _should_dump_alignment_debug(cfg, episode_idx) and not debug_saved:
                            debug_request_snapshot = {
                                "images": get_libero_debug_images(obs, cfg.resolution),
                                "state": robot_state.copy(),
                            }
                        sync_chunk = parent_conn.recv()
                        if isinstance(sync_chunk, Exception):
                            raise sync_chunk
                        all_inference_latencies.append(time.time() - sync_request_start)
                        current_chunk, current_window_end = _truncate_sync_replan_window(sync_chunk, cfg.replan_steps)
                        current_k = 0
                        remaining_actions = _remaining_actions(current_chunk, current_k)
                        if debug_request_snapshot is not None and not debug_saved:
                            debug_chunk = current_chunk[:current_window_end]
                            clipped_chunk, chunk_clip_count, chunk_max_abs = _collect_chunk_clip_stats(
                                debug_chunk,
                                cfg.policy.type,
                            )
                            debug_dir = save_alignment_debug_artifacts(
                                debug_root=debug_root,
                                task_id=task_id,
                                trial_idx=episode_idx,
                                task_description=task_description,
                                debug_images=debug_request_snapshot["images"],
                                robot_state=debug_request_snapshot["state"],
                                raw_action_chunk=debug_chunk,
                                clipped_action_chunk=clipped_chunk,
                                clip_count=chunk_clip_count,
                                max_abs_before_clip=chunk_max_abs,
                            )
                            logger.info("Saved alignment debug bundle to %s", debug_dir)
                            debug_saved = True
                            debug_request_snapshot = None

                    # 4. Check async inference completion.
                    if inference_pending and parent_conn.poll():
                        new_chunk = parent_conn.recv()
                        if isinstance(new_chunk, Exception):
                            raise new_chunk

                        new_chunk = _to_action_chunk(new_chunk)
                        if new_chunk is None:
                            raise RuntimeError("Worker returned invalid action chunk shape")

                        latency = time.time() - inference_start_time
                        all_inference_latencies.append(latency)
                        inference_pending = False

                        # Async boundary semantics:
                        # z < async_wait  -> activate at [async_wait:]
                        # z >= async_wait -> immediate activation at [z:]
                        z = effective_t - inference_request_frame
                        if z >= cfg.async_wait:
                            current_chunk = new_chunk
                            current_window_end = int(new_chunk.shape[0])
                            current_k = min(max(int(z), 0), int(new_chunk.shape[0]))
                            pending_new_chunk = None
                            pending_new_start_idx = 0
                            pending_activation_frame = -1
                        else:
                            pending_new_chunk = new_chunk
                            pending_new_start_idx = min(int(cfg.async_wait), int(new_chunk.shape[0]))
                            pending_activation_frame = inference_request_frame + int(cfg.async_wait)
                        if debug_request_snapshot is not None and not debug_saved:
                            debug_chunk = new_chunk[: min(cfg.replan_steps, int(new_chunk.shape[0]))]
                            clipped_chunk, chunk_clip_count, chunk_max_abs = _collect_chunk_clip_stats(
                                debug_chunk,
                                cfg.policy.type,
                            )
                            debug_dir = save_alignment_debug_artifacts(
                                debug_root=debug_root,
                                task_id=task_id,
                                trial_idx=episode_idx,
                                task_description=task_description,
                                debug_images=debug_request_snapshot["images"],
                                robot_state=debug_request_snapshot["state"],
                                raw_action_chunk=debug_chunk,
                                clipped_action_chunk=clipped_chunk,
                                clip_count=chunk_clip_count,
                                max_abs_before_clip=chunk_max_abs,
                            )
                            logger.info("Saved alignment debug bundle to %s", debug_dir)
                            debug_saved = True
                            debug_request_snapshot = None

                    # 5. Activate deferred chunk if wait window reached.
                    if pending_new_chunk is not None and effective_t >= pending_activation_frame:
                        current_chunk = pending_new_chunk
                        current_window_end = int(pending_new_chunk.shape[0])
                        current_k = min(pending_new_start_idx, int(pending_new_chunk.shape[0]))
                        pending_new_chunk = None
                        pending_new_start_idx = 0
                        pending_activation_frame = -1

                    # 6. Execute action.
                    can_execute_action = (
                        _remaining_actions(current_chunk, current_k) > 0
                        if cfg.async_mode
                        else current_chunk is not None and current_k < current_window_end
                    )
                    if can_execute_action:
                        action, was_clipped, max_abs_before_clip = _sanitize_libero_action(
                            current_chunk[current_k],
                            cfg.policy.type,
                        )
                        episode_clip_count += int(was_clipped)
                        episode_max_abs_action = max(episode_max_abs_action, max_abs_before_clip)
                        current_k += 1
                        obs, reward, done, info = env.step(action)
                    else:
                        episode_idle_steps += 1
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.policy.type))

                    pbar.update(1)
                    t += 1

                    if done or _env_check_success(env):
                        task_successes += 1
                        total_successes += 1
                        break

                pbar.close()
                all_idle_steps.append(episode_idle_steps)
                episode_success = bool(done or _env_check_success(env))
                trial_duration_s = time.perf_counter() - trial_start_time
                task_trial_durations_s.append(trial_duration_s)
                if episode_success:
                    task_success_durations_s.append(trial_duration_s)

                task_episodes += 1
                total_episodes += 1

                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=episode_success,
                    task_description=task_description,
                    log_dir=cfg.local_log_dir,
                )

                trial_summary = (
                    f"Task {task_id} | Trial {episode_idx + 1}/{cfg.num_trials_per_task} | "
                    f"Success: {episode_success} | Idle Steps: {episode_idle_steps} | "
                    f"Replans: {episode_replans} | Clip Count: {episode_clip_count} | "
                    f"Max |Action|: {episode_max_abs_action:.4f} | "
                    f"Trial Time: {_format_duration_s(trial_duration_s)}"
                )
                print(trial_summary)
                logger.info(trial_summary)
                log_file.write(trial_summary + "\n")
                log_file.flush()

            avg_task_trial_time_s = float(np.mean(task_trial_durations_s)) if task_trial_durations_s else 0.0
            avg_task_success_time_s = (
                float(np.mean(task_success_durations_s)) if task_success_durations_s else None
            )

            logger.info(f"Task {task_id} Success Rate: {task_successes}/{task_episodes}")
            logger.info(
                "Task %d Avg Trial Time (all): %s over %d trials",
                task_id,
                _format_duration_s(avg_task_trial_time_s),
                len(task_trial_durations_s),
            )
            if avg_task_success_time_s is not None:
                logger.info(
                    "Task %d Avg Trial Time (success only): %s over %d successful trials",
                    task_id,
                    _format_duration_s(avg_task_success_time_s),
                    len(task_success_durations_s),
                )
            else:
                logger.info("Task %d Avg Trial Time (success only): N/A (0 successful trials)", task_id)

            log_file.write(
                f"Task {task_id} Summary | Success Rate: {task_successes}/{task_episodes} | "
                f"Avg Trial Time (all): {_format_duration_s(avg_task_trial_time_s)} | "
                f"Avg Trial Time (success only): "
                f"{_format_duration_s(avg_task_success_time_s) if avg_task_success_time_s is not None else 'N/A'}\n"
            )
            log_file.flush()

        logger.info(f"Total Success Rate: {total_successes}/{total_episodes}")
        avg_latency = np.mean(all_inference_latencies) if all_inference_latencies else 0.0
        avg_idle = np.mean(all_idle_steps) if all_idle_steps else 0.0
        logger.info(f"Average Inference Latency: {avg_latency*1000:.2f}ms")
        logger.info(f"Average Idle Steps per Ep: {avg_idle:.1f}")

    finally:
        if "log_file" in locals():
            log_file.close()
        try:
            parent_conn.send("TERMINATE")
        except Exception:
            pass
        worker_process.join(timeout=5)
        if worker_process.is_alive():
            worker_process.terminate()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Evaluate VLASH policy in LIBERO")
    arg_parser.add_argument("--config", type=str, required=True, help="Path to LIBERO simulation eval config yaml")
    arg_parser.add_argument("--task_suite", type=str, default=None, help="Override task suite")
    arg_parser.add_argument("--num_trials_per_task", type=int, default=None, help="Override episodes per task")
    arg_parser.add_argument("--resolution", type=int, default=None, help="Override image resolution")
    arg_parser.add_argument(
        "--env_render_resolution",
        type=int,
        default=None,
        help="Override LIBERO env camera render resolution before resize/pad",
    )
    arg_parser.add_argument("--max_steps", type=int, default=None, help="Override max environment steps")
    arg_parser.add_argument("--warmup_steps", type=int, default=None, help="Override warmup steps")
    arg_parser.add_argument("--replan_steps", type=int, default=None, help="Override sync replanning window")
    arg_parser.add_argument("--env_seed", type=int, default=None, help="Override LIBERO environment seed")
    arg_parser.add_argument(
        "--debug_alignment",
        type=_str_to_bool,
        default=None,
        help="Dump alignment debug artifacts for the first trial of each task",
    )
    arg_parser.add_argument("--local_log_dir", type=str, default=None, help="Override log directory")
    arg_parser.add_argument("--async_mode", type=_str_to_bool, default=None, help="Enable async evaluation")
    arg_parser.add_argument("--async_threshold", type=int, default=None, help="Override async threshold")
    arg_parser.add_argument("--async_wait", type=int, default=None, help="Override async wait")
    arg_parser.add_argument(
        "--inference_mode",
        type=str,
        choices=sorted(VALID_INFERENCE_MODES),
        default=None,
        help="Inference conditioning mode",
    )
    arg_parser.add_argument("--rtc_prefix_k", type=int, default=None, help="RTC prefix length from old chunk")
    arg_parser.add_argument("--vlash_delay_k", type=int, default=None, help="VLASH delay index parameter")
    args = arg_parser.parse_args()

    overrides = {
        "task_suite": args.task_suite,
        "num_trials_per_task": args.num_trials_per_task,
        "resolution": args.resolution,
        "env_render_resolution": args.env_render_resolution,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "replan_steps": args.replan_steps,
        "env_seed": args.env_seed,
        "debug_alignment": args.debug_alignment,
        "local_log_dir": args.local_log_dir,
        "async_mode": args.async_mode,
        "async_threshold": args.async_threshold,
        "async_wait": args.async_wait,
        "inference_mode": args.inference_mode,
        "rtc_prefix_k": args.rtc_prefix_k,
        "vlash_delay_k": args.vlash_delay_k,
    }
    eval_libero(config_path=args.config, cli_overrides=overrides)
