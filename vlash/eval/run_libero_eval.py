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

import os
import sys
import time
import argparse
import multiprocessing as mp
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as tmp
import tqdm
from libero.libero import benchmark

from lerobot.configs import parser
from lerobot.utils.utils import get_safe_torch_device

from vlash.configs import RunConfig
from vlash.policies.factory import get_policy_class
from vlash.eval.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_images,
    quat2axisangle,
    save_rollout_video,
)

logger = getLogger(__name__)


def run_inference_worker(config_path, pipe_conn):
    """Worker process that holds the GPU model and performs inference."""
    try:
        cfg = parser.parse(RunConfig, ["--config", config_path])
        device = get_safe_torch_device(cfg.policy.device)

        # Load policy
        policy_class = get_policy_class(cfg.policy.type)
        policy = policy_class.from_pretrained(
            cfg.policy.pretrained_path,
            revision=cfg.policy.revision,
        )
        policy.to(device)
        policy.eval()

        logger.info(f"Worker: Model loaded on {device}")
        
        # Signal ready
        pipe_conn.send("READY")
        
        # Need prev_action_chunk for RTC statefulness
        prev_action_chunk = None

        while True:
            # Wait for inference request
            if not pipe_conn.poll(0.01):
                time.sleep(0.005)
                continue
                
            msg = pipe_conn.recv()
            if msg == "TERMINATE":
                break
                
            # msg is the observation dict
            obs, task_description = msg
            
            # Prepare observation dict exactly as VLA expects
            model_obs = {}
            for k, v in obs.items():
                # Add batch dimension and move to device
                model_obs[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            
            # Fake language tasks/tools format
            model_obs["task"] = [task_description]
                
            with torch.inference_mode():
                # For pi05 policy predict_action_chunk
                kwargs = {}
                if hasattr(policy, "config") and getattr(policy.config, "inference_prefix_mask_steps", 0) > 0:
                    kwargs["prev_action_chunk"] = prev_action_chunk

                # Support basic unnormalized observation pass
                action_chunk = policy.predict_action_chunk(model_obs, **kwargs)
                
                if "prev_action_chunk" in kwargs:
                    prev_action_chunk = action_chunk.clone()
                    
                # Remove batch dim and send back to host as numpy array
                action_chunk_np = action_chunk.squeeze(0).cpu().numpy()
                pipe_conn.send(action_chunk_np)

    except Exception as e:
        logger.error(f"Inference worker failed: {e}")
        pipe_conn.send(e)
        raise e


def eval_libero(
    config_path: str,
    task_suite_name: str = "libero_spatial",
    num_trials_per_task: int = 20,
    async_mode: bool = True,
    async_threshold: int = 20,
    async_wait_steps: int = 5,
    local_log_dir: str = "./experiments/logs",
):
    """Main evaluation loop."""
    
    # Load config to get image size expectations
    cfg = parser.parse(RunConfig, ["--config", config_path])
    
    # Set up Multiprocessing
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
        os.makedirs(local_log_dir, exist_ok=True)
        log_file_path = os.path.join(local_log_dir, f"eval_{task_suite_name}.txt")
        log_file = open(log_file_path, "w")

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        
        logger.info(f"Task suite: {task_suite_name} ({num_tasks_in_suite} tasks)")

        total_episodes = 0
        total_successes = 0
        
        # Tracking metrics
        all_inference_latencies = []
        all_idle_steps = []

        for task_id in range(num_tasks_in_suite):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.policy.type, resolution=224)
            
            task_episodes = 0
            task_successes = 0

            for episode_idx in range(num_trials_per_task):
                logger.info(f"Task {task_id}: {task_description} | Trial {episode_idx+1}/{num_trials_per_task}")
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                max_steps = 600
                num_steps_wait = 10
                
                replay_images = []
                action_queue = []
                
                # Async state variables
                inference_pending = False
                inference_start_time = 0.0
                inference_request_frame = 0
                pending_new_chunk = None
                pending_activation_frame = -1
                
                episode_idle_steps = 0
                
                pbar = tqdm.tqdm(total=max_steps)
                
                while t < max_steps + num_steps_wait:
                    # 1. Warmup simulation
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.policy.type))
                        img_dict = get_libero_images(obs, 224)
                        replay_images.append(img_dict["observation.image"])
                        t += 1
                        continue

                    effective_t = t - num_steps_wait

                    # 2. Extract observations
                    img_dict = get_libero_images(obs, 224)
                    replay_images.append(img_dict["observation.image"])
                    
                    robot_state = np.concatenate((
                        obs["robot0_eef_pos"], 
                        quat2axisangle(obs["robot0_eef_quat"]), 
                        obs["robot0_gripper_qpos"]
                    )).astype(np.float32)
                    
                    current_model_obs = {
                        "observation.image": img_dict["observation.image"],
                        "observation.wrist_image": img_dict["observation.wrist_image"],
                        "observation.state": robot_state
                    }

                    # 3. Check for async inference trigger
                    if async_mode and not inference_pending and len(action_queue) <= async_threshold:
                        # Send request via IPC
                        parent_conn.send((current_model_obs, task_description))
                        inference_pending = True
                        inference_start_time = time.time()
                        inference_request_frame = effective_t
                        
                    # Backup for sync mode or if we run completely dry
                    if not async_mode and len(action_queue) == 0:
                        parent_conn.send((current_model_obs, task_description))
                        chunk = parent_conn.recv()
                        action_queue = chunk.tolist()
                        
                    # 4. Check for async inference completion
                    if inference_pending and parent_conn.poll():
                        new_chunk = parent_conn.recv()
                        if isinstance(new_chunk, Exception):
                            raise new_chunk
                            
                        latency = time.time() - inference_start_time
                        all_inference_latencies.append(latency)
                        inference_pending = False
                        
                        # Apply dynamic offset logic
                        z = effective_t - inference_request_frame
                        if z >= async_wait_steps:
                            # Dynamic adaptation: inference took longer than wait offset.
                            # Drop first z actions and immediately adopt the rest.
                            action_queue = new_chunk[z:].tolist()
                        else:
                            # Eager execution: inference finished early.
                            # We continue executing the old queue until 'wait' steps have passed,
                            # then we adopt the new chunk from wait_steps onwards.
                            pending_new_chunk = new_chunk[async_wait_steps:].tolist()
                            pending_activation_frame = inference_request_frame + async_wait_steps

                    # 5. Check if it is time to activate the pending chunk
                    if pending_new_chunk is not None and effective_t >= pending_activation_frame:
                        action_queue = pending_new_chunk
                        pending_new_chunk = None

                    # 6. Execute action
                    if len(action_queue) > 0:
                        action = action_queue.pop(0)
                        
                        # [dx, dy, dz, dax, day, daz, gripper]
                        # LIBERO expects [0, 1] mapped to [-1, 1] for grippers differently sometimes,
                        # but if dataset used native action space, no conversion needed.
                        action_to_exec = action
                        
                        obs, reward, done, info = env.step(action_to_exec)
                    else:
                        # Queue exhausted (idle waiting for inference)
                        episode_idle_steps += 1
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.policy.type))
                    
                    pbar.update(1)
                    t += 1
                    
                    if done or env._check_success():
                        task_successes += 1
                        total_successes += 1
                        break
                        
                pbar.close()
                all_idle_steps.append(episode_idle_steps)
                
                task_episodes += 1
                total_episodes += 1
                
                # Save Video
                save_rollout_video(
                    replay_images, 
                    total_episodes, 
                    success=(done or env._check_success()), 
                    task_description=task_description, 
                    log_dir=local_log_dir
                )
                
                log_file.write(f"Task: {task_id} | Ep: {episode_idx} | Success: {(done or env._check_success())} | Idle Steps: {episode_idle_steps}\n")
                log_file.flush()

            logger.info(f"Task {task_id} Success Rate: {task_successes}/{task_episodes}")
            
        logger.info(f"Total Success Rate: {total_successes}/{total_episodes}")
        avg_latency = np.mean(all_inference_latencies) if all_inference_latencies else 0.0
        avg_idle = np.mean(all_idle_steps) if all_idle_steps else 0.0
        logger.info(f"Average Inference Latency: {avg_latency*1000:.2f}ms")
        logger.info(f"Average Idle Steps per Ep: {avg_idle:.1f}")
        
    finally:
        # Cleanup
        if 'log_file' in locals():
            log_file.close()
        parent_conn.send("TERMINATE")
        worker_process.join(timeout=5)
        if worker_process.is_alive():
            worker_process.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLASH policy in LIBERO")
    parser.add_argument("--config", type=str, required=True, help="Path to vlash run config yaml")
    parser.add_argument("--task_suite", type=str, default="libero_spatial", help="Task suite name")
    parser.add_argument("--async_mode", type=bool, default=True, help="Enable async evaluation")
    parser.add_argument("--async_threshold", type=int, default=20, help="Steps before chunk end to trigger inference")
    parser.add_argument("--async_wait", type=int, default=5, help="Simulated fixed delay (wait offset)")
    args = parser.parse_args()
    
    eval_libero(
        config_path=args.config,
        task_suite_name=args.task_suite,
        async_mode=args.async_mode,
        async_threshold=args.async_threshold,
        async_wait_steps=args.async_wait,
    )
