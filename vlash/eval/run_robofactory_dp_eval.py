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
RoboFactory DP (DataPoint) evaluation script for VLASH policies.

Evaluates pi0.5 policies in RoboFactory multi-robot environments using
the DP inference server for multi-GPU serving. Each robot independently
sends async inference requests while eagerly executing actions, with
dynamic chunk switching logic and TOPP trajectory interpolation.

Usage:
    python -m vlash.eval.run_robofactory_dp_eval \
        --server_address localhost:50051 \
        --rf_config <robofactory_task_config.yaml> \
        --num_trials 20 \
        --async_threshold 20 \
        --async_wait 5
"""

import argparse
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import sapien
import yaml
from tqdm import tqdm

from mani_skill.envs.sapien_env import BaseEnv
from robofactory.tasks import *  # noqa: F401,F403 - register all envs
from robofactory.utils.wrappers.record import RecordEpisodeMA
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

from vlash.eval.dp.client import InferenceClient
from vlash.eval.robofactory_utils import build_model_observation

logger = logging.getLogger(__name__)


# ============================================================================
# Per-Robot Async State
# ============================================================================

@dataclass
class RobotAsyncState:
    """Tracks the async inference state for a single robot."""
    action_queue: list = field(default_factory=list)      # pending raw actions [joint_pos + gripper]
    inference_pending: bool = False                         # whether a request is in-flight
    inference_start_time: float = 0.0                       # when the request was sent
    inference_request_frame: int = 0                        # sim frame when request was sent
    pending_request_id: str = ""                            # request_id of in-flight request
    pending_new_chunk: list | None = None                   # chunk waiting to be activated
    pending_activation_frame: int = -1                      # frame to activate pending chunk
    request_counter: int = 0                                # monotonic counter for request IDs
    idle_steps: int = 0                                     # steps where queue was empty


# ============================================================================
# Dynamic Chunk Switching
# ============================================================================

def apply_chunk_switching(
    robot_state: RobotAsyncState, 
    new_chunk: np.ndarray,
    effective_frame: int,
    wait_steps: int,
):
    """Apply the dynamic chunk switching logic.
    
    Same logic as LIBERO async eval:
    - If inference took longer than wait_steps, drop first z actions and immediately adopt
    - If inference finished early, schedule adoption at request_frame + wait_steps
    
    Args:
        robot_state: The robot's async state to update.
        new_chunk: New action chunk [n_action_steps, action_dim].
        effective_frame: Current simulation frame.
        wait_steps: The configured wait offset.
    """
    z = effective_frame - robot_state.inference_request_frame
    chunk_list = new_chunk.tolist()
    
    if z >= wait_steps:
        # Late arrival: drop first z actions, immediately adopt the rest
        if z < len(chunk_list):
            robot_state.action_queue = chunk_list[z:]
        else:
            # Extremely late: chunk is entirely stale, still adopt what we can
            robot_state.action_queue = chunk_list[-1:] if chunk_list else []
    else:
        # Early arrival: schedule adoption at request_frame + wait_steps
        robot_state.pending_new_chunk = chunk_list[wait_steps:]
        robot_state.pending_activation_frame = robot_state.inference_request_frame + wait_steps


def check_pending_activation(robot_state: RobotAsyncState, effective_frame: int):
    """Activate a pending chunk if the activation frame has been reached."""
    if (robot_state.pending_new_chunk is not None 
            and effective_frame >= robot_state.pending_activation_frame):
        robot_state.action_queue = robot_state.pending_new_chunk
        robot_state.pending_new_chunk = None


# ============================================================================
# TOPP Execution
# ============================================================================

def execute_topp_step(
    env: BaseEnv,
    planner: PandaArmMotionPlanningSolver,
    raw_obs: dict,
    robot_actions: dict[int, np.ndarray],
    agent_num: int,
    dt: float = 0.05,
    verbose: bool = False,
) -> tuple[dict, dict, bool]:
    """Execute one action per robot through TOPP trajectory interpolation.
    
    For each robot, computes a smooth trajectory from current joint positions
    to the target joint positions using TOPP, then steps the environment
    through the interpolated trajectory. Synchronizes robots by padding
    shorter trajectories with their final positions.
    
    Args:
        env: The RoboFactory gym environment.
        planner: Motion planning solver with TOPP capability.
        raw_obs: Current raw observation from the environment.
        robot_actions: Dict mapping robot_id -> target action [joint_pos + gripper].
        agent_num: Total number of agents.
        dt: TOPP time step.
        verbose: Whether to print debug info.
        
    Returns:
        Tuple of (final_obs, info, success).
    """
    # Compute TOPP trajectories for each robot
    trajectories = {}   # robot_id -> list of interpolated joint positions
    grippers = {}       # robot_id -> gripper value
    
    for robot_id in range(agent_num):
        if robot_id not in robot_actions:
            # Robot has no action this step; hold current position
            current_qpos = raw_obs["agent"][f"panda-{robot_id}"]["qpos"].squeeze(0).numpy()
            trajectories[robot_id] = [current_qpos[:-2]]  # exclude gripper qpos (last 2)
            grippers[robot_id] = current_qpos[-1]  # last element as gripper
            continue
            
        action = robot_actions[robot_id]
        target_joint = action[:-1]  # first 7: joint positions
        gripper = action[-1]        # last 1: gripper
        grippers[robot_id] = gripper
        
        current_qpos = raw_obs["agent"][f"panda-{robot_id}"]["qpos"].squeeze(0)[:-2].numpy()
        path = np.vstack((current_qpos, target_joint))
        
        try:
            times, positions, vel, acc, duration = planner.planner[robot_id].TOPP(path, dt, verbose=False)
            n_steps = positions.shape[0]
            if n_steps == 0:
                trajectories[robot_id] = [current_qpos]
            else:
                trajectories[robot_id] = [positions[j] for j in range(n_steps)]
        except Exception as e:
            if verbose:
                logger.warning(f"TOPP failed for robot {robot_id}: {e}")
            trajectories[robot_id] = [current_qpos]
    
    # Synchronize: find max interpolation steps across all robots
    max_interp_steps = max(len(traj) for traj in trajectories.values())
    
    # Step through the interpolated trajectory
    observation = raw_obs
    info = {}
    success = False
    
    for step_idx in range(max_interp_steps):
        action_dict = {}
        for robot_id in range(agent_num):
            traj = trajectories[robot_id]
            # Clamp: if this robot's trajectory is shorter, hold at its last position
            actual_idx = min(step_idx, len(traj) - 1)
            joint_pos = traj[actual_idx]
            true_action = np.hstack([joint_pos, grippers[robot_id]])
            action_dict[f"panda-{robot_id}"] = true_action
        
        observation, reward, terminated, truncated, info = env.step(action_dict)
        
        if info.get("success", False):
            success = True
            break
    
    return observation, info, success


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def eval_robofactory_dp(
    server_address: str,
    rf_config: str,
    num_trials: int = 20,
    async_threshold: int = 20,
    async_wait_steps: int = 5,
    max_steps: int = 250,
    topp_dt: float = 0.05,
    resolution: int = 224,
    seed: int = 10000,
    sim_backend: str = "cpu",
    render_mode: str = "rgb_array",
    record_dir: str = "./experiments/dp_eval",
    task_description_override: str = "",
    verbose: bool = False,
):
    """Main DP evaluation loop for RoboFactory multi-robot tasks.
    
    Args:
        server_address: DP inference server address (e.g. "localhost:50051").
        rf_config: Path to RoboFactory task config YAML.
        num_trials: Number of evaluation episodes.
        async_threshold: Remaining actions before triggering new inference.
        async_wait_steps: Fixed offset for dynamic chunk switching.
        max_steps: Maximum iteration steps per episode.
        topp_dt: TOPP interpolation time step.
        resolution: Image resolution for model input.
        seed: Random seed.
        sim_backend: ManiSkill simulation backend ("cpu" or "gpu").
        render_mode: Render mode for the environment.
        record_dir: Directory for saving evaluation results.
        task_description_override: Override task description for the model.
        verbose: Enable verbose logging.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # ------------------------------------------------------------------
    # 1. Parse RoboFactory config to get env_id
    # ------------------------------------------------------------------
    with open(rf_config, "r") as f:
        config = yaml.safe_load(f)
    env_id = config["task_name"] + "-rf"
    task_desc = task_description_override or config.get("task_description", env_id)
    
    logger.info(f"Env: {env_id}, Task: {task_desc}")
    
    # ------------------------------------------------------------------
    # 2. Create environment
    # ------------------------------------------------------------------
    env_kwargs = dict(
        config=rf_config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        num_envs=1,
        sim_backend=sim_backend,
        enable_shadow=True,
    )
    env: BaseEnv = gym.make(env_id, **env_kwargs)
    
    # Optional recording
    os.makedirs(record_dir, exist_ok=True)
    env = RecordEpisodeMA(
        env, record_dir, 
        info_on_video=False, 
        save_trajectory=False, 
        max_steps_per_video=30000,
    )
    
    # ------------------------------------------------------------------
    # 3. Create planner for TOPP interpolation
    # ------------------------------------------------------------------
    raw_obs, _ = env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=[agent.robot.pose for agent in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=True,
    )
    agent_num = planner.agent_num
    logger.info(f"Environment created with {agent_num} agents.")
    
    # ------------------------------------------------------------------
    # 4. Connect to inference server
    # ------------------------------------------------------------------
    client = InferenceClient(server_address)
    client.wait_for_server()
    logger.info(f"Connected to DP server at {server_address}")
    
    # ------------------------------------------------------------------
    # 5. Run evaluation episodes
    # ------------------------------------------------------------------
    total_successes = 0
    all_inference_latencies = []
    all_idle_steps = []
    
    log_file_path = os.path.join(record_dir, f"eval_{env_id}.txt")
    log_file = open(log_file_path, "w")
    
    try:
        for trial_idx in range(num_trials):
            logger.info(f"Trial {trial_idx + 1}/{num_trials}")
            raw_obs, _ = env.reset(seed=seed + trial_idx)
            
            # Initialize per-robot async state
            robot_states = {i: RobotAsyncState() for i in range(agent_num)}
            
            iteration = 0
            episode_success = False
            
            pbar = tqdm(total=max_steps, desc=f"Trial {trial_idx + 1}", leave=False)
            
            while iteration < max_steps:
                # --------------------------------------------------------
                # A. For each robot: check if we need to fire async inference
                # --------------------------------------------------------
                for robot_id in range(agent_num):
                    rs = robot_states[robot_id]
                    
                    if not rs.inference_pending and len(rs.action_queue) <= async_threshold:
                        # Build observation for this robot
                        model_obs = build_model_observation(raw_obs, robot_id, resolution)
                        
                        # Create unique request ID
                        request_id = f"robot{robot_id}_{rs.request_counter}"
                        rs.request_counter += 1
                        
                        # Fire async request
                        client.submit_async(request_id, model_obs, task_desc)
                        rs.inference_pending = True
                        rs.inference_start_time = time.time()
                        rs.inference_request_frame = iteration
                        rs.pending_request_id = request_id
                
                # --------------------------------------------------------
                # B. For each robot: check for completed inference
                # --------------------------------------------------------
                for robot_id in range(agent_num):
                    rs = robot_states[robot_id]
                    
                    if not rs.inference_pending:
                        continue
                    
                    result = client.poll_result(rs.pending_request_id)
                    if result is not None:
                        latency = time.time() - rs.inference_start_time
                        all_inference_latencies.append(latency)
                        rs.inference_pending = False
                        
                        # Apply dynamic chunk switching
                        apply_chunk_switching(rs, result, iteration, async_wait_steps)
                
                # --------------------------------------------------------
                # C. For each robot: check pending chunk activation
                # --------------------------------------------------------
                for robot_id in range(agent_num):
                    check_pending_activation(robot_states[robot_id], iteration)
                
                # --------------------------------------------------------
                # D. Pop one action per robot and execute through TOPP
                # --------------------------------------------------------
                robot_actions = {}
                all_have_actions = True
                
                for robot_id in range(agent_num):
                    rs = robot_states[robot_id]
                    if len(rs.action_queue) > 0:
                        action = rs.action_queue.pop(0)
                        robot_actions[robot_id] = np.array(action, dtype=np.float32)
                    else:
                        rs.idle_steps += 1
                        all_have_actions = False
                
                if robot_actions:
                    # Execute TOPP step (handles synchronization internally)
                    raw_obs, info, success = execute_topp_step(
                        env, planner, raw_obs, robot_actions, agent_num,
                        dt=topp_dt, verbose=verbose,
                    )
                    
                    if success:
                        episode_success = True
                        break
                else:
                    # All robots idle: step with dummy/hold actions
                    hold_action = {}
                    for robot_id in range(agent_num):
                        qpos = raw_obs["agent"][f"panda-{robot_id}"]["qpos"].squeeze(0).numpy()
                        hold_action[f"panda-{robot_id}"] = qpos
                    raw_obs_new, _, _, _, info = env.step(hold_action)
                    raw_obs = raw_obs_new
                
                iteration += 1
                pbar.update(1)
                
                # Check success from info
                if info.get("success", False):
                    episode_success = True
                    break
            
            pbar.close()
            
            if episode_success:
                total_successes += 1
            
            # Record per-robot idle steps
            total_idle = sum(rs.idle_steps for rs in robot_states.values())
            all_idle_steps.append(total_idle)
            
            result_str = "SUCCESS" if episode_success else "FAILED"
            log_line = (
                f"Trial {trial_idx} | {result_str} | "
                f"Iterations: {iteration} | Idle Steps: {total_idle}"
            )
            logger.info(log_line)
            log_file.write(log_line + "\n")
            log_file.flush()
        
        # ------------------------------------------------------------------
        # 6. Summary
        # ------------------------------------------------------------------
        success_rate = total_successes / num_trials if num_trials > 0 else 0.0
        avg_latency = np.mean(all_inference_latencies) if all_inference_latencies else 0.0
        avg_idle = np.mean(all_idle_steps) if all_idle_steps else 0.0
        
        summary = (
            f"\n{'='*60}\n"
            f"Evaluation Summary: {env_id}\n"
            f"  Success Rate: {total_successes}/{num_trials} ({success_rate:.1%})\n"
            f"  Avg Inference Latency: {avg_latency*1000:.2f}ms\n"
            f"  Avg Idle Steps per Episode: {avg_idle:.1f}\n"
            f"{'='*60}"
        )
        logger.info(summary)
        log_file.write(summary + "\n")
        print(summary)
        
    finally:
        log_file.close()
        client.close()
        env.close()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLASH pi0.5 policy on RoboFactory tasks via DP server"
    )
    parser.add_argument("--server_address", type=str, default="localhost:50051",
                        help="DP inference server address")
    parser.add_argument("--rf_config", type=str, required=True,
                        help="Path to RoboFactory task config YAML")
    parser.add_argument("--num_trials", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--async_threshold", type=int, default=20,
                        help="Remaining actions before triggering new inference")
    parser.add_argument("--async_wait", type=int, default=5,
                        help="Wait offset for dynamic chunk switching")
    parser.add_argument("--max_steps", type=int, default=250,
                        help="Maximum iterations per episode")
    parser.add_argument("--topp_dt", type=float, default=0.05,
                        help="TOPP interpolation time step")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Image resolution")
    parser.add_argument("--seed", type=int, default=10000,
                        help="Random seed")
    parser.add_argument("--sim_backend", type=str, default="cpu",
                        help="ManiSkill sim backend (cpu/gpu)")
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                        help="Render mode")
    parser.add_argument("--record_dir", type=str, default="./experiments/dp_eval",
                        help="Directory for evaluation results")
    parser.add_argument("--task_description", type=str, default="",
                        help="Override task description for the model")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    eval_robofactory_dp(
        server_address=args.server_address,
        rf_config=args.rf_config,
        num_trials=args.num_trials,
        async_threshold=args.async_threshold,
        async_wait_steps=args.async_wait,
        max_steps=args.max_steps,
        topp_dt=args.topp_dt,
        resolution=args.resolution,
        seed=args.seed,
        sim_backend=args.sim_backend,
        render_mode=args.render_mode,
        record_dir=args.record_dir,
        task_description_override=args.task_description,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
