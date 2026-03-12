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
"""Configuration for LIBERO simulation evaluation."""

from dataclasses import dataclass
from typing import Any, Union

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class LiberoEvalConfig:
    """Configuration for VLASH policy evaluation in LIBERO simulation."""

    # Policy configuration
    policy: Union[PreTrainedConfig, dict[str, Any], None] = None

    # LIBERO evaluation setup
    task_suite: str = "libero_spatial"
    num_trials_per_task: int = 20
    resolution: int = 224
    env_render_resolution: int = 256
    max_steps: int | None = None
    warmup_steps: int = 10
    replan_steps: int = 5
    env_seed: int = 7
    debug_alignment: bool = False
    local_log_dir: str = "./experiments/logs"

    # Async scheduling controls
    async_mode: bool = True
    async_threshold: int = 20
    async_wait: int = 5

    # Inference input construction mode
    inference_mode: str = "none"  # none | rtc | vlash | rtc_vlash
    rtc_prefix_k: int = 0
    vlash_delay_k: int = 1

    def __post_init__(self):
        """Parse policy config and validate simulation parameters."""
        if isinstance(self.policy, PreTrainedConfig):
            pass
        else:
            policy_path = None
            cli_overrides = []

            if isinstance(self.policy, dict):
                if "path" not in self.policy:
                    raise ValueError("When specifying policy as a dict in YAML, 'path' key is required")

                policy_dict = dict(self.policy)
                policy_path = policy_dict.pop("path")

                for k, v in policy_dict.items():
                    if isinstance(v, bool):
                        cli_overrides.append(f"--{k}={str(v).lower()}")
                    else:
                        cli_overrides.append(f"--{k}={v}")

            cli_policy_path = parser.get_path_arg("policy")
            if cli_policy_path:
                policy_path = cli_policy_path

            cli_policy_overrides = parser.get_cli_overrides("policy")
            if cli_policy_overrides:
                cli_overrides.extend(cli_policy_overrides)

            if policy_path:
                self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
                self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError(
                "You must provide a policy checkpoint.\n"
                "Use CLI: --policy.path=path/to/model --policy.device=cuda\n"
                "Or YAML: policy: {path: path/to/model}"
            )

        valid_modes = {"none", "rtc", "vlash", "rtc_vlash"}
        if self.inference_mode not in valid_modes:
            raise ValueError(
                f"inference_mode must be one of {sorted(valid_modes)}, got '{self.inference_mode}'"
            )

        if self.num_trials_per_task <= 0:
            raise ValueError("num_trials_per_task must be > 0")
        if self.resolution <= 0:
            raise ValueError("resolution must be > 0")
        if self.env_render_resolution <= 0:
            raise ValueError("env_render_resolution must be > 0")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be > 0")
        if self.env_seed < 0:
            raise ValueError("env_seed must be >= 0")
        if self.async_threshold < 0:
            raise ValueError("async_threshold must be >= 0")
        if self.async_wait < 0:
            raise ValueError("async_wait must be >= 0")
        if self.rtc_prefix_k < 0:
            raise ValueError("rtc_prefix_k must be >= 0")
        if self.vlash_delay_k < 0:
            raise ValueError("vlash_delay_k must be >= 0")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enable draccus parser to load policy from path."""
        return ["policy"]
