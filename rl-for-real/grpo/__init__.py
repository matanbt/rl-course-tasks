"""GRPO Training Components for Reasoning Gym.

This module provides components for training language models using
Group Relative Policy Optimization (GRPO) on reasoning-gym tasks.
"""

from .config import DatasetConfig, DatasetConfigItem
from .dataset import ReasoningGymDataset, prepare_datasets
from .trainer import CustomGRPOTrainer
from .eval_callback import FixedEvalCallback
from .report import ReportCallback, build_report, run_info
from .utils import build_wandb_config, log_trainable_params, wandb_run_name

__all__ = [
    "DatasetConfig",
    "DatasetConfigItem",
    "ReasoningGymDataset",
    "prepare_datasets",
    "CustomGRPOTrainer",
    "FixedEvalCallback",
    "ReportCallback",
    "build_report",
    "run_info",
    "build_wandb_config",
    "log_trainable_params",
    "wandb_run_name",
]
