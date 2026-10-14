"""Custom GRPO Trainer with reasoning-gym reward functions."""

import json
import os
import re

import torch
from trl import GRPOConfig, GRPOTrainer

from reasoning_gym.utils import extract_answer

from .dataset import ReasoningGymDataset
from .formatting import format_issues

# Try to import wandb for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CustomGRPOTrainer(GRPOTrainer):
    """
    Extended GRPOTrainer with reasoning-gym reward functions.
    
    Includes:
        - Accuracy reward based on reasoning-gym's scoring (optionally binary: exact answers only, as in eval)
        - Format reward for proper <think>/<answer> tags (optionally strict: also requires the clean layout)
        - Optional reward variance logging
    """

    def __init__(
        self,
        model,
        args: GRPOConfig,
        tokenizer,
        train_dataset: ReasoningGymDataset,
        eval_dataset: ReasoningGymDataset,
        log_reward_variance: bool = True,
        verbose_variance_logging: bool = False,
        batch_counts_as_step: bool = False,
        binary_accuracy_reward: bool = False,
        strict_format_reward: bool = False,
    ):
        super().__init__(
            model=model,
            reward_funcs=[
                self._accuracy_reward,
                self._format_reward,
            ],
            args=args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.log_reward_variance = log_reward_variance
        self.verbose_variance_logging = verbose_variance_logging
        self.batch_counts_as_step = batch_counts_as_step
        self.binary_accuracy_reward = binary_accuracy_reward
        self.strict_format_reward = strict_format_reward
        self._step_count = 0
    
    def log(self, logs: dict, start_time: float = None) -> None:
        """Override log to manually log to wandb with correct x-axis."""
        # First, let parent class merge _metrics into logs
        # (GRPOTrainer.log merges self._metrics[mode] into logs before calling super().log)
        mode = "train" if self.model.training else "eval"
        if hasattr(self, '_metrics') and mode in self._metrics:
            metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if val}
            if mode == "eval":
                metrics = {f"eval_{key}": val for key, val in metrics.items()}
            logs = {**logs, **metrics}
        
        # Now log to wandb with our custom step
        if self.state is not None and WANDB_AVAILABLE and wandb.run is not None:
            # Determine which step metric to use
            if self.batch_counts_as_step:
                step_value = self.state.global_step * self.args.gradient_accumulation_steps
            else:
                step_value = self.state.global_step
            
            try:
                wandb_logs = {}
                
                # Add all train metrics
                for key, value in logs.items():
                    # Convert string numbers to float (HuggingFace logs some metrics as strings)
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            continue  # Skip non-numeric strings
                    
                    if isinstance(value, (int, float)):
                        # Prefix with train/ if not already prefixed
                        if not key.startswith("train/") and not key.startswith("eval/"):
                            wandb_logs[f"train/{key}"] = value
                        else:
                            wandb_logs[key] = value
                
                # Use step parameter to force wandb to use our step value as the x-axis
                if wandb_logs:
                    wandb.log(wandb_logs, step=step_value)
            except Exception as e:
                # Log error but don't crash training
                print(f"Warning: Failed to log to wandb at step {step_value}: {e}")
        
        # Clear metrics before calling parent (to avoid double logging in parent's log method)
        if hasattr(self, '_metrics') and mode in self._metrics:
            self._metrics[mode].clear()
        
        # Call grandparent's log (skip GRPOTrainer.log since we already handled metrics)
        from transformers import Trainer
        Trainer.log(self, logs, start_time)

    def _accuracy_reward(self, completions: list[str], **kwargs) -> list[float]:
        """Compute accuracy reward using reasoning-gym's scoring function."""
        assert "item" in kwargs, "The 'item' argument must be provided."
        assert len(kwargs["item"]) == len(completions)
        
        answers = [extract_answer(c) for c in completions]
        rewards = [
            self.train_dataset.data.score_answer(answer, item)
            for answer, item in zip(answers, kwargs["item"])
        ]
        if self.binary_accuracy_reward:
            # The eval only counts score >= 1 as correct. Partial credit (e.g. the true answer appearing inside a
            # longer answer) would reward the model for putting its working in the answer tags.
            rewards = [1.0 if r >= 1.0 else 0.0 for r in rewards]

        if self.log_reward_variance:
            self._log_variance(rewards, answers, completions)

        self._save_rollouts(completions, answers, rewards, kwargs["item"], kwargs.get("completion_ids"))
        return rewards

    def _save_rollouts(self, completions: list[str], answers: list[str], rewards: list[float], items: list[dict],
                       completion_ids: list[list[int]] | None = None):
        """Append every rollout to <output_dir>/rollouts/rollouts_rank<N>.jsonl (written as training goes)."""
        rollout_dir = os.path.join(self.args.output_dir, "rollouts")
        os.makedirs(rollout_dir, exist_ok=True)
        path = os.path.join(rollout_dir, f"rollouts_rank{self.accelerator.process_index}.jsonl")
        mode = "train" if self.model.training else "eval"
        format_rewards = self._format_reward(completions)
        # Several reward calls can share a global step (gradient accumulation), so group ids carry a call counter
        self._rollout_calls = getattr(self, "_rollout_calls", 0) + 1
        # Same truncation test TRL uses: a completion that doesn't end in EOS/pad hit max_completion_length
        stop_ids = {self.eos_token_id, self.pad_token_id} if hasattr(self, "eos_token_id") else set()
        ids_list = completion_ids or [None] * len(completions)
        with open(path, "a", encoding="utf-8") as f:
            for i, (completion, answer, reward, fmt, item, ids) in enumerate(
                zip(completions, answers, rewards, format_rewards, items, ids_list)
            ):
                f.write(json.dumps({
                    "step": self.state.global_step,
                    "mode": mode,
                    "group": f"{self._rollout_calls}.{i // self.args.num_generations}",
                    "num_tokens": len(ids) if ids is not None else None,
                    "truncated": (not ids or ids[-1] not in stop_ids) if ids is not None else None,
                    "task": item.get("metadata", {}).get("source_dataset"),
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "completion": completion,
                    "extracted": answer,
                    "accuracy_reward": reward,
                    "format_reward": fmt,
                }, ensure_ascii=False, default=str) + "\n")
    
    def _log_variance(self, rewards: list[float], answers: list[str], completions: list[str]):
        """Log the percentage of prompts with reward variance across rollouts."""
        num_gens = self.args.num_generations
        num_prompts = len(rewards) // num_gens
        
        if num_prompts == 0:
            return
        
        prompts_with_reward_variance = 0
        prompts_with_answer_variance = 0
        
        for i in range(num_prompts):
            start_idx = i * num_gens
            end_idx = (i + 1) * num_gens
            group_rewards = rewards[start_idx:end_idx]
            group_answers = answers[start_idx:end_idx]
            
            has_reward_variance = len(set(group_rewards)) > 1
            has_answer_variance = len(set(group_answers)) > 1
            
            if has_reward_variance:
                prompts_with_reward_variance += 1
            if has_answer_variance:
                prompts_with_answer_variance += 1
            
            # Verbose: print full completions when there's variance
            if self.verbose_variance_logging and has_answer_variance:
                group_completions = completions[start_idx:end_idx]
                print(f"\n  {'─'*60}")
                print(f"  📊 Prompt {i}: rewards={group_rewards}")
                for gen_idx, (comp, rew, ans) in enumerate(zip(group_completions, group_rewards, group_answers)):
                    comp_preview = comp[:300].replace('\n', ' ↵ ')
                    if len(comp) > 300:
                        comp_preview += "..."
                    print(f"    Gen {gen_idx} [reward={rew}]: {comp_preview}")
                    print(f"           extracted: {ans}")
                print(f"  {'─'*60}")
        
        reward_variance_pct = (prompts_with_reward_variance / num_prompts * 100)
        answer_variance_pct = (prompts_with_answer_variance / num_prompts * 100)
        
        self._step_count += 1
        
        if self.verbose_variance_logging:
            print(f"🎲 Step {self._step_count}: {reward_variance_pct:.0f}% prompts have reward variance, "
                  f"{answer_variance_pct:.0f}% have answer variance ({num_prompts} prompts × {num_gens} gens)")
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                wandb.log({
                    "train/reward_variance_pct": reward_variance_pct,
                    "train/answer_variance_pct": answer_variance_pct,
                }, commit=False)
            except:
                pass
        
        # Periodic GPU cleanup
        if self._step_count % 10 == 0:
            torch.cuda.empty_cache()

    def _format_reward(self, completions: list[str], **kwargs) -> list[float]:
        """Reward proper formatting with <think> and <answer> tags."""
        def count_tags(text: str) -> float:
            count = 0.0
            if re.search(r"\s*<think>\s*", text):
                count += 0.25
            if re.search(r"\s*</think>\s*", text):
                count += 0.25
            if re.search(r"\s*<answer>\s*", text):
                count += 0.25
            if re.search(r"\s*</answer>\s*", text):
                count += 0.25
            return count

        if not self.strict_format_reward:
            return [count_tags(c) for c in completions]
        # Strict: the tags earn up to 0.5, and the other 0.5 needs a clean layout (one <think>, then one single-line
        # <answer>, nothing else). The eval scores only the last <answer>, so extra or chatty answers lose points.
        # Keeping the per-tag part leaves a gradient from untagged answers towards tagged ones.
        return [0.5 * count_tags(c) + (0.5 if not format_issues(c) else 0.0) for c in completions]
