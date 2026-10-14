"""Fixed evaluation callback for reliable progress tracking."""

import json
import logging
import os
import re

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from reasoning_gym.utils import extract_answer

from .dataset import ReasoningGymDataset

# Try to import wandb for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _dist_rank_and_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _is_main_process() -> bool:
    return _dist_rank_and_world()[0] == 0


class FixedEvalCallback(TrainerCallback):
    """
    Callback for reliable evaluation on a fixed set of samples.
    
    - Evaluates the same samples every time for comparable metrics
    - Under multi-GPU (accelerate/DDP), shards the samples across ranks and gathers the results
    - Logs to wandb
    - Saves results to JSON
    - Prints example outputs
    """
    
    def __init__(
        self,
        eval_dataset: ReasoningGymDataset,
        tokenizer,
        num_samples: int = 30,
        num_examples_to_print: int = 3,
        max_new_tokens: int = 512,
        output_dir: str = None,
        eval_steps: int = 50,
        temperature: float = 0.6,
        eval_batch_size: int = 16,
    ):
        self.tokenizer = tokenizer
        self.num_samples = min(num_samples, len(eval_dataset))
        self.num_examples_to_print = num_examples_to_print
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.last_eval_step = -1
        self.temperature = temperature
        self.do_sample = temperature > 0
        self.eval_batch_size = eval_batch_size
        
        # Store the underlying procedural dataset for proper scoring
        self.procedural_dataset = eval_dataset.data
        
        # Create fixed eval samples
        self.fixed_samples = []
        for i in range(self.num_samples):
            item = eval_dataset[i]
            # Get task name from composite dataset metadata (key is "source_dataset")
            task_name = item["item"].get("metadata", {}).get("source_dataset", "unknown")
            self.fixed_samples.append({
                "prompt": item["prompt"],
                "item": item["item"],
                "question": item["item"]["question"],
                "answer": item["item"]["answer"],
                "task": task_name,
            })
        
        # Count samples per task for logging
        task_counts = {}
        for sample in self.fixed_samples:
            task_counts[sample["task"]] = task_counts.get(sample["task"], 0) + 1
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FixedEvalCallback: {self.num_samples} samples, eval every {eval_steps} steps")
        self.logger.info(f"  Tasks: {task_counts}")
    
    def on_log(self, args, state, control, model=None, **kwargs):
        """Run evaluation at fixed intervals."""
        if model is None:
            return
        
        step = state.global_step
        if step > 0 and step % self.eval_steps == 0 and step != self.last_eval_step:
            self._run_eval(model, step)
            self.last_eval_step = step
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Also run evaluation when checkpoints are saved."""
        if model is None:
            return
        
        step = state.global_step
        if step != self.last_eval_step:
            self._run_eval(model, step)
            self.last_eval_step = step
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Evaluate the final model unless the last step was already evaluated."""
        if model is None:
            return

        step = state.global_step
        if step != self.last_eval_step:
            self._run_eval(model, step)
            self.last_eval_step = step

    def _run_eval(self, model, step: int):
        """Run evaluation and log/save results."""
        if _is_main_process():
            print(f"\n{'='*60}")
            print(f"Running fixed evaluation at step {step}")
            print(f"{'='*60}")

        results = self._evaluate(model, step)
        self._log_to_wandb(results, step)
        self._save_results(results, step)
        self._print_results(results, step)
        torch.cuda.empty_cache()
    
    def _evaluate(self, model, step: int) -> dict:
        """Evaluate model on fixed samples using batched generation."""
        model.eval()
        device = next(model.parameters()).device
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        # A response with no EOS token ran into max_new_tokens (Llama 3 has several EOS ids)
        eos = model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}
        stop_ids.discard(None)

        examples = []
        
        # Each rank evaluates its own shard of the fixed samples (all of them when not distributed)
        rank, world_size = _dist_rank_and_world()
        indexed_samples = list(enumerate(self.fixed_samples))[rank::world_size]
        
        # Process samples in batches
        for batch_start in range(0, len(indexed_samples), self.eval_batch_size):
            batch_indexed = indexed_samples[batch_start:batch_start + self.eval_batch_size]
            batch_samples = [s for _, s in batch_indexed]
            batch_prompts = [s["prompt"] for s in batch_samples]
            
            # Tokenize batch with left-padding for generation
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            self.tokenizer.padding_side = original_padding_side
            
            # Generate for entire batch
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": self.do_sample,
                    "pad_token_id": pad_token_id,
                }
                if self.do_sample:
                    generate_kwargs["temperature"] = self.temperature
                outputs = model.generate(**inputs, **generate_kwargs)
            
            # Process each sample in the batch
            input_length = inputs["input_ids"].shape[1]
            for i, (sample_idx, sample) in enumerate(batch_indexed):
                generated_ids = outputs[i][input_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                ids = generated_ids.tolist()
                stop_at = next((j for j, t in enumerate(ids) if t in stop_ids), None)
                truncated = stop_at is None
                num_tokens = len(ids) if truncated else stop_at + 1
                extracted = extract_answer(generated_text)

                score = self.procedural_dataset.score_answer(extracted, sample["item"])
                is_correct = score >= 1.0
                format_score = self._score_format(generated_text)
                
                examples.append({
                    "index": sample_idx,
                    "question": sample["question"],
                    "ground_truth": sample["answer"],
                    "generated": generated_text,
                    "extracted": extracted,
                    "correct": is_correct,
                    "accuracy_score": score,
                    "format_score": format_score,
                    "task": sample["task"],
                    "num_tokens": num_tokens,
                    "truncated": truncated,
                })
        
        torch.cuda.empty_cache()
        model.train()
        
        # Collect all shards so every rank holds the full, ordered result
        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, examples)
            examples = [ex for shard in gathered for ex in shard]
        examples.sort(key=lambda ex: ex["index"])
        assert len(examples) == len(self.fixed_samples)
        
        num_correct = sum(ex["correct"] for ex in examples)
        format_scores = [ex["format_score"] for ex in examples]
        num_truncated = sum(ex["truncated"] for ex in examples)
        task_stats = {}  # task_name -> {"correct": int, "total": int, "format_scores": list}
        for ex in examples:
            stats = task_stats.setdefault(ex["task"], {"correct": 0, "total": 0, "format_scores": []})
            stats["correct"] += int(ex["correct"])
            stats["total"] += 1
            stats["format_scores"].append(ex["format_score"])
        
        accuracy = (num_correct / len(self.fixed_samples)) * 100 if self.fixed_samples else 0
        avg_format = (sum(format_scores) / len(format_scores)) * 100 if format_scores else 0
        
        # Compute per-task metrics
        per_task_results = {}
        for task_name, stats in task_stats.items():
            task_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            task_format = (sum(stats["format_scores"]) / len(stats["format_scores"]) * 100) if stats["format_scores"] else 0
            per_task_results[task_name] = {
                "accuracy": task_accuracy,
                "format_score": task_format,
                "num_correct": stats["correct"],
                "num_samples": stats["total"],
            }
        
        return {
            "accuracy": accuracy,
            "format_score": avg_format,
            "num_correct": num_correct,
            "num_samples": len(self.fixed_samples),
            "truncated_ratio": num_truncated / len(self.fixed_samples) * 100 if self.fixed_samples else 0,
            "examples": examples,
            "per_task": per_task_results,
        }
    
    def _score_format(self, text: str) -> float:
        """Score format based on presence of tags."""
        score = 0.0
        if re.search(r"<think>", text, re.IGNORECASE):
            score += 0.25
        if re.search(r"</think>", text, re.IGNORECASE):
            score += 0.25
        if re.search(r"<answer>", text, re.IGNORECASE):
            score += 0.25
        if re.search(r"</answer>", text, re.IGNORECASE):
            score += 0.25
        return score
    
    def _log_to_wandb(self, results: dict, step: int):
        """Log evaluation results to wandb."""
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        try:
            # Build log dict with eval metrics
            log_dict = {
                "eval/accuracy": results["accuracy"],
                "eval/format_score": results["format_score"],
                "eval/num_correct": results["num_correct"],
            }
            
            # Add per-task metrics
            for task_name, task_results in results.get("per_task", {}).items():
                log_dict[f"eval/{task_name}/accuracy"] = task_results["accuracy"]
            
            # Use step parameter to force wandb to use train/global_step as x-axis
            wandb.log(log_dict, step=step)
            self.logger.info(f"📊 Logged to wandb: accuracy={results['accuracy']:.1f}% at global_step={step}")
        except Exception as e:
            self.logger.warning(f"Failed to log to wandb: {e}")
    
    def _save_results(self, results: dict, step: int):
        """Save evaluation results to JSON file."""
        if self.output_dir is None or not _is_main_process():
            return
        
        eval_dir = os.path.join(self.output_dir, "eval_results")
        os.makedirs(eval_dir, exist_ok=True)
        
        save_data = {
            "step": step,
            "accuracy": results["accuracy"],
            "format_score": results["format_score"],
            "num_correct": results["num_correct"],
            "num_samples": results["num_samples"],
            "truncated_ratio": results.get("truncated_ratio"),
            "per_task": results.get("per_task", {}),
            "examples": results["examples"],
        }
        
        filepath = os.path.join(eval_dir, f"eval_step_{step}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, default=str)
            self.logger.info(f"📁 Saved eval results to {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save eval results: {e}")
    
    def _print_results(self, results: dict, step: int):
        """Print evaluation results summary and examples."""
        if not _is_main_process():
            return
        print(f"\n{'='*70}")
        print(f"📊 Evaluation Results (Step {step})")
        print(f"{'='*70}")
        print(f"   Overall Accuracy:     {results['accuracy']:.1f}% ({results['num_correct']}/{results['num_samples']})")
        print(f"   Overall Format Score: {results['format_score']:.1f}%")
        
        # Print per-task breakdown
        per_task = results.get("per_task", {})
        if per_task:
            print(f"\n   {'─'*50}")
            print(f"   Per-Task Breakdown:")
            print(f"   {'─'*50}")
            for task_name in sorted(per_task.keys()):
                task_results = per_task[task_name]
                print(f"   {task_name:25s} {task_results['accuracy']:5.1f}% "
                      f"({task_results['num_correct']}/{task_results['num_samples']}) "
                      f"| format: {task_results['format_score']:.0f}%")
        
        print(f"{'='*70}")
        
        examples = results["examples"]
        to_print = examples[:self.num_examples_to_print]
        for i, ex in enumerate(to_print):
            status = "✓ CORRECT" if ex["correct"] else "✗ WRONG"
            task_name = ex.get("task", "unknown")
            accuracy_score = ex.get("accuracy_score", 0.0)
            print(f"\n{'─'*70}")
            print(f"Example {i+1} [{status}] | Task: {task_name} | Score: {accuracy_score:.2f} | Format: {ex['format_score']*100:.0f}%")
            print(f"{'─'*70}")
            
            q = ex['question']
            if len(q) > 1000:
                q = q[:1000] + "..."
            print(f"QUESTION:\n{q}\n")
            print(f"EXPECTED: {ex['ground_truth']}")
            print(f"EXTRACTED: {ex['extracted']} (score: {accuracy_score:.2f})")
            
            print(f"\nMODEL OUTPUT:")
            output = ex['generated']
            if len(output) > 1000:
                output = output[:1000] + "\n... [truncated]"
            print(output)
        
        print("=" * 70 + "\n")
