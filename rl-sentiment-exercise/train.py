"""
Main Training Script for RL Sentiment Fine-tuning Exercise.

This script uses TRL's GRPOTrainer to fine-tune GPT-2 for positive sentiment
generation using GRPO (Group Relative Policy Optimization).

Usage:
    # Basic training
    python train.py
    
    # With custom configuration
    python train.py --max_steps 200 --num_generations 8
    
    # Using accelerate for multi-GPU
    accelerate launch train.py

Configuration presets are provided for different hardware capabilities.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
import transformers
import trl

transformers.logging.set_verbosity_error()

# Optional wandb import
try:
    import trackio as wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data import get_train_dataset, get_validation_dataset, VALIDATION_PROMPTS
from sentiment import get_sentiment_scores
from reward_utils import make_reward_function


# =============================================================================
# VALIDATION CALLBACK
# =============================================================================

class ValidationCallback(transformers.TrainerCallback):
    """
    Custom callback to run validation on held-out prompts during training.
    
    Generates 1 completion per validation prompt, computes sentiment scores,
    prints first 4 examples and statistics for all 64 prompts.
    
    Optionally computes reference perplexity (how natural the outputs are
    according to the original model) to detect reward hacking / degenerate text.
    
    If wandb is enabled, logs validation metrics and sample completions.
    """
    
    def __init__(self, tokenizer, validation_prompts, eval_steps=10, num_display=4, 
                 use_wandb=False, temperature=1.0, max_new_tokens=48,
                 ref_model=None, compute_perplexity=True):
        self.tokenizer = tokenizer
        self.validation_prompts = validation_prompts
        self.eval_steps = eval_steps
        self.num_display = num_display
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.validation_history = []  # Store history for plotting
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.ref_model = ref_model
        self.compute_perplexity = compute_perplexity and (ref_model is not None)
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Run validation at step 0 before training starts."""
        self.max_steps = state.max_steps
        if model is not None:
            self._run_validation(model, 0)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Run validation at logging steps."""
        # Only run at specified intervals (skip 0 since we do it in on_train_begin)
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        
        model = kwargs.get("model")
        if model is None:
            return
            
        self._run_validation(model, state.global_step)
    
    def _run_validation(self, model, step, batch_size=16):
        """Generate completions and compute statistics using batched generation."""
        model.eval()
        device = next(model.parameters()).device
        
        completions = []
        
        # Generate completions in batches for efficiency
        with torch.no_grad():
            for i in range(0, len(self.validation_prompts), batch_size):
                batch_prompts = self.validation_prompts[i:i + batch_size]
                
                # Tokenize batch with left padding (required for decoder-only generation)
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Decode each output in the batch
                for output in outputs:
                    completion = self.tokenizer.decode(output, skip_special_tokens=True)
                    completions.append(completion)
                
                # Restore default padding side
                self.tokenizer.padding_side = "right"
        
        # Compute sentiment scores for all completions
        scores = get_sentiment_scores(completions)
        scores_array = np.array(scores)
        
        # Compute completion lengths (words) without the prompt
        gen_parts = []
        for prompt, completion in zip(self.validation_prompts, completions):
            gen = completion[len(prompt):] if completion.startswith(prompt) else completion
            gen_parts.append(gen)
        lengths = np.array([len(g.split()) for g in gen_parts])
        
        # Compute statistics
        stats = {
            "val/sentiment_mean": float(np.mean(scores_array)),
            "val/sentiment_std": float(np.std(scores_array)),
            "val/sentiment_min": float(np.min(scores_array)),
            "val/sentiment_max": float(np.max(scores_array)),
            "val/positive_ratio": float(np.mean(scores_array > 0.5)),
            "val/length_mean": float(np.mean(lengths)),
        }
        
        # QUESTION Q5: Below, we compute "reference perplexity": the perplexity of
        # text generated by the trained model, under the reference GPT-2 (before fine-tuning).
        # Why is there a tradeoff between the sentiment score (or any other metric we optimize)
        # and the reference perplexity?
        #
        # Bonus: in Exercise 2, you may observe a fairly low perplexity metric despite a
        # significant model behavior shift. Can you explain why?
        # Hint: observe perplexity_mean and first_tokens_perplexity.

        # Compute reference perplexity (how natural the text is under original GPT-2)
        perplexities = None
        if self.compute_perplexity:
            from reward_utils import compute_log_probs
            token_log_probs = compute_log_probs(
                gen_parts, list(self.validation_prompts),
                self.ref_model, self.tokenizer,
            )
            perplexities = []
            perplexities_first = []
            for lp_tokens in token_log_probs:
                if len(lp_tokens) > 0:
                    avg_lp = sum(lp_tokens) / len(lp_tokens)
                    perplexities.append(math.exp(-avg_lp))
                    first = lp_tokens[:2]
                    perplexities_first.append(math.exp(-sum(first) / len(first)))
                else:
                    perplexities.append(float("nan"))
                    perplexities_first.append(float("nan"))
            
            ppl_array = np.array([p for p in perplexities if not math.isnan(p)])
            ppl_first_array = np.array([p for p in perplexities_first if not math.isnan(p)])
            if len(ppl_array) > 0:
                stats["val/perplexity_mean"] = float(np.mean(ppl_array))
            if len(ppl_first_array) > 0:
                stats["val/first_tokens_perplexity"] = float(np.mean(ppl_first_array))
        
        # Store in history
        self.validation_history.append({"step": step, **stats})
        
        # Print header
        print(f"\n{'='*70}")
        print(f"VALIDATION (Step {step}/{getattr(self, 'max_steps', '?')}) - {len(completions)} prompts")
        print(f"{'='*70}")
        
        # Print first N examples
        sample_data = []
        for i in range(min(self.num_display, len(completions))):
            prompt = self.validation_prompts[i]
            completion = completions[i]
            score = scores[i]
            
            # Extract just the generated part (after prompt)
            generated = completion[len(prompt):] if completion.startswith(prompt) else completion
            
            ppl_str = ""
            if perplexities:
                ppl_str = f"  Perplexity: {perplexities[i]:.1f} (first: {perplexities_first[i]:.1f})"
            print(f"\n[{i+1}] Sentiment: {score:.3f}  Words: {lengths[i]}{ppl_str}")
            print(f"    Prompt: {prompt}")
            n_chars = 200
            print(f"    Generated:{generated[:n_chars]}{'...' if len(generated) > n_chars else ''}")
            
            sample_data.append({
                "prompt": prompt,
                "generated": generated[:n_chars],
                "sentiment": score
            })
        
        # Print statistics for all prompts
        print(f"\n{'-'*70}")
        print(f"STATISTICS (all {len(scores)} validation prompts):")
        print(f"  Mean sentiment:          {stats['val/sentiment_mean']:.3f}")
        print(f"  Std:                     {stats['val/sentiment_std']:.3f}")
        print(f"  Min:                     {stats['val/sentiment_min']:.3f}")
        print(f"  Max:                     {stats['val/sentiment_max']:.3f}")
        print(f"  Positive ratio:          {stats['val/positive_ratio']:.1%}")
        print(f"  Mean length (words):     {stats['val/length_mean']:.1f}")
        if "val/perplexity_mean" in stats:
            print(f"  Mean perplexity:         {stats['val/perplexity_mean']:.1f}")
            print(f"  First-token perplexity:  {stats['val/first_tokens_perplexity']:.1f}")
        print(f"{'='*70}\n")
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({"step": step, **stats})
            
            # # Log sample completions as a table
            # if sample_data:
            #     table = wandb.Table(
            #         columns=["prompt", "generated", "sentiment"],
            #         data=[[s["prompt"], s["generated"], s["sentiment"]] for s in sample_data]
            #     )
            #     wandb.log({f"val/samples_step_{step}": table})
        
        model.train()


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

@dataclass
class TrainingPreset:
    """Training configuration preset."""
    name: str
    max_steps: int
    num_generations: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_completion_length: int
    learning_rate: float
    temperature: float


PRESETS = {
    "default": TrainingPreset(
        name="default",
        max_steps=200,
        num_generations=8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_completion_length=64,
        learning_rate=1e-4,
        temperature=1.5,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _init_wandb(
    wandb_project: str,
    wandb_run_name: Optional[str],
    *,
    preset: str,
    negate_reward: bool,
    reward_shaping: str,
    kl_type: str,
    kl_coef: float,
    config: dict,
) -> Optional[str]:
    """Initialize trackio logging. Returns the run name, or None if tracking is inactive."""
    if not WANDB_AVAILABLE:
        print("Warning: trackio not installed. Install with: pip install trackio")
        return None

    if wandb_run_name:
        run_name = wandb_run_name
    else:
        parts = [preset]
        if negate_reward:
            parts.append("NEG")
        parts.append(reward_shaping)
        parts.append(f"lr{config['learning_rate']}")
        if kl_type != "none":
            parts.append(f"kl-{kl_type}")
            if kl_coef != 5.0:
                parts.append(f"c{kl_coef}")
        run_name = "_".join(parts)

    wandb.init(
        project=wandb_project,
        name=run_name,
        config=config,
    )
    print(f"Logging to trackio: {wandb_project}/{run_name}")
    return run_name


def _load_models(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    kl_type: str,
    compute_perplexity: bool,
):
    """Load policy and/or reference models as needed.

    Returns (policy_model, ref_model) -- either may be None.
    """
    policy_model = None
    ref_model = None

    def _load_frozen_ref(reason: str):
        print(f"Loading reference model ({reason})...")
        m = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        m = m.to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        print("Reference model loaded (frozen)")
        return m

    if kl_type != "none":
        print(f"Loading policy model for {kl_type} KL regularization...")
        policy_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        )
        policy_model = policy_model.to(device)
        print("Policy model loaded")
        ref_model = _load_frozen_ref(f"for {kl_type} KL regularization")
    elif compute_perplexity:
        ref_model = _load_frozen_ref("for perplexity evaluation")

    return policy_model, ref_model


def _print_config_summary(
    config_preset: "TrainingPreset",
    max_steps: int,
    num_generations: int,
    beta: float,
):
    """Print a human-readable GRPO configuration summary."""
    print(f"\nGRPO Configuration:")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Batch size: {config_preset.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {config_preset.gradient_accumulation_steps}")
    print(f"  - Group size (num_generations): {num_generations}")
    print(f"  - Max completion length: {config_preset.max_completion_length}")
    print(f"  - Learning rate: {config_preset.learning_rate}")
    print(f"  - Temperature: {config_preset.temperature}")
    print(f"  - KL beta (TRL internal): {beta}")


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(
    model_name: str = "gpt2",
    output_dir: str = "./outputs",
    preset: str = "default",
    max_steps: Optional[int] = None,
    num_generations: Optional[int] = None,
    reward_shaping: str = "expectation",
    kl_type: str = "none",
    kl_coef: float = 5.0,
    negate_reward: bool = False,
    beta: float = 0.0,
    use_solution: bool = False,
    use_peft: bool = False,
    log_completions: bool = False,
    compute_perplexity: bool = True,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "rl-sentiment",
    wandb_run_name: Optional[str] = None,
):
    """
    Train GPT-2 for positive sentiment generation using GRPO.
    
    Args:
        model_name: HuggingFace model name (default: "gpt2")
        output_dir: Directory to save checkpoints and logs
        preset: Training preset ("default")
        max_steps: Override preset's max_steps
        num_generations: Override preset's num_generations (GRPO group size)
        reward_shaping: "five_stars", "expectation", or "custom" (see reward_utils.py)
        kl_type: Custom KL regularization in reward ("none", "forward", "backward", "k3")
        kl_coef: Coefficient for custom KL regularization (default: 5.0)
        negate_reward: Negate reward to optimize for negative sentiment
        beta: TRL's internal KL regularization coefficient (0.0 = disabled)
        use_solution: Use rewards_solution.py instead of rewards.py (for testing)
        use_peft: Whether to use LoRA for parameter-efficient training
        log_completions: Whether to log TRL's internal training completions (default: False,
                         we use our own validation callback instead)
        compute_perplexity: Compute reference perplexity during validation (default: True)
        seed: Random seed for reproducibility
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name (default: "rl-sentiment")
        wandb_run_name: Optional W&B run name (auto-generated from config if None)
    """
    # Get preset configuration
    config_preset = PRESETS.get(preset, PRESETS["default"])
    print(f"\n{'='*60}")
    print(f"Training Configuration: {preset}")
    print(f"{'='*60}\n")
    
    # Override preset values if specified
    actual_max_steps = max_steps or config_preset.max_steps
    actual_num_generations = num_generations or config_preset.num_generations
    
    # Initialize trackio if enabled
    run_name = None
    if use_wandb:
        run_name = _init_wandb(
            wandb_project,
            wandb_run_name,
            preset=preset,
            negate_reward=negate_reward,
            reward_shaping=reward_shaping,
            kl_type=kl_type,
            kl_coef=kl_coef,
            config={
                "model_name": model_name,
                "preset": preset,
                "max_steps": actual_max_steps,
                "num_generations": actual_num_generations,
                "reward_shaping": reward_shaping,
                "kl_type": kl_type,
                "kl_coef": kl_coef,
                "negate_reward": negate_reward,
                "beta": beta,
                "use_peft": use_peft,
                "learning_rate": config_preset.learning_rate,
                "batch_size": config_preset.per_device_train_batch_size,
                "gradient_accumulation_steps": config_preset.gradient_accumulation_steps,
                "max_completion_length": config_preset.max_completion_length,
                "seed": seed,
            },
        )
        use_wandb = run_name is not None

    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"Device: {device}, Dtype: {dtype}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    # QUESTION Q6: The training dataset loaded below contains only prompts,
    # with no target outputs or labels. Where does the learning signal come from
    # instead? Trace it: what provides the "labels" and where is it passed to the
    # trainer? (Hint: look for reward_func)

    # Load dataset
    print("Loading dataset...")
    train_dataset = get_train_dataset()
    print(f"Training samples: {len(train_dataset)}")
    
    # Load models
    policy_model, ref_model = _load_models(
        model_name, device, dtype, kl_type, compute_perplexity,
    )
    
    # Import reward module (student code or solution)
    if use_solution:
        import rewards_solution as reward_module
        print("Using rewards_solution.py (solution code)")
    else:
        import rewards as reward_module
        print("Using rewards.py (student code)")
    
    if negate_reward:
        print("NEGATING REWARD: Optimizing for NEGATIVE sentiment!")
    
    reward_func = make_reward_function(
        shaping=reward_shaping,
        kl_type=kl_type,
        kl_coef=kl_coef,
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        negate=negate_reward,
        reward_module=reward_module,
    )
    
    # Print reward configuration
    shaping_desc = {
        "five_stars": "binary P(5 stars)",
        "expectation": "continuous expected stars",
        "custom": "custom (from rewards module)",
    }
    kl_desc = {
        "none": "None",
        "forward": f"Forward KL (coef={kl_coef})",
        "backward": f"Backward KL (coef={kl_coef})",
        "k3": f"Forward KL, k3 estimator (coef={kl_coef})",
    }
    negate_str = " [NEGATED]" if negate_reward else ""
    print(f"Reward: Sentiment model, shaping={shaping_desc.get(reward_shaping, reward_shaping)}{negate_str}")
    if kl_type != "none":
        print(f"KL regularization: {kl_desc.get(kl_type, kl_type)}")
    
    # QUESTION Q7: Look at the GRPOConfig parameters below. GRPO generates
    # num_generations completions per prompt, then ranks them by reward to compute
    # advantages. What is the current value of num_generations? What is the total
    # number of completions per optimizer update step, considering
    # per_device_train_batch_size and gradient_accumulation_steps?

    # Configure GRPO training
    training_args = trl.GRPOConfig(
        output_dir=output_dir,
        
        # Training parameters
        max_steps=actual_max_steps,
        per_device_train_batch_size=config_preset.per_device_train_batch_size,
        gradient_accumulation_steps=config_preset.gradient_accumulation_steps,
        learning_rate=config_preset.learning_rate,
        
        # GRPO specific
        num_generations=actual_num_generations,
        max_completion_length=config_preset.max_completion_length,
        # QUESTION Q8: This if the model temperature on training.
        # Find the temperature set for validation, and verify that it is currently the same.
        # Should we use different temperatures for training and validation? Which one would
        # be higher? What are the pros and cons of that approach?
        temperature=config_preset.temperature,
        # QUESTION Q9 (optional): The beta parameter below is TRL's built-in KL regularization
        # (which you will not be using in this exercise).
        # The kl_type/kl_coef parameters control a separate, custom KL regularization
        # that you will implement in rewards.py. You already observed this in reward_utils.py (Q4).
        # 
        # Optional: after implementing your custom KL, find TRL's KL regularization
        # and compare it to your custom KL.
        beta=beta,
        
        # Optimization
        warmup_steps=int(actual_max_steps * 0.1),  # 10% warmup
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Logging
        logging_steps=20,
        log_completions=log_completions,
        report_to="trackio" if use_wandb else "none",
        project=wandb_project,  # same project + run name -> trainer resumes our trackio run
        run_name=run_name,
        trackio_space_id=None,  # log locally instead of to a public HF Space
        
        # Saving
        save_steps=100,
        save_total_limit=2,
        
        # Memory optimization
        gradient_checkpointing=True,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        
        # Reproducibility
        seed=seed,
    )
    
    _print_config_summary(config_preset, actual_max_steps, actual_num_generations, beta)
    
    # Initialize trainer
    print("\nInitializing GRPOTrainer...")
    
    # Optional: PEFT/LoRA configuration
    peft_config = None
    if use_peft:
        try:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["c_attn", "c_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            print("Using LoRA for parameter-efficient training")
        except ImportError:
            print("Warning: PEFT not installed, training full model")
            peft_config = None
    
    # Create validation callback
    validation_callback = ValidationCallback(
        tokenizer=tokenizer,
        validation_prompts=VALIDATION_PROMPTS,
        eval_steps=training_args.logging_steps,
        num_display=4,
        use_wandb=use_wandb,
        temperature=config_preset.temperature,
        max_new_tokens=config_preset.max_completion_length,
        ref_model=ref_model,
        compute_perplexity=compute_perplexity,
    )
    
    # Use policy_model if loaded (for custom KL), otherwise let TRL load from model_name
    model_for_trainer = policy_model if policy_model is not None else model_name
    
    # QUESTION Q10: Look at the arguments passed to GRPOTrainer below, and make sure
    # you understand why each one is needed.

    trainer = trl.GRPOTrainer(
        model=model_for_trainer,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        peft_config=peft_config,
        callbacks=[validation_callback],
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    # Log final validation metrics and finish trackio
    if use_wandb and WANDB_AVAILABLE:
        # The trainer finished the run at train end; resume it to log the summary
        wandb.init(project=wandb_project, name=run_name, resume="allow")
        # Log validation history as a summary
        if validation_callback.validation_history:
            final_stats = validation_callback.validation_history[-1]
            initial_stats = validation_callback.validation_history[0]
            wandb.log({
                "final_sentiment_mean": final_stats["val/sentiment_mean"],
                "final_positive_ratio": final_stats["val/positive_ratio"],
                # Log improvement from start to end
                "sentiment_improvement": final_stats["val/sentiment_mean"] - initial_stats["val/sentiment_mean"],
            })

        wandb.finish()
        print("trackio run finished.")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {os.path.join(output_dir, 'final')}")
    print("="*60)
    
    return trainer


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GPT-2 for positive sentiment using GRPO"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", type=str, default="gpt2",
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    
    # Training preset
    parser.add_argument(
        "--preset", type=str, default="default",
        choices=["default"],
        help="Training preset"
    )
    
    # Override preset values
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Override max training steps"
    )
    parser.add_argument(
        "--num_generations", type=int, default=None,
        help="Override GRPO group size"
    )
    
    # Reward configuration
    parser.add_argument(
        "--reward_shaping", type=str, default="expectation",
        choices=["five_stars", "expectation", "custom"],
        help="Reward shaping: five_stars (binary), expectation (continuous), custom (shaped_reward)"
    )
    parser.add_argument(
        "--kl_type", type=str, default="none",
        choices=["none", "forward", "backward", "k3"],
        help="Custom KL regularization type (none, forward, backward, k3)"
    )
    parser.add_argument(
        "--kl_coef", type=float, default=5.0,
        help="Coefficient for custom KL regularization (default: 5.0)"
    )
    parser.add_argument(
        "--negate_reward", action="store_true",
        help="Negate reward to optimize for negative sentiment"
    )
    parser.add_argument(
        "--beta", type=float, default=0.0,
        help="TRL's internal KL regularization coefficient (default: 0.0 = disabled)"
    )
    
    # Training options
    parser.add_argument(
        "--use_solution", action="store_true",
        help="Use rewards_solution.py instead of rewards.py (for testing/development)"
    )
    parser.add_argument(
        "--use_peft", action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    parser.add_argument(
        "--log_trl_completions", action="store_true",
        help="Enable TRL's internal completion logging (shows GRPO group generations)"
    )
    parser.add_argument(
        "--no_perplexity", action="store_true",
        help="Disable reference perplexity computation during validation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Wandb arguments
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable logging to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="rl-sentiment",
        help="W&B project name (default: rl-sentiment)"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="W&B run name (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Run training
    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        preset=args.preset,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        reward_shaping=args.reward_shaping,
        kl_type=args.kl_type,
        kl_coef=args.kl_coef,
        negate_reward=args.negate_reward,
        beta=args.beta,
        use_solution=args.use_solution,
        use_peft=args.use_peft,
        log_completions=args.log_trl_completions,
        compute_perplexity=not args.no_perplexity,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
