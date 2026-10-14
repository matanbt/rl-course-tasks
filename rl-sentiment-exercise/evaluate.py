"""
Evaluation and Visualization Module for RL Sentiment Fine-tuning Exercise.

This module provides utilities for:
1. Generating samples from models (before/after training)
2. Computing sentiment statistics
3. Plotting training curves and sentiment distributions
4. Comparing different training configurations

Usage:
    # Generate comparison samples
    python evaluate.py --model_path ./outputs/final --num_samples 20
    
    # Compare base vs trained model
    python evaluate.py --compare gpt2 ./outputs/final
"""

import argparse
import json
import os
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

transformers.logging.set_verbosity_error()

from data import get_validation_dataset, VALIDATION_PROMPTS
from sentiment import get_sentiment_scores


# =============================================================================
# GENERATION UTILITIES
# =============================================================================

def load_model(model_path: str, device: Optional[str] = None):
    """
    Load a model and tokenizer from a path or HuggingFace name.
    
    Args:
        model_path: Path to saved model or HuggingFace model name
        device: Device to load model on
    
    Returns:
        tuple: (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
) -> list[str]:
    """
    Generate completions for a list of prompts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling
        num_return_sequences: Number of completions per prompt
    
    Returns:
        List of generated completions (prompt + generation)
    """
    device = next(model.parameters()).device
    
    all_completions = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        for output in outputs:
            completion = tokenizer.decode(output, skip_special_tokens=True)
            all_completions.append(completion)
    
    return all_completions


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(
    model_path: str,
    prompts: Optional[list[str]] = None,
    num_samples: int = 20,
    max_new_tokens: int = 50,
) -> dict:
    """
    Evaluate a model's sentiment generation capabilities.
    
    Args:
        model_path: Path to model or HuggingFace name
        prompts: Optional list of prompts (uses VALIDATION_PROMPTS if None)
        num_samples: Number of samples to generate
        max_new_tokens: Maximum tokens per completion
    
    Returns:
        dict with completions, scores, and statistics
    """
    print(f"Loading model: {model_path}")
    model, tokenizer = load_model(model_path)
    
    if prompts is None:
        prompts = VALIDATION_PROMPTS[:num_samples]
    else:
        prompts = prompts[:num_samples]
    
    print(f"Generating {len(prompts)} completions...")
    completions = generate_completions(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens
    )
    
    print("Computing sentiment scores...")
    scores = get_sentiment_scores(completions)
    
    # Compute completion lengths (words)
    gen_parts = []
    for prompt, completion in zip(prompts, completions):
        gen = completion[len(prompt):] if completion.startswith(prompt) else completion
        gen_parts.append(gen)
    lengths = [len(g.split()) for g in gen_parts]
    lengths_array = np.array(lengths)
    
    # Compute statistics
    scores_array = np.array(scores)
    stats = {
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
        "median": float(np.median(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "positive_ratio": float(np.mean(scores_array > 0.5)),
        "length_mean": float(np.mean(lengths_array)),
    }
    
    return {
        "model_path": model_path,
        "prompts": prompts,
        "completions": completions,
        "scores": scores,
        "lengths": lengths,
        "stats": stats,
    }


def compare_models(
    base_model: str,
    trained_model: str,
    prompts: Optional[list[str]] = None,
    num_samples: int = 10,
) -> dict:
    """
    Compare base model vs trained model.
    
    Args:
        base_model: Path/name of base model
        trained_model: Path to trained model
        prompts: Optional list of prompts
        num_samples: Number of samples per model
    
    Returns:
        dict with comparison results
    """
    print("\n" + "="*60)
    print("Evaluating BASE model...")
    print("="*60)
    base_results = evaluate_model(base_model, prompts, num_samples)
    
    print("\n" + "="*60)
    print("Evaluating TRAINED model...")
    print("="*60)
    trained_results = evaluate_model(trained_model, prompts, num_samples)
    
    return {
        "base": base_results,
        "trained": trained_results,
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_sentiment_distribution(
    results: dict,
    title: str = "Sentiment Score Distribution",
    save_path: Optional[str] = None,
):
    """
    Plot histogram of sentiment scores.
    
    Args:
        results: Output from evaluate_model()
        title: Plot title
        save_path: Optional path to save figure
    """
    scores = results["scores"]
    stats = results["stats"]
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Neutral (0.5)')
    plt.axvline(x=stats["mean"], color='g', linestyle='-', label=f'Mean ({stats["mean"]:.3f})')
    
    plt.xlabel("Sentiment Score (P(positive))")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add stats text box
    length_line = f"\nMean length: {stats['length_mean']:.1f} words" if "length_mean" in stats else ""
    stats_text = (
        f"Mean: {stats['mean']:.3f}\n"
        f"Std: {stats['std']:.3f}\n"
        f"Positive ratio: {stats['positive_ratio']:.1%}"
        f"{length_line}"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_comparison(
    comparison: dict,
    title: str = "Base vs Trained Model Comparison",
    save_path: Optional[str] = None,
):
    """
    Plot comparison of base vs trained model sentiment distributions.
    
    Args:
        comparison: Output from compare_models()
        title: Plot title
        save_path: Optional path to save figure
    """
    base_scores = comparison["base"]["scores"]
    trained_scores = comparison["trained"]["scores"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram comparison
    ax1 = axes[0]
    ax1.hist(base_scores, bins=15, alpha=0.5, label='Base', color='blue', edgecolor='black')
    ax1.hist(trained_scores, bins=15, alpha=0.5, label='Trained', color='green', edgecolor='black')
    ax1.axvline(x=0.5, color='r', linestyle='--', label='Neutral')
    ax1.set_xlabel("Sentiment Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    ax2.boxplot([base_scores, trained_scores], labels=['Base', 'Trained'])
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Sentiment Score")
    ax2.set_title("Score Distribution")
    ax2.grid(True, alpha=0.3)
    
    # Add improvement text
    base_mean = comparison["base"]["stats"]["mean"]
    trained_mean = comparison["trained"]["stats"]["mean"]
    improvement = trained_mean - base_mean
    
    improvement_text = (
        f"Base mean: {base_mean:.3f}\n"
        f"Trained mean: {trained_mean:.3f}\n"
        f"Improvement: {improvement:+.3f}"
    )
    ax2.text(0.02, 0.98, improvement_text, transform=ax2.transAxes,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def print_samples(results: dict, num_samples: int = 5):
    """
    Print sample completions with their sentiment scores.
    
    Args:
        results: Output from evaluate_model()
        num_samples: Number of samples to print
    """
    print("\n" + "="*60)
    print("SAMPLE COMPLETIONS")
    print("="*60)
    
    lengths = results.get("lengths", [None] * len(results["completions"]))
    for i in range(min(num_samples, len(results["completions"]))):
        prompt = results["prompts"][i]
        completion = results["completions"][i]
        score = results["scores"][i]
        
        sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"
        length_str = f"  Words: {lengths[i]}" if lengths[i] is not None else ""
        
        print(f"\n[{i+1}] Score: {score:.3f} ({sentiment}){length_str}")
        print(f"    Prompt: {prompt}")
        print(f"    Output: {completion}")


def print_comparison_samples(comparison: dict, num_samples: int = 5):
    """
    Print side-by-side samples from base and trained models.
    
    Args:
        comparison: Output from compare_models()
        num_samples: Number of samples to print
    """
    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*70)
    
    base = comparison["base"]
    trained = comparison["trained"]
    
    for i in range(min(num_samples, len(base["prompts"]))):
        prompt = base["prompts"][i]
        
        base_completion = base["completions"][i]
        base_score = base["scores"][i]
        
        trained_completion = trained["completions"][i]
        trained_score = trained["scores"][i]
        
        print(f"\n[{i+1}] Prompt: {prompt}")
        print("-" * 70)
        print(f"  BASE ({base_score:.3f}): {base_completion}")
        print(f"  TRAINED ({trained_score:.3f}): {trained_completion}")


# =============================================================================
# TRAINING LOG PARSING
# =============================================================================

def plot_training_curves(log_file: str, save_path: Optional[str] = None):
    """
    Plot training curves from a training log file.
    
    Args:
        log_file: Path to trainer_state.json or similar log file
        save_path: Optional path to save figure
    """
    # Try to load from trainer_state.json
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        if "log_history" in data:
            log_history = data["log_history"]
        else:
            print(f"No log_history found in {log_file}")
            return
    else:
        print(f"Log file not found: {log_file}")
        return
    
    # Extract metrics
    steps = []
    rewards = []
    losses = []
    
    for entry in log_history:
        if "step" in entry:
            steps.append(entry["step"])
            if "reward" in entry:
                rewards.append(entry["reward"])
            if "loss" in entry:
                losses.append(entry["loss"])
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if rewards:
        axes[0].plot(steps[:len(rewards)], rewards)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Training Reward")
        axes[0].grid(True, alpha=0.3)
    
    if losses:
        axes[1].plot(steps[:len(losses)], losses)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize sentiment generation models"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Evaluate single model
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single model")
    eval_parser.add_argument("model_path", type=str, help="Path to model")
    eval_parser.add_argument("--num_samples", type=int, default=20)
    eval_parser.add_argument("--save_plot", type=str, default=None)
    
    # Compare models
    compare_parser = subparsers.add_parser("compare", help="Compare base vs trained")
    compare_parser.add_argument("base_model", type=str, help="Base model path/name")
    compare_parser.add_argument("trained_model", type=str, help="Trained model path")
    compare_parser.add_argument("--num_samples", type=int, default=10)
    compare_parser.add_argument("--save_plot", type=str, default=None)
    
    # Plot training curves
    plot_parser = subparsers.add_parser("plot", help="Plot training curves")
    plot_parser.add_argument("log_file", type=str, help="Path to trainer_state.json")
    plot_parser.add_argument("--save_plot", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        results = evaluate_model(args.model_path, num_samples=args.num_samples)
        print_samples(results)
        print(f"\nStatistics: {results['stats']}")
        plot_sentiment_distribution(results, save_path=args.save_plot)
        
    elif args.command == "compare":
        comparison = compare_models(
            args.base_model,
            args.trained_model,
            num_samples=args.num_samples
        )
        print_comparison_samples(comparison)
        print(f"\nBase stats: {comparison['base']['stats']}")
        print(f"Trained stats: {comparison['trained']['stats']}")
        plot_comparison(comparison, save_path=args.save_plot)
        
    elif args.command == "plot":
        plot_training_curves(args.log_file, save_path=args.save_plot)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
