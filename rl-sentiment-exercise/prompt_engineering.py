"""
Prompt Engineering vs RL - Extension Exercise

Can you match your RL-trained model's sentiment by just changing the prompt?
Try different prompt prefixes and compare the results to RL training.

Usage:
    # Try the default prefix strategies (sentiment scoring)
    python prompt_engineering.py

    # Compare against an RL-trained model
    python prompt_engineering.py --trained_model ./outputs/final

    # Use your shaped_reward from Exercise 5 as the scoring metric
    python prompt_engineering.py --use_shaped_reward --trained_model ./outputs/final

    # Try your own custom prefixes
    python prompt_engineering.py --prefix "This is a very positive review. " --prefix "Short and sweet: "
"""

import argparse
import numpy as np

from data import VALIDATION_PROMPTS
from evaluate import load_model, generate_completions
from sentiment import get_sentiment_scores

PROMPT_STRATEGIES = {
    "no prefix": "",
    "descriptive": "Positive review: ",
    "instructive": "Write a very positive movie review. ",
    "enthusiastic": "I absolutely loved this movie! ",
}


def _strip_prefix(prefix, completion):
    """Return the completion with the prefix removed (for fair scoring)."""
    if prefix and completion.startswith(prefix):
        return completion[len(prefix):]
    return completion


def _load_scorer(use_shaped_reward, use_solution=False):
    """Return a scoring function: shaped_reward if requested, else raw sentiment."""
    if not use_shaped_reward:
        return get_sentiment_scores, "sentiment"

    if use_solution:
        from rewards_solution import shaped_reward
    else:
        from rewards import shaped_reward

    def scorer(completions):
        raw_scores = get_sentiment_scores(completions)
        return shaped_reward(raw_scores, completions)

    return scorer, "shaped_reward"


def evaluate_strategy(model, tokenizer, strategy_name, prefix, prompts, scorer,
                      max_new_tokens=50):
    prefixed_prompts = [prefix + p for p in prompts]
    completions = generate_completions(
        model, tokenizer, prefixed_prompts, max_new_tokens=max_new_tokens
    )

    stripped = [_strip_prefix(prefix, c) for c in completions]
    scores = scorer(stripped)
    scores_array = np.array(scores)

    stats = {
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
    }

    print(f"\n{'─'*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Prefix:   \"{prefix}\"")
    print(f"  Mean score:  {stats['mean']:.3f}")

    print(f"\n  Samples (stripped of prefix):")
    for i in range(min(5, len(stripped))):
        print(f"    [{scores[i]:.3f}] {stripped[i][:120]}...")

    return {"strategy": strategy_name, "prefix": prefix, "scores": scores, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Prompt engineering vs RL comparison")
    parser.add_argument(
        "--trained_model", type=str, default=None,
        help="Path to an RL-trained model to compare against",
    )
    parser.add_argument(
        "--prefix", type=str, action="append", default=None,
        help="Custom prompt prefix (can be repeated for multiple strategies)",
    )
    parser.add_argument(
        "--use_shaped_reward", action="store_true",
        help="Score with your shaped_reward() from rewards.py instead of raw sentiment",
    )
    parser.add_argument(
        "--use_solution", action="store_true",
        help="Use rewards_solution.py instead of rewards.py (for testing/development)",
    )
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    prompts = VALIDATION_PROMPTS
    if args.num_prompts < len(VALIDATION_PROMPTS):
        prompts = prompts[-args.num_prompts:]

    # Build strategy list: defaults + any custom prefixes
    if args.prefix is not None:
        strategies = {"no prefix": ""}
        for i, p in enumerate(args.prefix):
            label = p.strip()[:30] if len(args.prefix) > 1 else "custom"
            strategies[label] = p
    else:
        strategies = dict(PROMPT_STRATEGIES)

    scorer, scorer_name = _load_scorer(args.use_shaped_reward, args.use_solution)
    print(f"Scoring metric: {scorer_name}")

    print("Loading base GPT-2...")
    base_model, base_tokenizer = load_model("gpt2")

    trained_model, trained_tokenizer = None, None
    if args.trained_model:
        print(f"Loading RL-trained model: {args.trained_model}")
        trained_model, trained_tokenizer = load_model(args.trained_model)

    base_results = {}
    trained_results = {}

    for name, prefix in strategies.items():
        print(f"\n{'='*60}")
        print(f"BASE GPT-2 — {name}")
        print(f"{'='*60}")
        base_results[name] = evaluate_strategy(
            base_model, base_tokenizer, name, prefix, prompts, scorer,
            max_new_tokens=args.max_new_tokens,
        )

        if trained_model is not None:
            print(f"\n{'='*60}")
            print(f"RL-TRAINED — {name}")
            print(f"{'='*60}")
            trained_results[name] = evaluate_strategy(
                trained_model, trained_tokenizer, name, prefix, prompts, scorer,
                max_new_tokens=args.max_new_tokens,
            )

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY  (metric: {scorer_name})")
    print(f"{'='*60}")

    if trained_model is not None:
        print(f"  {'Strategy':<30} {'Base GPT-2':>12} {'RL-trained':>12}")
        print(f"  {'─'*30} {'─'*12} {'─'*12}")
        for name in strategies:
            b = base_results[name]["stats"]["mean"]
            t = trained_results[name]["stats"]["mean"]
            print(f"  {name:<30} {b:>12.3f} {t:>12.3f}")
    else:
        print(f"  {'Strategy':<30} {'Base GPT-2':>12}")
        print(f"  {'─'*30} {'─'*12}")
        for name in strategies:
            b = base_results[name]["stats"]["mean"]
            print(f"  {name:<30} {b:>12.3f}")

    if trained_model is None:
        print(f"\n  Tip: pass --trained_model to compare against an RL-trained model.")


if __name__ == "__main__":
    main()
