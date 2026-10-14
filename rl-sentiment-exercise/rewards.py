"""
Reward Functions for RL Sentiment Fine-tuning - STUDENT VERSION

This module contains reward function implementations for the RL exercise.
Students should implement the functions marked with TODO.

Functions to implement:
1. expected_stars() - Continuous sentiment reward (Exercise 3)
2. kl_penalty_forward() - Forward KL divergence penalty (Exercise 4)
3. kl_penalty_backward() - Backward KL divergence penalty (Exercise 4)
4. shaped_reward() - Apply reward shaping to rewards (Exercise 5)

The binary five_stars_reward() is provided as a working example.
"""

import math

import torch

from sentiment import get_star_probs


# =============================================================================
# BINARY REWARD FUNCTION (Provided - Exercise 2)
# =============================================================================

# QUESTION Q3: The reward is based on a sentiment model, which is a k-class classifier:
# for every text input, the sentiment model assigns probability to every class, i.e.,
# every star rating. We convert this output of the sentiment model into a binary reward.
# Why is the conversion needed?
# How is the conversion done here?
# What other ways are there to convert the sentiment probability outputs into a reward?

def five_stars_reward(completions: list[str], threshold: float = 0.9) -> list[float]:
    """
    Binary reward: 1 if P(5 stars) >= threshold, else 0.
    
    Args:
        completions: List of generated text completions
        threshold: Probability threshold for the 5-star class
    
    Returns:
        List of binary rewards (0.0 or 1.0)
    """
    probs = get_star_probs(completions)[:, -1].cpu().tolist()
    return [1.0 if p >= threshold else 0.0 for p in probs]


# =============================================================================
# EXPECTED STARS REWARD (Exercise 3)
# =============================================================================

def expected_stars(completions: list[str]) -> list[float]:
    """
    Continuous reward based on expected star rating, rescaled to [0, 1].
    
    Args:
        completions: List of generated text completions
    
    Returns:
        List of sentiment scores in [0, 1]
    
    TODO: Implement this function:
    - Use get_star_probs(completions) to get a (batch, 5) tensor of P(completion i is rated with j stars).
      - See five_stars_reward() above as reference for using get_star_probs().
    - Compute the expected star rating E[stars], then rescale from [1, 5] to [0, 1].
    """
    # =========================================================================
    # YOUR CODE HERE (~5 lines)
    # =========================================================================
    probs = get_star_probs(completions)
    probs = probs * torch.tensor([[1,2,3,4,5]], device=probs.device)
    probs = (probs.sum(dim=-1) - 1) / 4
    return probs.cpu().tolist()
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


# =============================================================================
# KL REGULARIZATION (Exercise 4)
#
# CONTEXT: TRL already includes built-in KL regularization (the `beta` parameter),
# applied per-token during advantage computation. Here you re-implement KL
# regularization yourself at the token level, to understand how it works.
#
# You receive per-token log probabilities as list[list[float]]: one list per
# completion, with one float per generated token. Your functions should compute
# per-token KL terms and average them to produce one penalty scalar per completion.
# =============================================================================

def kl_penalty_forward(
    log_probs_policy: list[list[float]],
    log_probs_ref: list[list[float]],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Forward KL regularization penalty.
    
    Recall: KL(π || π_ref) = E_π[log π - log π_ref].
    Since the data is already sampled from π, KL estimate is straightforward.
    Return a positive penalty (≥ 0 when policy diverges from reference).
    The infrastructure will SUBTRACT this penalty from the reward.
    
    Args:
        log_probs_policy: Per-token log probs under current policy.
            Shape: list of N completions, each a list of T_i floats.
        log_probs_ref: Per-token log probs under reference model (same shape).
        kl_coef: Coefficient controlling regularization strength.
    
    Returns:
        List of N penalty values (one per completion, ≥ 0).
    
    TODO: Implement this function
    """
    # =========================================================================
    # YOUR CODE HERE (~9 lines)
    # =========================================================================
    penalties = []
    for pol_comp, ref_comp in zip(log_probs_policy, log_probs_ref):
        assert len(pol_comp) == len(ref_comp)
        if len(pol_comp) == 0:                 # empty completion: no tokens, no divergence
            penalties.append(0.0)
            continue
        kl_terms = [p - r for p, r in zip(pol_comp, ref_comp)]   # k1 per token
        penalties.append(kl_coef * sum(kl_terms) / len(kl_terms))
    return penalties
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


def kl__k3_penalty_forward(
    log_probs_policy: list[list[float]],
    log_probs_ref: list[list[float]],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Forward KL regularization penalty.
    Adheres GRPO paper & TRL implementation (`k3` in Shulman's blogpost)
    """
    # =========================================================================
    # YOUR CODE HERE (~9 lines)
    # =========================================================================
    penalties = []
    for pol_comp, ref_comp in zip(log_probs_policy, log_probs_ref):
        assert len(pol_comp) == len(ref_comp)
        if len(pol_comp) == 0:                 # empty completion: no tokens, no divergence
            penalties.append(0.0)
            continue
        ratios = [r - p for p, r in zip(pol_comp, ref_comp)]   # k1 per token
        kl_terms = [math.exp(ratio) - ratio - 1 for ratio in ratios]
        penalties.append(kl_coef * sum(kl_terms) / len(kl_terms))
    return penalties
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


def kl_penalty_backward(
    log_probs_policy: list[list[float]],
    log_probs_ref: list[list[float]],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Backward (reverse) KL regularization penalty.
    
    Recall: KL(π_ref || π) = E_π_ref[log(π_ref / π)].
    Since we sample from π (not π_ref), you need importance sampling to correct
    the distribution mismatch.
    Return a positive penalty (≥ 0 in expectation when policy diverges).
    The infrastructure will SUBTRACT this penalty from the reward.

    Args:
        log_probs_policy: Per-token log probs under current policy.
            Shape: list of N completions, each a list of T_i floats.
        log_probs_ref: Per-token log probs under reference model (same shape).
        kl_coef: Coefficient controlling regularization strength.
    
    Returns:
        List of N penalty values (one per completion, ≥ 0 in expectation).
    
    TODO: Implement this function
    """
    # =========================================================================
    # YOUR CODE HERE (~13 lines)
    # =========================================================================
    pass
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


# =============================================================================
# REWARD SHAPING (Exercise 5)
# =============================================================================

def shaped_reward(scores: list[float], completions: list[str], prompts: list[str] = None) -> list[float]:
    """
    Apply custom reward shaping to transform raw sentiment scores.
    
    Reward shaping modifies the raw reward signal to change learning dynamics.
    This is your chance to experiment with different shaping strategies.
    
    Args:
        scores: Raw sentiment scores in [0, 1] from the sentiment model
        completions: The generated text completions (just the generated part, without the prompt)
        prompts: The original prompts (if you need the full sentence: prompt + completion)
    
    Returns:
        List of shaped reward values
    
    Potential ideas: numeric transformation (e.g. exponential, polynomial, log); penalize or encourage long responses;
                     penalize word repetitions; or any idea you think might help.
    
    TODO: Implement your chosen shaping strategy
    """
    # =========================================================================
    # YOUR CODE HERE
    # =========================================================================
    raise NotImplementedError(
        "Exercise: Implement reward shaping"
    )
    # Option 1: reduce word repeatitions
    for comp in completions:
        pass
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing reward functions module...\n")
    
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute.",
        "Terrible film. Complete waste of time.",
        "It was okay, nothing special.",
    ]
    
    # Test five_stars_reward (provided)
    print("1. Five Stars Reward (binary, threshold=0.8):")
    rewards = five_stars_reward(test_texts, threshold=0.8)
    for text, reward in zip(test_texts, rewards):
        print(f"   {reward:.1f}: {text[:40]}...")
    
    # Test expected_stars (student implementation)
    print("\n2. Expected Stars (student exercise):")
    try:
        rewards = expected_stars(test_texts)
        for text, reward in zip(test_texts, rewards):
            print(f"   {reward:.3f}: {text[:40]}...")
    except NotImplementedError:
        print(f"   Not implemented yet")
    
    # Test reward shaping (student implementation)
    print("\n3. Reward Shaping (student exercise):")
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_completions = ["bad", "meh", "okay", "good movie", "amazing film!"]
    try:
        shaped = shaped_reward(test_scores, test_completions)
        for score, completion, s in zip(test_scores, test_completions, shaped):
            print(f"   {score:.1f} '{completion}' -> {s:.3f}")
    except NotImplementedError:
        print(f"   Not implemented yet")
    
    # Test KL penalties (student implementation)
    # Values should be ≥ 0 (subtracted from reward by infrastructure)
    print("\n4. KL Penalties (student exercise):")
    test_lp_policy = [[-2.0, -3.0, -1.5], [-1.0, -2.0]]  # Per-token log probs
    test_lp_ref = [[-2.5, -2.5, -2.5], [-1.5, -1.5]]
    try:
        fwd = kl_penalty_forward(test_lp_policy, test_lp_ref, 0.1)
        print(f"   Forward KL: {[f'{p:.4f}' for p in fwd]}  (should be ≥ 0)")
    except NotImplementedError:
        print(f"   Forward KL: Not implemented yet")
    try:
        bwd = kl_penalty_backward(test_lp_policy, test_lp_ref, 0.1)
        print(f"   Backward KL: {[f'{p:.4f}' for p in bwd]}  (should be ≥ 0)")
    except NotImplementedError:
        print(f"   Backward KL: Not implemented yet")
    
    print("\nDone!")
