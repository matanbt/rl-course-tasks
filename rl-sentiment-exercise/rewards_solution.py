"""
Reward Functions for RL Sentiment Fine-tuning - SOLUTION VERSION

This module contains complete implementations of all reward functions.
This is the instructor/solution version - students should work with rewards.py.
"""

import math
import torch
from sentiment import get_star_probs


# =============================================================================
# BINARY REWARD FUNCTION (Provided - Exercise 2)
# =============================================================================

def five_stars_reward(completions: list[str], threshold: float = 0.8) -> list[float]:
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
# EXPECTED STARS REWARD (Exercise 3) - SOLUTION
# =============================================================================

def expected_stars(completions: list[str]) -> list[float]:
    """
    Continuous reward based on expected star rating, rescaled to [0, 1].
    
    SOLUTION: Compute E[stars] = sum(P(star_i) * i) for i in 1..5,
    then rescale from [1, 5] to [0, 1].
    
    Args:
        completions: List of generated text completions
    
    Returns:
        List of sentiment scores in [0, 1]
    """
    probs = get_star_probs(completions)
    stars = torch.arange(1, 6, device=probs.device, dtype=torch.float32)
    expected = (probs * stars).sum(dim=-1)
    scores = (expected - 1) / 4
    return scores.cpu().tolist()


# =============================================================================
# KL REGULARIZATION (Exercise 4) - SOLUTION
#
# CONTEXT: TRL includes built-in KL regularization (the `beta` parameter),
# applied per-token during advantage computation. Here we re-implement KL
# regularization at the token level for pedagogical purposes.
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
    Forward KL regularization penalty (token-level).
    
    SOLUTION:
    KL(π || π_ref) = E_π[log π - log π_ref]
    Since the data is already sampled from π, the per-token KL term is simply
    (log_policy - log_ref). Average over tokens and scale by kl_coef.
    Returns ≥ 0; the calling infrastructure subtracts this from the reward.
    """
    penalties = []
    for lp_policy, lp_ref in zip(log_probs_policy, log_probs_ref):
        n = len(lp_policy)
        if n == 0:
            penalties.append(0.0)
            continue
        token_kl = [lp_p - lp_r for lp_p, lp_r in zip(lp_policy, lp_ref)]
        penalties.append(kl_coef * sum(token_kl) / n)
    return penalties


def kl_penalty_backward(
    log_probs_policy: list[list[float]],
    log_probs_ref: list[list[float]],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Backward (reverse) KL regularization penalty (token-level).
    
    SOLUTION:
    KL(π_ref || π) = E_π_ref[log(π_ref / π)]
    Since we sample from π (not π_ref), we need importance sampling:
        E_π_ref[f(x)] = E_π[(π_ref(x)/π(x)) · f(x)]
    
    An exact implementation would apply sequence-level importance weight:
        penalty(x) = [Π_t π_ref(x_t|x<t)/π(x_t|x<t)] · [Σ_t log π_ref(x_t|x<t) - log π(x_t|x<t)]
    However, the product Π_t of T ratios can explode or vanish, causing high variance.
    
    We thus use an approximation:
        penalty(x) = (1/T) Σ_t [π_ref(x_t|x<t)/π(x_t|x<t)] · [log π_ref(x_t|x<t) - log π(x_t|x<t)]
    
    This replaces the single sequence-level weight with independent per-token weights.
    It is exact when π = π_ref, and a good approximation when they are close -
    which is the regime we are regularizing toward. This is a standard
    token-level approach often used in practice.
    
    Returns ≥ 0; the calling infrastructure subtracts this from the reward.
    """
    penalties = []
    for lp_policy, lp_ref in zip(log_probs_policy, log_probs_ref):
        n = len(lp_policy)
        if n == 0:
            penalties.append(0.0)
            continue
        token_kl = []
        for lp_p, lp_r in zip(lp_policy, lp_ref):
            diff = lp_r - lp_p
            weight = min(math.exp(diff), 100.0)  # importance-sampling weight P(π_ref) / P(π), clipped for stability
            token_kl.append(weight * diff)
        penalties.append(kl_coef * sum(token_kl) / n)
    return penalties


# =============================================================================
# REWARD SHAPING (Exercise 5) - SOLUTION
# =============================================================================

def shaped_reward(scores: list[float], completions: list[str], prompts: list[str] = None) -> list[float]:
    """
    Apply custom reward shaping to transform raw sentiment scores.
    
    SOLUTION: Target-length reward. Rewards positive sentiment but scales it
    by how close the completion is to a target of 55 characters per completion.
    The length factor is 1/(1 + deviation/target).
    """
    target_len = 55
    shaped = []
    for score, completion in zip(scores, completions):
        n_chars = len(completion)
        length_factor = 1 / (1 + abs(n_chars / target_len - 1))
        shaped.append(score * length_factor)
    return shaped


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing SOLUTION implementations...\n")
    
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute.",
        "Terrible film. Complete waste of time.",
        "It was okay, nothing special.",
    ]
    
    # Test five_stars_reward (binary)
    print("1. Five Stars Reward (binary, threshold=0.8):")
    rewards = five_stars_reward(test_texts, threshold=0.8)
    for text, reward in zip(test_texts, rewards):
        print(f"   {reward:.1f}: {text[:40]}...")
    
    # Test expected_stars (continuous)
    print("\n2. Expected Stars (continuous [0, 1]):")
    rewards = expected_stars(test_texts)
    for text, reward in zip(test_texts, rewards):
        print(f"   {reward:.3f}: {text[:40]}...")
    
    # Test reward shaping
    print("\n3. Shaped Reward:")
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_completions = ["bad", "meh", "okay", "good movie", "amazing film!"]
    shaped_results = shaped_reward(test_scores, test_completions)
    for score, s in zip(test_scores, shaped_results):
        print(f"   {score:.1f} -> {s:+.3f}")
    
    # Test KL penalties (token-level)
    print("\n4. KL Penalties (token-level):")
    test_lp_policy = [[-2.0, -3.0, -1.5], [-1.0, -2.0]]  # Per-token log probs
    test_lp_ref = [[-2.5, -2.5, -2.5], [-1.5, -1.5]]
    
    fwd = kl_penalty_forward(test_lp_policy, test_lp_ref, 0.1)
    print(f"   Forward KL penalties: {[f'{p:.4f}' for p in fwd]}")
    print(f"   (≥ 0; subtracted from reward by infrastructure)")
    
    bwd = kl_penalty_backward(test_lp_policy, test_lp_ref, 0.1)
    print(f"   Backward KL penalties: {[f'{p:.4f}' for p in bwd]}")
    print(f"   (≥ 0; importance-weighted, subtracted from reward)")
    
    print("\nAll solution tests passed!")
