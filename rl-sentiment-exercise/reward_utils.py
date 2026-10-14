"""
Reward utility functions for RL sentiment fine-tuning.

This module provides infrastructure for computing rewards, including:
- Log probability computation for KL regularization
- Factory function for creating combined reward functions

Students should focus on rewards.py for the exercise implementations.
"""

import torch
from typing import Optional


def compute_log_probs(
    completions: list[str],
    prompts: list[str],
    model,
    tokenizer,
) -> list[list[float]]:
    """
    Compute per-token log probabilities of completions under a model.
    
    This computes how "likely" each token is according to the given model.
    Used for KL regularization to compare policy vs reference distributions.
    
    Args:
        completions: List of generated text completions
        prompts: List of prompts that generated these completions
        model: The model to compute log probs under (policy or reference)
        tokenizer: Tokenizer for the model
    
    Returns:
        List of lists of per-token log probabilities (one list per completion).
        Each inner list has one float per completion token.
        - Values are negative (log probs)
        - Higher (less negative) = more likely under model
        - Lower (more negative) = less likely under model
    """
    device = next(model.parameters()).device
    
    # Concatenate prompts and completions
    full_texts = [p + c for p, c in zip(prompts, completions)]
    
    # Tokenize full sequences
    inputs = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prompt lengths to know where completions start
    prompt_inputs = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=512,
    )
    prompt_lengths = [len(ids) for ids in prompt_inputs["input_ids"]]
    
    # Forward pass through model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = inputs["input_ids"]
    
    # Collect per-token log probs for completion tokens only
    token_log_probs = []
    for i in range(len(completions)):
        start_idx = prompt_lengths[i]
        seq_len = int(inputs["attention_mask"][i].sum().item())
        
        # logits[t] predicts token[t+1], so we look at positions start_idx to seq_len-2
        completion_log_probs = []
        for t in range(start_idx, seq_len - 1):
            token_id = input_ids[i, t + 1].item()
            completion_log_probs.append(log_probs[i, t, token_id].item())
        
        token_log_probs.append(completion_log_probs)
    
    return token_log_probs


def make_reward_function(
    shaping: str = "expectation",
    kl_type: str = "none",
    kl_coef: float = 0.1,
    policy_model=None,
    ref_model=None,
    tokenizer=None,
    negate: bool = False,
    reward_module=None,
):
    """
    Factory function to create a combined reward function.
    
    This creates a reward function that:
    1. Computes base reward using sentiment scores (with optional shaping)
    2. Optionally negates the base reward (for negative sentiment optimization)
    3. Subtracts KL regularization penalty (if enabled)
    
    Args:
        shaping: Type of reward shaping:
            - "five_stars": Binary reward, 1 if P(5 stars)>=0.8, else 0
            - "expectation": Continuous expected-stars score in [0, 1]
            - "custom": Student-defined shaped_reward()
        kl_type: Type of KL regularization ("none", "forward", "backward", "k3")
        kl_coef: Coefficient for KL term
        policy_model: Current policy model (required if kl_type != "none")
        ref_model: Reference model (required if kl_type != "none")
        tokenizer: Tokenizer (required if kl_type != "none")
        negate: If True, negate the base reward (optimize for negative sentiment)
        reward_module: Module containing reward functions (rewards or rewards_solution)
    
    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    if kl_type != "none" and (policy_model is None or ref_model is None or tokenizer is None):
        raise ValueError(f"policy_model, ref_model, and tokenizer required for kl_type={kl_type}")
    
    if reward_module is None:
        import rewards_solution as reward_module
    
    # QUESTION Q4: This function receives completions and prompts as arguments from
    # TRL. Read through the 3 steps below. What is the final reward composed of?
    # Which steps are affected by the arguments --reward_shaping and --kl_type, and how?

    def reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        # Step 1: Compute completions' base reward
        if shaping == "five_stars":
            base_rewards = reward_module.five_stars_reward(completions)
        elif shaping == "expectation":
            base_rewards = reward_module.expected_stars(completions)
        elif shaping == "custom":
            expected_stars = reward_module.expected_stars(completions)
            base_rewards = reward_module.shaped_reward(expected_stars, completions, prompts)
        else:
            raise ValueError(f"Unknown shaping: {shaping}")
        
        # Step 2: Negate if optimizing for negative sentiment (before KL)
        if negate:
            base_rewards = [-r for r in base_rewards]
        
        # Step 3: Subtract KL regularization penalty if enabled
        if kl_type != "none" and prompts is not None:
            log_probs_policy = compute_log_probs(completions, prompts, policy_model, tokenizer)
            log_probs_ref = compute_log_probs(completions, prompts, ref_model, tokenizer)
            
            if kl_type == "forward":
                kl_terms = reward_module.kl_penalty_forward(
                    log_probs_policy, log_probs_ref, kl_coef
                )
            elif kl_type == "backward":
                kl_terms = reward_module.kl_penalty_backward(
                    log_probs_policy, log_probs_ref, kl_coef
                )
                # TODO let's add k3's kl term (both forward and backword)
            elif kl_type == "k3":
                kl_terms = reward_module.kl__k3_penalty_forward(
                    log_probs_policy, log_probs_ref, kl_coef
                )
            else:
                raise ValueError(f"Unknown kl_type: {kl_type}")
            
            base_rewards = [base - kl for base, kl in zip(base_rewards, kl_terms)]
        
        return base_rewards
    
    return reward_fn
