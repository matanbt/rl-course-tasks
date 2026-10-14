# RL Fine-tuning for Positive Sentiment Generation

RL fine-tuning of Language Models for positive sentiment using GRPO.

This repo was written by Ido Greenberg with advisement by Oran Lang, Yoad Tewel and Gal Chechik for the course [RL-for-Real](https://docs.google.com/document/d/1fmfYp7EH9fqcB7CWWBvrZ40MtCN89Sr_o3o3EG9hWyE/edit?usp=sharing).
The course is organized by NVIDIA Research in collaboration with Google Research, Mentee Robotics, Tel-Aviv University, Bar-Ilan University, and the Technion.

## Overview

In this exercise, you will:
1. Train GPT-2 to generate positive sentiment text.
2. Face the challenge of sparse rewards and resolve it via reward design.
3. Face reward hacking and resolve it via KL divergence regularization.
4. Reshape rewards to obtain different model behaviors.
5. Compare RL vs. prompt engineering.

## Prerequisites

- Python 3.8+
- GPU with 8GB VRAM
- Basic understanding of PyTorch, Transformers, and RL concepts

---

## Setup

### 1. Install Miniconda (if needed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r
```

### 2. Create Environment

```bash
conda create -n sentiment python=3.10 -y
conda activate sentiment
```

### 3. Install PyTorch and Dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python rewards.py  # Should show NotImplementedError for student functions
```

---

## Exercises

### Exercise 1: Review the Pipeline

(Note: to save time, you may start running Exercise 2 before reviewing the questions in Exercise 1.)

Read through the codebase to understand the full RL fine-tuning pipeline. There are **10 comprehension questions** (Q1–Q10) embedded as comments throughout the code. Search for `QUESTION Q` across these files:

- `data.py` — Q1, Q2 (data and prompts)
- `rewards.py` — Q3 (reward conversion)
- `reward_utils.py` — Q4 (reward function wiring)
- `train.py` — Q5–Q10 (validation, training data, GRPO config, temperature, KL, and trainer)

Read each question in context, and make sure you can answer it before moving on.

> You may want to disable autocomplete agents like Cursor, which may accidently answer for you :)

### Exercise 2: Train for Positive Sentiment

Run the vanilla GRPO fine-tuning using:

```bash
python train.py --reward_shaping five_stars
```

* Observe the results. Does the model learn to generate more positive text?
* `--reward_shaping five_stars` tells the trainer to use the reward function `five_stars_reward()` in `rewards.py`. Considering this reward, how can you explain the result of the training? (Hint: check the train rewards on WandB)

### Exercise 3: Non-sparse Reward

Implement `expected_stars()` in `rewards.py`, and train with this new reward function:

```bash
python train.py
```
> Note: `--reward_shaping expectation` is already the default, so no need to state it explicitly.

Observe the results - both the numeric validation scores and model output examples.
* Did the model learn to be positive? Compare the learning curves to the binary reward from Exercise 2.
* Do the outputs look like natural language? What might be going wrong?
* Given the model output examples, would you expect the perplexity score to be high or low? Does the actual perplexity metric match your expectation? How can you explain this?

### Exercise 4: KL Regularization

* In `rewards.py`, implement `kl_penalty_forward()` and `kl_penalty_backward()` to prevent the model from drifting too far from the original GPT-2.
* Run and compare the forward and backward regularizations.
* Is the learned model highly positive? Does it provide sensible writing?
* Tune the regularization coefficient to achieve a model that is both sensible and positive. Try to get the mean sentiment score above 0.8.

To train with your KL regularization, use `kl_type` (with `forward` or `backward`) and `kl_coef` (with a numeric coefficient). For example:

```bash
python train.py --kl_type forward
```

> **Technical note**: TRL has built-in KL regularization, but in this exercise you implement it yourself. You receive pre-computed per-token log probabilities for both the current policy and the reference model. These are re-calculated outside TRL so that you can access them without modifying TRL's interface.

### Exercise 5: Reward Shaping

Now that KL regularization keeps the model grounded, this is your chance to get creative.
Shape the reward however you like - force your will on the language model
and watch how different incentives translate within minutes into entirely different outputs.

Implement `shaped_reward()` in `rewards.py`  to modify the raw sentiment scores, and train via:

```bash
python train.py --reward_shaping custom
```

Among the reward metrics you tried, which ones learned well and which ones were harder to optimize?
Try to explain why.

We encourage you to come up with your own reward function. Below are a few ideas just to get you in the mood (see `shaped_reward()` in `rewards_solution.py` for a concrete example):
- Numerical transformation: Exponent, negation, etc.
- Length penalty: Encourage responses of certain lengths
- Repetition penalty: Penalize repetitive outputs
- Rhyme bonus: Reward words that share the same suffix
- Mood whiplash: Start negative, end positive (or vice versa)
- Vocabulary spice: Reward uncommon words or penalize boring ones

### Exercise 6: RL vs. Prompt Engineering

So far, you demonstrated that properly-used RL is a powerful tool for tuning an LLM to your needs.
But could one obtain a similar effect using *Prompt Engineering*, to guide the LLM toward the desired behavior?

In this exercise, you will (a) compare RL vs. Prompt Engineering, (b) demonstrate that they can also work together.

#### Code infrastructure

`prompt_engineering.py` lets you test the Base GPT-2 against your RL-trained model, with various modified prompts.
The code allows you to try different prompt-prefixes, which are just strings that are concatenated before the beginning of every test prompt.

By default, `prompt_engineering.py` uses built-in sentiment prefixes:

```bash
# Default: built-in prompt strategies, sentiment scoring
python prompt_engineering.py

# Compare against your RL-trained model
python prompt_engineering.py --trained_model ./outputs/final
```

However, in the exercise you will supply your own prefixes and your own score metric, using your custom reward function from Exercise 5.

#### Your task

* Choose one of the metrics you optimized in Exercise 5. Make sure your chosen metric is the one currently implemented in `shaped_reward()`.
* Suggest a few prompts that may improve this metric, and add them via `PROMPT_STRATEGIES` in `prompt_engineering.py`. Keep `no prefix` as a baseline.
* Compare the RL agent vs. the prompt-enhanced GPT-2, with your shaped reward as a score metric:
```bash
# Use your shaped_reward() as the scoring metric
python prompt_engineering.py --use_shaped_reward --trained_model ./outputs/final
```

Summarize your findings.
Where does prompt engineering fall short? Why is prompt engineering especially hard with a base (non-instruction-tuned) model like GPT-2? Did prompting and RL together do better than each of the two on their own?

---

## Training parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--reward_shaping` | five_stars, expectation, custom | expectation |
| `--kl_type` | none, forward, backward | none |
| `--kl_coef` | Custom KL strength | 5.0 |

Extended parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--beta` | TRL's internal KL strength | 0.0 |
| `--negate_reward` | Optimize for negative sentiment | False |
| `--use_solution` | Use solution code instead of student code | False |
| `--no_wandb` | Disable Weights & Biases logging | False |
| `--no_perplexity` | Disable reference perplexity evaluation | False |

---

## Project Structure

```
sentiment/
├── train.py              # Main training script (GRPO + TRL)
├── rewards.py            # Student exercises (implement TODOs here)
├── rewards_solution.py   # Reference solutions
├── reward_utils.py       # Reward infrastructure (wiring + log probs)
├── sentiment.py          # Sentiment scoring model
├── data.py               # Prompt datasets (train + validation)
├── evaluate.py           # Post-training evaluation
├── prompt_engineering.py # RL vs prompt engineering comparison
└── README.md
```

---

## Troubleshooting

**Out of Memory**: Add `--use_peft` for LoRA training

**Import Errors**: `pip install -r requirements.txt --upgrade`

---

## Expected Results

### Base model

```
Prompt: I would describe this film as
Generated: something out of pop theater's past: it could make many a girl shudder. I was in college when people took photographs of...
```

### RL tuning with sparse reward (Ex. 2)

Same as above (no learning).

### RL tuning with non-sparse reward (Ex. 3)

```
Prompt: I would describe this film as
Generated: incredible! amazing! amazing, all. truly! amazing. absolutely and really absolutely. really.! absolutely. truly!. truly! absolutely and so. truly. truly! my. perfectly. perfect. perfect. incredible...
```

### Regularized RL tuning (Ex. 4)

```
Prompt: I would describe this film as
Generated: having the most unique and creative story, both visually and narratively. For that I would say that this character has become a key player both inside the theater and in real life for the series.
```

Hint: the default KL coefficient in the code is very conservative. The example above was generated by a model trained with a modified KL coefficient.

---

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
