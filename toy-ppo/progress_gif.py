"""
Make a single GIF showing how a PPO agent improves over training.

Plays one episode per intermediate checkpoint (saved by run_ppo every SAVE_EVERY
iterations), with the training iteration and score drawn on each frame.

Usage:
    python -c "from main import run_ppo; run_ppo('LunarLander-v3')"   # train + save checkpoints
    python progress_gif.py   # (add --total N --iters ... if trained longer)  -> plots/PPO_progress_LunarLander-v3.gif
"""

import argparse
import os
import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from main import CHECKPOINT_DIR, NUM_ITERATIONS
from rl_utils import PolicyNetwork, DEVICE
from visualize import save_gif

BAR_H = 56


def load(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    policy = PolicyNetwork(ckpt["obs_dim"], ckpt["act_dim"]).to(DEVICE)
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy.eval()


def play(env, policy, max_steps):
    """Run one episode; return frames and the running score at each frame."""
    obs, _ = env.reset()
    frames, scores, score = [], [], 0.0
    for _ in range(max_steps):
        frames.append(env.render())
        scores.append(score)
        with torch.no_grad():
            action, _ = policy.get_action(torch.tensor(obs, dtype=torch.float32, device=DEVICE))
        obs, reward, terminated, truncated, _ = env.step(action.item())
        score += reward
        if terminated or truncated:
            break
    frames.append(env.render())
    scores.append(score)
    return frames, scores


def annotate(frame, it, total, score, font):
    """Add a top banner: training iteration, progress bar, episode score."""
    h, w = frame.shape[:2]
    img = Image.new("RGB", (w, h + BAR_H), (20, 24, 32))
    img.paste(Image.fromarray(frame), (0, BAR_H))
    d = ImageDraw.Draw(img)
    d.text((12, 10), f"Training iteration {it} / {total}", fill="white", font=font)
    d.text((w - 12, 10), f"Score: {score:.0f}", fill="white", font=font, anchor="ra")
    d.rectangle([12, BAR_H - 12, w - 12, BAR_H - 6], fill=(60, 66, 80))
    d.rectangle([12, BAR_H - 12, 12 + (w - 24) * it / total, BAR_H - 6], fill=(80, 200, 120))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--iters", type=int, nargs="+", default=[0, 10, 20, 40, 70, 100])
    parser.add_argument("--total", type=int, default=NUM_ITERATIONS, help="iterations the agent was trained for")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--skip", type=int, default=2, help="keep every Nth frame (smaller GIF)")
    args = parser.parse_args()

    font = ImageFont.load_default(size=24)
    env = gym.make(args.env, render_mode="rgb_array")
    env.reset(seed=0)
    out = []
    for it in args.iters:
        # the final checkpoint is saved without an _iter suffix
        path = f"{CHECKPOINT_DIR}/PPO_{args.env}_iter{it}.pt"
        if not os.path.exists(path):
            path = f"{CHECKPOINT_DIR}/PPO_{args.env}.pt"
        frames, scores = play(env, load(path), args.max_steps)
        print(f"  iter {it:3d}: score {scores[-1]:7.1f}  ({len(frames)} steps)")
        keep = list(range(0, len(frames), args.skip)) + [len(frames) - 1]  # always keep the final frame
        clip = [annotate(frames[k], it, args.total, scores[k], font) for k in keep]
        out += clip + [clip[-1]] * 20  # hold the last frame so the outcome is readable
    env.close()

    save_gif([np.asarray(f) for f in out], f"plots/PPO_progress_{args.env}.gif", fps=30)


if __name__ == "__main__":
    main()
