"""
Export Tetris replays + evaluation stats for every saved PPO checkpoint (for the HTML visualization).

For each checkpoint: play GAMES evaluation games, record their stats, and keep one representative
replay (the game whose score is closest to the median) frame by frame.

Run from the repo root:
    python -m tetris.ppo_scaled      # train (saves checkpoints every SAVE_EVERY iterations)
    python -m tetris.replays         # -> tetris/plots/tetris_replays.json
"""

import argparse
import glob
import json
import re
from multiprocessing import Pool

import numpy as np
import torch

from main_tetris import make_tetris_env
from rl_utils import PolicyNetwork

GAMES = 30
MAX_STEPS = 1500   # cap per game (a good agent could otherwise play for a very long time)


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy = PolicyNetwork(ckpt["obs_dim"], ckpt["act_dim"], ckpt.get("hidden", 64), ckpt.get("layers", 2))
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy.eval(), ckpt.get("obs_scale", 1.0)


def play(env, policy, obs_scale, seed):
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    frames, actions, scores, lines = [], [], [], []
    score = n_lines = 0
    for _ in range(MAX_STEPS):
        # board incl. the falling piece; 0 = empty, 2..8 = piece ids -> encode as '0'..'7'
        board = u.crop_padding(u.project_tetromino())
        frames.append("".join(str(max(int(v) - 1, 0)) for v in board.flat))
        scores.append(score)
        lines.append(n_lines)
        with torch.no_grad():
            action, _ = policy.get_action(torch.tensor(obs, dtype=torch.float32) * obs_scale)
        obs, reward, terminated, truncated, info = env.step(action.item())
        score += float(reward)
        n_lines += int(info["lines_cleared"])
        actions.append(action.item())
        if terminated or truncated:
            break
    frames.append(frames[-1])  # final board, paired with the final score/lines
    scores.append(score)
    lines.append(n_lines)
    return {"score": score, "lines": n_lines, "steps": len(actions),
            "frames": frames, "actions": actions, "scores": scores, "lines_t": lines}


def evaluate(job):
    it, path = job
    torch.set_num_threads(1)
    torch.manual_seed(it)
    env = make_tetris_env("Tetris")
    policy, obs_scale = load(path)
    results = [play(env, policy, obs_scale, 1000 + g) for g in range(GAMES)]
    scores = np.array([r["score"] for r in results])
    lines = np.array([r["lines"] for r in results])
    steps = np.array([r["steps"] for r in results])
    replay = results[int(np.argmin(np.abs(scores - np.median(scores))))]
    print(f"  iter {it:4d}: score {scores.mean():6.1f}  lines {lines.mean():5.2f}/game"
          f"  replay {replay['score']:.0f} pts, {len(replay['frames'])} frames", flush=True)
    return {"iter": it, "score_mean": float(scores.mean()), "score_p25": float(np.percentile(scores, 25)),
            "score_p75": float(np.percentile(scores, 75)), "score_max": float(scores.max()),
            "lines_mean": float(lines.mean()), "lines_max": int(lines.max()),
            "steps_mean": float(steps.mean()), "capped": int((steps >= MAX_STEPS).sum()),
            "replay": replay}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="tetris/runs/ppo_scaled")
    parser.add_argument("--out", default="tetris/plots/tetris_replays.json")
    args = parser.parse_args()

    jobs = sorted((int(re.search(r"iter(\d+)", p).group(1)), p) for p in glob.glob(f"{args.run}/PPO_iter*.pt"))
    with Pool(min(len(jobs), 8)) as pool:  # each worker holds a torch copy; 8 keeps memory modest
        ckpts = pool.map(evaluate, jobs)

    with open(f"{args.run}/history.json") as f:
        history = json.load(f)
    for h in history:  # iterations where no game ended have no return (NaN/None)
        if h["return"] is not None and h["return"] != h["return"]:
            h["return"] = None
    env = make_tetris_env("Tetris").unwrapped
    out = {"width": env.width, "height": env.height, "games": GAMES, "max_steps": MAX_STEPS,
           "history": history, "checkpoints": ckpts}
    data = json.dumps(out, separators=(",", ":"))
    with open(args.out, "w") as f:
        f.write(data)
    with open("tetris/viz_template.html", encoding="utf-8") as f:
        html = f.read().replace("/*__DATA__*/null", data.replace("</", "<\\/"))
    html_path = args.out.replace(".json", ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved {args.out} and {html_path} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
