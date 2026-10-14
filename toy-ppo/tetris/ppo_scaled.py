"""
Scaled-up PPO for Tetris: parallel games + GPU.

Same algorithm as main.run_ppo (reuses its GAE, PPO and value losses), but:
  - NUM_ENVS games run in parallel worker processes (one CPU core each)
  - the networks run on the GPU (if available), on a whole batch of games at once
  - bigger, deeper networks, bigger batches, many more iterations
  - board features are scaled down so the tanh units don't saturate
  - GAE with lambda=0.95 and a small entropy bonus (standard PPO settings)

Run from the repo root:
    python -m tetris.ppo_scaled
"""

import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from main import compute_gae, compute_ppo_loss, compute_value_loss
from main_tetris import make_tetris_env, GAMMA
from rl_utils import PolicyNetwork, ValueNetwork, SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ENVS       = 16       # parallel games
NUM_STEPS      = 512      # steps per game per iteration -> 8192 transitions / iteration
NUM_ITERATIONS = 300
HIDDEN         = 256
LAYERS         = 4        # hidden layers (main.py uses 2)
LR             = 3e-4
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
UPDATE_EPOCHS  = 10
MINIBATCH      = 1024
ENT_COEF       = 0.01
OBS_SCALE      = 1 / 20   # features are column heights (0-20), holes and bumpiness
SAVE_EVERY     = 10
RUN_DIR        = "tetris/runs/ppo_scaled"


def save(policy, obs_dim, act_dim, it):
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim, "hidden": HIDDEN, "layers": LAYERS, "obs_scale": OBS_SCALE,
                "policy_state_dict": policy.state_dict()}, f"{RUN_DIR}/PPO_iter{it}.pt")


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    torch.set_num_threads(2)  # more threads fight the NUM_ENVS game workers for cores (measured: 20 threads ~8x slower)
    torch.manual_seed(SEED)
    envs = gym.vector.AsyncVectorEnv(
        [lambda i=i: make_tetris_env("Tetris", seed=SEED + i) for i in range(NUM_ENVS)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,  # like the single-env loop: reset right when done
    )
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = int(envs.single_action_space.n)
    policy = PolicyNetwork(obs_dim, act_dim, HIDDEN, LAYERS).to(DEVICE)
    value_net = ValueNetwork(obs_dim, HIDDEN, LAYERS).to(DEVICE)
    params = list(policy.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    print(f"Device: {DEVICE} | {NUM_ENVS} envs x {NUM_STEPS} steps x {NUM_ITERATIONS} iterations")

    obs, _ = envs.reset(seed=SEED)
    ep_ret = np.zeros(NUM_ENVS)
    history = []
    t0 = time.time()
    for it in range(NUM_ITERATIONS):
        if it % SAVE_EVERY == 0:
            save(policy, obs_dim, act_dim, it)

        # ── rollout: all games step together; the networks see a (NUM_ENVS, obs_dim) batch ──
        b_obs = torch.zeros(NUM_STEPS, NUM_ENVS, obs_dim)
        b_act, b_logp, b_val, b_rew, b_done = (torch.zeros(NUM_STEPS, NUM_ENVS) for _ in range(5))
        finished = []
        for t in range(NUM_STEPS):
            o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE) * OBS_SCALE
            with torch.no_grad():
                action, log_prob = policy.get_action(o)
                value = value_net(o)
            obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done = terminated | truncated
            ep_ret += reward
            finished += ep_ret[done].tolist()
            ep_ret[done] = 0
            b_obs[t], b_act[t], b_logp[t], b_val[t] = o.cpu(), action.cpu(), log_prob.cpu(), value.cpu()
            b_rew[t], b_done[t] = torch.as_tensor(reward), torch.as_tensor(done, dtype=torch.float32)
        with torch.no_grad():
            last_val = value_net(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE) * OBS_SCALE).cpu()

        # ── advantages: main.compute_gae, once per game ──
        adv, ret = zip(*[compute_gae(b_rew[:, e], b_val[:, e], b_done[:, e], last_val[e], GAMMA, GAE_LAMBDA)
                         for e in range(NUM_ENVS)])
        adv = torch.stack(adv, 1).flatten().to(DEVICE)
        ret = torch.stack(ret, 1).flatten().to(DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        f_obs, f_act = b_obs.flatten(0, 1).to(DEVICE), b_act.flatten().to(DEVICE)
        f_logp = b_logp.flatten().to(DEVICE)

        # ── PPO update (same losses as main.py, plus an entropy bonus) ──
        n = len(f_obs)
        for _ in range(UPDATE_EPOCHS):
            perm = torch.randperm(n, device=DEVICE)
            for start in range(0, n, MINIBATCH):
                mb = perm[start:start + MINIBATCH]
                dist = torch.distributions.Categorical(logits=policy(f_obs[mb]))
                loss = (compute_ppo_loss(f_logp[mb], dist.log_prob(f_act[mb]), adv[mb], CLIP_EPS)
                        + 0.5 * compute_value_loss(value_net, f_obs[mb], ret[mb])
                        - ENT_COEF * dist.entropy().mean())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                optimizer.step()

        mean_ret = float(np.mean(finished)) if finished else None  # no game ended this iteration
        history.append({"iter": it + 1, "return": mean_ret, "episodes": len(finished),
                        "steps": (it + 1) * NUM_ENVS * NUM_STEPS, "seconds": round(time.time() - t0, 1)})
        with open(f"{RUN_DIR}/history.json", "w") as f:
            json.dump(history, f)
        if (it + 1) % 5 == 0:
            print(f"  iter {it+1:4d} | return {mean_ret or 0:7.1f} | {len(finished):4d} games"
                  f" | {time.time() - t0:6.0f}s", flush=True)

    save(policy, obs_dim, act_dim, NUM_ITERATIONS)
    envs.close()


if __name__ == "__main__":
    main()
