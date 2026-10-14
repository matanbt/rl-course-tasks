"""
REINFORCE vs PPO on Tetris — a minimal adaptation of main.py.

Requires:  pip install tetris-gymnasium   (only opencv-python is needed besides; jax is not)

Run:
    python main_tetris.py
"""

from functools import partial

import gymnasium as gym
import tetris_gymnasium.envs  # registers "tetris_gymnasium/Tetris"
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

import main
from main import run_reinforce, run_ppo, compute_returns, compute_gae
from rl_utils import plot_comparison, SEED

GAMMA = 0.99  # Tetris episodes are long; gamma=1 makes returns too high-variance


def make_tetris_env(env_id, seed=SEED):
    env = gym.make("tetris_gymnasium/Tetris")
    env = FeatureVectorObservation(env)  # board -> column heights, max height, holes, bumpiness
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


# run_reinforce / run_ppo look these up as main.py globals at call time, so patch them there
main.make_env = make_tetris_env
main.compute_returns = partial(compute_returns, gamma=GAMMA)
main.compute_gae = partial(compute_gae, gamma=GAMMA)
main.CHECKPOINT_DIR = "tetris/checkpoints"


if __name__ == "__main__":
    # "Tetris" only names the checkpoints/plots; make_tetris_env builds the actual env
    results = {"REINFORCE": run_reinforce("Tetris"), "PPO": run_ppo("Tetris")}
    plot_comparison(results, "REINFORCE vs PPO — Tetris", save_path="tetris/plots/Tetris.png")
