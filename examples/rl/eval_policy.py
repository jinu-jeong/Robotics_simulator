"""Evaluate a trained PPO policy on RoboSim/Reach-v0.

Usage::

    # Headless: print distance trajectory + success rate
    python examples/rl/eval_policy.py runs/ppo_reach/policy.zip --episodes 20

    # Visual: open Taichi viewer and watch the policy execute
    python examples/rl/eval_policy.py runs/ppo_reach/policy.zip --render
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
import robosim.envs  # noqa: F401  (registers ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=str, help="path to .zip from train_ppo_reach.py")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render",   action="store_true", help="open Taichi viewer")
    parser.add_argument("--seed",     type=int, default=123)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = gym.make("RoboSim/Reach-v0")
    model = PPO.load(args.policy, env=env)
    print(f"Loaded policy: {Path(args.policy).resolve()}")

    if args.render:
        run_render(env, model, seed=args.seed)
    else:
        run_headless(env, model, n_episodes=args.episodes, seed=args.seed)


def run_headless(env, model, n_episodes: int, seed: int) -> None:
    successes = 0
    total_returns = []
    final_dists   = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        last_dist = None
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            # Last 3 elements of obs = (target - ee); norm = distance.
            last_dist = float(np.linalg.norm(obs[-3:]))
            if term:
                successes += 1
                break
            if trunc:
                break
        total_returns.append(ep_ret)
        final_dists.append(last_dist)
        print(f"  ep {ep:2d}: return={ep_ret:7.2f}  final_dist={last_dist*100:5.2f} cm  "
              f"{'SUCCESS' if last_dist < 0.03 else ''}")

    print(f"\nSuccess rate     : {successes}/{n_episodes}  ({100*successes/n_episodes:.0f} %)")
    print(f"Mean return      : {np.mean(total_returns):.2f}")
    print(f"Mean final dist  : {100*np.mean(final_dists):.2f} cm")
    print(f"Median final dist: {100*np.median(final_dists):.2f} cm")


def run_render(env, model, seed: int) -> None:
    """Loop the policy in a viewer window. Press the window's close button to quit."""
    # Drill through gym's wrapper to reach the underlying RoboSimEnv (so we can
    # call scene.render() — gym wrappers don't forward custom methods).
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    scene = inner.scene

    obs, _ = env.reset(seed=seed)
    ep = 0
    step = 0
    while scene.render():
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        step += 1
        if term or trunc:
            tag = "SUCCESS" if term else "timeout"
            dist_cm = float(np.linalg.norm(obs[-3:])) * 100
            print(f"episode {ep}: {tag}  ({step} steps, final dist {dist_cm:.2f} cm)")
            ep += 1
            step = 0
            obs, _ = env.reset(seed=seed + ep)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
