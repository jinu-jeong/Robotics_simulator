"""Train PPO on RoboSim/Reach-v0 with Stable-Baselines3.

Install once::

    pip install robosim[rl] stable-baselines3

Run::

    python examples/rl/train_ppo_reach.py --timesteps 100000

The trained policy is saved to ``runs/ppo_reach/policy.zip``.
Evaluate with::

    python examples/rl/eval_policy.py runs/ppo_reach/policy.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import robosim.envs  # noqa: F401  (registers ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs",    type=int, default=4)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--out",       type=str, default="runs/ppo_reach")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 not installed.  Try:\n"
            "  pip install stable-baselines3"
        ) from exc

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_env(rank: int):
        def _init():
            env = gym.make("RoboSim/Reach-v0")
            env = Monitor(env, str(out_dir / f"monitor_{rank}.csv"))
            env.reset(seed=args.seed + rank)
            return env
        return _init

    if args.n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        vec_env = gym.make("RoboSim/Reach-v0")

    try:
        import tensorboard  # noqa: F401
        tb_log = str(out_dir / "tb")
    except ImportError:
        tb_log = None

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        n_steps=512,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log=tb_log,
    )
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401
        progress_bar = True
    except ImportError:
        progress_bar = False

    model.learn(total_timesteps=args.timesteps, progress_bar=progress_bar)
    model.save(str(out_dir / "policy.zip"))
    print(f"\nSaved policy → {out_dir / 'policy.zip'}")


if __name__ == "__main__":
    main()
