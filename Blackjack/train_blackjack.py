import ray
from ray import tune
from ray.tune.registry import register_env
import gym
from gym.envs.registration import register
from bj_env import BlackjackEnv


def env_creator(_):
        env = BlackjackEnv()
        return env

if __name__ == '__main__':

    register_env('Blackjack', lambda config: env_creator(None))
    
    ray.init()
    tune.run(
        "PPO",
        stop={"episode_reward_mean": 1},
        config={
            "env": "Blackjack",
            "num_gpus": 0,
            "num_workers": 1,
            "lr": 0.001,
            "framework": "torch"
        },
    local_dir="Blackjack/checkpoints/",
    checkpoint_freq=20
    )
    