import ray
from ray import tune
from ray.tune.registry import register_env
from cliffwalking import CliffWalkingEnv

def env_creator(_):
    env = CliffWalkingEnv()
    return env

if __name__ == '__main__':
    register_env('CliffWalking', lambda config: env_creator(None))
    ray.init()
    tune.run(
        "PPO",
        stop={"episode_reward_mean": -13},
        config={
            "env": "CliffWalking",
            "num_gpus": 0,
            "num_workers": 1,
            "lr": 0.0001,
            "framework": "torch"
        },
    local_dir="Gridworld/checkpoints/",
    checkpoint_freq=10
  )