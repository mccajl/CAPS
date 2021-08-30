import ray
from ray import tune
from ray.tune.registry import register_env
from ll_env import LunarLander

def env_creator(_):
    env = LunarLander()
    return env

if __name__ == '__main__':
    register_env('LunarLander', lambda config: env_creator(None))
    ray.init()
    tune.run(
        "PPO",
        stop={"episode_reward_mean": 240},
        config={
            "env": "LunarLander",
            "num_gpus": 0,
            "num_workers": 1,
            "lr": 0.0001,
            "framework": "torch"
        },
    local_dir="LunarLander/checkpoints/",
    checkpoint_freq=20
  )