import ray
from ray import tune
from ray.tune.registry import register_env
from mc_env import MountainCarEnv

def env_creator(_):
    env = MountainCarEnv()
    return env

if __name__ == '__main__':
    #register_env('MountainCar', lambda config: env_creator(None))
    ray.init()
    tune.run(
        "PPO",
        stop={"training_iteration": 10000},
        config={
            "env": "MountainCar-v0",
            "num_gpus": 0,
            "num_workers": 1,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "framework": "torch"
        },
    local_dir="MountainCar/checkpoints/",
    checkpoint_freq=20
  )