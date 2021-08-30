from bj_env import BlackjackEnv
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
import torch
from torch.autograd import Variable
import sys
import numpy as np


def test(model_path, num_episodes=10):

    ray.init()
    def env_creator(_):
        env = BlackjackEnv()
        return env
    
    register_env('Blackjack', env_creator)
    env = env_creator(None)
    act_dim = 2
    obs_dim = 3
    num_feats = 3

    config = {
          "env": "Blackjack",
          "num_gpus": 0,
          "num_workers": 1,
          "framework": "torch"
          }
    agent = PPOTrainer(config=config, env='Blackjack') #Will this work if I train with default cartpole env
    agent.restore(model_path)
    policy = agent.get_policy()
    model = policy.model

    def get_action(obs):
        obs = Variable(torch.from_numpy(obs))
        rnn_state = model.get_initial_state()
        seq_len = torch.Tensor([1])
        logits, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=rnn_state, seq_lens=seq_len)
        action = np.argmax(logits.detach().numpy())
        value = model.value_function().detach().numpy()
        logits = np.squeeze(logits.detach().numpy())
        probs = np.exp(logits) / sum(np.exp(logits))
        entropy = -sum(probs * np.log(probs))
        return action, value, entropy
    
    def to_one_hot(obs):
        s = np.zeros(45)
        obs = np.squeeze(obs)
        p = obs[0]
        d = obs[1]
        a = obs[2]
        s[p] = 1
        s[32+d] = 1
        s[43+a] = 1
        s = np.reshape(s, [1, -1])
        return s

    highlights_data = []
    print('Num episodes: ', num_episodes)
    for episode in range(num_episodes):
        #print("Episode: ", episode+1)
        episode_data = {'states': [], 'actions': [], 'entropy': [], 'dones': [], 'rewards': []}
        done = False
        obs = env.reset()
        obs = np.reshape(obs, [1, -1])
        total_reward = 0
        num_steps = 0
        while not done:
            action, value, entropy = get_action(to_one_hot(obs))
            episode_data['states'].append(obs)
            episode_data['actions'].append(action)
            episode_data['entropy'].append(entropy)
            next_obs, reward, done, _ = env.step(action)
            episode_data['dones'].append(int(done))
            episode_data['rewards'].append(reward)
            next_obs = np.reshape(next_obs, [1, -1])
            obs = next_obs
            total_reward = total_reward + reward
            num_steps = num_steps + 1
        
        highlights_data.append(episode_data)
        print("Reward: ", total_reward)
    
    return highlights_data, model, num_feats, act_dim



def calculate_fidelity(model_path, all_clusters, data, num_episodes=5):
    #ray.init()
    def env_creator(_):
        env = BlackjackEnv()
        return env
    
    register_env('Blackjack', env_creator)
    env = env_creator(None)
    act_dim = 2
    obs_dim = 3

    config = {
          "env": "Blackjack",
          "num_gpus": 0,
          "num_workers": 1,
          "framework": "torch"
          }
    agent = PPOTrainer(config=config, env='Blackjack') #Will this work if I train with default cartpole env
    agent.restore(model_path)
    policy = agent.get_policy()
    model = policy.model

    def get_action(obs):
        obs = Variable(torch.from_numpy(obs))
        rnn_state = model.get_initial_state()
        seq_len = torch.Tensor([1])
        logits, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=rnn_state, seq_lens=seq_len)
        action = np.argmax(logits.detach().numpy())

        return action
    
    all_actions = data.actions

    def get_cluster_action(clusters, num_feats=3, num_actions=2):
        if clusters == []:
            action = np.random.randint(0, num_actions)
            return action
        
        taken_actions = np.zeros(num_actions)
        for cluster in clusters:
            ids = cluster.getInstanceIds()
            actions = all_actions[ids]
            for i in range(len(actions)):
                taken_actions[actions[i]] = taken_actions[actions[i]] + 1
        
        policy = taken_actions / np.sum(taken_actions)

        action = np.random.choice(np.arange(num_actions), p=policy)
        return action

    
    def find_clusters(obs, clusters, num_feats=3):
        obs = np.reshape(obs, [-1])
        valid_clusters = []
        for cluster in clusters:
            in_cluster = True
            for i in range(num_feats):
                bounds = cluster.get_bounds(i)
                if obs[i] < bounds[1] or obs[i] > bounds[0]: #if feat not in cluster
                    in_cluster = False
            
            if in_cluster:
                valid_clusters.append(cluster)
        
        return valid_clusters

    action_matches = []
    for episode in range(num_episodes):
        done = False
        obs = env.reset()
        obs = np.reshape(obs, [1, -1])
        total_reward = 0
        num_steps = 0
        while not done:

            cls = find_clusters(obs, all_clusters)
            abstract_action = get_cluster_action(cls)
            
            action = get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            next_obs = np.reshape(next_obs, [1, -1])
            obs = next_obs
            total_reward = total_reward + reward

            action_matches.append(int(abstract_action==action))

            num_steps = num_steps + 1
        
    fidelity = sum(action_matches) / len(action_matches)
    
    return fidelity

def run_abstract_episode(all_clusters, data, num_episodes=3):

    def env_creator(_):
        env = MountainCarEnv()
        return env

    env = env_creator(None)
    act_dim = 2
    obs_dim = 3

    
    all_actions = data.actions

    def get_cluster_action(clusters, num_feats=3, num_actions=2):
        if clusters == []:
            action = np.random.randint(0, num_actions)
            return action
        
        taken_actions = np.zeros(num_actions)
        for cluster in clusters:
            ids = cluster.getInstanceIds()
            actions = all_actions[ids]
            for i in range(len(actions)):
                taken_actions[actions[i]] = taken_actions[actions[i]] + 1
        
        policy = taken_actions / np.sum(taken_actions)

        action = np.random.choice(np.arange(num_actions), p=policy)
        return action

    
    def find_clusters(obs, clusters, num_feats=3):
        obs = np.reshape(obs, [-1])
        valid_clusters = []
        for cluster in clusters:
            in_cluster = True
            for i in range(num_feats):
                bounds = cluster.get_bounds(i)
                if obs[i] < bounds[1] or obs[i] > bounds[0]: #if feat not in cluster
                    in_cluster = False
            
            if in_cluster:
                valid_clusters.append(cluster)
        
        return valid_clusters


    for episode in range(num_episodes):
        done = False
        obs = env.reset()
        obs = np.reshape(obs, [1, -1])
        total_reward = 0
        num_steps = 0
        while not done:

            cls = find_clusters(obs, all_clusters)
            abstract_action = get_cluster_action(cls)
            
            next_obs, reward, done, _ = env.step(abstract_action)
            next_obs = np.reshape(next_obs, [1, -1])
            obs = next_obs
            total_reward = total_reward + reward


            num_steps = num_steps + 1
            if num_steps > 200:
                done = True
        
        print("Episode {} with Abstract Policy. Reward: {}".format(episode+1, total_reward))
        


if __name__ == '__main__':
    path = sys.argv[1]
    test(path)
    



    



