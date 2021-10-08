import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='grid') #Environment (grid, cart, mountain)
    parser.add_argument('--path', default='') #Path to RLlib pre-trained model
    parser.add_argument('--num_episodes', type=int, default=3) #Number of episodes to collect data from
    parser.add_argument('--calc_fidelity', default=False) #calculate fidelity of generated graphs
    parser.add_argument('--alpha', type=float, default=0.015) #Alpha parameter
    parser.add_argument('--k', type=int, default=3) #Number of graphs to produce
    parser.add_argument('--max_height', type=int, default=10) #Maximum height of CLTree
    parser.add_argument('--lmbda', type=float, default=1) #Lambda value from RL training
    parser.add_argument('--hayes_baseline', default=False) #Whether to use Hayes and Shah 2017 baseline for explanations
    parser.add_argument('--topin_baseline', default=False) #Whether to use Topin and Veloso 2019 baseline for apg gen
    parser.add_argument('--zahavy_baseline', default=False) #Whether to cluster states according to Zahavy methodology
    parser.add_argument('--alg', default='DQN') #Training algorithm. DQN and PPO supported currently
    args = parser.parse_args()

    return args
