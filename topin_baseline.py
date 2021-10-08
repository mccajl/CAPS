import torch
import numpy as np
import matplotlib.pyplot as plt


def get_apg_action(abs_t, obs, num_actions):
    """
    abs_t is an array of shape [num_clusters, num states in cluster, (state, act, ...)]
    """
    states_actions = []
    for cluster in abs_t:
        for t in cluster:
            if np.array_equal(t[0], obs):
                return t[1]
    
    """
    states = [t[0] for t in abs_t]
    
    for i, c in enumerate(states):
        for s in c:
            if np.array_equal(s, obs):
                return i
    """
    return np.random.randint(num_actions)

def get_mean_values(abs_t):
    c_means = []
    for c in abs_t:
        pos = []
        v = []
        for s in c:
            state = s[0]
            pos.append(state[0])
            v.append(state[1])
        pos_mean = sum(pos) / len(pos)
        v_mean = sum(v) / len(v)
        c_means.append((pos_mean, v_mean))
    return c_means

def gen_apg(apg_baseline, model_path=None, calc_fidelity=None, mode="PPO"):
    num_runs = 10
    fidelities = []
    for run in range(num_runs):
        abstract_t, abstract_t_binary = apg_baseline.compute_abstractions()
        #print(np.array(abstract_t).shape)
        #print(np.array(abstract_t[0]).shape)
        l, transitions, taken_actions = apg_baseline.compute_graph_info(abstract_t)

        #vals = get_mean_values(abstract_t)
        if calc_fidelity is not None:
            fidelity = calc_fidelity(model_path, abstract_t, None, topin=True, apg_act=get_apg_action, mode=mode)
            print("Fidelity: ", fidelity)
            fidelities.append(fidelity)
        """
        for j in range(len(vals)):
            print("Group {} Pos Mean: {}. Vel Mean: {}".format(j+1, vals[j][0], vals[j][1]))
        
        for j in range(len(transitions)):
            nonzero_idx = np.where(np.array(transitions[j])!=0)[0]
            for idx in nonzero_idx:
                print('Group {} to Group {} with p={} and action {}'.format(j+1, idx+1, transitions[j][idx], int(np.mean(taken_actions[j][idx]))))
        """

    np.save('fidelity_results/topin_lander.npy', np.array(fidelities))