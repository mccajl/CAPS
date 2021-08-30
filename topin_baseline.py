import torch
import numpy as np
import matplotlib.pyplot as plt



def gen_apg(apg_baseline):

    abstract_t, abstract_t_binary = apg_baseline.compute_abstractions()
    l, transitions, taken_actions = apg_baseline.compute_graph_info(abstract_t)

    for j in range(len(transitions)):
        nonzero_idx = np.where(np.array(transitions[j])!=0)[0]
        for idx in nonzero_idx:
            print('Group {} to Group {} with p={} and action {}'.format(j+1, idx+1, transitions[j][idx], int(np.mean(taken_actions[j][idx]))))
