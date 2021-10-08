import torch
import numpy as np
import matplotlib.pyplot as plt
from explain_utils import graph_scores
from explain_utils import cluster_data


"""
Env specific info:
run episode function
predicate class
calculate fidelity function
number of actions
number of features
value function
env name
alpha
max height
lambda
feature groups (include in predicate class)
"""
def explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn=None, apg_baseline=None, mode="PPO"):


    attr_names = translator.attr_names
    attr_names.append('State Value')
    attr_names.append('Action')
    
    num_runs = 1
    fidelities = []
    cluster_v_scores = []
    ls = []
    e_scores = []
    for run in range(num_runs):
        all_clusters, best_heights, cluster_scores, value_scores, entropy_scores, lengths = cluster_data(translator, 
                                                                                                        apg_baseline, 
                                                                                                        dataset,
                                                                                                        attr_names,
                                                                                                        args.alpha,
                                                                                                        num_actions=num_actions,
                                                                                                        lmbda=args.lmbda,
                                                                                                        k=args.k,
                                                                                                        max_height=args.max_height,
                                                                                                        model_path=model_path,
                                                                                                        env=args.env
                                                                                                    )

        

        #fname = 'lunar_2.npy'
        #np.save('./Plotting_data/score/{}'.format(fname), np.array(cluster_scores))
        #np.save('./Plotting_data/entropy/{}'.format(fname), np.array(entropy_scores))
        #np.save('./Plotting_data/value/{}'.format(fname), np.array(value_scores))
        #np.save('./Plotting_data/lengths/{}'.format(fname), np.array(lengths))

        """
        graph_scores('cart', alpha, lengths, 
                    cluster_scores=cluster_scores, 
                    value_scores=value_scores, 
                    entropy_scores=entropy_scores,
                    fidelity_scores=fidelity_scores)
        """

        all_clusters = np.array(all_clusters)
        best_clusters = all_clusters[best_heights]
        

        fidelity_scores = []
        cluster_v_scores.append(value_scores[best_heights[0]])
        
        
        for h, clusters in enumerate(best_clusters):
            #if h != 0:
                #break
            print('***********************************************')
            print('Clusters at height {}'.format(best_heights[h]+1))
            
            c = 0
            cluster_state_indices = []
            for i, node in enumerate(clusters):
                
                c += node.getNrInstancesInNode()
                
                cluster_state_indices.append(node.getInstanceIds())

            if fidelity_fn is not None:
                if h == 0:
                    assert model_path is not None
                    fidelity = fidelity_fn(model_path, clusters, dataset, mode=mode)
                    print('Fidelity: ', fidelity)
                    
                    fidelities.append(fidelity)
                    fidelity_scores.append(fidelity)


            abstract_state_groups = []
            abstract_binary_state_groups = []
            for cluster in cluster_state_indices:
                abs_t = []
                bin_t = []
                for idx in cluster:
                    idx = int(idx)
                    abs_t.append((dataset.states[idx], dataset.actions[idx], dataset.next_states[idx], dataset.dones[idx], dataset.entropies[idx], dataset.rewards[idx]))
                    binary = translator.state_to_binary(dataset.states[idx])
                    bin_t.append((binary, dataset.actions[idx]))
                abstract_state_groups.append(abs_t)
                abstract_binary_state_groups.append(bin_t)
            

            abs_t = abstract_state_groups
            bin_t = abstract_binary_state_groups
            

            critical_values, group_ent = apg_baseline.get_critical_groups(abs_t)

            l, transitions, taken_actions = apg_baseline.compute_graph_info(abs_t)

            
            for j in range(len(transitions)):
                nonzero_idx = np.where(np.array(transitions[j])!=0)[0]
                for idx in nonzero_idx:
                    print('Group {} to Group {} with p={} and action {}'.format(j+1, idx+1, transitions[j][idx], int(np.mean(taken_actions[j][idx]))))
            
            if args.hayes_baseline: #Hayes and Shah baseline
                hayes_translations = translator.reduce_logic(bin_t)
            
            #CAPS explanation producer
            print('----------------------------------------')
            translations = translator.my_translation_algo(bin_t)
            for j, t in enumerate(translations):
                print('Group {}: {}'.format(j+1, t))
                if args.hayes_baseline:
                    print('(Hayes) Group {}: {}'.format(j+1, hayes_translations[j]))
                print('Critical value: {}. Entropy: {:.2f}'.format(critical_values[j], group_ent[j]))
            print('----------------------------------------')
        
        

    #e_data = [ls, e_scores]
    #np.save('entropy_results/caps_mc_4.npy', np.array(e_data))    
    #np.save('fidelity_results/caps_lander_dqn.npy', np.array(fidelities))

