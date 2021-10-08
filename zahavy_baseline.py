import numpy as np
import torch
from CLTree import CLTree
from data import Data_binary, InstanceData

def explain_zahavy(args, dataset, translator, abstraction_helper, num_actions, fidelity_fn=None, model_path=None, mode="PPO"):
    num_runs = 3
    fidelities = []
    cluster_v_scores = []
    for run in range(num_runs):

        states = dataset.states
        actions = dataset.actions
        next_states = dataset.next_states
        dones = dataset.dones
        entropies = dataset.entropies
        rewards = dataset.rewards
        ids = dataset.ids
        values = dataset.values

        binary_states = []
        for s in states:
            b = translator.state_to_binary(s)
            binary_states.append(b)
        binary_states = np.array(binary_states)
        data = Data_binary(states, actions, next_states, dones, entropies, rewards, values, binary_states, ids)

        cluster_data = InstanceData(data.cluster_input, data.num_feats, translator.language_set)
        cltree = CLTree(cluster_data)     
        cltree.buildTree()
        height = args.max_height
        alpha = args.alpha
        interactive = True

        all_clusters = []
        cluster_scores = []
        lengths = []
        value_scores = []
        entropy_scores = []

        print('Starting graph generation...')
        for i in range(height):
            print('Height: ', i+1)
            interactive_config = {'height': i+1}
            cltree.pruneTree(interactive, interactive_config)        
            clusters = cltree.getClustersList(min_nr_instances=1)
            all_clusters.append(clusters)
            
            c = 0
            cluster_state_indices = []
            for i, node in enumerate(clusters):
                
                c += node.getNrInstancesInNode()
                
                cluster_state_indices.append(node.getInstanceIds())
            #print('Number of clusters: ', len(clusters))
            #print("Total instances clustered: ", c)
            #print("Percent instances clusered: ", c/dataset.num_entries)

            abstract_state_groups = []
            abstract_binary_state_groups = []
            cluster_values = []
            cluster_policies = []
            for cluster in cluster_state_indices:
                abs_t = []
                bin_t = []
                v = []
                a = np.zeros(num_actions)
                for idx in cluster:
                    idx = int(idx)
                    val = dataset.values[idx]
                    v.append(val)
                    a[dataset.actions[idx]] += 1
                    abs_t.append((dataset.states[idx], dataset.actions[idx], dataset.next_states[idx], dataset.dones[idx], dataset.entropies[idx], dataset.rewards[idx]))
                    binary = translator.state_to_binary(dataset.states[idx])
                    bin_t.append((binary, dataset.actions[idx]))
                abstract_state_groups.append(abs_t)
                abstract_binary_state_groups.append(bin_t)
                cluster_values.append(sum(v)/len(v))
                a = a / np.sum(a)
                cluster_policies.append(a)
            
            cluster_values = np.array(cluster_values)
            pred_cluster_values = []
            abs_t = abstract_state_groups
            bin_t = abstract_binary_state_groups
            l, transitions, taken_actions = abstraction_helper.compute_graph_info(abs_t)
            cl_entropies = []


            
            for j, cluster in enumerate(clusters):
                transition_probs = np.array(transitions[j][:-1])
                pred_cluster_values.append(args.lmbda * sum(transition_probs * cluster_values))
                cl_pol = np.array(cluster_policies[j])
                cl_pol_nonzero = np.where(cl_pol != 0)[0]
                cl_pol_nonzero = cl_pol[cl_pol_nonzero]
                entr = -sum(cl_pol_nonzero * np.log2(cl_pol_nonzero))
                cl_entropies.append(entr)

            #Should value score be weighted according to number of states in a cluster?
            val_score = np.linalg.norm(cluster_values - pred_cluster_values) / np.linalg.norm(cluster_values)
            val_score = np.square(cluster_values - pred_cluster_values).mean()
            entropy_score = sum(cl_entropies) / len(cl_entropies)

            print('Val score: ', val_score)
            print('Entropy score: ', entropy_score)

            score = val_score + entropy_score

            a_score = score + alpha * len(clusters)
            
            print('Cluster score (lower is better): ', a_score)

            value_scores.append(val_score)
            entropy_scores.append(entropy_score)
            cluster_scores.append(a_score)
            lengths.append(len(clusters))

        cluster_scores = np.array(cluster_scores) #shape num_graphs, num_alphas
        lengths = np.array(lengths)
        
        all_clusters = np.array(all_clusters)
        best_graph_idx = np.argsort(cluster_scores)
        best_heights = best_graph_idx[:args.k]
        best_heights = np.squeeze(best_heights)
        best_heights = np.array(best_heights, dtype=np.int32)
        if best_heights[0] == 0: #Don't want to include the low graph since its value score is always low
            best_heights = best_graph_idx[1:args.k+1]
        

        all_clusters = np.array(all_clusters)
        best_clusters = all_clusters[best_heights]
        fidelity_scores = []
        
        cluster_v_scores.append(value_scores[best_heights[0]])
        for h, clusters in enumerate(best_clusters):
            if h != 0:
                break
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
                    binary = binary_states[idx]
                    bin_t.append((binary, dataset.actions[idx]))
                abstract_state_groups.append(abs_t)
                abstract_binary_state_groups.append(bin_t)
            

            abs_t = abstract_state_groups
            bin_t = abstract_binary_state_groups
            

            critical_values, group_ent = abstraction_helper.get_critical_groups(abs_t)

            l, transitions, taken_actions = abstraction_helper.compute_graph_info(abs_t)

            """
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
            """

    np.save('fidelity_results/zahavy_lander_dqn.npy', np.array(fidelities))

