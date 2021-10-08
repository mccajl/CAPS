import matplotlib.pyplot as plt
import numpy as np
from CLTree import CLTree
from data import InstanceData


def cluster_data(translator, abstraction_helper, dataset, attr_names, alpha, num_actions, lmbda, k, max_height=20, model_path=None, env='grid'):
    cluster_data = InstanceData(dataset.cluster_input, dataset.num_feats+2, attr_names)
    cltree = CLTree(cluster_data)     
    cltree.buildTree()
                    

    height = max_height
    interactive_config = {'height': height}
    interactive = True

    cluster_scores = []
    lengths = []
    value_scores = []
    entropy_scores = []
    all_clusters = []

    test_alphas = np.arange(21)
    test_alphas = test_alphas / 100
    test_alpha_scores = []

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
            pred_cluster_values.append(lmbda * sum(transition_probs * cluster_values))
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

        """
        test_a_scores = []
        for test_a in test_alphas:
            t_a = score + test_a * len(clusters)
            test_a_scores.append(t_a)
        test_alpha_scores.append(test_a_scores)
        """
        
        print('Cluster score (lower is better): ', a_score)

        value_scores.append(val_score)
        entropy_scores.append(entropy_score)
        cluster_scores.append(a_score)
        lengths.append(len(clusters))

    cluster_scores = np.array(cluster_scores) #shape num_graphs, num_alphas
    lengths = np.array(lengths)

    """
    best_scores_by_alpha = np.argmin(test_alpha_scores, axis=0)
    best_graph_size_by_alpha = lengths[best_scores_by_alpha]
    plot_data = np.array([best_graph_size_by_alpha, test_alphas])
    np.save('temp_plot_data/{}_num_nodes_vs_alpha.npy'.format(env), plot_data)
    """


    best_graph_idx = np.argsort(cluster_scores)
    best_graphs = best_graph_idx[:k]
    best_graphs = np.squeeze(best_graphs)
    best_graphs = np.array(best_graphs, dtype=np.int32)
    if best_graphs[0] == 0: #Don't want to include the low graph since its value score is always low
        best_graphs = best_graph_idx[1:k+1]
    
    return all_clusters, best_graphs, cluster_scores, value_scores, entropy_scores, lengths

def graph_scores(env, alpha, lengths, cluster_scores=None, value_scores=None, entropy_scores=None, fidelity_scores=None):
    plt.style.use('ggplot')
    if cluster_scores is not None:
        """
        plt.plot(lengths, cluster_scores, color='orange')
        plt.xlabel('Number of Graph Nodes')
        plt.ylabel('Heuristic Score')
        plt.title('Score vs Graph Size (alpha={0:.3f})'.format(alpha))
        plt.savefig('results/heuristic_graphs/{}_score_vs_size.png'.format(env))
        plt.clf()
        """
        plot_data = np.array([cluster_scores, lengths])
        np.save('temp_plot_data/{}_score_vs_size.npy'.format(env), plot_data)

        log_scores = np.log(cluster_scores)
        plot_data = np.array([log_scores, lengths])
        np.save('temp_plot_data/{}_log_score_vs_size.npy'.format(env), plot_data)

    if value_scores is not None:
        """
        plt.plot(lengths, value_scores, color='orange')
        plt.xlabel('Number of Graph Nodes')
        plt.ylabel('Value Score')
        plt.title('Value Score vs Graph Size')
        plt.savefig('results/heuristic_graphs/{}_val_score_vs_size.png'.format(env))
        plt.clf()
        """

        plot_data = np.array([value_scores, lengths])
        np.save('temp_plot_data/{}_value_vs_size.npy'.format(env), plot_data)

    if entropy_scores is not None:
        """
        plt.plot(lengths, entropy_scores, color='orange')
        plt.xlabel('Number of Graph Nodes')
        plt.ylabel('Graph Policy Entropy')
        plt.title('Entropy Score vs Graph Size')
        plt.savefig('results/heuristic_graphs/{}_entropy_score_vs_size.png'.format(env))
        plt.clf()
        """

        plot_data = np.array([entropy_scores, lengths])
        np.save('temp_plot_data/{}_entropy_vs_size.npy'.format(env), plot_data)

    if fidelity_scores is not None:
        
        plt.plot(lengths, fidelity_scores, color='orange')
        plt.xlabel('Number of Graph Nodes')
        plt.ylabel('Graph Policy Fidelity')
        plt.title('Fidelity vs Graph Size')
        plt.savefig('results/heuristic_graphs/{}_fidelity_vs_size.png'.format(env))
        plt.clf()