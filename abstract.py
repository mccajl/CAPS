import numpy as np
import math
import copy
import math

class APG:
    def __init__(self, num_actions, value_fn, translator, info=None, epsilon=1):
        self.info = info
        if self.info is not None: #Temporary
            self.epsilon = epsilon
            self.states = info['states']
            self.actions = info['actions']
            self.next_states = info['next_states']
            self.dones = info['dones']
            self.criticals = info['entropies']
            self.num_actions = num_actions
            self.num_features = len(self.states[0])
        self.value_fn = value_fn
        self.translator = translator
        self.num_binary = self.translator.num_predicates()
        self.lookup = []
        


    def compute_abstractions(self):
        
        c = [[] for i in range(self.num_actions)]
        c_binary = [[] for i in range(self.num_actions)]
        trajectories = [(self.states[i], self.actions[i], self.next_states[i], self.dones[i], self.criticals[i]) for i in range(len(self.states))]
        

        for t in trajectories: #t is a (state, action, next_state, done) tuple
            c[t[1]].append(t) #Sort trajectories by which action was taken
            t_binary = (self.translator.state_to_binary(t[0]), t[1])
            c_binary[t_binary[1]].append(t_binary)
        

        importance = [[] for i in range(len(c))]
        #Importance is shape [num_actions, num_binary]
        for i in range(len(c)): #Calculate feature importance in each grouping of states
            importance[i] = np.abs(np.array(self.calculate_importance(c[i], c_binary[i])))
        

        while np.max(importance) > self.epsilon:
            i_max = np.argmax(np.max(importance, axis=-1), axis=-1)
            j_max = np.argmax(importance[i_max])
            #i_max = int(np.argmax(np.max(np.max(importance, axis=-1), axis=-1))) #action with most important binary feat
            #j_max = int(np.argmax(np.max(importance[i_max], axis=-1))) #cts feat idx with most important binary feat
            #k_max = int(np.argmax(importance[i_max][j_max])) #most import binary feat idx
            c0 = []
            c1 = []
            c0_binary = []
            c1_binary = []
            for i, t in enumerate(c_binary[i_max]): #loop through each (binary) state in the most un-abstracted group
                if t[0][j_max] == 0: #Split into two groups based on values of the important feature
                    c0.append(c[i_max][i])
                    c0_binary.append(t)
                else:
                    c1.append(c[i_max][i])
                    c1_binary.append(t)
            
            
            importance[i_max] = copy.deepcopy(np.abs(np.array(self.calculate_importance(c0, c0_binary))))
            c[i_max] = copy.deepcopy(c0)
            c_binary[i_max] = copy.deepcopy(c0_binary)

            importance.append(np.abs(np.array(self.calculate_importance(c1, c1_binary))))
            c.append(c1)
            c_binary.append(c1_binary)
        

        return c, c_binary

    

    def binary_to_state(self, binary):
        pass
    

    def condense_predicate_set(self, c_binary):
        binary_states = [np.array(t[0]) for t in c_binary] #Each binary state in the abstract state group
        for i, state in enumerate(binary_states):
            if i == 0:
                result = state
            else:
                result = result * state
        
        return result


    def state_value(self, state):
        return self.value_fn(state)
    
    def calculate_importance(self, c, c_binary):
        
        c = [t[0] for t in c]
        c_binary = [t[0] for t in c_binary]

        #c is shape [num_states, num_features]
        #c_binary is shape [num_states, num_features, num_bins]

        #Function returns an array of shape [num_features, num_bins]
        #Each value is the importance in the set for each binary feature

        q_total = 0
        p0 = np.zeros(self.num_binary)
        q0 = np.zeros(self.num_binary)
        p1 = np.zeros(self.num_binary)
        q1 = np.zeros(self.num_binary)
        q_diff = np.zeros(self.num_binary)

        for i, state in enumerate(c):
            v = self.state_value(state) #get value of cts state
            q_total = q_total + v

            for f in range(self.num_binary): #Loop through all (binary) features in state to see which ones are 0
                if c_binary[i][f] == 0:
                    p0[f] = p0[f] + 1
                    q0[f] = q0[f] + v
        
        if len(c) != 0:
            q_total = q_total / len(c)

        for f in range(self.num_binary):
            if p0[f] != 0:
                q0[f] = q0[f] / p0[f]
                p0[f] = p0[f] / len(c)
            p1[f] = 1 - p0[f]
            if p1[f] != 0:
                q1[f] = (q_total - (p0[f] * q0[f])) / p1[f]
            q_diff[f] = q0[f] - q1[f]
        
        return [q_diff[f] * math.sqrt(p0[f] * p1[f]) for f in range(self.num_binary)]
    
    def compute_graph_info(self, c):
        transitions = np.zeros([len(c)+1, len(c)+1])
        actions = [[[] for i in range(len(c)+1)] for j in range(len(c) + 1)]
        
        for i in range(len(c)):
            for t in c[i]:
                state = t[0]
                self.lookup.append((state, i))

        n = None
        for i in range(len(c)):
            for t in c[i]:
                
                if t[3]:
                    n = len(c)
                    action = t[1]
                
                else:
                    for tup in self.lookup:
                        if np.array_equal(tup[0], t[2]):
                            n = tup[1]
                            action = t[1]
                    
                if n is None:
                    continue
                #assert n is not None #Sometimes this triggers. Why is this? Could happen if tup[0] is the very first state?
                transitions[i, n] = transitions[i, n] + (1/len(c[i]))
                actions[i][n].append(action)
                
        
        return self.lookup, transitions, actions
    

    def generate_lookup(self, c):
        lookup = []
        for i in range(len(c)):
            for t in c[i]:
                state = t[0]
                lookup.append((state, i))
        
        return lookup
                
    def compute_cluster_values_cts(self, c, gt_values, lmbda):
        #Need to compute, for each cluster:
        #Probability of each state appearing in each cluster
        #Reward for each state in the cluster
        #Probabilities for each state moving to each cluster
        lookup = self.generate_lookup(c)
        cluster_idx = np.arange(len(c))
        est_cluster_values = []
        for j, cluster in enumerate(c):
            size = len(cluster)
            value = 0
            for t in cluster:
                next_state_clusters = np.zeros(len(c))
                state = t[0]
                next_state = t[2]
                done = t[3]
                reward = t[5]
                for tup in lookup:
                    if np.array_equal(tup[0], next_state):
                        next_state_clusters[tup[1]] += 1
                if np.sum(next_state_clusters) == 0: #Next state was not clustered
                    continue
                next_state_clusters = next_state_clusters / np.sum(next_state_clusters)
                r_term = 0
                for i in range(len(c)):
                    r_term += next_state_clusters[i] * gt_values[i]
                r_term = r_term * lmbda
                r_term = r_term + reward
                value += r_term / size
            est_cluster_values.append(value)

        
        return est_cluster_values
            

                    
    
    def compute_cluster_values_discrete(self, c):
        #Need to compute, for each cluster:
        #Probability of each state appearing in each cluster
        #Reward for each state in the cluster
        #Probabilities for each state moving to each cluster

        pass





    def get_critical_groups(self, c):
        critical_values = np.zeros(len(c))
        group_entropies = np.zeros(len(c))
        num_criticals = math.ceil(len(c) * .15)
        for i, group in enumerate(c): #Loop through group in set of abstract states
            entropies = []
            for t in group: #Loop through states in group
               entropies.append(t[4])
            mean_entropy = np.mean(entropies)
            group_entropies[i] = mean_entropy
        
        sorted_idx = np.argsort(group_entropies)
        top = sorted_idx[:num_criticals]
        critical_values[top] = 1
        
        return critical_values, group_entropies


    








