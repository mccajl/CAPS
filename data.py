import numpy as np
import copy

class Data_binary:
    def __init__(self, states, actions, next_states, dones, entropies, rewards, values, bin_states, ids):
        """
        Same as data except made to accept binary predicate data for the Zahavy et al baseline
        """
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.entropies = entropies
        self.rewards = rewards
        self.values = values
        self.dones = dones
        self.bin_states = bin_states
        self.num_feats = len(self.bin_states[0])
        self.num_entries = len(self.bin_states)
        self.ids = ids
        self.cluster_input = np.concatenate([self.bin_states, np.expand_dims(self.ids, axis=1)], axis=1)
        self.data_bounds = self.get_bounds()
    
    def get_bounds(self):

        bounds = {'max': [], 'min': []}
        data = self.bin_states
        for feat in range(self.num_feats):
            bounds['max'].append(np.max(data[:, feat]))
            bounds['min'].append(np.min(data[:, feat]))
        
        return bounds


class Data:
    def __init__(self, dataset, value_fn, normalize_value=True):
        """
        dataset: list of length num_episodes
            each element is a dictionary with:
                'states': np array of episode states
                'actions': np array of episode actions
                'entropy': np array of entropy measure for critical-ness.
                'dones': episode dones for each timestep
                'rewards': episode reward for each timestep
        value_fn: value function from RL environment.
        normalize_value: whether state values should be normalized between 0 and 1
        """

        self.dataset = dataset
        self.value_fn = value_fn
        self.normalize_value = normalize_value
        self.num_feats, self.states, self.actions, self.next_states, self.entropies, self.dones, self.values, self.rewards = self.parse_data()
        self.ids = np.arange(self.num_entries)
        
        self.states_and_values = np.concatenate([self.states, np.expand_dims(self.values, axis=1)], axis=1)
        self.states_values_actions = np.concatenate([self.states_and_values, np.expand_dims(self.actions, axis=1)], axis=1)
        self.cluster_input = np.concatenate([self.states_values_actions, np.expand_dims(self.ids, axis=1)], axis=1)
        self.data_bounds = self.get_bounds()

    def parse_data(self):
        states = []
        actions = []
        next_states = []
        entropy = []
        done = []
        reward = []

        num_feats = len(np.reshape(np.array(self.dataset[0]['states'][0]), [-1]))

        for e in self.dataset:
            if e['dones'][-1]: #if episode ends on a done
                state = np.reshape(e['states'], [-1, num_feats])
                next_state = state[1:]
                next_state = np.reshape(next_state, [-1, num_feats])
                dummy_state = np.zeros(shape=[1, num_feats])
                next_state = np.concatenate([next_state, dummy_state], axis=0)
                states.append(state)
                next_states.append(np.array(next_state))
                actions.append(np.squeeze(e['actions']))
                entropy.append(np.squeeze(e['entropy']))
                done.append(np.squeeze(e['dones']))
                reward.append(np.squeeze(e['rewards']))
            else:
                s = np.squeeze(e['states'])
                state = s[:-1]
                next_state = s[1:]
                states.append(state)
                next_states.append(next_state)
                actions.append(np.squeeze(e['actions'])[:-1])
                entropy.append(np.squeeze(e['entropy'])[:-1])
                done.append(np.squeeze(e['dones'])[:-1])
                reward.append(np.squeeze(e['rewards'])[:-1])
        
        #Manually flatten in the case of unequal length episodes
        
    
        s = np.array(states[0])
        ns = np.array(next_states[0])
        a = np.array(actions[0])
        e = np.array(entropy[0])
        d = np.array(done[0])
        r = np.array(reward[0])
        if np.array(a).ndim == 0:
                a = np.array([a])
                e = np.array([e])
                d = np.array([d])
                r = np.array([r])
        for i in range(len(states)-1):
            if np.array(actions[i+1]).ndim == 0:
                actions[i+1] = [actions[i+1]]
                entropy[i+1] = [entropy[i+1]]
                done[i+1] = [done[i+1]]
                reward[i+1] = [reward[i+1]]
            s = np.concatenate([s, np.array(states[i+1])], axis=0) #concat along timestep axis
            ns = np.concatenate([ns, np.array(next_states[i+1])], axis=0)
            a = np.concatenate([a, np.array(actions[i+1])], axis=0)
            e = np.concatenate([e, np.array(entropy[i+1])], axis=0)
            d = np.concatenate([d, np.array(done[i+1])], axis=0)
            r = np.concatenate([r, np.array(reward[i+1])], axis=0)



        states = np.reshape(s, [-1, num_feats])
        self.num_entries = len(states)
        next_states = np.reshape(ns, [self.num_entries, num_feats])
        actions = np.reshape(a, [-1])
        entropy = np.reshape(e, [-1])
        done = np.reshape(d, [-1])
        reward = np.reshape(r, [-1])

        values = []
        for entry in states:
            values.append(self.value_fn(entry))
        values = np.reshape(np.array(values), [-1])

        if self.normalize_value:
            values = (values - np.min(values)) / (np.max(values) - np.min(values))

        return num_feats, states, actions, next_states, entropy, done, values, reward
    
    def get_bounds(self):

        bounds = {'max': [], 'min': []}
        data = self.states_and_values
        for feat in range(self.num_feats+1):
            bounds['max'].append(np.max(data[:, feat]))
            bounds['min'].append(np.min(data[:, feat]))
        
        return bounds
    

class InstanceData:
    def __init__(self, instance_values, num_feats, attr_names):
        self.instance_values = instance_values #Data shape [num_points, num_feats+1] where the +1 is the unique id given to each data point
        self.num_feats = num_feats #Should include value feature
        self._init_max_min()
                    
        self.nr_virtual_points = len(self.instance_values)
        self.nr_total_instances = 2*self.nr_virtual_points

        self.attr_names = attr_names #Should include value

    def _init_max_min(self):
        #print(len(self.instance_values))
        if len(self.instance_values) > 1:          
            self.max_values = np.amax(self.instance_values[:, :-1], 0) 
            self.min_values = np.amin(self.instance_values[:, :-1], 0)             
        else:
            self.max_values = np.squeeze(copy.copy(self.instance_values[:, :-1]))
            self.min_values = np.squeeze(copy.copy(self.instance_values[:, :-1]))

    def calculate_limits(self):
        self._init_max_min()

    def length(self):
        return len(self.instance_values)
    
    def sort(self, idx):
        feat_values = self.instance_values[:, idx]
        sorted_idx = np.argsort(feat_values)
        self.instance_values = self.instance_values[sorted_idx] #instance values, sorted from low to high on the index

    def get_max(self, idx):
        return self.max_values[idx]

    def get_min(self, idx):
        return self.min_values[idx]

    def set_max(self, idx, value):
        if len(self.max_values) > 0:
            self.max_values[idx] = value

    def set_min(self, idx, value):
        if len(self.min_values) > 0:
            self.min_values[idx] = value
    
    def getInstances(self, idx):
        if self.length() > 0:
            return self.instance_values[:,idx]
        else:
            return []


    def getInstanceIndex(self, id_num):
        #print(id_num)
        if self.length() > 1:                
            idx = np.argwhere(self.instance_values[:,-1] == id_num)
            return idx[0]
        elif self.length() == 1 and id_num == np.squeeze(self.instance_values)[-1]:            
            return [0]
        else:
            return None
    
    def getId(self, idx):
        if self.length() > 0:        
            return self.instance_values[idx, -1]
        else:
            return None
         
        
        
        



