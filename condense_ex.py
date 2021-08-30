import numpy as np
import math


class Explainer:

    def __init__(self, target_states, group_indices, num_preds, pred_set):
        """
        Write explanation for group indices
        """
        self.ungrouped_states = target_states
        self.group_indices = group_indices
        self.num_feats = num_preds
        self.pred_set = pred_set

    def full_translate(self):
        self.group_states()
        p, c = self.calc_statistics()
        #print(p)
        f = self.pick_feats(p)
        c_sets, n_c_sets = self.get_corr_sets(f, c)
        n = self.group_feats(n_c_sets)
        ex = self.gen_explanations(c_sets, n)
        
        return ex

    def group_states(self):
        self.states = []
        for s in self.ungrouped_states:
            s = np.array(s)
            new_s = []
            for group in self.group_indices:
                new_s.append(s[group])
            self.states.append(new_s)
        
    def calc_statistics(self):
        proportions = np.zeros(self.num_feats)
        binary_counts = np.zeros(shape=[self.num_feats, self.num_feats, 4])

        for s in self.ungrouped_states:
            for i, f in enumerate(s):
                for j, o_f in enumerate(s):
                    #This will count for the correlation between a feat and itself. account for this later
                    if f and o_f:
                        binary_counts[i, j, 0] += 1
                    elif f and not o_f:
                        binary_counts[i, j, 1] += 1
                    elif not f and o_f:
                        binary_counts[i, j, 2] += 1
                    else:
                        binary_counts[i, j, 3] += 1
                    
            
            proportions = proportions + s
        
        proportions = proportions / len(self.ungrouped_states)
        #print(proportions)
        correlations = self.pearson_coefficient(binary_counts)

        return proportions, correlations

    def pearson_coefficient(self, counts):
        #feats within subgroups should have corr = -1
        correlations = np.zeros(shape=[self.num_feats, self.num_feats])
        for i in range(len(counts)):
            for j, count in enumerate(counts[i]):
                denom = math.sqrt((count[0]+count[1]) * (count[2]+count[3]) * (count[0]+count[2]) * (count[1]+count[3]))
                if denom != 0:
                    correlations[i, j] = ((count[0] * count[3]) - (count[1] * count[2])) / denom
        
        return correlations
    
    def pick_feats(self, p):
        include_feats = []
        for i in range(len(p)):
            if p[i] >= 0.2: #We include all features that exist at least somewhat within the group.
                include_feats.append(i)
        return include_feats
    
    def get_corr_sets(self, include_feats, correlations):
        corr_sets = []
        print(include_feats)
        for i, feat in enumerate(include_feats):
            corr_feats = np.where(correlations[feat] > 0.7)[0]
            if len(corr_feats) > 1:
                corr_sets.append(corr_feats)
        
        
        for i, c_set in enumerate(corr_sets):
            for j, o_c_set in enumerate(corr_sets):
                if i != j:
                    if np.array_equal(c_set, o_c_set):
                        corr_sets.pop(i)
            
        non_corr_set = include_feats
        for c_set in corr_sets:
            print(c_set)
            for f in c_set:
                non_corr_set.remove(f)

        return corr_sets, non_corr_set

    def group_feats(self, not_corr):
        not_c = []
        for group in self.group_indices:
            g_c = []
            for f in not_corr:
                if f in group:
                    g_c.append(f)
            if len(g_c):
                not_c.append(g_c)
        
        return not_c
    
    def gen_explanations(self, corr_set, grouped_n_corr):
        explanations = []
        for c in corr_set:
            preds = [self.pred_set[i]['true'] for i in c]
            string = ' and '.join(preds)
            string = '(' + string + ')'
            explanations.append(string)
        
        ex = ''
        if len(explanations):
            ex = ' or '.join(explanations)
            ex = '(' + ex + ')'
        
        explanations = []
        for group in grouped_n_corr:
            preds = [self.pred_set[i]['true'] for i in group]
            string = ' or '.join(preds)
            string = '(' + string + ')'
            explanations.append(string)
        
        n_ex = ' and '.join(explanations)
        if ex != '':
            full_ex = ex + ' and ' + n_ex
        else:
            full_ex = n_ex
        
        return full_ex



if __name__ == '__main__':

    translations = [{'true': 'zero'},
                    {'true': 'one'},
                    {'true': 'two'},
                    {'true': 'three'},
                    {'true': 'four'},
                    {'true': 'five'}]
    ex_states = [[0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1],
                 [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1]]
    group_ind = [[0, 1], [2, 3], [4 ,5]]
    e = Explainer(ex_states, group_ind, 6, translations)
    e.group_states()
    p, c = e.calc_statistics()
    print(c)
    f = e.pick_feats(p)
    print(f)
    c_sets, n_c_sets = e.get_corr_sets(f, c)
    print(c_sets)
    n = e.group_feats(n_c_sets)
    print(n)
    ex = e.gen_explanations(c_sets, n)
    print(ex)

