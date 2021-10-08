import numpy as np
import math
from qm import QuineMcCluskey
from condense_ex import Explainer

class PredicateTemplate:
    def __init__(self, num_feats):
        self.num_feats = num_feats
        self.attr_names = None
    
    def predicate_set(self):
        raise NotImplementedError

    def feat_groups(self):
        raise NotImplementedError

    def translate_state(self, state):
        raise NotImplementedError
    
    def num_predicates(self):
        raise NotImplementedError
    
    def state_to_binary(self, state):
        raise NotImplementedError
    
    def reduce_logic(self, c_binary):
        #Sections of code adapted from https://gitlab.tue.nl/ha800-hri/hayes-shah/-/blob/master/hayes_shah/hs.py
        """
        c_binary: The set of abstract classes. An array of length num_abstract_classes
            c_binary[k] is an array which contains all states of the kth class
            that array contains tuples of the form (binary_state, action)
        Simplify the predicates of an abstract state into the minimal form
        This algorithm from Hayes and Shah 2017 requires the predicate features from all binary states in the target
        class (the abstract state, in this case), and the non-target class (all other abstract states)
        The target class and non-target classes should be mutually exclusive for the algorithm
        This is not reasonable to assume, given that binary states that are the same could be in different classes
            due to stochasticity in the policy and inaccuracies in the predicates
        So, the initial solution would be to loop through the other abstract states and remove the predicate states
            which are the exact same. The resulting explanations may not be completely faithful, but higher accuracy would
            come with better predicates/
        There could be an issue where no reliable predicate explanation is found, since there are just so many states in the
            other classes. In that case, I would come up with my own way of simplifying the predicates
        """

        qm = QuineMcCluskey()

        use_qm = True

        predicates = self.predicate_set()
        explanations = []
        condensed_sets = []

        for i, abs_class in enumerate(c_binary):
            target_states = [t[0] for t in abs_class]
            non_target_states = []
            for j in range(len(c_binary)):
                if j != i:
                    for k in range(len(c_binary[j])):
                        s = c_binary[j][k][0]
                        non_target_states.append(s)
            

            for s in target_states: #Loop through sets to ensure the intersection is empty
                for j, non_s in enumerate(non_target_states):
                    if np.array_equal(s, non_s):
                        non_target_states.pop(j)
            
            if use_qm:
                target_state_strings = []
                non_target_state_strings = []
        
                for s in target_states:
                    string = ''
                    for f in s:
                        string = string + str(f)
                    target_state_strings.append(string)
        
                
                for s in non_target_states:
                    string = ''
                    for f in s:
                        string = string + str(f)
                    non_target_state_strings.append(string)
                    
                
                n = len(target_states[0])
                all_bin = [bin(x)[2:].rjust(n, '0') for x in range(2**n)]
                not_valid = list(set(all_bin) - set(target_state_strings) - set(non_target_state_strings)) #All states which never appear

                
                
                minterms = qm.simplify_los(target_state_strings, not_valid)
                #print('{}: {}'.format(i+1,minterms))
                clauses = self.minterm_to_clause(minterms, predicates)
                #print('{}: {}'.format(i+1,clauses))
                explanations.append(' or '.join(clauses))
            
            else:
                proportions = np.zeros(self.num_predicates())
                for s in target_states:
                    proportions = proportions + np.array(s)
                pos_proportions = proportions / len(target_states)
                pos_explans = []
                neg_explans = []
                for j, p in enumerate(pos_proportions):
                    predicate = predicates[j]
                    if p >= 0.9:
                        pos_explans.append(predicate['true'])
                    elif p <= 0.1:
                        neg_explans.append(predicate['false'])
                
                if pos_explans == []:
                    most_common = np.argmax(pos_proportions) #Add most common occurence
                    neg_explans.append(predicates[most_common]['true'])
                    explanations.append(' and '.join(neg_explans)) #Only use neg explans if no pos exist
                else:
                    explanations.append(' and '.join(pos_explans))
                condensed_sets.append(pos_proportions)

        return explanations

    def minterm_to_clause(self, minterms, predicates):
        

        clauses = []

        for min_term in minterms:
            str_terms = []
            for i in range(len(min_term)):
                predicate = predicates[i]
                if min_term[i] == '0':
                    str_terms.append(predicate['false'])
                elif min_term[i] == '1':
                    str_terms.append(predicate['true'])

            clauses.append(' and '.join(str_terms))

        return clauses
    

    def my_translation_algo(self, c_binary):
        predicates = self.predicate_set()
        explanations = []
        condensed_sets = []

    
        for i, abs_class in enumerate(c_binary):
            target_states = [t[0] for t in abs_class]
            e = Explainer(target_states, self.feat_groups(), len(predicates), predicates)
            ex = e.full_translate()
            explanations.append(ex)
        
        return explanations

class LunarLanderPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['X Coordinate',
                           'Y Coordinate',
                           'X Velocity',
                           'Y Velocity',
                           'Lander Angle',
                           'Angular Velocity',
                           'Left leg on ground',
                           'Right leg on ground']
        self.language_set = np.array(['Left of the goal', 'Right of the goal',
                                      'On top of goal', 'Higher than goal', 'Same height as goal',
                                      'Left leg on ground', 'Right leg on ground',
                                      'Lander tilted left', 'Lander tilted right',
                                      'Moving right', 'Moving left'])
    
    def state_to_binary(self, state):
        b = [self.left_of_goal(state),
             self.right_of_goal(state),
             self.on_top_of_goal(state),
             self.higher_than_goal(state),
             self.at_same_height(state),
             self.left_leg_on_ground(state),
             self.right_leg_on_ground(state),
             self.tilted_left(state),
             self.tilted_right(state),
             self.moving_right(state),
             self.moving_left(state)]
        return np.array(b)
    
    def translate_state(self, binary_set):
        idx = np.where(binary_set==1)[0]
        true_set = self.language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string

    def feat_groups(self):
        groups = [[0, 1, 2], [3, 4], [5], [6], [7, 8], [9, 10]]
        return groups
    
    def predicate_set(self):
        predicates = [{'true': 'Left of the goal', 'false': 'Not left of the goal'},
                      {'true': 'Right of the goal', 'false': 'Not right of the goal'},
                      {'true': 'Directly on top of goal', 'false': 'Not directly on top of goal'},
                      {'true': 'Higher than goal', 'false': 'Not higher than goal'},
                      {'true': 'At same height as goal', 'false': 'Not at same height as goal'},
                      {'true': 'Left leg on the ground', 'false': 'Left leg not on the ground'},
                      {'true': 'Right leg on the ground', 'false': 'Right leg not on the ground'},
                      {'true': 'Lander tilted left', 'false': 'Lander not tilted left'},
                      {'true': 'Lander tilted right', 'false': 'Lander not tilted right'},
                      {'true': 'Moving right', 'false': 'Not moving right'},
                      {'true': 'Moving left', 'false': 'Not moving left'}]
        return predicates
    
    def num_predicates(self):
        return len(self.predicate_set())
    
    def left_of_goal(self, state):
        if state[0] < -0.08:
            return 1
        else:
            return 0
    
    def right_of_goal(self, state):
        if state[0] > 0.08:
            return 1
        else:
            return 0
    
    def on_top_of_goal(self, state):
        if np.abs(state[0]) <= 0.08:
            return 1
        else:
            return 0
    
    def higher_than_goal(self, state):
        if state[1] > 0.08:
            return 1
        else:
            return 0
    
    def at_same_height(self, state):
        if state[1] <= 0.08:
            return 1
        else:
            return 0
    
    def left_leg_on_ground(self, state):
        if state[6] == 1:
            return 1
        else:
            return 0
    
    def right_leg_on_ground(self, state):
        if state[7] == 1:
            return 1
        else:
            return 0
    
    def tilted_left(self, state):
        if state[4] < -0.3:
            return 1
        else:
            return 0
    def tilted_right(self, state):
        if state[4] > 0.3:
            return 1
        else:
            return 0
    
    def moving_right(self, state):
        if state[2] > 0.01:
            return 1
        else:
            return 0
    
    def moving_left(self, state):
        if state[2] < -0.01:
            return 1
        else:
            return 0


class BlackjackPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Current sum', 'Dealer card', 'Usable ace']
        self.language_set = np.array(['sum less than 14', 'sum 14-16','sum 17-19',
                                      'sum 20-21', ' d sum less 7', 'd sum 7-9',
                                      'd sum 10-ace','ace 11'])
    

    def predicate_set(self):
        predicates = [{'true': 'Sum less than 14', 'false': 'Sum not less than 14'},
                      {'true': 'Sum of 14-16', 'false': 'Sum not of 14-16'},
                      {'true': 'Sum of 17-19', 'false': 'Sum not of 17-19'},
                      {'true': 'Sum of 20-21', 'false': 'Sum not of 20-21'},
                      {'true': 'Dealer card less than 7', 'false': 'Dealer card 7 or more'},
                      {'true': 'Dealer card 7-9', 'false': 'Dealer card not 7-9'},
                      {'true': 'Dealer card 10 or ace', 'false': 'Dealer card not 10 or ace'},
                      {'true': 'Ace is 11', 'false': 'No ace or ace is not 11'}]
        
        return predicates
    
    def num_predicates(self):
        return len(self.predicate_set())
    
    def feat_groups(self):
        groups = [[0, 1, 2, 3], [4, 5, 6], [7]]
        return groups


    def state_to_binary(self, state):
        b = [self.less_14(state),
             self.p14_16(state),
             self.p17_19(state),
             self.p20_21(state),
             self.dless_7(state),
             self.d7_9(state),
             self.d10_ace(state),
             self.use_ace(state)]
        return np.array(b)
    
    def translate_state(self, binary_set):
        idx = np.where(binary_set==1)[0]
        true_set = self.language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string


    def less_14(self, state):
        if state[0] < 14:
            return 1
        else:
            return 0
    
    def p14_16(self, state):
        if state[0] >= 14 and state[0] <= 16:
            return 1
        else:
            return 0
    
    def p17_19(self, state):
        if state[0] >= 17 and state[0] <= 19:
            return 1
        else:
            return 0
    
    def p20_21(self, state):
        if state[0] == 20 or state[0] == 21:
            return 1
        else:
            return 0

    def dless_7(self, state):
        if state[1] < 7 and state[1] != 1:
            return 1
        else:
            return 0
    
    def d7_9(self, state):
        if state[1] >= 7 and state[1] <= 9:
            return 1
        else:
            return 0
    
    def d10_ace(self, state):
        if state[1] == 10 or state[1] == 1:
            return 1
        else:
            return 0
    
    
    def use_ace(self, state):
        if state[2] == 1:
            return 1
        else:
            return 0
    
class MountainCarPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Car Position', 'Car Velocity']
        self.language_set = np.array(['At the bottom',
                    'On the left slope',
                    'On the right slope',
                    'High up on the left slope',
                    'High up on the right slope',
                    'Moving left slowly',
                    'Moving right slowly',
                    'Not moving',
                    'Moving left quickly',
                    'Moving right quickly'])

    def predicate_set(self):
        predicates = [{'true': 'At the bottom', 'false': 'Not at the bottom'},
                      {'true': 'On the left slope', 'false': 'Not on the left slope'},
                      {'true': 'On the right slope', 'false': 'Not on the right slope'},
                      {'true': 'High up on the left slope', 'false': 'Not high up on the left slope'},
                      {'true': 'High up on the right slope', 'false': 'Not high up on the right slope'},
                      {'true': 'Moving left slowly', 'false': 'Not moving left slowly'},
                      {'true': 'Moving right slowly', 'false': 'Not moving right slowly'},
                      {'true': 'Not moving', 'false': 'Not moving'},
                      {'true': 'Moving left quickly', 'false': 'Not moving left quickly'},
                      {'true': 'Moving right quickly', 'false': 'Not moving right quickly'}]
        return predicates

    def state_to_binary(self, state):
        binary_set = [self.at_bottom(state),
                      self.on_left_slope(state),
                      self.on_right_slope(state),
                      self.high_on_left(state),
                      self.high_on_right(state),
                      self.moving_left_slow(state),
                      self.moving_right_slow(state),
                      self.not_moving(state),
                      self.moving_left_fast(state),
                      self.moving_right_fast(state)]
        
        return np.array(binary_set)
    
    def translate_state(self, binary_set):
        language_set = np.array(['At the bottom',
                    'On the left slope',
                    'On the right slope',
                    'High up on the left slope',
                    'High up on the right slope',
                    'Moving left slow',
                    'Moving right slow',
                    'Not moving',
                    'Moving left fast',
                    'Moving right fast'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string

    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        return groups

    def num_predicates(self):
        return 10
    
    def at_bottom(self, state):
        if state[0] >= -0.6 and state[0] <= -0.4:
            return 1
        else:
            return 0
    
    def on_left_slope(self, state):
        if state[0] < -0.6 and state[0] > -0.9:
            return 1
        else:
            return 0
    
    def on_right_slope(self, state):
        if state[0] > -0.4 and state[0] < 0.3:
            return 1
        else:
            return 0
    
    def high_on_left(self, state):
        if state[0] <= -0.9:
            return 1
        else:
            return 0
    
    def high_on_right(self, state):
        if state[0] >= 0.3:
            return 1
        else:
            return 0
    
    def moving_left_slow(self, state):
        if state[1] < 0 and state[1] > -0.025:
            return 1
        else:
            return 0
    
    def moving_right_slow(self, state):
        if state[1] > 0 and state[1] < 0.025:
            return 1
        else:
            return 0
    
    def not_moving(self, state):
        if state[1] == 0:
            return 1
        else:
            return 0
    
    def moving_left_fast(self, state):
        if state[1] <= -0.025:
            return 1
        else:
            return 0
    
    def moving_right_fast(self, state):
        if state[1] >= 0.025:
            return 1
        else:
            return 0

class GridworldPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Position']
        self.language_set = np.array(['At the start',
                    'Reached the goal',
                    'A cliff is below',
                    'On the left border',
                    'On the right border',
                    'In free space',
                    'Near the goal'])

    def state_to_coords(self, state):
        obs = state[0]
        coords = (obs // 12, obs % 12)
        return coords

    def predicate_set(self):
        predicates = [{'true': 'At the start', 'false': 'Not at the start'},
                      {'true': 'Reached the goal', 'false': 'Has not reached the goal'},
                      {'true': 'A cliff is below', 'false': 'A cliff is not below'},
                      {'true': 'On the left border', 'false': 'Not on the left border'},
                      {'true': 'On the right border', 'false': 'Not on the right border'},
                      {'true': 'In free space', 'false': 'Not in free space'},
                      {'true': 'Near the goal', 'false': 'Not near the goal'}]
    
        return predicates
    
    def state_to_binary(self, state):
        coords = self.state_to_coords(state)
        binary_set = [self.at_start(coords),
                      self.at_goal(coords),
                      self.cliff_below(coords),
                      self.at_left_edge(coords),
                      self.at_right_edge(coords),
                      self.in_free_space(coords),
                      self.near_goal(coords)]
        
        return np.array(binary_set)
    
    def translate_state(self, binary_set):
        language_set = np.array(['At the start',
                    'Reached the goal',
                    'A cliff is below',
                    'On the left border',
                    'On the right border',
                    'In free space',
                    'Near the goal'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string


    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4, 5], [6]]
        return groups

    def num_predicates(self):
        return 7

    def at_start(self, coords):
        if coords[0] == 3 and coords[1] == 0:
            return 1
        else:
            return 0
    
    def at_goal(self, coords):
        if coords[0] == 3 and coords[1] == 11:
            return 1
        else:
            return 0

    def cliff_below(self, coords):
        if coords[0] == 2 and coords[1] > 0 and coords[1] < 11:
            return 1
        else:
            return 0
    
    def at_left_edge(self, coords):
        if coords[0] < 3 and coords[1] == 0:
            return 1
        else:
            return 0
    
    def at_right_edge(self, coords):
        if coords[0] < 3 and coords[1] == 11:
            return 1
        else:
            return 0
    
    def in_free_space(self, coords):
        if coords[0] < 2 and coords[1] > 0 and coords[1] < 11:
            return 1
        else:
            return 0

    def near_goal(self, coords):
        if coords[0] > 1 and coords[1] > 9:
            return 1
        else:
            return 0




class CartpolePredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
        self.language_set = np.array(['Pole is falling to the left',
                    'Pole is falling to the right',
                    'Pole is stabilizing from left',
                    'Pole is stabilizing from right',
                    'Pole is standing up',
                    'Cart is moving left',
                    'Cart is moving right',
                    'Cart is on the left',
                    'Cart is on the right',
                    'Cart is in the middle'])

    def predicate_set(self):
        predicates = [{'true': 'Pole is falling to the left', 'false': 'Pole is not falling to the left'},
                      {'true': 'Pole is falling to the right', 'false': 'Pole is not falling to the right'},
                      {'true': 'Pole is stabilizing to the left', 'false': 'Pole is not stabilizing to the left'},
                      {'true': 'Pole is stabilizing to the right', 'false': 'Pole is not stabilizing to the right'},
                      {'true': 'Pole is standing up', 'false': 'Pole is not standing up'},
                      {'true': 'Cart is moving left', 'false': 'Cart is not moving left'},
                      {'true': 'Cart is moving right', 'false': 'Cart is not moving right'},
                      {'true': 'Cart is on the left', 'false': 'Cart is not on the left'},
                      {'true': 'Cart is on the right', 'false': 'Cart is not on the right'},
                      {'true': 'Cart is in the middle', 'false': 'Cart is not in the middle'}]
    
        return predicates
    
    def translate_state(self, binary_set):
        language_set = np.array(['Pole is falling to the left',
                    'Pole is falling to the right',
                    'Pole is stabilizing from left',
                    'Pole is stabilizing from right',
                    'Pole is standing up',
                    'Cart is moving left',
                    'Cart is moving right',
                    'Cart is on the left',
                    'Cart is on the right',
                    'Cart is in the middle'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string
    
    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]
        return groups

    def num_predicates(self):
        return 10
    
    def state_to_binary(self, state):
        state = np.reshape(state, [-1])
        binary = [self.pole_fall_left(state),
                  self.pole_fall_right(state),
                  self.pole_stabilize_left(state),
                  self.pole_stabilize_right(state),
                  self.pole_standing_up(state),
                  self.cart_moving_left(state),
                  self.cart_moving_right(state),
                  self.cart_pos_left(state),
                  self.cart_pos_right(state),
                  self.cart_near_middle(state)]
        
        return np.array(binary)
    
    def pole_fall_left(self, state):
        if state[2] < -0.01 and state[3] < 0:
            return 1
        else:
            return 0
    
    def pole_fall_right(self, state):
        if state[2] > 0.01 and state[3] > 0:
            return 1
        else:
            return 0
    
    def pole_stabilize_left(self, state):
        if state[2] < -0.01 and state[3] > 0:
            return 1
        else:
            return 0
    
    def pole_stabilize_right(self, state):
        if state[2] > 0.01 and state[3] < 0:
            return 1
        else:
            return 0
    
    def pole_standing_up(self, state):
        if np.abs(state[2]) <= 0.01:
            return 1
        else:
            return 0
    
    def cart_moving_left(self, state):
        if state[1] < 0:
            return 1
        else:
            return 0
    
    def cart_moving_right(self, state):
        if state[1] >= 0:
            return 1
        else:
            return 0
    
    def cart_pos_left(self, state):
        if state[0] < -0.05:
            return 1
        else:
            return 0
    
    def cart_pos_right(self, state):
        if state[0] > 0.05:
            return 1
        else:
            return 0
    
    def cart_near_middle(self, state):
        if np.abs(state[0]) < 0.05:
            return 1
        else:
            return 0

class AcornPredicates(PredicateTemplate):
    def __init__(self, num_feats, num_bins=10):
        super().__init__(num_feats)
        self.num_bins = num_bins
        self.predicate_set = []
        self.binned_indices = []

        self.agent1_features = ['Percent Tweets', 'Percent Replies',
                                'Percent Retweets', 'Percent Mentions',
                                'Percent Followers']

    
    def translate_state(self, state): #Temporary implementation. Later, will include other ways of predicate grounding
        self.predicate_set = []
        binned = self.binning(state)
        for i, feature in enumerate(binned):
            self.binned_indices.append(len(self.predicate_set))
            self.predicate_set.append(feature)
        
        nl_predicate_set = self.nl_grounding()

        return self.predicate_set, nl_predicate_set

    def state_to_binary(self, state):
        return self.binning(state)
    
    def binning(self, state):
        binned_state = []
        for feature in state:
            assert 0 <= feature and feature <= 1

            idx = math.floor(feature * self.num_bins)
            binary = np.zeros(self.num_bins)
            binary[idx] = 1
            binned_state.append(binary)
        binned_state = np.reshape(np.array(binned_state), [-1])
    
        return binned_state
    
    def nl_grounding(self): #Temporary. Later, will include translations for other types of predicates
        for i, pred in enumerate(self.predicate_set):
            if i in self.binned_indices:
                feat_idx = np.where(np.array(self.binned_indices==i))[0][0]
                self.nl_predicate_set.append(self.translate_bins(pred, feat_idx))
        
        return self.nl_predicate_set
        
        

    
    def translate_bins(self, predicate, feat_idx):
        idx = np.argmax(predicate)
        low_bound = idx / self.num_bins
        high_bound = low_bound + (1 / self.num_bins)

        string = "{} is between {} and {}".format(self.agent1_features[feat_idx], low_bound, high_bound)
        return string

    def num_predicates(self):
        return self.num_feats * self.num_bins



