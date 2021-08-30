import sys
import torch
from torch.autograd import Variable
import numpy as np
import random
import ray
from CAPS import explain
from topin_baseline import gen_apg
from config import argparser
from Cartpole.test_cartpole import calculate_fidelity as calculate_fidelity_cart
from Cartpole.test_cartpole import test as test_cart
from data import Data
from abstract import APG
from translation import CartpolePredicates
from Gridworld.test_gridworld import calculate_fidelity as calculate_fidelity_grid
from Gridworld.test_gridworld import test as test_grid
from translation import GridworldPredicates
from MountainCar.test_mountaincar import calculate_fidelity as calculate_fidelity_mountain
from MountainCar.test_mountaincar import test as test_mountain
from translation import MountainCarPredicates
from zahavy_baseline import explain_zahavy
from LunarLander.test_lunarlander import calculate_fidelity as calculate_fidelity_lunar
from LunarLander.test_lunarlander import test as test_lunar
from translation import LunarLanderPredicates
from Blackjack.test_blackjack import calculate_fidelity as calculate_fidelity_blackjack
from Blackjack.test_blackjack import test as test_blackjack
from translation import BlackjackPredicates




if __name__ == '__main__':

    args = argparser()
    model_path = args.path
    assert model_path != ''
    fidelity_fn = None

    if args.env == 'cart':
        data, model, num_feats, num_actions = test_cart(model_path, args.num_episodes)

        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_cart
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        
        dataset = Data(data, value_fn)
        translator = CartpolePredicates(num_feats=num_feats)
        if args.zahavy_baseline:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain_zahavy(args, dataset, translator, abstract_baseline, num_actions)
        elif args.topin_baseline:
            info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
            abstract_baseline = APG(num_actions, value_fn, translator, info=info)
            gen_apg(abstract_baseline)
        else:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline)
    
    elif args.env == 'grid':
        data, model, num_feats, num_actions = test_grid(model_path, args.num_episodes)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_grid
        def value_fn(obs):
            int_state = obs[0]
            obs = np.zeros(48)
            obs[int_state] = 1
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        
        dataset = Data(data, value_fn)
        translator = GridworldPredicates(num_feats=num_feats)
        if args.zahavy_baseline:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain_zahavy(args, dataset, translator, abstract_baseline, num_actions)
        elif args.topin_baseline:
            info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
            abstract_baseline = APG(num_actions, value_fn, translator, info=info)
            gen_apg(abstract_baseline)
        else:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline)
    
    elif args.env == 'mountain':
        data, model, num_feats, num_actions = test_mountain(model_path, args.num_episodes)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_mountain
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = MountainCarPredicates(num_feats=num_feats)
        if args.zahavy_baseline:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain_zahavy(args, dataset, translator, abstract_baseline, num_actions)
        elif args.topin_baseline:
            info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
            abstract_baseline = APG(num_actions, value_fn, translator, info=info)
            gen_apg(abstract_baseline)
        else:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline)
    
    elif args.env == 'lunar':
        data, model, num_feats, num_actions = test_lunar(model_path, args.num_episodes)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_lunar
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = LunarLanderPredicates(num_feats=num_feats)
        if args.zahavy_baseline:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain_zahavy(args, dataset, translator, abstract_baseline, num_actions)
        elif args.topin_baseline:
            info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
            abstract_baseline = APG(num_actions, value_fn, translator, info=info)
            gen_apg(abstract_baseline)
        else:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline)

    elif args.env == 'blackjack':
        data, model, num_feats, num_actions = test_blackjack(model_path, args.num_episodes)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_blackjack
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = np.squeeze(obs)
            p = obs[0]
            d = obs[1]
            a = obs[2]
            s = np.zeros(45)
            s[p] = 1
            s[32+d] = 1
            s[43+a] = 1
            s = np.reshape(s, [1, -1])
            obs = Variable(torch.from_numpy(s))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = BlackjackPredicates(num_feats=num_feats)
        if args.zahavy_baseline:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain_zahavy(args, dataset, translator, abstract_baseline, num_actions)
        elif args.topin_baseline:
            info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
            abstract_baseline = APG(num_actions, value_fn, translator, info=info)
            gen_apg(abstract_baseline)
        else:
            abstract_baseline = APG(num_actions, value_fn, translator)
            explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline)

    else:
        raise ValueError('Enter valid environment')

        







