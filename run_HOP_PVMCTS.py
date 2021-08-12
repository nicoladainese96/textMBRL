# +
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
import os
import sys
import argparse 
import string
import random

# custom imports
import utils
import mcts
import train
from rtfm import featurizer as X



parser = argparse.ArgumentParser(description='Policy and Value HOP MCTS parser')
# Training arguments
parser.add_argument('--game_name', type=str, help='Name of the rtfm environment to use, e.g. groups_simple', default="groups_simple")
parser.add_argument('--ucb_C', type=float, help='Exploration constant in UBC formula', default=1.0)
parser.add_argument('--discount', type=float, help='Discount factor for future rewards', default=0.9)
parser.add_argument('--episode_length', type=int, help='Length of trajectories. Restarts the game if end is reached before time', default=32)
parser.add_argument('--max_actions', type=int, help='Maximum number of random rollouts used in simulate mode for evaluating a leaf node in MCTS', default=20)
parser.add_argument('--num_simulations', type=int, help='Number of searches for node in MCTS', default=50)
parser.add_argument('--device', type=str, help='Device used by the value network (cuda or cpu)', default=mcts.device)
parser.add_argument('--n_episodes', type=int, help='Number of episodes played during the run', default=5000)
parser.add_argument('--memory_size', type=int, help='Number of episodes stored in the replay buffer', default=1024)
parser.add_argument('--batch_size', type=int, help='Number of samples used in a mini-batch for updates', default=32)
parser.add_argument('--n_steps', type=int, help='Number of steps used for bootstrapping', default=5)
parser.add_argument('--tau', type=float, help='Coefficient for exponential moving average update of the target network', default=0.5)
parser.add_argument('--num_trees', type=int, help='Number of trees used in parallel during HOP-MCTS step', default=3)
# Algorithmic parameters
parser.add_argument('--dirichlet_alpha', type=float, help='Alpha parameter of root Dirichlet noise', default=0.5)
parser.add_argument('--exploration_fraction', type=float, help='Mixing parameter between policy net prior and Dirichlet noise', default=0.25)
parser.add_argument('--temperature', type=float, help='Starting temperature used in softmax policy for Q values', default=1.)
parser.add_argument('--full_cross_entropy', dest='full_cross_entropy', help='If True, uses the full probability as a target for the cross-entropy loss', default=False, action='store_true')
parser.add_argument('--entropy_bonus', dest='entropy_bonus', help='If True, adds the negative entropy of the policy (multiplied by a const.) to the policy loss', default=False,  action='store_true')
parser.add_argument('--entropy_weight', type=float, help='Weight that multiplies the entropy bonus, if used', default=1e-2)
parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
parser.add_argument('--no_support_values', dest='discrete_support_values', help='If True, signals that the network is using a discrete support value head.', default=True,  action='store_false')
parser.add_argument('--ucb_method', type=str, help='Which method to use in p-UCT formula', default="old")
# Architecture parameters
parser.add_argument('--emb_dim', type=int, help='Number of dimensions in the embedding layers', default=10)
parser.add_argument('--support_size', type=int, help='Number of support values in [0,1]. Full range is 2*support_values+1 intervals in [-1,1]', default=10)
parser.add_argument('--conv_channels', type=int, help='Number of convolutional channels used in most convolutional layers of the policy-value net', default=64)
parser.add_argument('--conv_layers', type=int, help='Number of 1x1 convolutional layers', default=2)
parser.add_argument('--residual_layers', type=int, help='Number residual convolutional layers', default=2)
parser.add_argument('--linear_features_in', type=int, help='Number of features in input to the value and policy MLPs', default=128)
parser.add_argument('--linear_feature_hidden', type=int, help='Number of features used in the hidden layer of the value and policy MLPs', default=128)
# Paths
parser.add_argument('--ID', type=str, help='ID of the experiment', default=None)
parser.add_argument('--save_dir', type=str, help='Path to save directory', default='./save_dir')
parser.add_argument('--checkpoint_period', type=int, help='Every how many steps to save a checkpoint', default=500)



args, unknown_flags = parser.parse_known_args()  # Let argparse parse known flags from sys.argv.

def gen_PID():
    ID = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
    ID = ID.upper()
    return ID

def main():
    start = time.time()
    # define them by the parser values
    print("args.full_cross_entropy: ", args.full_cross_entropy)
    print("args.entropy_bonus: ", args.entropy_bonus)
    print("args.discrete_support_values: ", args.discrete_support_values)
    if args.ucb_method == "old":
        ucb_method = "p-UCT-old"
    elif args.ucb_method == "AlphaGo":
        ucb_method = "p-UCT-AlphaGo"
    elif args.ucb_method == "Rosin":
        ucb_method = "p-UCT-Rosin"
    else:
        raise Exception("ucb_method should be one of 'old', 'AlphaGo', 'Rosin'.")
        
    training_params = dict(
        ucb_C = args.ucb_C,
        discount = args.discount, 
        episode_length = args.episode_length,
        max_actions = args.max_actions,
        num_simulations = args.num_simulations,
        device = "cpu", # disable GPU usage 
        n_episodes = args.n_episodes,
        memory_size = args.memory_size,
        batch_size = args.batch_size,
        n_steps = args.n_steps,
        tau = args.tau,
        dirichlet_alpha = args.dirichlet_alpha,
        exploration_fraction = args.exploration_fraction,
        temperature = args.temperature,
        full_cross_entropy = args.full_cross_entropy,
        entropy_bonus = args.entropy_bonus,
        entropy_weight = args.entropy_weight,
        discrete_support_values = args.discrete_support_values,
        ucb_method=ucb_method,
        num_trees=args.num_trees
    )
    
    device = "cpu" # disable GPU usage
    temperature = args.temperature
    
    
    network_params = {
        "emb_dim":args.emb_dim,
        "conv_channels":args.conv_channels,
        "conv_layers":args.conv_layers,
        "residual_layers":args.residual_layers,
        "linear_features_in":args.linear_features_in,
        "linear_feature_hidden":args.linear_feature_hidden
    }
    
    # Environment and simulator 
    flags = utils.Flags(env="rtfm:%s-v0"%args.game_name)
    gym_env = utils.create_env(flags)
    featurizer = X.Render()
    game_simulator = mcts.FullTrueSimulator(gym_env, featurizer)
    object_ids = utils.get_object_ids_dict(game_simulator)
    
    # Networks
    if args.discrete_support_values:
        network_params["support_size"] = args.support_size
        pv_net = mcts.DiscreteSupportPVNet_v3(gym_env, **network_params).to(device)
        target_net = mcts.DiscreteSupportPVNet_v3(gym_env, **network_params).to(device)
    else:
        pv_net = mcts.FixedDynamicsPVNet_v3(gym_env, **network_params).to(device)
        target_net = mcts.FixedDynamicsPVNet_v3(gym_env, **network_params).to(device)
        
    # Init target_net with same parameters of value_net
    for trg_params, params in zip(target_net.parameters(), pv_net.parameters()):
        trg_params.data.copy_(params.data)

    # Training and optimization
    optimizer = torch.optim.Adam(pv_net.parameters(), lr=args.lr)
    gamma = 10**(-2/(args.n_episodes-1)) # decrease lr of 2 order of magnitude during training
    gamma_T = 10**(-1/(args.n_episodes-1)) # decrease lr of 2 order of magnitude during training
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    replay_buffer = train.PolicyValueReplayBuffer(args.memory_size, args.discount)
    
    # Experiment ID
    if args.ID is None:
        ID = gen_PID()
    else:
        ID = args.ID
    print("Experiment ID: ", ID)
    
    total_rewards = []
    entropies = []
    accuracies = []
    losses = []
    policy_losses = []
    value_losses = []

    for i in range(args.n_episodes):
        ### Generate experience ###
        t0 = time.time()
        mode = "predict"
        target_net.eval() # just to make sure
        pv_net.eval()
        results = train.play_rollout_pv_net_hop_mcts(
            pv_net,
            game_simulator,
            args.episode_length,
            args.ucb_C,
            args.discount,
            args.max_actions,
            args.num_simulations,
            args.num_trees,
            object_ids,
            args.dirichlet_alpha,
            args.exploration_fraction,
            temperature,
            mode=mode,
            ucb_method=ucb_method
            )
        
        results = train.play_rollout_pv_net_hop_mcts(
            args.episode_length,
            object_ids,
            game_simulator, 
            args.ucb_C, 
            args.discount, 
            args.max_actions, 
            pv_net,
            args.num_simulations, 
            args.num_trees,
            temperature,
            dirichlet_alpha=args.dirichlet_alpha, 
            exploration_fraction=args.exploration_fraction,
            ucb_method=ucb_method
            )
        total_reward, frame_lst, reward_lst, done_lst, action_lst, best_action_lst, probs_lst = results
        replay_buffer.store_episode(frame_lst, reward_lst, done_lst, action_lst, best_action_lst, probs_lst)
        total_rewards.append(total_reward)
        rollout_time = (time.time()-t0)/60
        if (i+1)%10==0:
            print("\nEpisode %d - Total reward %d "%(i+1, total_reward))
            print("Rollout time: %.2f"%(rollout_time))

        if i >= args.batch_size:
            ### Update ###
            target_net.eval() # just to make sure
            frames, target_values, actions, best_actions, probs = replay_buffer.get_batch(
                args.batch_size, args.n_steps, target_net, device
            )
            pv_net.train()
            update_results = train.compute_PV_net_update(
                pv_net, 
                frames, 
                target_values, 
                actions, 
                best_actions, 
                probs,
                optimizer,
                args.full_cross_entropy,
                args.entropy_bonus,
                args.entropy_weight,
                args.discrete_support_values
            )
            loss, entropy, accuracy, policy_loss, value_loss = update_results
            scheduler.step()
            temperature = gamma_T*temperature

            # update target network only from time to time
            if (i+1)%8==0:
                train.update_target_net(target_net, pv_net, args.tau)

            if (i+1)%10==0:
                print("Loss: %.4f - Policy loss: %.4f - Value loss: %.4f"%(loss, policy_loss, value_loss))
                print("Entropy: %.4f - Accuracy: %.1f %%"%(entropy, accuracy*100))
            losses.append(loss)
            entropies.append(entropy)
            accuracies.append(accuracy)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        if (i+1)%50==0:
            # Print update
            print("\nAverage reward over last 50 rollouts: %.2f\n"%(np.mean(total_rewards[-50:])))
            print("Percentage of optimal actions: %.1f %%"%(np.mean(accuracies[-50:])*100))

        if (i+1)%args.checkpoint_period==0:
            # Plot histograms of value stats and save checkpoint
            target_net.eval()
            pv_net.eval()  
                
            # No plots in the script
            #train.plot_value_stats(value_net, target_net, rb, batch_size, n_steps, discount, device)

            d = dict(
                episodes_played=i,
                training_params=training_params,
                object_ids=object_ids,
                pv_net=pv_net,
                target=target_net,
                losses=losses,
                policy_losses=policy_losses,
                value_losses=value_losses,
                total_rewards=total_rewards,
                accuracies=accuracies,
                entropies=entropies, 
                optimizer=optimizer,
            )

            experiment_path ="%s/%s/"%(args.save_dir, ID)
            if not os.path.isdir(experiment_path):
                os.mkdir(experiment_path)
            torch.save(d, experiment_path+'training_dict_%d'%(i+1))
            torch.save(replay_buffer, experiment_path+'replay_buffer')
            torch.save(network_params, experiment_path+'network_params')
            print("Saved checkpoint.")

    end = time.time()
    elapsed = (end-start)/60
    print("Run took %.1f min."%elapsed)
   


if __name__ == "__main__":
    main()
