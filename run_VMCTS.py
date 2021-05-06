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



parser = argparse.ArgumentParser(description='Value MCTS parser')
# Game arguments
parser.add_argument('--ucb_C', type=float, help='Exploration constant in UBC formula', default=1.0)
parser.add_argument('--discount', type=float, help='Discount factor for future rewards', default=0.9)
parser.add_argument('--episode_length', type=int, help='Length of trajectories. Restarts the game if end is reached before time', default=32)
parser.add_argument('--max_actions', type=int, help='Maximum number of random rollouts used in simulate mode for evaluating a leaf node in MCTS', default=5)
parser.add_argument('--num_simulations', type=int, help='Number of searches for node in MCTS', default=50)
parser.add_argument('--device', type=str, help='Device used by the value network (cuda or cpu)', default=mcts.device)
parser.add_argument('--n_episodes', type=int, help='Number of episodes played during the run', default=500)
parser.add_argument('--memory_size', type=int, help='Number of episodes stored in the replay buffer', default=256)
parser.add_argument('--batch_size', type=int, help='Number of samples used in a mini-batch for updates', default=32)
parser.add_argument('--n_steps', type=int, help='Number of steps used for bootstrapping', default=5)
parser.add_argument('--tau', type=float, help='Coefficient for exponential moving average update of the target network', default=0.5)
parser.add_argument('--target_update_period', type=int, help='Number of value network updates to do before every target update', default=8)
parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
# Paths
parser.add_argument('--ID', type=str, help='ID of the experiment', default=None)
parser.add_argument('--save_dir', type=str, help='Path to save directory', default='save_dir')
parser.add_argument('--checkpoint_period', type=int, help='Every how many steps to save a checkpoint', default=100)



args, unknown_flags = parser.parse_known_args()  # Let argparse parse known flags from sys.argv.

def gen_PID():
    ID = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
    ID = ID.upper()
    return ID

def main():
    start = time.time()
    # define them by the parser values
    training_params = dict(
        ucb_C = args.ucb_C,
        discount = args.discount, 
        episode_length = args.episode_length,
        max_actions = args.max_actions,
        num_simulations = args.num_simulations,
        device = args.device,
        n_episodes = args.n_episodes,
        memory_size = args.memory_size,
        batch_size = args.batch_size,
        n_steps = args.n_steps,
        tau = args.tau 
    )
    
    device = args.device
    
    # Environment and simulator 
    flags = utils.Flags(env="rtfm:groups_simple_stationary-v0")
    gym_env = utils.create_env(flags)
    featurizer = X.Render()
    game_simulator = mcts.FullTrueSimulator(gym_env, featurizer)
    object_ids = utils.get_object_ids_dict(game_simulator)
    
    # Networks
    value_net = mcts.FixedDynamicsValueNet_v2(gym_env).to(device)
    target_net = mcts.FixedDynamicsValueNet_v2(gym_env).to(device)
    # Init target_net with same parameters of value_net
    for trg_params, params in zip(target_net.parameters(), value_net.parameters()):
                trg_params.data.copy_(params.data)

    # Training and optimization
    optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
    gamma = 10**(-2/(args.n_episodes-1)) # decrease lr of 2 order of magnitude during training
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    loss_fn = F.mse_loss
    rb = train.nStepsReplayBuffer(args.memory_size, args.discount)
    
    # Experiment ID
    if args.ID is None:
        ID = gen_PID()
    else:
        ID = args.ID
    print("Experiment ID: ", ID)
    
    total_rewards = []
    losses = []
    for i in range(args.n_episodes):
        ### Generate experience ###
        t0 = time.time()
        value_net.eval()
        total_reward, frame_lst, reward_lst, done_lst = train.play_rollout_value_net(value_net,
                                                                                    game_simulator,
                                                                                    args.episode_length,
                                                                                    args.ucb_C,
                                                                                    args.discount,
                                                                                    args.max_actions,
                                                                                    args.num_simulations,
                                                                                    mode="predict",
                                                                                    bootstrap="no"
                                                                                    )
        t1 = time.time()
        total_rewards.append(total_reward)
        print("\nEpisode %d - Total reward %d"%(i+1, total_reward))
        rollout_time = (t1-t0)/60
        print("Rollout time: %.2f"%(rollout_time))
        rb.store_episode(frame_lst, reward_lst, done_lst)

        ### Train value_net ###

        try:
            # update value network all the time
            target_net.eval()
            frames, targets = rb.get_batch(args.batch_size, args.n_steps, args.discount, target_net, device)
            value_net.train()
            loss = train.compute_update_v1(value_net, frames, targets, loss_fn, optimizer)
            scheduler.step()
            # update target network only from time to time
            if (i+1)%args.target_update_period==0:
                train.update_target_net(target_net, value_net, args.tau)
            print("Loss: %.4f"%loss)
            losses.append(loss)

        except:
            pass

        if (i+1)%50==0:
            # Print update
            print("\nAverage reward over last 50 rollouts: %.2f\n"%(np.mean(total_rewards[-50:])))

        if (i+1)%args.checkpoint_period==0:
            # Plot histograms of value stats and save checkpoint
            target_net.eval()
            value_net.eval()
                
            # No plots in the script
            #train.plot_value_stats(value_net, target_net, rb, batch_size, n_steps, discount, device)

            d = dict(
                episodes_played=i,
                training_params=training_params,
                object_ids=object_ids,
                value_net=value_net,
                target_net=target_net,
                rb=rb,
                losses=losses,
                total_rewards=total_rewards
            )

            experiment_path ="./%s/%s/"%(args.save_dir, ID)
            if not os.path.isdir(experiment_path):
                os.mkdir(experiment_path)
            torch.save(d, experiment_path+'training_dict_%d'%(i+1))
            print("Saved checkpoint.")

    end = time.time()
    elapsed = (end-start)/60
    print("Run took %.1f min."%elapsed)
   


if __name__ == "__main__":
    main()
