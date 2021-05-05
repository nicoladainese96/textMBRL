import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
# custom modules
from rtfm import featurizer as X
import utils
import mcts

verbose = False
vprint = print if verbose else lambda *args, **kwargs: None

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print("Using device "+device)

def show_root_summary(root, discount):
    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    
    for action, child in root.children.items():
        Q =  child.reward + discount*child.value()
        visits = child.visit_count
        print("Action ", action_dict[action], ": Q-value=%.3f - Visit counts=%d"%(Q,visits))

def play_episode_v0(
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    render = True,
    debug_render=False
):
    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    valid_actions = env.reset()
    if render:
        env.render()
    total_reward = 0
    done = False
    for i in range(episode_length):
        tree = mcts.MCTS(env, valid_actions, ucb_C, discount, max_actions, render=debug_render)
        print("\n","-"*40)
        print("Performing MCTS step")
        root, info = tree.run(num_simulations)
        show_root_summary(root, discount)
        print("-"*40)
        print("Tree info: ", info)
        action = root.best_action(discount)
        print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        valid_actions, reward, done = env.step(action)
        if render:
            env.render()
        print("Reward received: ", reward)
        print("Done: ", done)
        total_reward += reward
        if done:
            break
    return total_reward

def play_episode_v1(
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    render = True,
    debug_render=False
):
    """
    W.r.t. version 0 it re-uses the information cached in the child node selected 
    """
    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    valid_actions = env.reset()
    if render:
        env.render()
    total_reward = 0
    done = False
    new_root = None
    for i in range(episode_length):
        tree = mcts.MCTS(env, valid_actions, ucb_C, discount, max_actions, render=debug_render, root=new_root)
        print("Performing MCTS step")
        root, info = tree.run(num_simulations)
        show_root_summary(root, discount)
        print("Tree info: ", info)
        action = root.best_action(discount)
        print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        new_root = tree.get_subtree(action)
        valid_actions, reward, done = env.step(action)
        if render:
            env.render()
        print("Reward received: ", reward)
        print("Done: ", done)
        total_reward += reward
        if done:
            break
    return total_reward

def play_episode_value_net(
    value_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    mode="simulate",
    render = False,
    debug_render=False
):
    """
    W.r.t. version 0 it re-uses the information cached in the child node selected 
    """
    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    frame, valid_actions = env.reset()
    if render:
        env.render()
    total_reward = 0
    done = False
    new_root = None
    # variables used for training of value net
    frame_lst = [frame]
    reward_lst = []
    done_lst = []
    for i in range(episode_length):
        tree = mcts.ValueMCTS(frame, 
                         env, 
                         valid_actions, 
                         ucb_C, 
                         discount, 
                         max_actions, 
                         value_net,
                         render=debug_render, 
                         root=new_root
                        )
        #print("Performing MCTS step")
        root, info = tree.run(num_simulations, mode=mode)
        #show_root_summary(root, discount)
        #print("Tree info: ", info)
        action = root.best_action(discount)
        #print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        new_root = tree.get_subtree(action)
        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
        #print("Reward received: ", reward)
        #print("Done: ", done)
        total_reward += reward
        if done:
            break
    return total_reward, frame_lst, reward_lst, done_lst

def play_rollout_value_net(
        value_net,
        env,
        episode_length,
        ucb_C,
        discount,
        max_actions,
        num_simulations,
        mode="simulate",
        bootstrap="no",
        render = False,
        debug_render=False
    ):
    """
    W.r.t. version 0 it re-uses the information cached in the child node selected 
    """
    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    frame, valid_actions = env.reset()
    if render:
        env.render()
    total_reward = 0
    done = False
    new_root = None
    # variables used for training of value net
    frame_lst = [frame]
    reward_lst = []
    done_lst = []
    for i in range(episode_length):
        tree = mcts.ValueMCTS(frame, 
                         env, 
                         valid_actions, 
                         ucb_C, 
                         discount, 
                         max_actions, 
                         value_net,
                         render=debug_render, 
                         root=new_root
                        )
        if render:
            print("Performing MCTS step")
        root, info = tree.run(num_simulations, mode=mode)
        
        if render:
            show_root_summary(root, discount)
            print("Tree info: ", info)
        action = root.best_action(discount)
        if render:
            print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        new_root = tree.get_subtree(action)
        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
            print("Reward received: ", reward)
            print("Done: ", done)
        total_reward += reward
        if done:
            frame, valid_actions = env.reset()
            if render:
                print("\nNew episode begins.")
                env.render()
            done = False
            new_root = None

    if not done_lst[-1]:
        if bootstrap=="no":
            pass
        elif bootstrap=="root_value":
            reward_lst[-1] = new_root.value()
        elif bootstrap=="value_net":
            reward_lst[-1] = value_net(new_root.frame).item()
        else:
            raise Exception("Value of bootstrap variable should be one of 'no','root_value' and 'value_net'.")
    return total_reward, frame_lst, reward_lst, done_lst

def play_episode_given_IC(
    value_net,
    env,
    frame,
    valid_actions,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    mode="simulate",
    render = False,
    debug_render=False,
    action_selection="Q_value"
):
    """
    Plays an episode from a starting point which is specified with frame and valid_actions
    outside the function. Used to compare under the same initial conditions different ways
    of estimating the value of a leaf node in MCTS.
    """
    if render:
        env.render()
    total_reward = 0
    done = False
    new_root = None
    for i in range(episode_length):
        tree = mcts.ValueMCTS(frame, 
                         env, 
                         valid_actions, 
                         ucb_C, 
                         discount, 
                         max_actions, 
                         value_net,
                         render=debug_render, 
                         root=new_root
                        )
        root, info = tree.run(num_simulations, mode=mode)
        
        if action_selection == "Q_value":
            action = root.best_action(discount)
        elif action_selection == "highest_count":
            action = root.highest_visit_count()
        else:
            raise Exception("action_selection parameter not valid.")
            
        new_root = tree.get_subtree(action)
        frame, valid_actions, reward, done = env.step(action)

        
        if render:
            env.render()
        total_reward += reward
        if done:
            break
    return total_reward

### Replay buffer stuff ###
def get_cumulative_rewards(rewards, discount, dones):
    cum_disc_rewards = []
    cum_r = 0
    for i,r in enumerate(reversed(rewards)):
        not_done = 1 - dones[-(i+1)]
        cum_r = not_done*discount*cum_r + r
        cum_disc_rewards.append (cum_r)
    cum_disc_rewards = torch.tensor(cum_disc_rewards[::-1])
    return cum_disc_rewards

class ReplayBuffer:
    def __init__(self, mem_size, discount):
        self.mem_size = mem_size
        self.discount = discount
        self.frame_buffer = []
        self.V_target_buffer = []
    
    def store_episode(self, frame_lst, reward_lst, done_lst):
        frames, targets = self.batch_episode(frame_lst, reward_lst, done_lst)
        self.frame_buffer.append(frames)
        self.V_target_buffer.append(targets)
        if len(self.frame_buffer) > self.mem_size:
            self.frame_buffer.pop(0)
            self.V_target_buffer.pop(0)
            
    def batch_episode(self, frame_lst, reward_lst, done_lst):
        episode_len = len(reward_lst)
        frames = {}
        for k in frame_lst[0].keys():
            k_value_lst = []
            for b in range(episode_len): # this is a problem, since the frame_lst has one more state!!!
                k_value_lst.append(frame_lst[b][k])
            k_value_lst = torch.cat(k_value_lst, axis=0)
            frames[k] = k_value_lst.unsqueeze(0)
        rewards = torch.tensor(reward_lst).float()
        dones = torch.tensor(done_lst).float()
        targets = get_cumulative_rewards(rewards, self.discount, dones)
        
        return frames, targets.unsqueeze(0)
    
    def get_batch(self, batch_size):
        id_range = len(self.V_target_buffer)
        assert id_range >= batch_size, "Not enough samples stored to get this batch size"
        sampled_ids = np.random.choice(id_range, size=batch_size, replace=False)
        sampled_targets = torch.cat([self.V_target_buffer[i] for i in sampled_ids], axis=0)
        # batch together frames 
        sampled_frames = {}
        for k in self.frame_buffer[0].keys():
            key_values = torch.cat([self.frame_buffer[i][k] for i in sampled_ids], axis=0)
            sampled_frames[k] = key_values
        return sampled_frames, sampled_targets

def compute_update(value_net, frames, targets, loss_fn, optimizer):
    reshaped_frames = {}
    for k in frames.keys():
        shape = frames[k].shape
        reshaped_frames[k] = frames[k].reshape(-1,*shape[2:])
    targets = targets.reshape(-1).to(device)
    values = value_net(reshaped_frames).squeeze(1)
    
    loss = loss_fn(values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

class nStepsReplayBuffer:
    def __init__(self, mem_size, discount):
        self.mem_size = mem_size
        self.discount = discount
        self.frame_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        
    def store_episode(self, frame_lst, reward_lst, done_lst):
        frames, rewards, done = self.batch_episode(frame_lst, reward_lst, done_lst)
        self.frame_buffer.append(frames)
        self.reward_buffer.append(rewards)
        self.done_buffer.append(done)
        if len(self.frame_buffer) > self.mem_size:
            self.frame_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.done_buffer.pop(0)
            
    def batch_episode(self, frame_lst, reward_lst, done_lst):
        """
        Unifies the time dimension fo the data and adds a batch dimension of 1 in front
        """
        episode_len = len(reward_lst)+1 # counting also initial frame with no reward
        frames = {}
        for k in frame_lst[0].keys():
            k_value_lst = []
            for b in range(episode_len):
                k_value_lst.append(frame_lst[b][k])
            k_value_lst = torch.cat(k_value_lst, axis=0)
            frames[k] = k_value_lst.unsqueeze(0) # add batch size dimension in front
            
        rewards = np.array(reward_lst, dtype=np.float)
        done = np.array(done_lst, dtype=np.bool)
        
        return frames, rewards, done
    
    def get_batch(self, batch_size, n_steps, discount, target_net, device="cpu"):
        # Decide which indexes to sample
        id_range = len(self.frame_buffer)
        assert id_range >= batch_size, "Not enough samples stored to get this batch size"
        sampled_ids = np.random.choice(id_range, size=batch_size, replace=False)
        
        # Sample frames, rewards and done
        sampled_rewards = np.array([self.reward_buffer[i] for i in sampled_ids])
        sampled_done = np.array([self.done_buffer[i] for i in sampled_ids])
        
        # batch together frames 
        sampled_frames = {}
        for k in self.frame_buffer[0].keys():
            key_values = torch.cat([self.frame_buffer[i][k] for i in sampled_ids], axis=0)
            sampled_frames[k] = key_values
            
        trg_frames = {}
        src_frames = {}
        for k in sampled_frames.keys():
            trg_frames[k] = sampled_frames[k][:,1:] # all but first frame
            src_frames[k] = sampled_frames[k][:,:-1] # all but ast frame     
            
        # sampled_targets of shape (B*T,)
        sampled_targets = self.compute_n_step_V_trg(n_steps, discount, sampled_rewards, sampled_done, 
                                                    trg_frames, target_net, device)
        # Flatten also the src_frames
        reshaped_frames = {}
        for k in src_frames.keys():
            shape = src_frames[k].shape
            reshaped_frames[k] = src_frames[k].reshape(-1,*shape[2:])

        return reshaped_frames, sampled_targets
    
    def compute_n_step_V_trg(self, n_steps, discount, rewards, done, trg_states, value_net, device="cpu"):
        """
        Compute m-steps value target, with m = min(n_steps, steps-to-episode-end).
        Formula (for precisely n-steps):
            V^{(n)}(t) = \sum_{k=}^{n-1} gamma^k r_{t+k+1} + gamma^n * V(s_{t+n})

        Input
        -----
        n_steps: int
            How many steps in the future to consider before bootstrapping while computing the value target
        discount: float in (0,1)
            Discount factor of the MDP
        rewards: np.array of shape (B,T), type float
        done: np.array of shape (B,T), type bool
        trg_states: dictionary of tensors all of shape (B,T,...)
        value_net: instance of nn.Module
            outputs values of shape (B*T,) given states reshaped as (B*T,...)

        """
        done_plus_ending = done.copy()
        done_plus_ending[:,-1] = True
        n_step_rewards, episode_mask, n_steps_mask_b = self.compute_n_step_rewards(rewards, done_plus_ending, n_steps, discount)
        new_states, Gamma_V, done = self.compute_n_step_states(trg_states, done, episode_mask, n_steps_mask_b, discount)

        new_states_reshaped = {}
        for k in new_states.keys():
            new_states_reshaped[k] = new_states[k].reshape((-1,)+new_states[k].shape[2:])
        done = torch.LongTensor(done.astype(int)).to(device).reshape(-1)
        n_step_rewards = torch.tensor(n_step_rewards).float().to(device).reshape(-1)
        Gamma_V = torch.tensor(Gamma_V).float().to(device).reshape(-1)

        with torch.no_grad():
            V_pred = value_net(new_states_reshaped).squeeze()
            V_trg = (1-done)*Gamma_V*V_pred + n_step_rewards
            V_trg = V_trg.squeeze()
        return V_trg
    
    def compute_n_step_rewards(self, rewards, done, n_steps, discount):
        """
        Computes n-steps discounted reward. 
        Note: the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """

        B = done.shape[0]
        T = done.shape[1]

        # Compute episode mask (i-th row contains 1 if col j is in the same episode of col i, 0 otherwise)
        episode_mask = [[] for _ in range(B)]
        last = [-1 for _ in range(B)]
        xs, ys = np.nonzero(done)

        # Add done at the end of every batch to avoid exceptions -> not used in real target computations
        xs = np.concatenate([xs, np.arange(B)])
        ys = np.concatenate([ys, np.full(B, T-1)])
        for x, y in zip(xs, ys):
            m = [1 if (i > last[x] and i <= y) else 0 for i in range(T)]
            for _ in range(y-last[x]):
                episode_mask[x].append(m)
            last[x] = y
        episode_mask = np.array(episode_mask)

        # Compute n-steps mask and repeat it B times
        n_steps_mask = []
        for i in range(T):
            m = [1 if (j>=i and j<i+n_steps) else 0 for j in range(T)]
            n_steps_mask.append(m)
        n_steps_mask = np.array(n_steps_mask)
        n_steps_mask_b = np.repeat(n_steps_mask[np.newaxis,...] , B, axis=0)

        # Broadcast rewards to use multiplicative masks
        rewards_repeated = np.repeat(rewards[:,np.newaxis,:], T, axis=1)

        # Exponential discount factor
        Gamma = np.array([discount**i for i in range(T)]).reshape(1,-1)
        n_steps_r = (Gamma*rewards_repeated*episode_mask*n_steps_mask_b).sum(axis=2)/Gamma
        return n_steps_r, episode_mask, n_steps_mask_b

    def compute_n_step_states(self, trg_states, done, episode_mask, n_steps_mask_b, discount):
        """
        Computes n-steps target states (to be used by the critic as target values together with the
        n-steps discounted reward). For last n-1 elements the target state is the last one available.
        Adjusts also the `done` mask used for disabling the bootstrapping in the case of terminal states
        and returns Gamma_V, that are the discount factors for the target state-values, since they are 
        n-steps away (except for the last n-1 states, whose discount is adjusted accordingly).

        Return
        ------
        new_states, Gamma_V, done: arrays with first dimension = len(states)-1
        """

        B = done.shape[0]
        T = done.shape[1]
        V_mask = episode_mask*n_steps_mask_b
        b, x, y = np.nonzero(V_mask)
        V_trg_index = [[] for _ in range(B)]
        for b_i in range(B):
            valid_x = (b==b_i)
            for i in range(T):
                matching_x = (x==i)
                V_trg_index[b_i].append(y[valid_x*matching_x][-1])
        V_trg_index = np.array(V_trg_index)

        cols = np.array([], dtype=np.int)
        rows = np.array([], dtype=np.int)
        for i, v in enumerate(V_trg_index):
            cols = np.concatenate([cols, v], axis=0)
            row = np.full(V_trg_index.shape[1], i)
            rows = np.concatenate([rows, row], axis=0)

        new_states = {}
        for k in trg_states.keys(): 
            new_states[k] = trg_states[k][rows, cols].reshape(trg_states[k].shape)

        pw = V_trg_index - np.arange(V_trg_index.shape[1]) + 1
        Gamma_V = discount**pw
        shifted_done = done[rows, cols].reshape(done.shape)
        return new_states, Gamma_V, shifted_done


def compute_update_v1(value_net, frames, targets, loss_fn, optimizer):
    values = value_net(frames).squeeze(1)
    
    loss = loss_fn(values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def update_target_net(target_net, value_net, tau):
     # Update critic_target: (1-tau)*old + tau*new
        
    for trg_params, params in zip(target_net.parameters(), value_net.parameters()):
            trg_params.data.copy_((1.-tau)*trg_params.data + tau*params.data)
            
    # in case BatchNorm2d is used, copy also running_mean and running_var, which are not parameters
    for src_module, trg_module in zip(value_net.modules(), target_net.modules()):
        if isinstance(src_module, nn.BatchNorm2d):
            assert isinstance(trg_module, nn.BatchNorm2d), "src_module is instance of  BatchNorm2d but rg_module is not"
            trg_module.running_mean = src_module.running_mean
            trg_module.running_var = src_module.running_var


def plot_value_stats(value_net, target_net, rb, batch_size, n_steps, discount, device):
    # Value vs target hist on a batch
    target_net.eval()
    frames, targets = rb.get_batch(batch_size, n_steps, discount, target_net, device)
    targets = targets.reshape(-1).cpu().numpy()
    with torch.no_grad():
        value_net.eval()
        trg_values = target_net(frames).squeeze(1).cpu().numpy()
        values = value_net(frames).squeeze(1).cpu().numpy()
    diff = values - targets
    diff1 = values - trg_values

    print("Confronting ")
    fig, ax = plt.subplots(2,2, figsize=(12,12))

    # Targets distribution
    ns, xs, _ = ax[0,0].hist(targets, bins=50, label="Total counts = %d"%len(targets))
    ax[0,0].set_xlabel("Target values (using bootstrap)", fontsize=15)
    ax[0,0].set_ylabel("Counts", fontsize=15)
    ax[0,0].legend(fontsize=13)

    # Values distribution
    ns, xs, _ = ax[0,1].hist(values, bins=50, label="Total counts = %d"%len(values))
    ax[0,1].set_xlabel("Predicted values", fontsize=15)
    ax[0,1].set_ylabel("Counts", fontsize=15)
    ax[0,1].legend(fontsize=13)

    # Distribution of the difference
    ns, xs, _ = ax[1,0].hist(diff, bins=50, label="Total counts = %d"%len(diff))
    ax[1,0].set_xlabel("Difference (values - targets)", fontsize=15)
    ax[1,0].set_ylabel("Counts", fontsize=15)
    ax[1,0].legend(fontsize=13)

    # Scatterplot targets (x) vs values (y)
    m, M = targets.min(), targets.max()
    x = np.linspace(m,M,100)
    ax[1,1].scatter(targets, values, s=4, label='datapoints')
    ax[1,1].plot(x,x, 'r--', label="Identity line")
    ax[1,1].set_xlabel("Target values (using bootstrap)", fontsize=15)
    ax[1,1].set_ylabel("Values predicted", fontsize=15)
    ax[1,1].legend(fontsize=13)

    plt.tight_layout()
    plt.show()

    
    fig, ax = plt.subplots(2,2, figsize=(12,12))

    # Targets distribution
    ns, xs, _ = ax[0,0].hist(trg_values, bins=50, label="Total counts = %d"%len(trg_values))
    ax[0,0].set_xlabel("Values predicted by target net (no bootstrap)", fontsize=15)
    ax[0,0].set_ylabel("Counts", fontsize=15)
    ax[0,0].legend(fontsize=13)

    # Values distribution
    ns, xs, _ = ax[0,1].hist(values, bins=50, label="Total counts = %d"%len(values))
    ax[0,1].set_xlabel("Predicted values", fontsize=15)
    ax[0,1].set_ylabel("Counts", fontsize=15)
    ax[0,1].legend(fontsize=13)

    # Distribution of the difference
    ns, xs, _ = ax[1,0].hist(diff1, bins=50, label="Total counts = %d"%len(diff1))
    ax[1,0].set_xlabel("Difference (values - targets)", fontsize=15)
    ax[1,0].set_ylabel("Counts", fontsize=15)
    ax[1,0].legend(fontsize=13)

    # Scatterplot targets (x) vs values (y)
    m, M = trg_values.min(), trg_values.max()
    x = np.linspace(m,M,100)
    ax[1,1].scatter(trg_values, values, s=4, label='datapoints')
    ax[1,1].plot(x,x, 'r--', label="Identity line")
    ax[1,1].set_xlabel("Values predicted by target net (no bootstrap)", fontsize=15)
    ax[1,1].set_ylabel("Values predicted", fontsize=15)
    ax[1,1].legend(fontsize=13)

    plt.tight_layout()
    plt.show()


# ## Performance tests ###

def compare_modes(game_simulator, value_net, episode_length, ucb_C, discount, max_actions, num_simulations, n_episodes=10):
    simulate_score = 0
    predict_score = 0
    sim_and_pred_score = 0
    hybrid_score = 0
    
    for i in range(n_episodes):
        #print("\nEpisode number %d"%(i+1))
        if i == 0:
            render=True
        else:
            render=False
        frame, valid_actions = game_simulator.reset()
        original_sim_state = game_simulator.save_state_dict()
        
        if render:
            print("\nMode: simulate")
        game_simulator.load_state_dict(copy.deepcopy(original_sim_state))
        simulate_reward = play_episode_given_IC(value_net,
                                                game_simulator,
                                                frame,
                                                valid_actions,
                                                episode_length,
                                                ucb_C,
                                                discount,
                                                max_actions,
                                                num_simulations,
                                                mode="simulate",
                                                render=render)
        #print("Total reward: ", simulate_reward)
        simulate_score += simulate_reward
        
        if render:
            print("\nMode: predict")
        game_simulator.load_state_dict(copy.deepcopy(original_sim_state))
        predict_reward = play_episode_given_IC(value_net,
                                                game_simulator,
                                                frame,
                                                valid_actions,
                                                episode_length,
                                                ucb_C,
                                                discount,
                                                max_actions,
                                                num_simulations,
                                                mode="predict",
                                                render=render)
        #print("Total reward: ", predict_reward)
        predict_score += predict_reward
        
        if render:
            print("\nMode: simulate and predict")
        game_simulator.load_state_dict(copy.deepcopy(original_sim_state))
        sim_and_pred_reward = play_episode_given_IC(value_net,
                                                game_simulator,
                                                frame,
                                                valid_actions,
                                                episode_length,
                                                ucb_C,
                                                discount,
                                                max_actions,
                                                num_simulations,
                                                mode="simulate_and_predict",
                                                render=render)
        #print("Total reward: ", sim_and_pred_reward)
        sim_and_pred_score += sim_and_pred_reward
        
        if render:
            print("\nMode: hybrid")
        game_simulator.load_state_dict(copy.deepcopy(original_sim_state))
        hybrid_reward = play_episode_given_IC(value_net,
                                                game_simulator,
                                                frame,
                                                valid_actions,
                                                episode_length,
                                                ucb_C,
                                                discount,
                                                max_actions,
                                                num_simulations,
                                                mode="hybrid",
                                                render=render)
        #print("Total reward: ", sim_and_pred_reward)
        hybrid_score += hybrid_reward
        
    simulate_score = simulate_score/n_episodes
    predict_score = predict_score/n_episodes
    sim_and_pred_score = sim_and_pred_score/n_episodes 
    hybrid_score = hybrid_score/n_episodes
    return simulate_score, predict_score, sim_and_pred_score, hybrid_score

def compare_selection_methods(game_simulator, value_net, episode_length, ucb_C, 
                              discount, max_actions, num_simulations, n_episodes=10):
    Q_value_selection = 0
    count_selection = 0
    
    for i in range(n_episodes):
        #print("\nEpisode number %d"%(i+1))
        if i == 0:
            render=True
        else:
            render=False
        frame, valid_actions = game_simulator.reset()
        original_sim_state = game_simulator.save_state_dict()
        
        if render:
            print("\nSelection mode: Q value")
        game_simulator.load_state_dict(copy.deepcopy(original_sim_state))
        Q_value_reward = play_episode_given_IC(value_net,
                                                game_simulator,
                                                frame,
                                                valid_actions,
                                                episode_length,
                                                ucb_C,
                                                discount,
                                                max_actions,
                                                num_simulations,
                                                mode="predict",
                                                action_selection="Q_value",
                                                render=render)
        #print("Total reward: ", simulate_reward)
        Q_value_selection += Q_value_reward
        
        if render:
            print("\nMode: highest count")
        game_simulator.load_state_dict(copy.deepcopy(original_sim_state))
        count_reward = play_episode_given_IC(value_net,
                                                game_simulator,
                                                frame,
                                                valid_actions,
                                                episode_length,
                                                ucb_C,
                                                discount,
                                                max_actions,
                                                num_simulations,
                                                mode="predict",
                                                action_selection="highest_count",
                                                render=render)
        #print("Total reward: ", predict_reward)
        count_selection += count_reward
        
       
        
    Q_value_selection = Q_value_selection/n_episodes
    count_selection = count_selection/n_episodes
    return Q_value_selection, count_selection

def compare_with_untrained_net(game_simulator, value_net, untrained_value_net, episode_length, ucb_C, 
                              discount, max_actions, num_simulations, n_episodes=10):
    trained_score = 0
    untraind_score = 0
    
    for i in range(n_episodes):
        #print("\nEpisode number %d"%(i+1))
        if i == 0:
            render=True
        else:
            render=False
        frame, valid_actions = game_simulator.reset()
        original_sim_state = game_simulator.save_state_dict()
        
        if render:
            print("\nNetwork: trained")
        game_simulator.load_state_dict(copy.deepcopy(original_sim_state))
        trained_reward = play_episode_given_IC(value_net,
                                                game_simulator,
                                                frame,
                                                valid_actions,
                                                episode_length,
                                                ucb_C,
                                                discount,
                                                max_actions,
                                                num_simulations,
                                                mode="predict",
                                                render=render)
        #print("Total reward: ", simulate_reward)
        trained_score += trained_reward
        
        if render:
            print("\nNetwork: untrained")
        game_simulator.load_state_dict(copy.deepcopy(original_sim_state))
        untrained_reward = play_episode_given_IC(untrained_value_net,
                                                game_simulator,
                                                frame,
                                                valid_actions,
                                                episode_length,
                                                ucb_C,
                                                discount,
                                                max_actions,
                                                num_simulations,
                                                mode="predict",
                                                render=render)
        #print("Total reward: ", predict_reward)
        untraind_score += untrained_reward
        
       
        
    trained_score = trained_score/n_episodes
    untraind_score = untraind_score/n_episodes
    return trained_score, untraind_score

class Pi:
    def __init__(self,
                 simulator,
                 value_net,
                 discount,
                 temperature=0.5
                ):
        self.simulator = simulator
        self.value_net = value_net
        self.discount = discount
        self.T = temperature
        
    def get_action(self, valid_actions, simulator_state):
        Q_values = []
        action_dict = {
            0:"Stay",
            1:"Up",
            2:"Down",
            3:"Left",
            4:"Right"
            }
        print("valid_actions: ", valid_actions)
        for a in valid_actions:
            print("a:", a, action_dict[a])
            self.simulator.load_state_dict(copy.deepcopy(simulator_state))
            frame, possible_actions, reward, done = self.simulator.step(a)
            print("reward: ", reward)
            with torch.no_grad():
                value = self.value_net(frame).item()
                print("value: ", value)
            Q = reward + (1-float(done))*self.discount*value
            Q_values.append(Q)
        Q_values = np.array(Q_values)
        print("Q values: ", Q_values)
        soft_Q = np.exp(Q_values/self.T)/np.exp(Q_values/self.T).sum()
        print("soft_Q: ", soft_Q)
        print("valid_actions: ", valid_actions)
        best_action = np.random.choice(valid_actions, p=soft_Q)
        #print("Best id: ", best_id)
        #best_action = valid_actions[best_id]
        print("best action: ", best_action)
        return best_action

def test_value_policy(value_net, env, episode_length, discount, render = False):
    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    
    policy = Pi(env, value_net, discount)
    frame, valid_actions = env.reset()
    print("valid actions: ", valid_actions)
    simulator_state = env.save_state_dict()
    if render:
        env.render()
    total_reward = 0
    done = False
    
    for i in range(episode_length):
        a = policy.get_action(valid_actions, simulator_state)
        print("Action selected from value policy: ", a, "({})".format(action_dict[a]))
        # reset internal state of the env
        env.load_state_dict(simulator_state)
        #if render:
        #    env.render()
        frame, valid_actions, reward, done = env.step(a)
        print("valid actions: ", valid_actions)
        simulator_state = env.save_state_dict()
        if render:
            env.render()
        print("Reward received: ", reward)
        print("Done: ", done)
        total_reward += reward
        if done:
            break
    return total_reward
