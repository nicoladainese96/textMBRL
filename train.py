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
from play_functions import *
from replay_buffers import *

verbose = False
vprint = print if verbose else lambda *args, **kwargs: None

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print("Using device "+device)

### Updates functions for training ###

def compute_update(value_net, frames, targets, loss_fn, optimizer):
    """
    Computes the update for a standard value network.
    """
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

def compute_update_v1(value_net, frames, targets, loss_fn, optimizer):
    """
    Computes the update for a standard value network.
    
    Frames and targets already assumed to be dictionaries containing tensors with first shape B*T.
    """
    values = value_net(frames).squeeze(1)
    
    loss = loss_fn(values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def compute_update_discrete_value_net(value_net, frames, targets, optimizer):
    """
    Computes the update for a discrete support set value network.
    
    Frames and targets already assumed to be dictionaries containing tensors with first shape B*T.
    """
    discrete_targets = mcts.scalar_to_support_v1(targets.view(-1,1), value_net.support_size).squeeze()
    values = value_net.logits(frames)
    loss = (-discrete_targets * F.log_softmax(values, dim=1)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def update_target_net(target_net, value_net, tau):
    """
    Updates the target network with an exponential moving average of the weights of the critic.
    
    Copies all stored statistics of the BatchNorm2d layers (mean and variance of each input channel).
    """
     # Update critic_target: (1-tau)*old + tau*new
        
    for trg_params, params in zip(target_net.parameters(), value_net.parameters()):
            trg_params.data.copy_((1.-tau)*trg_params.data + tau*params.data)
            
    # in case BatchNorm2d is used, copy also running_mean and running_var, which are not parameters
    for src_module, trg_module in zip(value_net.modules(), target_net.modules()):
        if isinstance(src_module, nn.BatchNorm2d):
            assert isinstance(trg_module, nn.BatchNorm2d), "src_module is instance of  BatchNorm2d but rg_module is not"
            trg_module.running_mean = src_module.running_mean
            trg_module.running_var = src_module.running_var

def compute_policy_update(pv_net, frames, target_probs, optimizer):
    """
    Computes the update only for the policy head of PV network using the full cross-entropy loss 
    (target is a distribution over the actions and not a single action).
    """
    target_probs = target_probs.to(device)
    #print("target_probs: \n", target_probs)
    mask = (target_probs==0)
    #print("mask: ", mask)
    values, probs = pv_net(frames) # (B*T, A)
    #print("probs: \n", probs)
    # regularize nan, since they won't contribute anyways because the target prob is 0
    probs = torch.clamp(probs, 1e-9, 1 - 1e-9)
    log_probs = torch.log(probs)
    #print("log_probs: \n", log_probs)
    loss = -(target_probs*log_probs).sum(axis=1).mean() # cross-entropy averaged over batch dim
    #print("loss: \n", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def compute_policy_update_action_targets(pv_net, frames, target_actions, target_best_actions, optimizer):
    """
    Computes the update only for the policy head of PV network using the cross-entropy loss with the sampled action 
    as a target (like a classification task with a single right answer).
    """
    target_actions = target_actions.to(device)
    values, probs = pv_net(frames) # (B*T, A)
    # regularize nan, since they won't contribute anyways because the target prob is 0
    probs = torch.clamp(probs, 1e-9, 1 - 1e-9)
    log_probs = torch.log(probs)
  
    loss = F.nll_loss(log_probs, target_actions) # cross-entropy averaged over batch dim
    #print("loss: \n", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    entropy = compute_policy_entropy(probs, log_probs)
    accuracy = compute_policy_accuracy(target_best_actions, probs)
    
    return loss.item(), entropy, accuracy

def compute_policy_entropy(probs, log_probs):
    with torch.no_grad():
        H = -(probs*log_probs).sum(axis=1).mean()
    return H.item()

def compute_policy_accuracy(target_best_actions, probs):
    """
    Computes the percentage of optimal actions that would be chosen according to the arg max of the policy probabilities
    (usually actions are chosen after the PV-MCTS step, which involves the 'probs' as they are the prior of the root node,
    but the resulting action can be different).
    """
    with torch.no_grad():
        best_actions = probs.argmax(axis=1)
        correct = target_best_actions[torch.arange(len(best_actions)), best_actions]
        accuracy = correct.float().mean().item()
    return accuracy

def compute_policy_value_update_action_targets(pv_net, frames, target_values, target_actions, target_best_actions, optimizer):
    """
    compute_update_discrete_value_net +  compute_policy_update_action_targets 
    
    Assumes that the value network uses a discrete support set and that the policy and the value are the outputs of a single 
    network with two heads.
    """
    ### Common prediction ###
    v_logits, probs = pv_net(frames, return_v_logits=True) # (B*T, A)
    
    ### Value loss ###
    discrete_targets = mcts.scalar_to_support_v1(target_values.view(-1,1), pv_net.support_size).squeeze()
    value_loss = (-discrete_targets * F.log_softmax(v_logits, dim=1)).mean()

    ### Policy loss ###
    target_actions = target_actions.to(device)
    # regularize nan, since they won't contribute anyways because the target prob is 0
    probs = torch.clamp(probs, 1e-9, 1 - 1e-9)
    log_probs = torch.log(probs)
    policy_loss = F.nll_loss(log_probs, target_actions) # cross-entropy averaged over batch dim
    
    ### Update ###
    loss = value_loss + policy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    ### Extra metrics for monitoring training ###
    entropy = compute_policy_entropy(probs, log_probs)
    accuracy = compute_policy_accuracy(target_best_actions, probs)
    
    return loss.item(), entropy, accuracy, policy_loss.item(), value_loss.item()


def compute_PV_net_update(
    pv_net,
    frames,
    target_values,
    target_actions,
    target_best_actions,
    target_probs,
    optimizer,
    full_cross_entropy=False,
    entropy_bonus=False,
    h=1e-2,
    discrete_support_values=True
):
    """

    """
    ### Value loss ###
    if discrete_support_values:
        v_logits, probs = pv_net(frames, return_v_logits=True) # (B*T, A)
        discrete_targets = mcts.scalar_to_support_v1(target_values.view(-1,1), pv_net.support_size).squeeze()
        value_loss = (-discrete_targets * F.log_softmax(v_logits, dim=1)).mean()

    else:
        values, probs = pv_net(frames)
        value_loss = loss = F.mse_loss(values, target_values)
        
    ### Policy loss ###
    probs = torch.clamp(probs, 1e-9, 1 - 1e-9)
    log_probs = torch.log(probs)
        
    if full_cross_entropy:
        target_probs = target_probs.to(device)
        mask = (target_probs==0)
        policy_loss = -(target_probs*log_probs).sum(axis=1).mean() # cross-entropy averaged over batch dim
    else:
        target_actions = target_actions.to(device)
        policy_loss = F.nll_loss(log_probs, target_actions) # cross-entropy averaged over batch dim
    
    entropy = -(probs*log_probs).sum(axis=1).mean()
    if entropy_bonus:
        policy_loss = policy_loss - h*entropy # negative entropy so that when minimizing entropy increases
        
    ### Update ###
    loss = value_loss + policy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    ### Extra metrics for monitoring training ###
    accuracy = compute_policy_accuracy(target_best_actions, probs)
    
    return loss.item(), entropy.item(), accuracy, policy_loss.item(), value_loss.item()


############################################################################################################################

def plot_value_stats(value_net, target_net, rb, batch_size, n_steps, device, n_peaks=10):
    # Value vs target hist on a batch
    target_net.eval()
    frames, targets = rb.get_batch(batch_size, n_steps, target_net, device)
    targets = targets.reshape(-1).cpu().numpy()
    with torch.no_grad():
        value_net.eval()
        trg_values = target_net(frames).flatten().cpu().numpy()
        values = value_net(frames).flatten().cpu().numpy()
    diff = values - targets
    diff1 = values - trg_values

    print("Confronting target and value distributions")
    fig, ax = plt.subplots(2,2, figsize=(14,12))

    # Targets distribution
    ns, xs, _ = ax[0,0].hist(targets, bins=50, label="Total counts = %d"%len(targets))
    ax[0,0].set_xlabel("Target values (using bootstrap)", fontsize=15)
    ax[0,0].set_ylabel("Counts", fontsize=15)
    ax[0,0].legend(fontsize=13)
    ax[0,0].set_title('Distribution of target values', fontsize=16)
    
    # Values distribution
    ns, xs, _ = ax[0,1].hist(values, bins=50, label="Total counts = %d"%len(values))
    ax[0,1].set_xlabel("Predicted values", fontsize=15)
    ax[0,1].set_ylabel("Counts", fontsize=15)
    ax[0,1].legend(fontsize=13)
    ax[0,1].set_title('Distribution of predicted values', fontsize=16)
    
    # Distribution of the difference
    ns, xs, _ = ax[1,0].hist(diff, bins=50, label="Total counts = %d"%len(diff))
    ax[1,0].set_xlabel("Difference (values - targets)", fontsize=15)
    ax[1,0].set_ylabel("Counts", fontsize=15)
    ax[1,0].legend(fontsize=13)
    ax[1,0].set_title('Distribution of the residuals', fontsize=16)
    
    """# Scatterplot targets (x) vs values (y)
    m, M = targets.min(), targets.max()
    x = np.linspace(m,M,100)
    ax[1,1].scatter(targets, values, s=4, label='datapoints')
    ax[1,1].plot(x,x, 'r--', label="Identity line")
    ax[1,1].set_xlabel("Target values (using bootstrap)", fontsize=15)
    ax[1,1].set_ylabel("Values predicted", fontsize=15)
    ax[1,1].legend(fontsize=13)
    """
    # Boxplots of values grouped by targets
    V_exact =[rb.discount**i for i in range(n_steps)][::-1]
    # Divide the pairs (target,value) based on the value of the target
    epsilon = 1e-3
    classes = []
    for i in range(n_steps):
        if i==0:
            mask_greater_than = np.ones(len(targets))
        else:
            mask_greater_than = (targets > V_exact[i-1] + epsilon) # slighly bigger than previous value

        mask_smaller_than = (targets < V_exact[i] + epsilon)
        class_mask = mask_greater_than*mask_smaller_than # logic 'and' with boolean masks
        class_indexes = np.nonzero(class_mask)[0]
        classes.append(class_indexes)
    data = [np.concatenate([targets[indexes].reshape(-1,1), values[indexes].reshape(-1,1)], axis=1) for indexes in classes]
   
    n = n_peaks
    min_width = V_exact[-n+1] - V_exact[-n] 
    ax[1,1].set_title('Box plots for %d highest target value peaks'%n, fontsize=16)

    ax[1,1].set_xlim(V_exact[-n]-0.05,V_exact[-1]+0.05)
    labels=["%.3f"%v for v in V_exact[-n:]]
    ax[1,1].boxplot(data[-n:], positions=V_exact[-n:], labels=labels, widths=0.9*min_width)
    ax[1,1].set_xlabel("Target values (grouped by peaks)", fontsize=16)
    ax[1,1].set_ylabel("Predicted values", fontsize=16)

    x = np.linspace(V_exact[-n] -0.1, V_exact[-1] +0.1)
    ax[1,1].plot(x,x,"--", label="Identity line")
    N = len(values)
    frequencies = np.array([len(c) for c in classes])
    perc = frequencies[-n:]/N*100
    ax[1,1].legend(fontsize=13)
    ax[1,1].set_xticklabels(["%.3f\n(%.1f%%)"%(l,p) for l,p in zip(ax[1,1].get_xticks(),perc)], rotation = 60)

    plt.tight_layout()
    plt.show()


def plot_value_stats_v2(value_net, target_net, rb, batch_size, n_steps, device, n_peaks=10):
    # Value vs target hist on a batch
    target_net.eval()
    frames, targets, _, _, _ = rb.get_batch(batch_size, n_steps, target_net, device) # using action-value buffer
    targets = targets.reshape(-1).cpu().numpy()
    with torch.no_grad():
        value_net.eval()
        values, _ = value_net(frames)
        values = values.flatten().cpu().numpy()
    diff = values - targets

    print("Confronting target and value distributions")
    fig, ax = plt.subplots(2,2, figsize=(14,12))

    # Targets distribution
    ns, xs, _ = ax[0,0].hist(targets, bins=50, label="Total counts = %d"%len(targets))
    ax[0,0].set_xlabel("Target values (using bootstrap)", fontsize=15)
    ax[0,0].set_ylabel("Counts", fontsize=15)
    ax[0,0].legend(fontsize=13)
    ax[0,0].set_title('Distribution of target values', fontsize=16)
    
    # Values distribution
    ns, xs, _ = ax[0,1].hist(values, bins=50, label="Total counts = %d"%len(values))
    ax[0,1].set_xlabel("Predicted values", fontsize=15)
    ax[0,1].set_ylabel("Counts", fontsize=15)
    ax[0,1].legend(fontsize=13)
    ax[0,1].set_title('Distribution of predicted values', fontsize=16)
    
    # Distribution of the difference
    ns, xs, _ = ax[1,0].hist(diff, bins=50, label="Total counts = %d"%len(diff))
    ax[1,0].set_xlabel("Difference (values - targets)", fontsize=15)
    ax[1,0].set_ylabel("Counts", fontsize=15)
    ax[1,0].legend(fontsize=13)
    ax[1,0].set_title('Distribution of the residuals', fontsize=16)

    # Boxplots of values grouped by targets
    V_exact =[rb.discount**i for i in range(n_steps)][::-1]
    # Divide the pairs (target,value) based on the value of the target
    epsilon = 1e-3
    classes = []
    for i in range(n_steps):
        if i==0:
            mask_greater_than = np.ones(len(targets))
        else:
            mask_greater_than = (targets > V_exact[i-1] + epsilon) # slighly bigger than previous value

        mask_smaller_than = (targets < V_exact[i] + epsilon)
        class_mask = mask_greater_than*mask_smaller_than # logic and with boolean masks
        class_indexes = np.nonzero(class_mask)[0]
        classes.append(class_indexes)
    data = [np.concatenate([targets[indexes].reshape(-1,1), values[indexes].reshape(-1,1)], axis=1) for indexes in classes]
   
    n = n_peaks
    min_width = V_exact[-n+1] - V_exact[-n] 
    ax[1,1].set_title('Box plots for %d highest target value peaks'%n, fontsize=16)

    ax[1,1].set_xlim(V_exact[-n]-0.05,V_exact[-1]+0.05)
    labels=["%.3f"%v for v in V_exact[-n:]]
    ax[1,1].boxplot(data[-n:], positions=V_exact[-n:], labels=labels, widths=0.9*min_width)
    ax[1,1].set_xlabel("Target values (grouped by peaks)", fontsize=16)
    ax[1,1].set_ylabel("Predicted values", fontsize=16)

    x = np.linspace(V_exact[-n] -0.1, V_exact[-1] +0.1)
    ax[1,1].plot(x,x,"--", label="Identity line")
    N = len(values)
    frequencies = np.array([len(c) for c in classes])
    perc = frequencies[-n:]/N*100
    ax[1,1].legend(fontsize=13)
    ax[1,1].set_xticklabels(["%.3f\n(%.1f%%)"%(l,p) for l,p in zip(ax[1,1].get_xticks(),perc)], rotation = 60)

    plt.tight_layout()
    plt.show()

############################################################################################################################



# ## Performance tests - obsolete stuff ###

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
