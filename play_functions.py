# Custom modules
import mcts
from stochastic_mcts import StochasticPVMCTS
import utils
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch import multiprocessing as mp
import random

""
def show_root_summary(root, discount):
    """
    Show the Q-values and visit counts of a (root) node after the MCTS step. 
    """
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

""
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
    """
    Plays an episode with a standard MCTS. Tree is reconstructed from scratch at each step.
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

""
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
    Plays an episode with a standard MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
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

""
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
    Plays an episode with a value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a standard MCTS, if mode='predict', the value network is used to estimate the 
    value of the leaf nodes (instead of a MC rollout).
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

""
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
    Plays a rollout with a value MCTS; the difference with an episode is that a rollout is always composed by a given number of
    steps, thus if an episode ends before the episode_length (should be rollout_length) is reached, the environment is reset
    and the agent continues to play.
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a standard MCTS, if mode='predict', the value network is used to estimate the 
    value of the leaf nodes (instead of a MC rollout).
    
    There is the option of adding a bootstrap value at the end of the episode (either the predicted value or 
    the root value from MCTS of the last state), but as a functionality is deprecated, because this is better handled in the
    replay buffers when extracting a batch before un update.
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

""
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
    
    Notice that the variable 'env' passed has to be in the same internal state that frame is representing. 
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

""
def play_episode_policy_value_net(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    temperature,
    object_ids,
    mode="simulate",
    render = False,
    debug_render=False
):
    """
    Plays an episode with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Samples the next action based on the visit count of the root node's children and returns the MCTS policy as the target with
    which to train the policy network.
    
    Formula used for MCTS policy ('polynomial weighting' of the visit counts):
    
    p(a) = N(a)^{1/T} / \sum_b N(b)^{1/T}
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
    probs_lst = []
    reward_lst = []
    done_lst = []
    action_is_optimal = []
    for i in range(episode_length):
        tree = mcts.PolicyValueMCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root
                             )
        #print("Performing MCTS step")
        root, info = tree.run(num_simulations, mode=mode)
        #show_root_summary(root, discount)
        #print("Tree info: ", info)
        action, probs = root.sample_child(temperature)
        probs_lst.append(probs) # shape (A,)
        if render:
            print("probs from MCTS: ", probs)
            print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        
        # Evaluate chosen action against optimal policy
        best_actions = utils.get_optimal_actions(frame, object_ids)
        if render:
            print("Best actions: ", best_actions, [action_dict[a] for a in best_actions])
        if action in best_actions:
            action_is_optimal.append(True)
        else:
            action_is_optimal.append(False)
            
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
            break
    return total_reward, frame_lst, reward_lst, done_lst, probs_lst, action_is_optimal

""
def play_episode_policy_value_net_v1(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    object_ids,
    mode="simulate",
    dir_noise=False,
    render = False,
    debug_render=False
):
    """
    Plays an episode with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Chooses the best action as the one with highest Q-value according to the MCTS step and actually it's not returning
    any signal on which to train the policy (probably I used this to test a policy trained in a supervised fashion to
    predict the optimal actions given by a hard-coded policy; value net is not trained, thus this shoudl be used only
    in 'simulate' mode.
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
    action_is_optimal = []
    for i in range(episode_length):
        tree = mcts.PolicyValueMCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root
                             )
        #print("Performing MCTS step")
        root, info = tree.run(num_simulations, mode=mode, dir_noise=dir_noise)
        #show_root_summary(root, discount)
        #print("Tree info: ", info)
        action = root.best_action(discount)
        if render:
            #print("probs from MCTS: ", probs)
            show_root_summary(root, discount)
            print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        
        # Evaluate chosen action against optimal policy
        best_actions = utils.get_optimal_actions(frame, object_ids)
        if render:
            print("Best actions: ", best_actions, [action_dict[a] for a in best_actions])
        if action in best_actions:
            action_is_optimal.append(True)
        else:
            action_is_optimal.append(False)
            
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
            break
    return total_reward, frame_lst, reward_lst, done_lst, action_is_optimal

""
def show_policy_summary(pv_net, frame, root, discount, mcts_action, best_actions):
    """
    TO EDIT.
    """
    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    with torch.no_grad():
        v, prior = pv_net(frame)
        prior = prior.cpu().numpy().flatten()
        
    for action, child in root.children.items():
        Q =  child.reward + discount*child.value()
        visits = child.visit_count
        print("Action ", action_dict[action], ": Prior=%.3f - Q-value=%.3f - Visit counts=%d"%(prior[action],Q,visits))
    
    best_prior = prior.argmax()
    print("Action with best prior: ", best_prior, "({})".format(action_dict[best_prior]))
    print("Action selected from MCTS: ", mcts_action, "({})".format(action_dict[mcts_action]))
    print("Best actions: ", best_actions, [action_dict[a] for a in best_actions])
    return best_prior

""
def play_episode_policy_value_net_v2(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    object_ids,
    mode="simulate",
    dir_noise=False,
    render = False,
    debug_render=False
):
    """
    Plays an episode with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Chooses the best action as the one with highest Q-value according to the MCTS step and actually it's not returning
    any signal on which to train the policy (probably I used this to test a policy trained in a supervised fashion to
    predict the optimal actions given by a hard-coded policy; value net is not trained, thus this shoudl be used only
    in 'simulate' mode.
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
    action_is_optimal = []
    if render:
        prior_is_optimal = []
    for i in range(episode_length):
        tree = mcts.PolicyValueMCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root
                             )
        #print("Performing MCTS step")
        root, info = tree.run(num_simulations, mode=mode, dir_noise=dir_noise)
        #show_root_summary(root, discount)
        #print("Tree info: ", info)
        action = root.best_action(discount)
        best_actions = utils.get_optimal_actions(frame, object_ids)
        if render:
            #print("probs from MCTS: ", probs)
            best_prior = show_policy_summary(pv_net, frame, root, discount, action, best_actions)
            
            if best_prior in best_actions:
                prior_is_optimal.append(True)
            else:
                prior_is_optimal.append(False)

        # Evaluate chosen action against optimal policy
        if action in best_actions:
            action_is_optimal.append(True)
        else:
            action_is_optimal.append(False)
            
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
            break
    if render:
        return total_reward, frame_lst, reward_lst, done_lst, action_is_optimal, prior_is_optimal
    else:
        return total_reward, frame_lst, reward_lst, done_lst, action_is_optimal

""
def play_episode_policy_value_net_v3(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    object_ids,
    mode="simulate",
    ucb_method="p-UCT-old",
    dir_noise=False,
    render = False,
    debug_render=False,
    default_Q=0
):
    """
    Plays an episode with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Chooses the best action as the one with highest Q-value according to the MCTS step and actually it's not returning
    any signal on which to train the policy (probably I used this to test a policy trained in a supervised fashion to
    predict the optimal actions given by a hard-coded policy; value net is not trained, thus this shoudl be used only
    in 'simulate' mode.
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
    action_is_optimal = []
    if render:
        prior_is_optimal = []
    for i in range(episode_length):
        tree = mcts.PV_MCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root,
                             ucb_method=ucb_method
                             )
        #print("Performing MCTS step")
        root, info = tree.run(num_simulations, mode=mode, dir_noise=dir_noise, default_Q=default_Q)
        #show_root_summary(root, discount)
        #print("Tree info: ", info)
        action = root.best_action(discount)
        best_actions = utils.get_optimal_actions(frame, object_ids)
        if render:
            #print("probs from MCTS: ", probs)
            best_prior = show_policy_summary(pv_net, frame, root, discount, action, best_actions)
            
            if best_prior in best_actions:
                prior_is_optimal.append(True)
            else:
                prior_is_optimal.append(False)

        # Evaluate chosen action against optimal policy
        if action in best_actions:
            action_is_optimal.append(True)
        else:
            action_is_optimal.append(False)
            
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
            break
    if render:
        return total_reward, frame_lst, reward_lst, done_lst, action_is_optimal, prior_is_optimal
    else:
        return total_reward, frame_lst, reward_lst, done_lst, action_is_optimal

""
def play_rollout_policy_value_net(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    temperature,
    object_ids,
    mode="simulate",
    bootstrap="no",
    render = False,
    debug_render=False
):
    """
    Plays a rollout with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Samples the next action based on the visit count of the root node's children and returns the MCTS policy as the target with
    which to train the policy network.
    
    Formula used for MCTS policy ('polynomial weighting' of the visit counts):
    
    p(a) = N(a)^{1/T} / \sum_b N(b)^{1/T}
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
    probs_lst = []
    reward_lst = []
    done_lst = []
    action_is_optimal = []
    for i in range(episode_length):
        tree = mcts.PolicyValueMCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root
                             )
        root, info = tree.run(num_simulations, mode=mode)
        action, probs = root.sample_child(temperature)
        probs_lst.append(probs) # shape (A,)
        # Evaluate chosen action against optimal policy
        best_actions = utils.get_optimal_actions(frame, object_ids)
        if action in best_actions:
            action_is_optimal.append(True)
        else:
            action_is_optimal.append(False)
            
        new_root = tree.get_subtree(action)
        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
            
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
    action_is_optimal = np.array(action_is_optimal, dtype=np.float)
    return total_reward, frame_lst, reward_lst, done_lst, probs_lst, action_is_optimal

""
def play_rollout_policy_value_net_v1(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    object_ids,
    dirichlet_alpha, 
    exploration_fraction,
    mode="simulate",
    render = False,
    debug_render=False,
):
    """
    Plays a rolllout with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Selects the next action based on the highest Q-value of the root node's children and returns the list of 
    sampled actions as the target with which to train the policy network.

    This function is also mixing a prior sampled from a Dirichlet distribution (with parameters dirichlet_alpha for each 
    possible action) to the prior of the root node's children, in order to increase exploration at the base of the tree 
    even in cases where the policy is almost deterministic. The mixture coefficient between the prior and the categorical 
    distribution sampled by the Dirichelt distribution is the exploration_fraction, such that:
    
    p(a) = (1-exploration_fraction) Prior(a) + exploration_fraction Dir(a)
    
    """
    A = len(env.env.action_space)
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
    action_lst = []
    best_action_lst = []
    
    for i in range(episode_length):
        tree = mcts.PolicyValueMCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root
                             )
        root, info = tree.run(num_simulations, 
                              mode=mode, 
                              dir_noise=True, 
                              dirichlet_alpha=dirichlet_alpha, 
                              exploration_fraction=exploration_fraction
                             )
        
        action = root.best_action(discount)
        action_lst.append(action)
        
        if render:
            show_root_summary(root, discount)
            print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        best_actions = utils.get_optimal_actions(frame, object_ids)
        best_actions_tensor = torch.tensor([1 if i in best_actions else 0 for i in range(A)]).view(1,-1)
        best_action_lst.append(best_actions_tensor)
        
        new_root = tree.get_subtree(action)
        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
        total_reward += reward
        
        if done:
            frame, valid_actions = env.reset()
            if render:
                print("\nNew episode begins.")
                env.render()
            done = False
            new_root = None


    return total_reward, frame_lst, reward_lst, done_lst, action_lst, torch.cat(best_action_lst, axis=0)

""
def play_rollout_policy_value_net_softQ(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    object_ids,
    dirichlet_alpha, 
    exploration_fraction,
    temperature,
    mode="simulate",
    render = False,
    debug_render=False,
):
    """
    Plays a rolllout with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Samples the next action based on the Q-values of the root node's children and returns both the MCTS policy and the list of 
    sampled actions as possible targets with which to train the policy network.
    
    Formula used for MCTS policy (softmax of Q-values with temperature):
    
    p(a) = exp{Q(a)/T} / \sum_b exp{Q(b)/T}

    Note: the softmax function with T=0 is the argmax function.
    
    This function is also mixing a prior sampled from a Dirichlet distribution (with parameters dirichlet_alpha for each 
    possible action) to the prior of the root node's children, in order to increase exploration at the base of the tree 
    even in cases where the policy is almost deterministic. The mixture coefficient between the prior and the categorical 
    distribution sampled by the Dirichelt distribution is the exploration_fraction, such that:
    
    p(a) = (1-exploration_fraction) Prior(a) + exploration_fraction Dir(a)
    
    """
    
    A = len(env.env.action_space)
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
    action_lst = []
    best_action_lst = []
    probs_lst = []
    
    for i in range(episode_length):
        tree = mcts.PolicyValueMCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root
                             )
        root, info = tree.run(num_simulations, 
                              mode=mode, 
                              dir_noise=True, 
                              dirichlet_alpha=dirichlet_alpha, 
                              exploration_fraction=exploration_fraction
                             )
        
        #action = root.best_action(discount)
        action, probs = root.softmax_Q(temperature, discount)
        action_lst.append(action)
        probs_lst.append(probs)
        
        if render:
            show_root_summary(root, discount)
            print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        best_actions = utils.get_optimal_actions(frame, object_ids)
        best_actions_tensor = torch.tensor([1 if i in best_actions else 0 for i in range(A)]).view(1,-1)
        best_action_lst.append(best_actions_tensor)
        
        new_root = tree.get_subtree(action)
        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
        total_reward += reward
        
        if done:
            frame, valid_actions = env.reset()
            if render:
                print("\nNew episode begins.")
                env.render()
            done = False
            new_root = None


    return total_reward, frame_lst, reward_lst, done_lst, action_lst, torch.cat(best_action_lst, axis=0), probs_lst

""
def play_rollout_policy_value_net_softQ_v1(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    object_ids,
    dirichlet_alpha, 
    exploration_fraction,
    temperature,
    mode="simulate",
    ucb_method="p-UCT-old",
    render=False,
    debug_render=False,
):
    """
    Plays a rolllout with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Samples the next action based on the Q-values of the root node's children and returns both the MCTS policy and the list of 
    sampled actions as possible targets with which to train the policy network.
    
    Formula used for MCTS policy (softmax of Q-values with temperature):
    
    p(a) = exp{Q(a)/T} / \sum_b exp{Q(b)/T}

    Note: the softmax function with T=0 is the argmax function.
    
    This function is also mixing a prior sampled from a Dirichlet distribution (with parameters dirichlet_alpha for each 
    possible action) to the prior of the root node's children, in order to increase exploration at the base of the tree 
    even in cases where the policy is almost deterministic. The mixture coefficient between the prior and the categorical 
    distribution sampled by the Dirichelt distribution is the exploration_fraction, such that:
    
    p(a) = (1-exploration_fraction) Prior(a) + exploration_fraction Dir(a)
    
    Version v1: introduces the variable ucb_method, where different formulas for the p-UCT can be used.
    """
    
    A = len(env.env.action_space)
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
    action_lst = []
    best_action_lst = []
    probs_lst = []
    
    for i in range(episode_length):
        tree = mcts.PV_MCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root,
                             ucb_method=ucb_method
            
                             )
        root, info = tree.run(num_simulations, 
                              mode=mode, 
                              dir_noise=True, 
                              dirichlet_alpha=dirichlet_alpha, 
                              exploration_fraction=exploration_fraction
                             )
        
        #action = root.best_action(discount)
        action, probs = root.softmax_Q(temperature, discount)
        action_lst.append(action)
        probs_lst.append(probs)
        
        if render:
            show_root_summary(root, discount)
            print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        best_actions = utils.get_optimal_actions(frame, object_ids)
        best_actions_tensor = torch.tensor([1 if i in best_actions else 0 for i in range(A)]).view(1,-1)
        best_action_lst.append(best_actions_tensor)
        
        new_root = tree.get_subtree(action)
        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
        total_reward += reward
        
        if done:
            frame, valid_actions = env.reset()
            if render:
                print("\nNew episode begins.")
                env.render()
            done = False
            new_root = None


    return total_reward, frame_lst, reward_lst, done_lst, action_lst, torch.cat(best_action_lst, axis=0), probs_lst


""
def play_rollout_policy_value_net_softQ_v2(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    max_actions,
    num_simulations,
    object_ids,
    dirichlet_alpha, 
    exploration_fraction,
    temperature,
    mode="simulate",
    ucb_method="p-UCT-old",
    render=False,
    debug_render=False,
):
    """
    Plays a rolllout with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Samples the next action based on the Q-values of the root node's children and returns both the MCTS policy and the list of 
    sampled actions as possible targets with which to train the policy network.
    
    Formula used for MCTS policy (softmax of Q-values with temperature):
    
    p(a) = exp{Q(a)/T} / \sum_b exp{Q(b)/T}

    Note: the softmax function with T=0 is the argmax function.
    
    This function is also mixing a prior sampled from a Dirichlet distribution (with parameters dirichlet_alpha for each 
    possible action) to the prior of the root node's children, in order to increase exploration at the base of the tree 
    even in cases where the policy is almost deterministic. The mixture coefficient between the prior and the categorical 
    distribution sampled by the Dirichelt distribution is the exploration_fraction, such that:
    
    p(a) = (1-exploration_fraction) Prior(a) + exploration_fraction Dir(a)
    
    Version v2: same as v1, but it's not re-using the old sub-tree in the new mcts step. 
    This has be done if we want to use the deterministic PV-MCTS as a baseline for the stochastic environment.
    As it is, this function it's not convinient to use in the deterministic setup (altough it can be useful to 
    study the search tree properties from scratch at every step).
    """
    
    A = len(env.env.action_space)
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
    action_lst = []
    best_action_lst = []
    probs_lst = []
    
    for i in range(episode_length):
        tree = mcts.PV_MCTS(
                             frame, 
                             env, 
                             valid_actions, 
                             ucb_C, 
                             discount, 
                             max_actions, 
                             pv_net,
                             render=debug_render, 
                             root=new_root,
                             ucb_method=ucb_method
            
                             )
        root, info = tree.run(num_simulations, 
                              mode=mode, 
                              dir_noise=True, 
                              dirichlet_alpha=dirichlet_alpha, 
                              exploration_fraction=exploration_fraction
                             )
        
        #action = root.best_action(discount)
        action, probs = root.softmax_Q(temperature, discount)
        action_lst.append(action)
        probs_lst.append(probs)
        
        if render:
            show_root_summary(root, discount)
            print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))
        best_actions = utils.get_optimal_actions(frame, object_ids)
        best_actions_tensor = torch.tensor([1 if i in best_actions else 0 for i in range(A)]).view(1,-1)
        best_action_lst.append(best_actions_tensor)
        
        #new_root = tree.get_subtree(action) #
        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
        total_reward += reward
        
        if done:
            frame, valid_actions = env.reset()
            if render:
                print("\nNew episode begins.")
                env.render()
            done = False
            new_root = None


    return total_reward, frame_lst, reward_lst, done_lst, action_lst, torch.cat(best_action_lst, axis=0), probs_lst

def PV_MCTS_process_pipes(
    worker_end,
    frame, 
    env, 
    valid_actions, 
    ucb_C, 
    discount, 
    max_actions, 
    pv_net,
    num_simulations,
    dir_noise, 
    dirichlet_alpha, 
    exploration_fraction,
    ucb_method="AlphaGo",
    mode="predict",
    debug_render=False
):
    """
    Execute single-threaded PV-MCTS step starting from 'frame' and communicate the root's children's Q-values
    and visit counts to the master process through a pipe.
    """
    tree = mcts.PV_MCTS(
                         frame, 
                         env, 
                         valid_actions, 
                         ucb_C, 
                         discount, 
                         max_actions, 
                         pv_net,
                         render=debug_render, 
                         ucb_method=ucb_method
                         )
    
    root, info = tree.run(num_simulations, 
                          mode=mode, 
                          dir_noise=dir_noise, 
                          dirichlet_alpha=dirichlet_alpha, 
                          exploration_fraction=exploration_fraction
                         )
    # get Q values and visit counts at the end of the run and send them to the master process
    Qs = root.get_Q_values(discount).cpu().numpy()
    Ns = root.get_children_visit_counts()
    print("Qs: ", Qs)
    print("Ns: ", Ns)
    worker_end.send((Qs, Ns))
    
def hop_pv_mcts_step(
    frame, 
    env, 
    valid_actions, 
    ucb_C, 
    discount, 
    max_actions, 
    pv_net,
    num_simulations, 
    num_trees,
    temperature=0.0,
    dir_noise=True, 
    dirichlet_alpha=1., 
    exploration_fraction=0.25,
    ucb_method="AlphaGo",
    mode="predict",
    debug_render=False, 
):
    """
    Executes one step of Hindsight Optimization Policy&Value MCTS.
    
    Hindsight Optimization estimates Q values of many determinizations of a stochastic process through a 
    deterministic (in terms of transitions of the environment) planning algorithm in order to find an 
    approximation of the real Q values.  
    
    Returns the mean Q values and visit counts, obtained by averaging those quantities along 'num_trees' 
    parallel simulations.
    """
    
    #pv_net.share_memory() # not sure it's needed here (should be called in the main function)
    
    # Init num_trees pair of pipes
    master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(num_trees)])
    
    workers = []
    for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
        p = mp.Process(target=PV_MCTS_process_pipes,
                       args=(
                            worker_end,
                            frame, 
                            env, 
                            valid_actions, 
                            ucb_C, 
                            discount, 
                            max_actions, 
                            pv_net,
                            num_simulations,
                            dir_noise, 
                            dirichlet_alpha, 
                            exploration_fraction,
                            ucb_method,
                            mode,
                            debug_render,
                       )
                      )
        p.start()
        workers.append(p)
    
    # Now that the workers have been started we need to get results from each pipe before joining the processes
    results = [master_end.recv() for master_end in master_ends]
    
    # Make sure all processes are closed before proceeding
    for p in workers:
        p.join()
        
    # Separate Qs and Ns and make 2 arrays of shape (num_trees, n_actions)
    Qs_realizations = np.concatenate([r[0].reshape(1,-1) for r in results])
    if debug_render:
        print("Qs_realizations: \n", Qs_realizations)
    
    Ns_realizations = np.concatenate([r[1].reshape(1,-1) for r in results])
    if debug_render:
        print("Ns_realizations: \n", Ns_realizations)
    
    mean_Qs = Qs_realizations.mean(axis=0) # average among different trees / realizations
    mean_Ns = Ns_realizations.mean(axis=0)
    
    return mean_Qs, mean_Ns

def softmax_Q(Qs, T):
    """
    Samples an action from the softmax probability obtained from the Q values and a temperature parameter.
    Returns both the action and the probability mass function over the actions.
    """
    Qs = torch.tensor(Qs)
    if T > 0:
        probs = F.softmax(Qs/T, dim=0)
    elif T==0:
        probs = torch.zeros(len(Qs)) 
        a = torch.argmax(Qs)
        probs[a] = 1.

    sampled_action = torch.multinomial(probs, 1).item()
    return sampled_action, probs.cpu().numpy()

def play_rollout_pv_net_hop_mcts(
    episode_length,
    object_ids,
    env, 
    ucb_C, 
    discount, 
    max_actions, 
    pv_net,
    num_simulations, 
    num_trees,
    temperature=0.0,
    dir_noise=True, 
    dirichlet_alpha=1., 
    exploration_fraction=0.25,
    ucb_method="p-UCT-AlphaGo",
    mode="predict",
    render=False,
    debug_render=False,
):
    """
    Plays a rolllout with a policy and value MCTS with the hindsight optimization technique 
    to deal with stochastic transitions. 
    
    If mode='simulate', leaf node's evaluation is done with MC rollout evaluations, if mode='predict', 
    the value network is used instead.
    
    Samples the next action based on the Q-values of the root node's children (averaged among 'num_trees' 
    realizations of the tree search, each with independently sampled transitions) and returns both the MCTS policy 
    and the list of sampled actions as possible targets with which to train the policy network.
    
    Formula used for MCTS policy (softmax of Q-values with temperature):
    
    p(a) = exp{Q(a)/T} / \sum_b exp{Q(b)/T}

    Note: the softmax function with T=0 is the argmax function.
    
    This function is also mixing a prior sampled from a Dirichlet distribution (with parameters dirichlet_alpha for each 
    possible action) to the prior of the root node's children, in order to increase exploration at the base of the tree 
    even in cases where the policy is almost deterministic. The mixture coefficient between the prior and the categorical 
    distribution sampled by the Dirichelt distribution is the exploration_fraction, such that:
    
    p(a) = (1-exploration_fraction) Prior(a) + exploration_fraction Dir(a)
    
    """
    
    A = len(env.env.action_space)
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
    action_lst = []
    probs_lst = []
    
    for i in range(episode_length):
        
        Qs, Ns = hop_pv_mcts_step(
            frame, 
            env, 
            valid_actions, 
            ucb_C, 
            discount, 
            max_actions, 
            pv_net,
            num_simulations, 
            num_trees,
            temperature,
            dir_noise, 
            dirichlet_alpha, 
            exploration_fraction,
            ucb_method=ucb_method,
            mode=mode,
            debug_render=debug_render
        )

        action, probs = softmax_Q(Qs, temperature)
        action_lst.append(action)
        probs_lst.append(probs)
        
        if render:
            print("Action selected from HOP-MCTS: ", action, "({})".format(action_dict[action]))
        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
        total_reward += reward
        
        if done:
            frame, valid_actions = env.reset()
            if render:
                print("\nNew episode begins.")
                env.render()
            done = False
            new_root = None


    return total_reward, frame_lst, reward_lst, done_lst, action_lst, probs_lst


def play_rollout_pv_net_stochastic(
    pv_net,
    env,
    episode_length,
    ucb_C,
    discount,
    num_simulations,
    dirichlet_alpha, 
    exploration_fraction,
    temperature,
    render=False,
    debug_render=False,
):
    """
    Plays a rolllout with a policy and value MCTS. 
    Starts building the tree from the sub-tree of the root's child node that has been selected at the previous step.
    
    If mode='simulate', it's identical to a policy MCTS with MC rollout evaluations, if mode='predict', the value network 
    is used to estimate the value of the leaf nodes (instead of a MC rollout).
    
    Samples the next action based on the Q-values of the root node's children and returns both the MCTS policy and the list of 
    sampled actions as possible targets with which to train the policy network.
    
    Formula used for MCTS policy (softmax of Q-values with temperature):
    
    p(a) = exp{Q(a)/T} / \sum_b exp{Q(b)/T}

    Note: the softmax function with T=0 is the argmax function.
    
    This function is also mixing a prior sampled from a Dirichlet distribution (with parameters dirichlet_alpha for each 
    possible action) to the prior of the root node's children, in order to increase exploration at the base of the tree 
    even in cases where the policy is almost deterministic. The mixture coefficient between the prior and the categorical 
    distribution sampled by the Dirichelt distribution is the exploration_fraction, such that:
    
    p(a) = (1-exploration_fraction) Prior(a) + exploration_fraction Dir(a)
    
    Version v2: same as v1, but it's not re-using the old sub-tree in the new mcts step. 
    This has be done if we want to use the deterministic PV-MCTS as a baseline for the stochastic environment.
    As it is, this function it's not convinient to use in the deterministic setup (altough it can be useful to 
    study the search tree properties from scratch at every step).
    """
    
    A = len(env.env.action_space)
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
    action_lst = []
    probs_lst = []
    
    for i in range(episode_length):
        tree = StochasticPVMCTS(
            frame, 
            env, 
            valid_actions, 
            ucb_C, 
            discount, 
            pv_net,
            root=new_root,
            render=debug_render, 
            )
        
        root, info = tree.run(num_simulations,
                              dir_noise=True, 
                              dirichlet_alpha=dirichlet_alpha, 
                              exploration_fraction=exploration_fraction
                             )
        
        action, probs = root.softmax_Q(temperature)
        action_lst.append(action)
        probs_lst.append(probs)
        
        if render:
            print("Action selected from MCTS: ", action, "({})".format(action_dict[action]))

        frame, valid_actions, reward, done = env.step(action)
        
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
        total_reward += reward
        
        new_root = tree.get_subtree(action, frame)
        if new_root is not None and render:
            # amount of simulations that we are going to re-use in the next step:
            print("new_root.visit_count: ", new_root.visit_count) 
        if done:
            frame, valid_actions = env.reset()
            if render:
                print("Final reward: ", reward)
                print("\nNew episode begins.")
                env.render()
            done = False
            new_root = None


    return total_reward, frame_lst, reward_lst, done_lst, action_lst, probs_lst
