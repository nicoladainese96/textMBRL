from rtfm import featurizer as X
from rtfm import tasks # needed to make rtfm visible as Gym env
from core import environment # env wrapper

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import copy

# simulate flags without argparse
class Flags():
    def __init__(self, 
                 env="rtfm:groups_simple_stationary-v0", # simplest env
                 #env="rtfm:groups-v0",
                 height=6,
                 width=6,
                 partial_observability=False,
                 max_placement=2,
                 shuffle_wiki=False,
                 time_penalty=0
                 ):
        self.env = env
        self.height = height
        self.width = width
        self.partial_observability = partial_observability
        self.max_placement = max_placement
        self.shuffle_wiki = shuffle_wiki 
        self.time_penalty = time_penalty

def create_env(flags, featurizer=None):
    f = featurizer or X.Concat([X.Text(), X.ValidMoves()])
    env = gym.make(flags.env, 
                   room_shape=(flags.height, flags.width), 
                   partially_observable=flags.partial_observability, 
                   max_placement=flags.max_placement, 
                   featurizer=f, 
                   shuffle_wiki=flags.shuffle_wiki, 
                   time_penalty=flags.time_penalty)
    return env


def get_object_ids_dict(game_simulator):
    
    def encode(sent, vocab, max_len=10, eos="pad", pad="pad"):
        if isinstance(sent, list):
            words = sent[:max_len-1] + [eos]
            length = len(words)
            if len(words) < max_len:
                words += [pad] * (max_len - len(words))
            return vocab.word2index([w.strip() for w in words]), length
        else:
            raise Exception("encode function on get_object_ids_dict not working. Wrong input.")
            
    # these ids are always fixed, just the items keep changing ids
    object_ids = dict(
        target_monster = 179,
        distractor_monster = 180,
        agent = 183
        )
    
    list_of_items = list(game_simulator.env.world.items)
    
    # not sure this is ever going to happen
    while len(list_of_items) != 2:
        game_simulator.reset()

    vocab = game_simulator.env.vocab
    for l in list_of_items:
        ids, lengths = encode(l.tokenized_name(), vocab)
        if l.char == "y":
            object_ids["yes_item"] = ids[0]
        elif l.char == "n":
            object_ids["no_item"] = ids[0]
            
    return object_ids


def render_frame(frame, object_ids):
    wall_ids = 3
    name_frame = frame["name"][0]
    W = H = name_frame.shape[0]
    empty_row = [" " for _ in range(W)]
    representation = np.array([empty_row for _ in range(H)])
    
    try:
        y, x = torch.nonzero(name_frame[:,:,0,0] == object_ids["agent"])[0]
        representation[y, x] = "@"
    except:
        pass
    
    try:
        y, x = torch.nonzero(name_frame[:,:,0,0] == object_ids["target_monster"])[0]
        representation[y, x] = "!"
    except:
        pass
    
    try:
        y, x = torch.nonzero(name_frame[:,:,0,0] == object_ids["distractor_monster"])[0]
        representation[y, x] = "?"
    except:
        pass
    
    try:
        y, x = torch.nonzero(name_frame[:,:,0,0] == object_ids["yes_item"])[0]
        representation[y, x] = "y"
    except:
        pass
    
    try:
        y, x = torch.nonzero(name_frame[:,:,0,0] == object_ids["no_item"])[0]
        representation[y, x] = "n"
    except:
        pass

    try:
        positions = torch.nonzero(name_frame[:,:,0,0] == wall_ids)
        for y,x in positions:
            representation[y, x] = u"\u2588"
    except:
        pass
    
    char_repr = ""
    for r in representation:
        for r_i in r:
            char_repr += r_i
        char_repr +="\n"
    print(char_repr)


def yes_in_inv(frame, object_ids):
    """
    Check if yes i
    """
    inv = frame["inv"][0,0]
    if inv == object_ids["yes_item"]:
        return True
    else:
        return False


def grid_distance(pos1, pos2):
    """
    pos1, pos2: shape (1,2) and (N,2) or 2 broadcastable shapes with axis 1 of dimension 2 (x,y)
    """
    return torch.abs(pos1-pos2).sum(axis=1)


def delta_pos_in_action(pos1, pos2):
    """
    Returns the action needed to go from pos1 to pos2, assuming that their grid distance is 1.
    
    Inputs
    ------
    pos1, pos2: tensors of shape (1,2)
    
    Returns
    -------
    a: int encoding the action needed to go from pos1 to pos2
    """
    action_dict = {
        "Stay":0,
        "Up":1,
        "Down":2,
        "Left":3,
        "Right":4
    }
    
    assert grid_distance(pos1, pos2)[0] == 1, "pos1 and pos2 are not adjacent"
    
    delta_row = pos2[0,0] - pos1[0,0]
    delta_col = pos2[0,1] - pos1[0,1]
    if delta_row == 1:
        return action_dict["Down"]
    elif delta_row == -1:
        return action_dict["Up"]
    else:
        if delta_col == 1:
            return action_dict["Right"]
        else:
            return action_dict["Left"]


def path_to_goal_exists(sorted_d, sorted_pos, d):
    """
    Returns the positions adjacent to the agent and says whether there is an optimal path
    passing through them (one at the time) that connects the agent to the goal in at most d steps.
    
    Inputs
    ------
    sorted_d: torch long tensor of shape (n,)
        Contains the distances of the sorted_pos from the initial position (usually the agent position)
    sorted_pos: torch tensor of shape (n,2) 
        Contains the (row,col) positions of all the n cells whose distance from the agent + distance from the
        goal is less or equal to d (third argument of the function)
    d: int
    
    Returns
    -------
    initial_pos: torch tensor of shape (m,2)
        Positions among the sorted_pos which are adjacent to the current agent position
    optimal_moves: torch tensor of shape (m,) and type bool
        Says whether each initial position corrsponds or not to an optimal move towards the goal from the 
        agent position
    """
    
    def path_to_goal_from_position(next_possible_pos, sorted_d, sorted_pos, d):
        """
        Checks if from next_possible_pos there is a path that goes to one of the sorted_pos
        with distance equal to d-1, which means a path that goes connects next_possible_pos
        to a cell adjacent to the goal.
        """
        for l in range(1,d-1):
            possible_cells_adjacent_to_next = sorted_pos[sorted_d==l+1]
            cells_adjacent_to_next = []
            
            for pos in next_possible_pos:
                dist = grid_distance(pos, possible_cells_adjacent_to_next)
                
                if (dist==1).any():
                    next_cell = possible_cells_adjacent_to_next[dist==1]
                    cells_adjacent_to_next+=[c for c in next_cell]
                    
            if len(cells_adjacent_to_next)>0:
                next_possible_pos = cells_adjacent_to_next

            else:
                return False
        # able to construct a path until cells that are d-1 steps away from the agent and 1 step away from the goal
        return True
    
    initial_pos = sorted_pos[sorted_d==1]
    optimal_moves = []
    
    for next_possible_pos in initial_pos:
        optimal_move = path_to_goal_from_position([next_possible_pos], sorted_d, sorted_pos, d)
        optimal_moves.append(optimal_move)
    
    optimal_moves = torch.tensor(optimal_moves)
    return initial_pos, optimal_moves


def get_optimal_actions(frame, object_ids, empty_id=170):
    """
    Given a frame dictionary and the dictionary with the ids of the various objects in the map,
    returns a list with all the possible optimal actions to win the game. If winning the game is impossible,
    returns the integer 0, which encodes the action Stay.
    """
    frame_name = frame["name"][0,:,:,0,0]
    
    # the optimal action is either to go and collect the yes item (if we don't have it in the inventory)
    # or to go to the target monster and kill it, if we have the yes item
    if yes_in_inv(frame, object_ids):
        next_goal_pos = torch.nonzero(frame_name==object_ids["target_monster"])
    else:
        next_goal_pos = torch.nonzero(frame_name==object_ids["yes_item"])
    
    # starting point
    agent_pos = torch.nonzero(frame_name==object_ids["agent"])
    
    # sometimes weird initial random conditions are found in which both items are placed on the agent 
    # but probably only the no_item remains... 
    if len(agent_pos)==0 or len(next_goal_pos)==0:
        #render_frame(frame, object_ids)
        return [0] 
    
    empty_mask = (frame_name==empty_id)
    free_positions = list(torch.nonzero((empty_mask).float()))
    
    # we can also pass through the no item on the way to the yes item; this is not possible instead if we
    # already have it, because the inventory has space for one item only
    if not yes_in_inv(frame, object_ids):
        no_item_mask = (frame_name==object_ids["no_item"])
        no_pos = list(torch.nonzero(no_item_mask.float()))
        free_positions = free_positions + no_pos
    
    free_positions = torch.cat([x.unsqueeze(0) for x in free_positions], axis=0)

    # if agent is adjacent to goal, just return a list containing the action that brings it there
    if grid_distance(agent_pos, next_goal_pos)==1:
        return [delta_pos_in_action(agent_pos, next_goal_pos)]

    # to get the optimal actions we need to identify the optimal paths
    # a 'straight' path in a grid, i.e. one that does not need to circumvent obstacles, has the property
    # that each of its cells has a total distance from the agent and the goal which is the same
    distances_from_agent = grid_distance(free_positions, agent_pos)
    distances_from_goal = grid_distance(free_positions, next_goal_pos)
    total_distance = distances_from_agent+distances_from_goal
    
    d_min = total_distance.min()
    d_max = total_distance.max()
    extra_distance = 0 # number of extra moves to reach the goal with shortest path due to obstacles
    best_actions = [0] # if no path is found, backup action is Stay with id=0
    while extra_distance <= d_max-d_min:
        # Look at cells that are distant at most d steps from the agent + goal (i.e. d = d_A + d_G)
        d = d_min + extra_distance 
        eligible_mask = (total_distance<=d)
        eligible_pos = free_positions[eligible_mask]
        
        # Focus on the distance of these cells from the agent, to create an ordering of them 
        d_from_agent = distances_from_agent[eligible_mask]
        # check that there is every value between 1 and d-1; if that's not the case, try to increase d
        unique = d_from_agent.unique() 
        if not torch.all((unique[..., None] == torch.arange(1,d)).any(-1)):
            extra_distance += 1
            continue
        else:
            # sort eligible pos according to their distance from the agent
            sorted_d, sorted_idx = d_from_agent.sort()
            sorted_pos = eligible_pos[sorted_idx]
            # Returns the positions adjacent to the agent and says whether there is an optimal path
            # passing through them (one at the time) that connects the agent to the goal in at most d steps.
            initial_pos, optimal_moves = path_to_goal_exists(sorted_d, sorted_pos, d)
            
            # no free cells near the agent within d (there might be some free ones adjacent but with 
            # greater distance from the goal)
            if len(initial_pos)==0:
                extra_distance += 1
                continue
            # there is at least one adjacent cell from which we can arrive at the goal within d steps
            elif optimal_moves.any():
                optimal_next_cells = initial_pos[optimal_moves]
                best_actions = []
                for pos in optimal_next_cells:
                    action = delta_pos_in_action(agent_pos, pos.view(1,2))
                    best_actions.append(action)
                break
            # free cells are available within distance d, but there is no path within d that connects
            # the agent to the goal
            else:
                extra_distance += 1
                continue
            
    return best_actions


def play_episode_optimal_policy(
    env,
    episode_length,
    object_ids,
    render = False,
    reset_if_done=False,
    return_frames=False
):

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
    frame_lst = [frame]
    
    for i in range(episode_length):
        
        best_actions = get_optimal_actions(frame, object_ids)
        if render:
            print("Best actions: ", [action_dict[a] for a in best_actions])
        action = np.random.choice(best_actions)
        frame, valid_actions, reward, done = env.step(action)
        frame_lst.append(frame)
        
        if render:
            env.render()
        total_reward += reward
        
        if done:
            if reset_if_done:
                frame, valid_actions = env.reset()
            else:
                break
    
    if return_frames:
        return total_reward, frame_lst
    else:
        return total_reward 


def play_episode_optimal_policy_v1(
    env,
    episode_length,
    object_ids,
    render = False,
    reset_if_done=True
):

    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    A = len(env.env.action_space)
    frame, valid_actions = env.reset()
    if render:
        env.render()
    total_reward = 0
    done = False
    frame_lst = [frame]
    action_lst = []
    best_action_lst = []
    for i in range(episode_length):
        
        best_actions = get_optimal_actions(frame, object_ids)
        if render:
            print("Best actions: ", [action_dict[a] for a in best_actions])
        action = np.random.choice(best_actions)
        action_lst.append(action)
        best_actions_tensor = torch.tensor([1 if i in best_actions else 0 for i in range(A)]).view(1,-1)
        best_action_lst.append(best_actions_tensor)
        
        frame, valid_actions, reward, done = env.step(action)
        frame_lst.append(frame)
        
        if render:
            env.render()
        total_reward += reward
        
        if done:
            if reset_if_done:
                frame, valid_actions = env.reset()
            else:
                break
    

    return total_reward, frame_lst, action_lst, torch.cat(best_action_lst, axis=0)


def play_episode_optimal_policy_v2(
    env,
    episode_length,
    object_ids,
    render = False,
    reset_if_done=True
):

    action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }
    A = len(env.env.action_space)
    frame, valid_actions = env.reset()
    if render:
        env.render()
    total_reward = 0
    done = False
    frame_lst = [frame]
    reward_lst = []
    done_lst = []
    action_lst = []
    best_action_lst = []
    for i in range(episode_length):
        
        best_actions = get_optimal_actions(frame, object_ids)
        if render:
            print("Best actions: ", [action_dict[a] for a in best_actions])
        action = np.random.choice(best_actions)
        action_lst.append(action)
        best_actions_tensor = torch.tensor([1 if i in best_actions else 0 for i in range(A)]).view(1,-1)
        best_action_lst.append(best_actions_tensor)
        
        frame, valid_actions, reward, done = env.step(action)
        frame_lst.append(frame)
        reward_lst.append(reward)
        done_lst.append(done)
        
        if render:
            env.render()
        total_reward += reward
        
        if done:
            if reset_if_done:
                frame, valid_actions = env.reset()
            else:
                break
    

    return total_reward, frame_lst, reward_lst, done_lst, action_lst, torch.cat(best_action_lst, axis=0)


# +
def plot_losses(losses, window=20, ylabel=None):
    plt.figure(figsize=(10,8))
    plot_window_average(losses, window)
    plt.xlabel("Number of optimizer steps", fontsize=16)
    if ylabel is None:
        plt.ylabel("Policy net cross entropy loss", fontsize=16)
    else:
        plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.show()
    
def plot_entropies(entropies, window=20):
    plt.figure(figsize=(10,8))
    plot_window_average(entropies, window)
    plt.xlabel("Number of optimizer steps", fontsize=16)
    plt.ylabel("Policy net entropy", fontsize=16)
    plt.legend()
    plt.show()
    
def plot_rewards(total_rewards, window=20):
    plt.figure(figsize=(10,8))
    plot_window_average(total_rewards, window)
    plt.xlabel("Number of optimizer steps", fontsize=16)
    plt.ylabel("Average total reward in 32 steps", fontsize=16)
    plt.legend()
    plt.show()
    
def plot_action_optimality(perc_optimal_actions, window=20):
    plt.figure(figsize=(10,8))
    plot_window_average(perc_optimal_actions, window)
    plt.xlabel("Number of optimizer steps", fontsize=16)
    plt.ylabel("Fraction of optimal actions per trajectory", fontsize=16)
    plt.legend()
    plt.show()
    
def plot_window_average(y, window, label=None):
    if window is None:
        average_y = y
    else:
        average_y = np.array([np.mean(y[i-window:i]) for i in range(window, len(y))])
        
    if label is None:
        plt.plot(np.arange(len(average_y)), average_y, label="Moving average over %d steps"%window)
    else:
        plt.plot(np.arange(len(average_y)), average_y, label=label)


# -

def get_problematic_traj(value_net, test_rb, test_size):
    # Check number of problematic trajectories in the test set according to current pv_net
    problematic_trajectories = []
    for i in range(test_size):
        # get single trajectory
        frame_lst = test_rb.frame_buffer[i]
        reward_lst = test_rb.reward_buffer[i]

        # remove batch axis from frames
        reshaped_frame_lst = {}
        for k in frame_lst.keys():
            shape = frame_lst[k].shape
            reshaped_frame_lst[k] = frame_lst[k].reshape(-1,*shape[2:])

        total_reward = np.sum(reward_lst)
        if total_reward == 1:
            value_net.eval()
            with torch.no_grad():
                #values = [value_net(f).cpu().squeeze() for f in reshaped_frame_lst]
                values = list(value_net(reshaped_frame_lst).cpu().squeeze())

            # check whether values are monotonically increasing or not (disregarding the terminal state, 
            # because in that case we should sum a reward of 1)    
            correct = True
            if len(values) > 2:
                for j in range(len(values)-2): # look at pairs (j,j+1), stop at one pair from the end
                    if values[j] > values[j+1]:
                        correct = False
            if not correct:
                problematic_trajectories.append((reshaped_frame_lst,values))
    
    frac_of_problematic_traj = len(problematic_trajectories)/test_size
    print("Percentage of problematic trajectories: %.1f %%"%(frac_of_problematic_traj*100))
    return frac_of_problematic_traj


def get_problematic_traj_pv_net(pv_net, test_rb, test_size):
    # Check number of problematic trajectories in the test set according to current pv_net
    problematic_trajectories = []
    for i in range(test_size):
        # get single trajectory
        frame_lst = test_rb.frame_buffer[i]
        reward_lst = test_rb.reward_buffer[i]

        # remove batch axis from frames
        reshaped_frame_lst = {}
        for k in frame_lst.keys():
            shape = frame_lst[k].shape
            reshaped_frame_lst[k] = frame_lst[k].reshape(-1,*shape[2:])

        total_reward = np.sum(reward_lst)
        if total_reward == 1:
            pv_net.eval()
            with torch.no_grad():
                raw_values, _ = pv_net(reshaped_frame_lst)
                values = list(raw_values.cpu().squeeze())

            # check whether values are monotonically increasing or not (disregarding the terminal state, 
            # because in that case we should sum a reward of 1)    
            correct = True
            if len(values) > 2:
                for j in range(len(values)-2): # look at pairs (j,j+1), stop at one pair from the end
                    if values[j] > values[j+1]:
                        correct = False
            if not correct:
                problematic_trajectories.append((reshaped_frame_lst,values))
    
    frac_of_problematic_traj = len(problematic_trajectories)/test_size
    print("Percentage of problematic trajectories: %.1f %%"%(frac_of_problematic_traj*100))
    return frac_of_problematic_traj
