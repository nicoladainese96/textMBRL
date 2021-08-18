import numpy as np
import torch

""
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
    """
    Naive replay buffer for training a value network based on frames, rewards and done signals.
    """
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

""
class nStepsReplayBuffer:
    """
    Advanced replay buffer for training a value network based on frames, rewards and done signals.
    Uses an n-step bootstrapping and it's suggested to use a target network to compute the values for bootstrapping. 
    """
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
    
    def get_batch(self, batch_size, n_steps, target_net, device="cpu"):
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
        sampled_targets = self.compute_n_step_V_trg(n_steps, self.discount, sampled_rewards, sampled_done, 
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

""
class PolicyReplayBuffer:
    """
    Policy replay buffer that can be used to train a policy with the cross entropy loss over the full probability
    distribution.
    """
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.frame_buffer = []
        self.probs_buffer = []
    
    def store_episode(self, frame_lst, probs_lst):
        frames, probs = self.batch_episode(frame_lst, probs_lst)
        self.frame_buffer.append(frames)
        self.probs_buffer.append(probs)
        if len(self.frame_buffer) > self.mem_size:
            self.frame_buffer.pop(0)
            self.probs_buffer.pop(0)
            
    def batch_episode(self, frame_lst, probs_lst):
        episode_len = len(probs_lst)
        frames = {}
        for k in frame_lst[0].keys():
            k_value_lst = []
            for b in range(episode_len): # this is a problem, since the frame_lst has one more state!!!
                k_value_lst.append(frame_lst[b][k])
            k_value_lst = torch.cat(k_value_lst, axis=0)
            frames[k] = k_value_lst.unsqueeze(0)
            
        probs = torch.tensor(probs_lst).float() # (T, A)
        
        return frames, probs.unsqueeze(0) #(1,T,A)
    
    def get_batch(self, batch_size):
        id_range = len(self.probs_buffer)
        assert id_range >= batch_size, "Not enough samples stored to get this batch size"
        sampled_ids = np.random.choice(id_range, size=batch_size, replace=False)
        sampled_probs = torch.cat([self.probs_buffer[i] for i in sampled_ids], axis=0)
        # batch together frames 
        sampled_frames = {}
        for k in self.frame_buffer[0].keys():
            key_values = torch.cat([self.frame_buffer[i][k] for i in sampled_ids], axis=0)
            sampled_frames[k] = key_values
            
        reshaped_frames = {}
        for k in sampled_frames.keys():
            shape = sampled_frames[k].shape
            reshaped_frames[k] = sampled_frames[k].reshape(-1,*shape[2:])
        
        reshaped_probs = sampled_probs.reshape(-1, sampled_probs.shape[-1]) #(B*T, A)
        return reshaped_frames, reshaped_probs

""
class ActionReplayBuffer:
    """
    Policy replay buffer that can be used to train a policy with the cross entropy loss with target the sampled action 
    (like if it was a class to predict with probability 1).
    """
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.frame_buffer = []
        self.actions_buffer = []
        self.best_actions_buffer = []
    
    def store_episode(self, frame_lst, actions_lst, best_action_lst):
        frames, actions = self.batch_episode(frame_lst, actions_lst)
        self.frame_buffer.append(frames)
        self.actions_buffer.append(actions)
        self.best_actions_buffer.append(best_action_lst.unsqueeze(0))
        if len(self.frame_buffer) > self.mem_size:
            self.frame_buffer.pop(0)
            self.actions_buffer.pop(0)
            self.best_actions_buffer.pop(0)
            
    def batch_episode(self, frame_lst, actions_lst):
        episode_len = len(actions_lst)
        frames = {}
        for k in frame_lst[0].keys():
            k_value_lst = []
            for b in range(episode_len): # this is a problem, since the frame_lst has one more state!!!
                k_value_lst.append(frame_lst[b][k])
            k_value_lst = torch.cat(k_value_lst, axis=0)
            frames[k] = k_value_lst.unsqueeze(0)
            
        actions = torch.tensor(actions_lst).long() # (T,)
        
        return frames, actions.unsqueeze(0) #(1,T,)
    
    def get_batch(self, batch_size):
        id_range = len(self.actions_buffer)
        assert id_range >= batch_size, "Not enough samples stored to get this batch size"
        sampled_ids = np.random.choice(id_range, size=batch_size, replace=False)
        sampled_actions = torch.cat([self.actions_buffer[i] for i in sampled_ids], axis=0)
        sampled_best_actions = torch.cat([self.best_actions_buffer[i] for i in sampled_ids], axis=0)
        # batch together frames 
        sampled_frames = {}
        for k in self.frame_buffer[0].keys():
            key_values = torch.cat([self.frame_buffer[i][k] for i in sampled_ids], axis=0)
            sampled_frames[k] = key_values
            
        reshaped_frames = {}
        for k in sampled_frames.keys():
            shape = sampled_frames[k].shape
            reshaped_frames[k] = sampled_frames[k].reshape(-1,*shape[2:])
        
        reshaped_actions = sampled_actions.flatten() #(B*T, )
        reshaped_best_actions = sampled_best_actions.reshape(-1, sampled_best_actions.shape[2])
        return reshaped_frames, reshaped_actions, reshaped_best_actions

""
class nStepsActionValueReplayBuffer:
    """
    nStepsReplayBuffer + ActionReplayBuffer
    """
    def __init__(self, mem_size, discount):
        self.mem_size = mem_size
        self.discount = discount
        self.frame_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.actions_buffer = []
        self.best_actions_buffer = []
        
    def store_episode(self, frame_lst, reward_lst, done_lst, actions_lst, best_action_lst):
        frames, rewards, done, actions = self.batch_episode(frame_lst, reward_lst, done_lst, actions_lst)
        self.frame_buffer.append(frames)
        self.reward_buffer.append(rewards)
        self.done_buffer.append(done)
        self.actions_buffer.append(actions)
        self.best_actions_buffer.append(best_action_lst.unsqueeze(0))
        if len(self.frame_buffer) > self.mem_size:
            self.frame_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.done_buffer.pop(0)
            self.actions_buffer.pop(0)
            self.best_actions_buffer.pop(0)
            
    def batch_episode(self, frame_lst, reward_lst, done_lst, actions_lst):
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
        actions = torch.tensor(actions_lst).long() # (T,)
        
        return frames, rewards, done, actions.unsqueeze(0)
    
    def get_batch(self, batch_size, n_steps, discount, target_net, device="cpu"):
        # Decide which indexes to sample
        id_range = len(self.frame_buffer)
        assert id_range >= batch_size, "Not enough samples stored to get this batch size"
        sampled_ids = np.random.choice(id_range, size=batch_size, replace=False)
        
        # Sample frames, rewards and done
        sampled_rewards = np.array([self.reward_buffer[i] for i in sampled_ids])
        sampled_done = np.array([self.done_buffer[i] for i in sampled_ids])
        sampled_actions = torch.cat([self.actions_buffer[i] for i in sampled_ids], axis=0)
        sampled_best_actions = torch.cat([self.best_actions_buffer[i] for i in sampled_ids], axis=0)
        
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

        reshaped_actions = sampled_actions.flatten() #(B*T, )
        reshaped_best_actions = sampled_best_actions.reshape(-1, sampled_best_actions.shape[2])
        return reshaped_frames, sampled_targets, reshaped_actions, reshaped_best_actions
    
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
            V_pred, _ = value_net(new_states_reshaped) # value net in this case has also a policy head
            V_trg = (1-done)*Gamma_V*V_pred.squeeze() + n_step_rewards
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

""
class PolicyValueReplayBuffer:
    """
    nStepsActionValueReplayBuffer + PolicyReplayBuffer
    """
    def __init__(self, mem_size, discount):
        self.mem_size = mem_size
        self.discount = discount
        self.frame_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.actions_buffer = []
        self.best_actions_buffer = []
        self.probs_buffer = []
        
    def store_episode(self, frame_lst, reward_lst, done_lst, actions_lst, best_action_lst, probs_lst):
        frames, rewards, done, actions, probs = self.batch_episode(frame_lst, reward_lst, done_lst, actions_lst, probs_lst)
        self.frame_buffer.append(frames)
        self.reward_buffer.append(rewards)
        self.done_buffer.append(done)
        self.actions_buffer.append(actions)
        self.best_actions_buffer.append(best_action_lst.unsqueeze(0))
        self.probs_buffer.append(probs)
        if len(self.frame_buffer) > self.mem_size:
            self.frame_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.done_buffer.pop(0)
            self.actions_buffer.pop(0)
            self.best_actions_buffer.pop(0)
            self.probs_buffer.pop(0)
            
    def batch_episode(self, frame_lst, reward_lst, done_lst, actions_lst, probs_lst):
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
        actions = torch.tensor(actions_lst).long() # (T,)
        probs = torch.tensor(probs_lst).float() # (T, A)
        
        return frames, rewards, done, actions.unsqueeze(0), probs.unsqueeze(0) 
    
    def get_batch(self, batch_size, n_steps, target_net, device="cpu"):
        # Decide which indexes to sample
        id_range = len(self.frame_buffer)
        assert id_range >= batch_size, "Not enough samples stored to get this batch size"
        sampled_ids = np.random.choice(id_range, size=batch_size, replace=False)
        
        # Sample frames, rewards and done
        sampled_rewards = np.array([self.reward_buffer[i] for i in sampled_ids])
        sampled_done = np.array([self.done_buffer[i] for i in sampled_ids])
        sampled_actions = torch.cat([self.actions_buffer[i] for i in sampled_ids], axis=0)
        sampled_best_actions = torch.cat([self.best_actions_buffer[i] for i in sampled_ids], axis=0)
        sampled_probs = torch.cat([self.probs_buffer[i] for i in sampled_ids], axis=0)
        
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
        sampled_targets = self.compute_n_step_V_trg(n_steps, self.discount, sampled_rewards, sampled_done, 
                                                    trg_frames, target_net, device)
        # Flatten also the src_frames
        reshaped_frames = {}
        for k in src_frames.keys():
            shape = src_frames[k].shape
            reshaped_frames[k] = src_frames[k].reshape(-1,*shape[2:])

        reshaped_actions = sampled_actions.flatten() #(B*T, )
        reshaped_best_actions = sampled_best_actions.reshape(-1, sampled_best_actions.shape[2])
        reshaped_probs = sampled_probs.reshape(-1, sampled_probs.shape[-1]) #(B*T, A)
        return reshaped_frames, sampled_targets, reshaped_actions, reshaped_best_actions, reshaped_probs
    
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
            V_pred, _ = value_net(new_states_reshaped) # value net in this case has also a policy head
            V_trg = (1-done)*Gamma_V*V_pred.squeeze() + n_step_rewards
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



""
class HopPolicyValueReplayBuffer:
    """
    PolicyValueReplayBuffer without storing the best actions 
    (because they become unknown in the stochastic environment).
    """
    def __init__(self, mem_size, discount):
        self.mem_size = mem_size
        self.discount = discount
        self.frame_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.actions_buffer = []
        self.probs_buffer = []
        
    def store_episode(self, frame_lst, reward_lst, done_lst, actions_lst, probs_lst):
        frames, rewards, done, actions, probs = self.batch_episode(frame_lst, reward_lst, done_lst, actions_lst, probs_lst)
        self.frame_buffer.append(frames)
        self.reward_buffer.append(rewards)
        self.done_buffer.append(done)
        self.actions_buffer.append(actions)
        self.probs_buffer.append(probs)
        if len(self.frame_buffer) > self.mem_size:
            self.frame_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.done_buffer.pop(0)
            self.actions_buffer.pop(0)
            self.probs_buffer.pop(0)
            
    def batch_episode(self, frame_lst, reward_lst, done_lst, actions_lst, probs_lst):
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
        actions = torch.tensor(actions_lst).long() # (T,)
        probs = torch.tensor(probs_lst).float() # (T, A)
        
        return frames, rewards, done, actions.unsqueeze(0), probs.unsqueeze(0) 
    
    def get_batch(self, batch_size, n_steps, target_net, device="cpu"):
        # Decide which indexes to sample
        id_range = len(self.frame_buffer)
        assert id_range >= batch_size, "Not enough samples stored to get this batch size"
        sampled_ids = np.random.choice(id_range, size=batch_size, replace=False)
        
        # Sample frames, rewards and done
        sampled_rewards = np.array([self.reward_buffer[i] for i in sampled_ids])
        sampled_done = np.array([self.done_buffer[i] for i in sampled_ids])
        sampled_actions = torch.cat([self.actions_buffer[i] for i in sampled_ids], axis=0)
        sampled_probs = torch.cat([self.probs_buffer[i] for i in sampled_ids], axis=0)
        
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
        sampled_targets = self.compute_n_step_V_trg(n_steps, self.discount, sampled_rewards, sampled_done, 
                                                    trg_frames, target_net, device)
        # Flatten also the src_frames
        reshaped_frames = {}
        for k in src_frames.keys():
            shape = src_frames[k].shape
            reshaped_frames[k] = src_frames[k].reshape(-1,*shape[2:])

        reshaped_actions = sampled_actions.flatten() #(B*T, )
        reshaped_probs = sampled_probs.reshape(-1, sampled_probs.shape[-1]) #(B*T, A)
        return reshaped_frames, sampled_targets, reshaped_actions, reshaped_probs
    
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
            V_pred, _ = value_net(new_states_reshaped) # value net in this case has also a policy head
            V_trg = (1-done)*Gamma_V*V_pred.squeeze() + n_step_rewards
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

