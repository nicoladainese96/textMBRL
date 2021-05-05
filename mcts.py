import utils
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from rtfm import featurizer as X

verbose = False
vprint = print if verbose else lambda *args, **kwargs: None

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print("Using device "+device)

# ## Standard MCTS ###

class MCTS():
    def __init__(self, 
                 simulator,
                 valid_actions,
                 ucb_c,
                 discount,
                 max_actions,
                 root=None,
                 render=False):
        """
        Monte Carlo Tree Search assuming deterministic dynamics.
        
        simulator: 
            wrapper of the environment that returns a scalar reward, a list of valid actions 
            and a 'done' boolean flag when presented with an action
        valid_moves:
            list of valid moves for the root node
        ucb_c:
            Constantused in the UCB1 formula for trees
            UCB(s,a) = Q(s,a) + ucb_c*sqrt(log N(s,a)/(\sum_b N(s,b)))
        discount:
            discoung factor gamma of the MDP
        max_actions:
            number of actions to be taken at most from the root node to the end of a rollout
        root: 
            might be the child of an old root node; use it to keep all the cached computations 
            from previous searches with a different root node. 
        """
        self.simulator = simulator
        self.original_dict = simulator.save_state_dict()
        self.valid_actions = valid_actions
        self.action_space = len(valid_actions)
        self.ucb_c = ucb_c
        self.discount = discount
        self.max_actions = max_actions
        self.root = root
        self.render = render
    
    def get_subtree(self, action):
        """
        Returns the subtree whose root node is the current root's child corresponding to
        the given action.
        """
        return self.root.children[action]
    
    def run(self, num_simulations):
        """
        Runs num_simulations searches starting from the root node corresponding to the internal
        state of the simulator given during initialization.
        Returns the root node and an extra_info dictionary
        """
        if self.root is None:
            self.root = Node()
            self.root.expand(
                self.valid_actions,
                0, # reward to get to root
                False, # terminal node
                self.simulator # state of the simulator at the root node 
            )
            # not sure about this
            self.root.visit_count += 1
        
        max_tree_depth = 0
        root = self.root
        for n in range(num_simulations):
            ### Start of a simulation/search ###
            vprint("\nSimulation %d started."%(n+1))
            node = root
            # make sure that the simulator internal state is reset to the original one
            self.simulator.load_state_dict(root.simulator_dict)
            search_path = [node]
            current_tree_depth = 0
            if self.render:
                node.render(self.simulator)
            ### Selection phase until leaf node is reached ###
            while node.expanded or (current_tree_depth<self.max_actions):
                current_tree_depth += 1
                action, node = self.select(node)
                if self.render and node.expanded:
                    node.render(self.simulator)
                vprint("Current tree depth: ", current_tree_depth)
                vprint("Action selected: ", action)
                vprint("Child node terminal: ", node.terminal)
                vprint("Child node expanded: ", node.expanded)
                if node.expanded or node.terminal:
                    search_path.append(node)
                    if node.terminal:
                        break
                else:
                    break
                
            ### Expansion of leaf node (if not terminal)###
            vprint("Expansion phase started")
            if not node.terminal:
                parent = search_path[-1] # last expanded node on the search path
                node = self.expand(node, parent, action)
                if self.render:
                    node.render(self.simulator)
                search_path.append(node)
            
            ### Simulation phase for self.max_actions - current_tree_depth steps ###
            vprint("Simulation  phase started")
            value = self.simulate(node, current_tree_depth)
            vprint("Simulated value: ", value)
            
            ### Backpropagation of the leaf node value along the seach_path ###
            vprint("Backpropagation phase started")
            self.backprop(search_path, value)
        
            max_tree_depth = max(max_tree_depth, current_tree_depth)
            vprint("Simulation %d done."%(n+1))
        extra_info = {
            "max_tree_depth": max_tree_depth
        }
        # just a check to see if root works as a shallow copy of self.root
        assert root.visit_count == self.root.visit_count, "self.root not updated during search"
        
        # make sure that the simulator internal state is reset to the original one
        self.simulator.load_state_dict(root.simulator_dict)
        return root, extra_info
        
    def select(self, node):
        """
        Use UCT formula on the input node; return the action selected and the corresponding child node 
        """
        actions = []
        ucb_values = []
        value_terms = []
        exploration_terms = []
        for action, child in node.children.items():
            actions.append(action)
            U, V, E = self.ucb_score(node, child)
            ucb_values.append(U)
            value_terms.append(V)
            exploration_terms.append(E)
        actions = np.array(actions)
        vprint("actions: ", actions)
        
        value_terms = np.array(value_terms)
        vprint("value_terms: ", value_terms)
        
        exploration_terms = np.array(exploration_terms)
        vprint("exploration_terms: ", exploration_terms)
        
        ucb_values = np.array(ucb_values)
        vprint("ucb_values: ", ucb_values)
        
        max_U = ucb_values.max()
        vprint("max_U: ", max_U)
        
        mask = (ucb_values==max_U)
        vprint("mask: ", mask)
        
        best_actions = actions[mask]
        vprint("best_actions: ", best_actions)
        
        action = np.random.choice(best_actions)
        return action, node.children[action]

    def ucb_score(self, parent, child, eps=1e-3):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        exploration_term = self.ucb_c*np.sqrt(np.log(parent.visit_count)/(child.visit_count+eps))

        if child.visit_count > 0:
            # Mean value Q
            value_term = child.reward + self.discount*child.value() 
        else:
            value_term = 0

        return value_term + exploration_term, value_term, exploration_term
    
    def expand(self, node, parent, action):
        """
        Expand the node obtained by taking the given action from the parent node 
        """
        simulator = parent.get_simulator(self.simulator) # get a deepcopy of the simulator with the parent's state stored
        valid_actions, reward, done = simulator.step(action) # this also updates the simulator's internal state
        vprint("reward: ", reward)
        vprint("done: ", done)
        node.expand(valid_actions, reward, done, simulator)
        return node
    
    def simulate(self, node, current_depth):
        """
        Simulate a rollout with a random policy starting from the input node
        until the end of the episode or self.max_actions are reached 
        (also considering the current depth of the input node from the root)
        """
        if not node.terminal:
            simulator = node.get_simulator(self.simulator)
            valid_actions = node.valid_actions
            steps = self.max_actions - current_depth
            cum_discounted_reward = 0
            for i in range(steps):
                action = np.random.choice(valid_actions)
                valid_actions, reward, done = simulator.step(action)
                cum_discounted_reward += (self.discount**i)*reward
                if done:
                    break
        else:
            cum_discounted_reward = 0
        return cum_discounted_reward
            
    def backprop(self, search_path, value):
        """
        Update the value sum and visit count of all nodes along the search path.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.discount*value

class Node:
    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.reward = 0
        self.simulator = None
        self.expanded = False
        self.terminal = False
        self.simulator_dict = None

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, valid_actions, reward, done, simulator):
        self.expanded = True
        vprint("Valid actions as child: ", valid_actions)
        vprint("Terminal node: ", done)
        self.reward = reward
        self.terminal = done
        self.valid_actions = valid_actions
        if not done:
            for action in valid_actions:
                self.children[action] = Node()
        self.simulator_dict = simulator.save_state_dict()
        
    def get_simulator(self, simulator):
        if self.simulator_dict is not None:
            # load a deepcoy of the simulator_dict, so that the internal variable remains unchanged
            simulator.load_state_dict(copy.deepcopy(self.simulator_dict)) 
            return simulator
        else:
            print("Trying to load simulator_dict, but it was never instantiated.")
            raise NotImplementedError()
    
    def best_action(self, discount):
        """
        Look among the children and take the one with higher Q-value. 
        Exclude children with 0 visits.
        """
        actions = []
        Qvalues = []
        for action, child in self.children.items():
            actions.append(action)
            Qvalues.append(child.reward + discount*child.value())
        actions = np.array(actions)
        Qvalues = np.array(Qvalues)
        max_Q = Qvalues.max()
        mask = (Qvalues==max_Q)
        best_actions = actions[mask]
        return np.random.choice(best_actions)
    
    def highest_visit_count(self):
        best_action = None
        highest_count = 0
        for action, child in self.children.items():
            if child.visit_count > highest_count:
                best_action = action
        return best_action
    
    def render(self, simulator):
        if self.simulator_dict is not None:
            simulator.load_state_dict(self.simulator_dict)
            simulator.render()
        else:
            raise Exception("Node simulator not initialized yet.")


class TrueSimulator():
    """
    Returns only valid actions, reward and done signal from env.step() - no state is returned
    """
    def __init__(self, env, featurizer=None):
        self.env = env
        self.action_space = len(gym_env.action_space)
        self.featurizer = featurizer
        
    def reset(self):
        frame = self.env.reset()
        valid_moves = frame['valid'].numpy().astype(bool) # boolean mask of shape (action_space)
        actions = np.arange(self.action_space)
        valid_actions = actions[valid_moves]
        return valid_actions
    
    def step(self, action, *args, **kwargs):
        frame, reward, done, _ = self.env.step(int(action), *args, **kwargs)
        valid_moves = frame['valid'].numpy().astype(bool) # boolean mask of shape (action_space)
        actions = np.arange(self.action_space)
        valid_actions = actions[valid_moves]
        return valid_actions, reward, done
    
    def render(self):
        self.featurizer.featurize(self.env)
        
    def save_state_dict(self):
        return self.env.save_state_dict()
        
    def load_state_dict(self, d):
        self.env.load_state_dict(d)

### Value-based MCTS ###
class FullTrueSimulator():
    """
    Returns the full state from the environment step.
    """
    def __init__(self, env, featurizer=None):
        self.env = env
        self.action_space = len(env.action_space)
        self.featurizer = featurizer
        
    def _process_frame(self, frame):
        """
        Extracts from frame a valid_action numpy array containing integers from 0 to self.action_space-1
        Adds batch dimension to all values stored inside frame dictionary
        """
        # do this before batch dim is added
        valid_moves = frame['valid'].numpy().astype(bool) # boolean mask of shape (action_space)
        actions = np.arange(self.action_space)
        valid_actions = actions[valid_moves]
        
        for k in frame.keys():
            frame[k] = frame[k].unsqueeze(0)
        
        return frame, valid_actions
    
    def reset(self):
        frame = self.env.reset()
        frame, valid_actions = self._process_frame(frame)
        return frame, valid_actions
    
    def step(self, action, *args, **kwargs):
        frame, reward, done, _ = self.env.step(int(action), *args, **kwargs)
        frame, valid_actions = self._process_frame(frame)
        return frame, valid_actions, reward, done
    
    def render(self):
        self.featurizer.featurize(self.env)
        
    def save_state_dict(self):
        return self.env.save_state_dict()
        
    def load_state_dict(self, d):
        self.env.load_state_dict(d)

class FixedDynamicsValueNet(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                ):
        super().__init__()
        self.embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v = self.value_mlp(z_flat)
        return v

class FixedDynamicsValueNet_v1(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v = self.value_mlp(z_flat)
        return v


class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ResidualConv, self).__init__()
        # YOUR CODE HERE
        
        # RESIDUAL PART
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SKIP CONNECTION PART
        if stride==1 and in_channels==out_channels:
            self.skip_net = None
        else:
            self.skip_net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        
    def forward(self, x):
        # YOUR CODE HERE
        
        h1 = F.relu(self.bn1(self.conv1(x)))
        res = F.relu(self.bn2(self.conv2(h1)))
        
        if self.skip_net is None:
            skip_x = x
        else:
            skip_x = self.skip_net(x)
        
        out = F.relu(res+skip_x)
        return out


class FixedDynamicsValueNet_v2(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            ResidualConv(64,64),
            ResidualConv(64,64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v = self.value_mlp(z_flat)
        return v


class ValueMLP(nn.Module):
    def __init__(self, gym_env, emb_dim=10):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_features = (name_shape[0]*name_shape[1]*name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        print("n_features: ", n_features)
        
        self.value_mlp = nn.Sequential(
            nn.Linear(n_features,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(B,-1)
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1)
        z = torch.cat([x,inv], axis=1)
        v = self.value_mlp(z)
        return v


class ValueNode(Node):
    def __init__(self):
        super().__init__()
        self.frame = None

    def expand(self, frame, valid_actions, reward, done, simulator):
        self.expanded = True
        vprint("Valid actions as child: ", valid_actions)
        vprint("Terminal node: ", done)
        self.frame = frame
        self.reward = reward
        self.terminal = done
        self.valid_actions = valid_actions
        if not done:
            for action in valid_actions:
                self.children[action] = ValueNode()
        self.simulator_dict = simulator.save_state_dict()

class ValueMCTS(MCTS):
    def __init__(self, 
                 root_frame,
                 simulator,
                 valid_actions,
                 ucb_c,
                 discount,
                 max_actions,
                 value_net,
                 root=None,
                 render=False):
        
        super().__init__(simulator,
                     valid_actions,
                     ucb_c,
                     discount,
                     max_actions,
                     root,
                     render)
        self.value_net = value_net
        self.root_frame = root_frame
        
    def run(self, num_simulations, mode="simulate"):
        """
        Runs num_simulations searches starting from the root node corresponding to the internal
        state of the simulator given during initialization.
        Returns the root node and an extra_info dictionary
        """
        if self.root is None:
            self.root = ValueNode()
            self.root.expand(
                self.root_frame,
                self.valid_actions,
                0, # reward to get to root
                False, # terminal node
                self.simulator # state of the simulator at the root node 
            )
            # not sure about this
            self.root.visit_count += 1
        
        max_tree_depth = 0
        root = self.root
        for n in range(num_simulations):
            ### Start of a simulation/search ###
            vprint("\nSimulation %d started."%(n+1))
            node = root
            # make sure that the simulator internal state is reset to the original one
            self.simulator.load_state_dict(root.simulator_dict)
            search_path = [node]
            current_tree_depth = 0
            if self.render:
                node.render(self.simulator)
            ### Selection phase until leaf node is reached ###
            while node.expanded or (current_tree_depth<self.max_actions):
                current_tree_depth += 1
                action, node = self.select(node)
                if self.render and node.expanded:
                    node.render(self.simulator)
                vprint("Current tree depth: ", current_tree_depth)
                vprint("Action selected: ", action)
                vprint("Child node terminal: ", node.terminal)
                vprint("Child node expanded: ", node.expanded)
                if node.expanded or node.terminal:
                    search_path.append(node)
                    if node.terminal:
                        break
                else:
                    break
                
            ### Expansion of leaf node (if not terminal)###
            vprint("Expansion phase started")
            if not node.terminal:
                parent = search_path[-1] # last expanded node on the search path
                node = self.expand(node, parent, action)
                if self.render:
                    node.render(self.simulator)
                search_path.append(node)
            
            ### Simulation phase for self.max_actions - current_tree_depth steps ###
            vprint("Value prediction/simulation phase started")
            if mode=="simulate":
                value = self.simulate(node, current_tree_depth)
            elif mode=="predict":
                value = self.predict(node)
            elif mode=="simulate_and_predict":
                value = self.simulate_and_predict(node, current_tree_depth)
            elif mode =="hybrid":
                value1 = self.simulate(node, current_tree_depth)
                value2 =self.predict(node)
                value = 0.5*value1 + 0.5*value2
            else:
                raise Exception("Mode "+mode+" not implemented.")
            vprint("Predicted/simulated value: ", value)
            
            ### Backpropagation of the leaf node value along the seach_path ###
            vprint("Backpropagation phase started")
            self.backprop(search_path, value)
        
            max_tree_depth = max(max_tree_depth, current_tree_depth)
            vprint("Simulation %d done."%(n+1))
        extra_info = {
            "max_tree_depth": max_tree_depth
        }
        # just a check to see if root works as a shallow copy of self.root
        assert root.visit_count == self.root.visit_count, "self.root not updated during search"
        
        # make sure that the simulator internal state is reset to the original one
        self.simulator.load_state_dict(root.simulator_dict)
        return root, extra_info
    
    def expand(self, node, parent, action):
        """
        Expand the node obtained by taking the given action from the parent node 
        """
        simulator = parent.get_simulator(self.simulator) # get a deepcopy of the simulator with the parent's state stored
        frame, valid_actions, reward, done = simulator.step(action) # this also updates the simulator's internal state
        vprint("reward: ", reward)
        vprint("done: ", done)
        node.expand(frame, valid_actions, reward, done, simulator)
        return node
    
    def predict(self, node):
        if not node.terminal:
            with torch.no_grad():
                value = self.value_net(node.frame).item()
        else:
            value = 0
        return value
    
    def simulate(self, node, current_depth):
        """
        Simulate a rollout with a random policy starting from the input node
        until the end of the episode or self.max_actions are reached 
        (also considering the current depth of the input node from the root)
        """
        if not node.terminal:
            simulator = node.get_simulator(self.simulator)
            valid_actions = node.valid_actions
            steps = self.max_actions - current_depth
            cum_discounted_reward = 0
            for i in range(steps):
                action = np.random.choice(valid_actions)
                frame, valid_actions, reward, done = simulator.step(action)
                cum_discounted_reward += (self.discount**i)*reward
                if done:
                    break
        else:
            cum_discounted_reward = 0
        return cum_discounted_reward
    
    def simulate_and_predict(self, node, current_depth, n_steps=5):
        """
        Simulate a rollout with a random policy starting from the input node
        until the end of the episode or self.max_actions are reached 
        (also considering the current depth of the input node from the root)
        or at most n_steps before calling the value_net to approximate the rest of the trajectory.
        """
        if not node.terminal:
            simulator = node.get_simulator(self.simulator)
            valid_actions = node.valid_actions
            steps = min(self.max_actions, n_steps) # FIX THIS
            cum_discounted_reward = 0
            for i in range(steps):
                action = np.random.choice(valid_actions)
                frame, valid_actions, reward, done = simulator.step(action)
                cum_discounted_reward += (self.discount**i)*reward
                if done:
                    break
            if not done:
                with torch.no_grad():
                    bootstrap_value = self.value_net(frame).item()
                cum_discounted_reward += (self.discount**steps)*bootstrap_value
        else:
            cum_discounted_reward = 0
        return cum_discounted_reward


# Discrete support
class DiscreteSupportValueNet(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                 support_size=2,
                ):
        super().__init__()
        self.embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.support_size = support_size
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, support_size*2+1)
        )
        
    def forward(self, frame):
        
        v_logits = self.logits(frame)
        v = support_to_scalar(v_logits, self.support_size)
        return v
    
    def logits(self, frame):
        device = next(self.parameters()).device
        x = self.embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v_logits = self.value_mlp(z_flat)
        return v_logits


# +
def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
