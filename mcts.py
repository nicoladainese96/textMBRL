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
from my_networks import *

verbose = False
vprint = print if verbose else lambda *args, **kwargs: None

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print("Using device "+device)

### Standard MCTS ###

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

############################################################################################################################
            
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

############################################################################################################################

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

############################################################################################################################

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

############################################################################################################################

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

############################################################################################################################

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

############################################################################################################################

action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }


# funtion to implement soft-Q policy from root node with T parameter; T=0 => argmax Q
class PriorValueNode(Node):
    def __init__(self, prior=0.):
        super().__init__()
        self.frame = None
        self.prior = prior
        self.full_action_space = None # comprehends also moves that are not allowed in the node
        
    def expand(self, frame, valid_actions, priors, reward, done, simulator):
        self.expanded = True
        vprint("Valid actions as child: ", valid_actions)
        vprint("Prior over the children: ", priors)
        vprint("Terminal node: ", done)
        self.full_action_space = len(priors) # trick to pass this information
        self.frame = frame
        self.reward = reward
        self.terminal = done
        self.valid_actions = valid_actions
        if not done:
            for action in valid_actions:
                self.children[action] = PriorValueNode(priors[action])
        self.simulator_dict = simulator.save_state_dict()
        
    def get_children_probs(self, T):
        """
        Use formula p(a) = N(a)^{1/T} / \sum_b N(b)^{1/T} to get the probabilities 
        of selecting the various actions.
        """
        
        Ns = np.zeros(self.full_action_space) # check this
        for action, child in self.children.items():
            Ns[action] = child.visit_count
        
        #print("Ns: ", Ns)
        scores = Ns**(1./T)
        #print("scores: ", scores)
        probs = scores/scores.sum()
        #print("probs: ", probs)
        return probs
    
    def softmax_Q(self, T, discount):
        Qs = -torch.ones(self.full_action_space)*np.inf
        for action, child in self.children.items():
            Qs[action] = child.reward + discount*child.value()
        if T > 0:
            probs = F.softmax(Qs/T, dim=0)
        elif T==0:
            probs = torch.zeros(self.full_action_space) 
            a = torch.argmax(Qs)
            probs[a] = 1.
            
        sampled_action = torch.multinomial(probs, 1).item()
        return sampled_action, probs.cpu().numpy()
    
    def sample_child(self, temperature):
        probs = self.get_children_probs(temperature)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        return action, probs
    
    def add_exploration_noise(self, dirichlet_alpha=0.5, exploration_fraction=0.25):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

############################################################################################################################

class PolicyValueMCTS(MCTS):
    def __init__(self, 
                 root_frame,
                 simulator,
                 valid_actions,
                 ucb_c,
                 discount,
                 max_actions,
                 pv_net,
                 root=None,
                 render=False):
        
        super().__init__(simulator,
                     valid_actions,
                     ucb_c,
                     discount,
                     max_actions,
                     root,
                     render)
        self.pv_net = pv_net 
        self.root_frame = root_frame
        
    def run(self, num_simulations, mode="simulate", dir_noise=False, dirichlet_alpha=1.0, exploration_fraction=0.25):
        """
        Runs num_simulations searches starting from the root node corresponding to the internal
        state of the simulator given during initialization.
        Returns the root node and an extra_info dictionary
        """
        if self.root is None or self.root.visit_count==0:
            self.root = PriorValueNode() 
            
            with torch.no_grad():
                _, root_prior = self.pv_net(self.root_frame)
                root_prior = root_prior.reshape(-1).cpu().numpy()
                
                
            self.root.expand(
                self.root_frame,
                self.valid_actions,
                root_prior,
                0, # reward to get to root
                False, # terminal node
                self.simulator # state of the simulator at the root node 
            )
                
            # not sure about this
            self.root.visit_count += 1
            
        if dir_noise:
            # add noise to root even if the tree was inherited from previous time-step 
            self.root.add_exploration_noise(dirichlet_alpha, exploration_fraction)
                
        max_tree_depth = 0
        root = self.root
        #print("root.simulator_dict :", root.simulator_dict) # not ok
        #print("root: ", root) # ok
        #print("self.simulator: ", self.simulator)
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
                vprint("Action selected: ", action, action_dict[action])
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
    
    def ucb_score(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        exploration_term = self.ucb_c*child.prior*np.sqrt(np.log(parent.visit_count)/(child.visit_count+1))

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
        frame, valid_actions, reward, done = simulator.step(action) # this also updates the simulator's internal state
        with torch.no_grad():
            value, prior = self.pv_net(frame)
            prior = prior.reshape(-1).cpu().numpy()
        vprint("valid_actions: ", valid_actions)
        vprint("prior: ", prior)
        vprint("reward: ", reward)
        vprint("done: ", done)
        node.expand(frame, valid_actions, prior, reward, done, simulator)
        return node
    
    def predict(self, node):
        if not node.terminal:
            with torch.no_grad():
                value, _ = self.pv_net(node.frame)
                value = value.item()
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
                    value, _ = self.pv_net(node.frame)
                    bootstrap_value = value.item()
                cum_discounted_reward += (self.discount**steps)*bootstrap_value
        else:
            cum_discounted_reward = 0
        return cum_discounted_reward
    
############################################################################################################################

class PVNode(PriorValueNode):
    def __init__(self, prior=0., weight=0.):
        super().__init__(prior)
        # this weight is sqrt(prior)/sum_{all_children}sqrt(prior), needs to be computed beforehand
        self.weight = weight 
    
    def expand(self, frame, valid_actions, priors, reward, done, simulator):
        self.expanded = True
        vprint("Valid actions as child: ", valid_actions)
        vprint("Prior over the children: ", priors)
        weights = np.sqrt(priors)/np.sqrt(priors).sum()
        vprint("Weights over the children: ", weights)
        vprint("Terminal node: ", done)
        self.full_action_space = len(priors) # trick to pass this information
        self.frame = frame
        self.reward = reward
        self.terminal = done
        self.valid_actions = valid_actions
        if not done:
            for action in valid_actions:
                self.children[action] = PVNode(priors[action], weights[action])
        self.simulator_dict = simulator.save_state_dict()
        
    def add_exploration_noise(self, dirichlet_alpha=0.5, exploration_fraction=0.25):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        priors = []
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
            priors.append(self.children[a].prior) 
        # recompute all weights as the square root of the prior and then normalize them to 1
        priors = np.array(priors)
        weights = np.sqrt(priors)/np.sqrt(priors).sum()
        for i,a in enumerate(actions):
            self.children[a].weight = weights[i]

############################################################################################################################

class PV_MCTS(PolicyValueMCTS):
    def __init__(self, 
             root_frame,
             simulator,
             valid_actions,
             ucb_c,
             discount,
             max_actions,
             pv_net,
             root=None,
             render=False,
             ucb_method="p-UCT-old"
                ):

        super().__init__(
            root_frame,
            simulator,
            valid_actions,
            ucb_c,
            discount,
            max_actions,
            pv_net,
            root,
            render
        )
        
        possible_ucb_methods = ["p-UCT-old", "p-UCT-AlphaGo", "p-UCT-Rosin"]
        assert ucb_method in possible_ucb_methods, \
            ("ucb method not recognized, should be one of: ", possible_ucb_methods)
        self.ucb_method = ucb_method
        
    def run(self, num_simulations, mode="simulate", dir_noise=False, dirichlet_alpha=1.0, exploration_fraction=0.25):
        """
        Runs num_simulations searches starting from the root node corresponding to the internal
        state of the simulator given during initialization.
        Returns the root node and an extra_info dictionary
        """
        if self.root is None or self.root.visit_count==0:
            self.root = PVNode() 
            
            with torch.no_grad():
                _, root_prior = self.pv_net(self.root_frame)
                root_prior = root_prior.reshape(-1).cpu().numpy()
                
                
            self.root.expand(
                self.root_frame,
                self.valid_actions,
                root_prior,
                0, # reward to get to root
                False, # terminal node
                self.simulator # state of the simulator at the root node 
            )
                
            # not sure about this
            self.root.visit_count += 1
            
        if dir_noise:
            # add noise to root even if the tree was inherited from previous time-step 
            self.root.add_exploration_noise(dirichlet_alpha, exploration_fraction)
                
        max_tree_depth = 0
        root = self.root
        #print("root.simulator_dict :", root.simulator_dict) # not ok
        #print("root: ", root) # ok
        #print("self.simulator: ", self.simulator)
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
                vprint("Action selected: ", action, action_dict[action])
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
    
    def ucb_score(self, parent, child):
        if self.ucb_method == "p-UCT-old":
            return self.ucb_score_old(parent, child)
        elif self.ucb_method == "p-UCT-AlphaGo":
            return self.ucb_score_AlphaGo(parent, child)
        else:
            return self.ucb_score_Rosin(parent, child)
        
    def ucb_score_old(self, parent, child, offset=1):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        exploration_term = self.ucb_c*child.prior*np.sqrt(np.log(parent.visit_count + offset)/(child.visit_count+1))

        if child.visit_count > 0:
            # Mean value Q
            value_term = child.reward + self.discount*child.value() 
        else:
            value_term = 0

        return value_term + exploration_term, value_term, exploration_term
    
    def ucb_score_AlphaGo(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        exploration_term = self.ucb_c*child.prior/(child.visit_count+1)*np.sqrt(parent.visit_count)

        if child.visit_count > 0:
            # Mean value Q
            value_term = child.reward + self.discount*child.value() 
        else:
            value_term = 0

        return value_term + exploration_term, value_term, exploration_term
    
    def ucb_score_Rosin(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        # c_term increases p-UCT with time (i.e. parent's visit counts) and decreases with child's visit count
        if child.visit_count > 0:
            c_term = np.sqrt(3*np.log(parent.visit_count)/(2*child.visit_count))
        else:
            c_term = 0
        
        # m_term is a penalty term; increases for small probabilities and decreases with parent's visit counts
        if parent.visit_count > 1:
            m_term = 2/child.weight*np.sqrt(np.log(parent.visit_count)/parent.visit_count)
        else:
            m_term = 2/child.weight
            
        exploration_term = c_term - m_term
        
        if child.visit_count > 0:
            # Mean value Q
            value_term = child.reward + self.discount*child.value() 
        else:
            value_term = 1 # max payoff

        return value_term + exploration_term, value_term, exploration_term