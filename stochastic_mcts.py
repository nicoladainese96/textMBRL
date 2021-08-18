import numpy as np
import copy
import torch
import torch.nn.functional as F

verbose = False
vprint = print if verbose else lambda *args, **kwargs: None

action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }

class StochasticStateNode():
    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.reward = 0
        self.simulator = None
        self.expanded = False
        self.terminal = False
        self.simulator_dict = None
        self.frame = None
        self.full_action_space = None # comprehends also moves that are not allowed in the node
        self.transition_id = None # hash 

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count 
   
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
                self.children[action] = StochasticStateActionNode(priors[action])
                self.children[action].action = action
        self.simulator_dict = simulator.save_state_dict()
    
    def softmax_Q(self, T):
        Qs = self.get_Q_values()
        if T > 0:
            probs = F.softmax(Qs/T, dim=0)
        elif T==0:
            probs = torch.zeros(self.full_action_space) 
            a = torch.argmax(Qs)
            probs[a] = 1.
            
        sampled_action = torch.multinomial(probs, 1).item()
        return sampled_action, probs.cpu().numpy()
    
    def get_Q_values(self):
        Qs = -torch.ones(self.full_action_space)*np.inf
        for action, child in self.children.items():
            Qs[action] = child.Q_value()
        return Qs
    
    def get_children_visit_counts(self):
        Ns = np.zeros(self.full_action_space) # check this
        for action, child in self.children.items():
            Ns[action] = child.visit_count
        return Ns
    
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
    def render(self, simulator):
        if self.simulator_dict is not None:
            simulator.load_state_dict(self.simulator_dict)
            simulator.render()
        else:
            raise Exception("Node simulator not initialized yet.")
            
    def get_simulator(self, simulator):
        if self.simulator_dict is not None:
            # load a deepcoy of the simulator_dict, so that the internal variable remains unchanged
            simulator.load_state_dict(copy.deepcopy(self.simulator_dict)) 
            return simulator
        else:
            print("Trying to load simulator_dict, but it was never instantiated.")
            raise NotImplementedError()
            
class StochasticStateActionNode():
    def __init__(self, prior):
        self.visit_count = 0
        self.Q_value_sum = 0
        self.children = {}
        self.action = None # has it ever been used?
        self.prior = prior
        
    def Q_value(self):
        if self.visit_count == 0:
            return 0
        return self.Q_value_sum / self.visit_count 

class StochasticPVMCTS():
    def __init__(self,
                 root_frame,
                 simulator,
                 valid_actions,
                 ucb_c,
                 discount,
                 pv_net,
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
        self.pv_net = pv_net 
        self.root_frame = root_frame
        self.root = root # probably it will be None all the time? maybe actually it's possible to re-use it
        self.render = render
        
    def get_subtree(self, action, new_frame):
        """
        Returns the subtree whose root node is the current root's child corresponding to
        the given action and transition id (since it's stochastic). 
        If nothing is found, then return None.
        """
        state_action_node = self.root.children[action]
        env_transition_id = hash(str(new_frame['name'].cpu().numpy()))
        transition_ids = np.array(list(state_action_node.children.keys()))
        # if the transition actually occurred in the environment has already been encountered
        # in the tree search, select the subtree which has that state node as root
        if np.any(env_transition_id==transition_ids):
            new_root = state_action_node.children[env_transition_id]
        else:
            new_root = None
            
        return new_root
    
    def run(self, num_simulations, dir_noise=True, dirichlet_alpha=1.0, exploration_fraction=0.25, default_Q=0.5):
        """
        Runs num_simulations searches starting from the root node corresponding to the internal
        state of the simulator given during initialization.
        Returns the root node and an extra_info dictionary.
        
        dir_noise: bool
            If True, modifies the root prior with a noisy categorical prior sampled from a Dirichlet distribution
        dirichlet_alpha: float
            Alpha parameter for the Dirichlet distribution, taken equal for all actions
        exploration_fraction: float in (0,1)
            Mixture coefficient between the root prior and the prior noisy distribution sampled from the
            Dirichlet distribution
        default_Q: float in (-1,1)
            What is default value for unexplored actions in the p-UCT formula. Higher is more optimistic and 
            increases exploration, but decreases the influence of the prior in the first few decisions.
            Suggested values: 0, 0.5 or 1.
        """
        self.default_Q = default_Q
        
        if self.root is None or self.root.visit_count==0:
            self.root = StochasticStateNode() 
        
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
                
            self.root.visit_count += 1 
                
        if dir_noise:
            # add noise to root even if the tree was inherited from previous time-step
            self.root.add_exploration_noise(dirichlet_alpha, exploration_fraction)
                
        max_tree_depth = 0 # keep track of the maximum depth of the tree (hard to compute at the end) 
        root = self.root
           
        for n in range(num_simulations):
            ### Start of a simulation/search ###
            vprint("\nSimulation %d started."%(n+1))
            state_node = root
            # make sure that the simulator internal state is reset to the original one
            self.simulator.load_state_dict(root.simulator_dict)
            search_path = [state_node]
            current_tree_depth = 0
            if self.render:
                state_node.render(self.simulator)
            ### Selection phase until leaf node is reached ###
            new_transition = False # leaf node iff state_node is terminal or reached through new transition
            while not new_transition:
                current_tree_depth += 1
                action, state_action_node, state_node, new_transition = self.select_and_expand(state_node)
                if self.render:
                    state_node.render(self.simulator)
                vprint("Current tree depth: ", current_tree_depth)
                vprint("Action selected: ", action, action_dict[action])
                vprint("New transition: ", new_transition)
                vprint("Child node terminal: ", state_node.terminal)
                # always append both nodes
                search_path.append(state_action_node) 
                search_path.append(state_node) 
                
                if state_node.terminal or new_transition:
                    # Exit from the search when a terminal node or a new node is encountered
                    break

            ### Value prediction of the leaf node ###
            vprint("Value prediction phase started")
            value = self.predict_value(state_node)
            
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
                
    def select_and_expand(self, state_node):
        """
        Select which action to take in the input node through the p-UCT formula.
        Sample a transition for that (state, action) pair (i.e. get a StochasticStateNode
        that is going to be a child of the StochasticStateActionNode) in the simulator.
        If the new state is already present in the list of children corresponding to the transitions 
        sampled in the past, then just select that node, otherwise initialize a StochasticStateNode,
        add it to the list of children and expand it.
        Return both the StochasticStateActionNode and the StochasticStateNode.
        """
        ### Usual part to select which action to take ###
        actions = []
        ucb_values = []
        value_terms = []
        exploration_terms = []
        for action, child in state_node.children.items():
            actions.append(action)
            U, V, E = self.ucb_score(state_node, child)
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
        vprint("Action selected: ", action, action_dict[action])
        
        state_action_node = state_node.children[action]
        
        ### New part for stochastic MCTS ###
        simulator = state_node.get_simulator(self.simulator) # get a deepcopy of the simulator with the parent's state stored
        frame, valid_actions, reward, done = simulator.step(action) # this also updates the simulator's internal state
            
        # hash the frame 
        transition_id = hash(str(frame['name'].cpu().numpy())) # devise a test to actually make sure of this
        
        # check if the transition has already been sampled in the past
        transition_ids = np.array(list(state_action_node.children.keys()))
        
        if np.any(transition_id==transition_ids):
            # if that is the case, just select the new_state_node and return 
            new_transition = False
            new_state_node = state_action_node.children[transition_id]
        
        else:
            new_transition = True
            
            # init the new_state_node
            new_state_node = StochasticStateNode()
            
            # add transition to the children of the state_action_node
            state_action_node.children[transition_id] = new_state_node
            
            # expand the new_state_node
            
            with torch.no_grad():
                value, prior = self.pv_net(frame)
                prior = prior.reshape(-1).cpu().numpy()
            vprint("valid_actions: ", valid_actions)
            vprint("prior: ", prior)
            vprint("reward: ", reward)
            vprint("done: ", done)
            new_state_node.expand(frame, valid_actions, prior, reward, done, simulator)

        return action, state_action_node, new_state_node, new_transition
    
    def ucb_score(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        exploration_term = self.ucb_c*child.prior/(child.visit_count+1)*np.sqrt(parent.visit_count)

        if child.visit_count > 0:
            # Mean value Q
            value_term = child.Q_value()
        else:
            value_term = self.default_Q # just trying

        return value_term + exploration_term, value_term, exploration_term
    
    def predict_value(self, state_node):
        if not state_node.terminal:
            with torch.no_grad():
                value, _ = self.pv_net(state_node.frame)
                value = value.item()
        else:
            value = 0
        return value
    
    def backprop(self, search_path, value):
        """
        Update the value sum and visit count of all state nodes along the search path. 
        Does the same but updating the Q values for the state action nodes.
        search_path starts with a StochasticStateNode and also ends with one, in the middle the two classes are alternated.
        """
        # How the chain of values and Q-values works
        # Q_{T-2} = r_{T-2} + \gamma V_{T-1} -> V_{T-1} = Q_{T-1} -> Q_{T-1} = r_{T-1} + \gamma V_T -> v_T
        for node in reversed(search_path):
            if isinstance(node, StochasticStateNode):
                node.value_sum += value
                node.visit_count += 1
                value = node.reward + self.discount*value # actually a Q-value
            else:
                node.Q_value_sum += value
                node.visit_count +=1
                value = value # just to show that the value flows unchanged through the state-action
            
            
            