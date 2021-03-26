from rtfm import featurizer as X
from rtfm import tasks # needed to make rtfm visible as Gym env
from core import environment # env wrapper

import numpy as np
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
