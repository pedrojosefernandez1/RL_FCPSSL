import numpy as np
import agents.approximation_methods.tiles3 as tc
import gymnasium as gym
from agents.approximation_methods.base import ApproximationAgent
from agents.policies.epsilon_greedy import EpsilonGreedy

class TileCodingFeatureExtractor:
    def __init__(self, num_tilings=8, bins=10, low=None, high=None, iht_size=4096):
        assert np.log2(iht_size).is_integer(), "iht_size must be a power of 2"  
        self.iht = tc.IHT(iht_size)  # Tabla hash para el tile coding
        self.iht_size = iht_size
        self.num_tilings = num_tilings
        self.low = np.where(np.isfinite(low), np.array(low), -1e12)
        self.high = np.where(np.isfinite(high), np.array(high), 1e12)
        self.scale = np.where((self.high - self.low) != 0, self.num_tilings / (self.high - self.low), 1.0)
    
    def __call__(self, state):
        scaled_state = state * self.scale
        indices = tc.tiles(self.iht, self.num_tilings, scaled_state)
        feature_vector = np.zeros(self.iht_size)
        feature_vector[indices] = 1  # Vector disperso con valores 1 en los índices activados
        return feature_vector
