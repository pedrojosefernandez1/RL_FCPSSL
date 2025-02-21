import numpy as np
from tqdm import tqdm
from agents.agent import MonteCarloAllAgent
import gymnasium as gym

class MonteCarloEpsilonSoftAgent(MonteCarloAllAgent):
    def __init__(self, env: gym.Env, gamma=0.99, epsilon=0.1):
        super().__init__(env, gamma=gamma)
        self.epsilon = epsilon
        
    def get_action(self, state) -> int:
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

