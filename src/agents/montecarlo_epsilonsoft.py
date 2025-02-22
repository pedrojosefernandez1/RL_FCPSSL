import numpy as np
from tqdm import tqdm
from agents.montecarlo_all import MonteCarloAllAgent
import gymnasium as gym

class MonteCarloEpsilonSoftAgent(MonteCarloAllAgent):
    def __init__(self, env: gym.Env, gamma=0.99, epsilon=0.1, decay_rate=0.99, min_epsilon=0.01):
        super().__init__(env, gamma=gamma)
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        
        
    def get_action(self, state) -> int:
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def __str__(self):
        return f'MonteCarloEpsilonSoftAgent(gamma={self.gamma}, epsilon={self.epsilon}, decay_rate={self.decay_rate}, min_epsilon={self.min_epsilon})'
