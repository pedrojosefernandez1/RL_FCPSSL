import numpy as np
from tqdm import tqdm
from agents.montecarlo_all import MonteCarloAllAgent
import gymnasium as gym

class MonteCarloEpsilonGreedyAgent(MonteCarloAllAgent):
    def __init__(self, env: gym.Env, gamma=0.99, epsilon=0.4, decay_rate=0.99, min_epsilon=0.01):
        super().__init__(env, gamma=gamma)
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.epsilon_episode = []

    def get_action(self, state, info) -> int:
        if 'action_mask' in info:
            pi_A = info['action_mask'] * self.epsilon / np.sum(info['action_mask'])
            valid_actions = np.where(info['action_mask'])[0] 
            best_action = valid_actions[np.argmax(self.Q[state, valid_actions])]
            
        else:
            pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
            best_action = np.argmax(self.Q[state])
            
        pi_A[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    
    def decay(self):
        self.epsilon_episode.append(self.epsilon)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def stats(self):
        """Devuelve estadísticas del entrenamiento"""
        stats = super().stats()
        stats["epsilon_episode"] = self.epsilon_episode
        return stats
    
    def __str__(self):
        return f'MonteCarloEpsilonGreedyAgent(gamma={self.gamma}, epsilon={self.epsilon}, decay_rate={self.decay_rate}, min_epsilon={self.min_epsilon})'