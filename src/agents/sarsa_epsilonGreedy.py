import numpy as np
from tqdm import tqdm
from agents.sarsa import SarsaAgent
import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo

class SarsaEpsilonGreedyAgent(SarsaAgent):
    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.1, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """SARSA con política ε-greedy"""
        super().__init__(env, gamma=gamma, alpha=alpha)
        self.epsilon = epsilon  # Exploración inicial
        self.epsilon_decay = epsilon_decay  # Decaimiento de epsilon
        self.min_epsilon = min_epsilon  # Valor mínimo de epsilon
        self.epsilon_episode = []  # Historial de epsilon

    def get_action(self, state) -> int:
        """Selecciona una acción usando política ε-greedy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.nA)  # Exploración (acción aleatoria)
        else:
            return np.argmax(self.Q[state])  # Explotación (mejor acción conocida)

    def decay(self):
        """Reduce epsilon gradualmente"""
        self.epsilon_episode.append(self.epsilon)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def stats(self):
        """Devuelve estadísticas del entrenamiento, incluyendo la evolución de ε."""
        stats = super().stats()
        stats["epsilon_episode"] = self.epsilon_episode
        return stats

    def __str__(self):
        return f'SarsaEpsilonGreedyAgent(gamma={self.gamma}, alpha={self.alpha}, epsilon={self.epsilon}, decay_rate={self.epsilon_decay}, min_epsilon={self.min_epsilon})'