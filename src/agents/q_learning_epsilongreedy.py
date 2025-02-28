import numpy as np
from agents.q_learning import QLearningAgent
import gymnasium as gym

class QLearningEpsilonGreedyAgent(QLearningAgent):
    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.1, alpha_decay=0.995, min_alpha=0.01, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """QLearning con política ε-greedy"""
        super().__init__(env, gamma=gamma, alpha=alpha)
        self.epsilon = epsilon  # Exploración inicial
        self.epsilon_decay = epsilon_decay  # Decaimiento de epsilon
        self.min_epsilon = min_epsilon  # Valor mínimo de epsilon
        self.epsilon_episode = []  # Historial de epsilon
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.alpha_episode = []

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
        """Reduce epsilon gradualmente"""
        self.epsilon_episode.append(self.epsilon)
        self.alpha_episode.append(self.epsilon)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)  # Decaimiento de epsilon
        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)  # Decaimiento de alpha

    def stats(self):
        """Devuelve estadísticas del entrenamiento, incluyendo la evolución de ε."""
        stats = super().stats()
        stats["epsilon_episode"] = self.epsilon_episode
        stats["apha_episode"] = self.alpha_episode
        return stats

    def __str__(self):
        return f'QLearningEpsilonGreedyAgent(gamma={self.gamma}, alpha={self.alpha}, epsilon={self.epsilon}, decay_rate={self.epsilon_decay}, min_epsilon={self.min_epsilon})'