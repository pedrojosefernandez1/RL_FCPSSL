"""
Módulo: QLearningEpsilonGreedyAgent
========================
Este módulo implementa la clase `QLearningEpsilonGreedyAgent`, que extiende
el algoritmo Q-Learning con una estrategia de exploración ε-greedy.
"""

import numpy as np
from agents.tabular_methods.qlearning.q_learning import QLearningAgent
import gymnasium as gym

class QLearningEpsilonGreedyAgent(QLearningAgent):
    """
    Agente basado en Q-Learning con política ε-greedy.
    Incorpora exploración reduciendo ε gradualmente a lo largo del entrenamiento.
    """

    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.1, alpha_decay=0.995, min_alpha=0.01, 
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Inicializa el agente Q-Learning con exploración ε-greedy.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento.
            alpha (float): Tasa de aprendizaje inicial.
            alpha_decay (float): Factor de decaimiento de alpha.
            min_alpha (float): Valor mínimo de alpha.
            epsilon (float): Probabilidad inicial de exploración.
            epsilon_decay (float): Factor de decaimiento de epsilon.
            min_epsilon (float): Valor mínimo de epsilon.
        """
        super().__init__(env, gamma=gamma, alpha=alpha)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon_episode = []
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.alpha_episode = []

    def get_action(self, state, info) -> int:
        """
        Selecciona una acción utilizando una política ε-greedy.
        
        Args:
            state: Estado actual del entorno.
            info: Información adicional del entorno.
        
        Returns:
            int: Acción seleccionada.
        """
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
        """
        Reduce gradualmente los valores de ε y α para mejorar la estabilidad del aprendizaje.
        """
        self.epsilon_episode.append(self.epsilon)
        self.alpha_episode.append(self.alpha)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)

    def stats(self):
        """
        Devuelve estadísticas del entrenamiento, incluyendo la evolución de ε y α.
        
        Returns:
            dict: Contiene la tabla Q, recompensas, evolución de epsilon y alpha.
        """
        stats = super().stats()
        stats["epsilon_episode"] = self.epsilon_episode
        stats["alpha_episode"] = self.alpha_episode
        return stats

    def __str__(self):
        """
        Devuelve una representación en cadena del agente con sus parámetros actuales.
        """
        return (f'QLearningEpsilonGreedyAgent(gamma={self.gamma}, alpha={self.alpha}, alpha_decay={self.alpha_decay}, '
                f'min_alpha={self.min_alpha}, epsilon={self.epsilon}, epsilon_decay={self.epsilon_decay}, '
                f'min_epsilon={self.min_epsilon})')
