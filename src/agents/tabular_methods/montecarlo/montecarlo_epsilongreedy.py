"""
Módulo: MonteCarloEpsilonGreedyAgent
========================
Este módulo implementa el agente `MonteCarloEpsilonGreedyAgent`, una variante del método
Monte Carlo que incorpora una estrategia ε-greedy para equilibrar la exploración y explotación.
"""

import numpy as np
from tqdm import tqdm
from agents.tabular_methods.montecarlo.montecarlo_all import MonteCarloAllAgent
import gymnasium as gym

class MonteCarloEpsilonGreedyAgent(MonteCarloAllAgent):
    """
    Agente basado en Monte Carlo con estrategia ε-greedy.
    Se diferencia del MonteCarloAllAgent en que incorpora exploración probabilística.
    """

    def __init__(self, env: gym.Env, gamma=0.99, epsilon=0.4, decay_rate=0.99, min_epsilon=0.01):
        """
        Inicializa el agente Monte Carlo con política ε-greedy.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento para la recompensa futura.
            epsilon (float): Probabilidad inicial de exploración.
            decay_rate (float): Factor de decaimiento de epsilon.
            min_epsilon (float): Valor mínimo de epsilon.
        """
        super().__init__(env, gamma=gamma)
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.epsilon_episode = []

    def get_action(self, state, info) -> int:
        """
        Selecciona una acción usando una política ε-greedy.
        
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
        Reduce el valor de ε de manera exponencial hasta alcanzar un valor mínimo.
        """
        self.epsilon_episode.append(self.epsilon)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def stats(self):
        """
        Devuelve estadísticas del entrenamiento, incluyendo la evolución de ε.
        
        Returns:
            dict: Contiene la tabla Q, recompensas y evolución de epsilon.
        """
        stats = super().stats()
        stats["epsilon_episode"] = self.epsilon_episode
        return stats
    
    def __str__(self):
        """
        Devuelve una representación en cadena del agente con sus parámetros.
        """
        return f'MonteCarloEpsilonGreedyAgent(gamma={self.gamma}, epsilon={self.epsilon}, decay_rate={self.decay_rate}, min_epsilon={self.min_epsilon})'
