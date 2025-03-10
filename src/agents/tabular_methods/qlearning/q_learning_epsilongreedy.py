"""
Módulo: QLearningEpsilonGreedyAgent
========================
Este módulo implementa la clase `QLearningEpsilonGreedyAgent`, que extiende
el algoritmo Q-Learning con una estrategia de exploración ε-greedy.
"""

import numpy as np
from agents.tabular_methods.qlearning.q_learning import QLearningAgent
from agents.policies.epsilon_greedy import EpsilonGreedy
import gymnasium as gym

class QLearningEpsilonGreedyAgent(EpsilonGreedy, QLearningAgent):
    """
    Agente basado en Q-Learning con política ε-greedy.
    Incorpora exploración reduciendo `epsilon` y `alpha` gradualmente a lo largo del entrenamiento.
    """

    def __init__(self, env: gym.Env,seed =32, gamma=0.99, alpha=0.1, epsilon=1.0, 
                 alpha_decay=0.995, min_alpha=0.01, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Inicializa el agente Q-Learning con exploración ε-greedy.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            seed (int, opcional): Semilla para la reproducibilidad. Por defecto 32.
            gamma (float): Factor de descuento.
            alpha (float): Tasa de aprendizaje inicial.
            epsilon (float): Probabilidad inicial de exploración.
            alpha_decay (float): Factor de decaimiento de alpha.
            min_alpha (float): Valor mínimo de alpha.
            epsilon_decay (float): Factor de decaimiento de epsilon.
            min_epsilon (float): Valor mínimo de epsilon.
        """
        QLearningAgent.__init__(self, env, seed=seed, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, min_alpha=min_alpha)
        EpsilonGreedy.__init__(self, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)

    def decay(self):
        """
        Reduce `alpha` y `epsilon` llamando a los métodos correspondientes.
        """
        QLearningAgent.decay(self)  # Reduce alpha (manejado por TDLearningAgent)
        EpsilonGreedy.decay(self)  # Reduce epsilon

    def stats(self):
        stats = QLearningAgent.stats(self)
        stats = stats | EpsilonGreedy.stats(self)
        return stats

    def __str__(self):
        """
        Devuelve una representación en cadena del agente con sus parámetros actuales.
        """
        return (f'QLearningEpsilonGreedyAgent(gamma={self.gamma}, alpha={self.alpha}, alpha_decay={self.alpha_decay}, '
                f'min_alpha={self.min_alpha}, epsilon={self.epsilon}, epsilon_decay={self.epsilon_decay}, '
                f'min_epsilon={self.min_epsilon})')

    