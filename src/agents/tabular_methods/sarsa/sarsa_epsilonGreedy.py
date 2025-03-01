##### agents/tabular_methods/sarsa/sarsa_epsilongreedy.py #####
"""
Módulo: SarsaEpsilonGreedyAgent
========================
Este módulo implementa la clase `SarsaEpsilonGreedyAgent`, que extiende el algoritmo
SARSA con una estrategia de exploración ε-greedy.
"""

import numpy as np
from agents.tabular_methods.sarsa.sarsa import SarsaAgent
from agents.policies.epsilon_greedy import EpsilonGreedy
import gymnasium as gym

class SarsaEpsilonGreedyAgent(EpsilonGreedy, SarsaAgent):
    """
    Agente basado en SARSA con política ε-greedy.
    Incorpora exploración reduciendo `epsilon` y `alpha` gradualmente a lo largo del entrenamiento.
    """

    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.1, epsilon=1.0, 
                 alpha_decay=0.995, min_alpha=0.01, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Inicializa el agente SARSA con exploración ε-greedy.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento.
            alpha (float): Tasa de aprendizaje inicial.
            epsilon (float): Probabilidad inicial de exploración.
            alpha_decay (float): Factor de decaimiento de alpha.
            min_alpha (float): Valor mínimo de alpha.
            epsilon_decay (float): Factor de decaimiento de epsilon.
            min_epsilon (float): Valor mínimo de epsilon.
        """
        SarsaAgent.__init__(self, env, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, min_alpha=min_alpha)
        EpsilonGreedy.__init__(self, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)

    def decay(self):
        """
        Reduce `alpha` y `epsilon` llamando a los métodos correspondientes.
        """
        SarsaAgent.decay(self)  # Reduce alpha (manejado por TDLearningAgent)
        EpsilonGreedy.decay(self)  # Reduce epsilon

    def stats(self):
        stats = SarsaAgent.stats(self)
        stats = stats | EpsilonGreedy.stats(self)
        return stats

    def __str__(self):
        """
        Devuelve una representación en cadena del agente con sus parámetros actuales.
        """
        return (f'SarsaEpsilonGreedyAgent(gamma={self.gamma}, alpha={self.alpha}, alpha_decay={self.alpha_decay}, '
                f'min_alpha={self.min_alpha}, epsilon={self.epsilon}, epsilon_decay={self.epsilon_decay}, '
                f'min_epsilon={self.min_epsilon})')