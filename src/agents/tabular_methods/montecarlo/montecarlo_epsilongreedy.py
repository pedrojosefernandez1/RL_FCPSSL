##### agents/tabular_methods/montecarlo/montecarlo_epsilongreedy.py #####
"""
Módulo: MonteCarloEpsilonGreedyAgent
========================
Este módulo implementa la clase `MonteCarloEpsilonGreedyAgent`, una variante del método
Monte Carlo que incorpora una estrategia ε-greedy para equilibrar la exploración y explotación.
"""

import numpy as np
from agents.tabular_methods.montecarlo.montecarlo_all import MonteCarloAllAgent
from agents.policies.epsilon_greedy_mixin import EpsilonGreedyMixin
import gymnasium as gym

class MonteCarloEpsilonGreedyAgent(EpsilonGreedyMixin, MonteCarloAllAgent):
    """
    Agente basado en Monte Carlo con estrategia ε-greedy.
    Se diferencia del MonteCarloAllAgent en que incorpora exploración probabilística.
    """

    def __init__(self, env: gym.Env,seed = 32, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Inicializa el agente Monte Carlo con política ε-greedy.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            seed (int, opcional): Semilla para la reproducibilidad. Por defecto 32.
            gamma (float): Factor de descuento.
            epsilon (float): Probabilidad inicial de exploración.
            epsilon_decay (float): Factor de decaimiento de epsilon.
            min_epsilon (float): Valor mínimo de epsilon.
        """
        MonteCarloAllAgent.__init__(self, env, seed=seed, gamma=gamma)
        EpsilonGreedyMixin.__init__(self, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)

    def decay(self):
        """
        Reduce el valor de `epsilon` gradualmente.
        """
        EpsilonGreedyMixin.decay(self)
