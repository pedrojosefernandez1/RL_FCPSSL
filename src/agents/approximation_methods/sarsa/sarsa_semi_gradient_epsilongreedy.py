
"""
Módulo: SarsaSemiGradientEpsilonGreedyAgent
========================
Este módulo implementa el algoritmo SARSA Semi-Gradiente con estrategia ε-greedy,
utilizando funciones de aproximación en lugar de tablas de valores discretas.
"""

import numpy as np
from agents.approximation_methods.sarsa.sarsa_semi_gradient import SarsaSemiGradientAgent
from agents.policies.epsilon_greedy import EpsilonGreedy
import gymnasium as gym

class SarsaSemiGradientEpsilonGreedyAgent(EpsilonGreedy, SarsaSemiGradientAgent):
    """
    Agente basado en SARSA Semi-Gradiente con política ε-greedy.
    """

    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.01, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, feature_extractor=None, seed=42):
        """
        Inicializa el agente SARSA Semi-Gradiente con política ε-greedy.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento.
            alpha (float): Tasa de aprendizaje.
            epsilon (float): Probabilidad inicial de exploración.
            epsilon_decay (float): Factor de decaimiento de epsilon.
            min_epsilon (float): Valor mínimo de epsilon.
            feature_extractor (callable, opcional): Función de extracción de características.
            seed (int): Semilla para la reproducibilidad.
        """
        env.reset(seed=seed)
        np.random.seed(seed)
        SarsaSemiGradientAgent.__init__(self, env, gamma=gamma, alpha=alpha, feature_extractor=feature_extractor)
        EpsilonGreedy.__init__(self, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self.seed = seed

    def get_action(self, state, info):
        """
        Selecciona una acción usando la política ε-greedy con función de aproximación.
        """
        return self.get_action(state, info, Q_function=lambda s: np.dot(self.weights, s), action_space=self.env.action_space.n)
    
    def decay(self):
        """
        Reduce `alpha` y `epsilon` llamando a las funciones de decaimiento correspondientes.
        """
        super().decay()  # Reduce alpha
        EpsilonGreedy.decay(self)  # Reduce epsilon
