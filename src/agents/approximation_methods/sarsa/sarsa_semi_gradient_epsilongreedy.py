
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

    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.01, epsilon=1.0, epsilon_decay=0.995, alpha_decay=0.995, min_epsilon=0.01, min_alpha=0.01, feature_extractor=None, seed=42):
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
        SarsaSemiGradientAgent.__init__(self, env, seed=seed, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, min_alpha=min_alpha, feature_extractor=feature_extractor)
        EpsilonGreedy.__init__(self, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)

    def get_action(self, state, info):
        """
        Selecciona una acción usando la política ε-greedy con función de aproximación.
        """


        #return EpsilonGreedy.get_action(self, state, info, Q_function=lambda s: np.array([np.dot(self.weights[a], self.feature_extractor(s)) for a in range(self.nA)]), action_space=self.nA)
        #return EpsilonGreedy.get_action(self, state, info, Q_function=lambda s: np.array([np.dot(self.weights[a], self._normalize(self.feature_extractor(s)))for a in range(self.nA)]), action_space=self.nA)
    
        return EpsilonGreedy.get_action(self, state, info, Q_function=lambda s: np.array([
            np.dot(self.weights[a], self.feature_extractor(s) / (np.linalg.norm(self.feature_extractor(s)) + 1e-8))
            for a in range(self.nA)
        ]), action_space=self.nA)

    
        #return EpsilonGreedy.get_action(self, state, info, Q_function=lambda s: np.dot(self.weights, self.feature_extractor(s)), action_space=self.nA)

    def decay(self):
        """
        Reduce `alpha` y `epsilon` llamando a las funciones de decaimiento correspondientes.
        """
        SarsaSemiGradientAgent.decay(self)  # Reduce alpha
        EpsilonGreedy.decay(self)  # Reduce epsilon

    def stats(self):
        stats = SarsaSemiGradientAgent.stats(self)
        stats = stats | EpsilonGreedy.stats(self)
        return stats

    def __str__(self):
        """
        Devuelve una representación en cadena del agente con sus parámetros actuales.
        """
        return (f'SarsaSemiGradientEpsilonGreedyAgent(gamma={self.gamma}, alpha={self.alpha}, alpha_decay={self.alpha_decay}, '
                f'min_alpha={self.min_alpha}, epsilon={self.epsilon}, epsilon_decay={self.epsilon_decay}, '
                f'min_epsilon={self.min_epsilon})')

    def pi_star(self):
        return super().pi_star()