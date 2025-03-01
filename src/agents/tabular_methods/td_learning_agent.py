##### agents/tabular_methods/td_learning_agent.py #####
"""
Módulo: TDLearningAgent
========================
Define la clase base `TDLearningAgent` para algoritmos de diferencias temporales como
Q-Learning y SARSA.
"""

import numpy as np
from agents.tabular_methods.base import TabularAgent

class TDLearningAgent(TabularAgent):
    """
    Clase base para agentes de aprendizaje por diferencias temporales (TD).
    Maneja `alpha` y su evolución a lo largo del entrenamiento.
    """

    def __init__(self, env, gamma=0.99, alpha=0.1, alpha_decay=0.995, min_alpha=0.01):
        """
        Inicializa un agente TD con una tasa de aprendizaje y su historial.

        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento.
            alpha (float): Tasa de aprendizaje inicial.
            alpha_decay (float): Factor de decaimiento de alpha.
            min_alpha (float): Valor mínimo de alpha.
        """
        super().__init__(env, gamma=gamma)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.alpha_history = []  

    def decay(self):
        """
        Reduce gradualmente `alpha` y lo registra en el historial.
        """
        self.alpha_history.append(self.alpha)  
        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)
