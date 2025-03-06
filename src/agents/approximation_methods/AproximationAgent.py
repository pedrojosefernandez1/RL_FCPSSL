"""
Módulo: ApproximationAgent
========================
Este módulo define la clase base `ApproximationAgent`, que proporciona
una estructura común para los agentes de aprendizaje por refuerzo basados en aproximación,
como SARSA Semi-Gradiente y Deep Q-Learning.
"""

import numpy as np
from agents.base.agent import Agent

class ApproximationAgent(Agent):
    """
    Clase base para agentes de métodos de aproximación.
    Proporciona estructuras comunes como el factor de descuento `gamma`, la tasa de aprendizaje `alpha`
    y la gestión de pesos o parámetros de modelos de aproximación.
    """

    def __init__(self, env, seed=32, gamma=0.99, alpha=0.01, alpha_decay=0.995, min_alpha=0.001):
        """
        Inicializa el agente con parámetros compartidos para métodos de aproximación.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento.
            alpha (float): Tasa de aprendizaje inicial.
            alpha_decay (float): Factor de decaimiento de alpha.
            min_alpha (float): Valor mínimo de alpha.
        """
        super().__init__(env, seed=seed)
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.episode_rewards = []
        self.episodes = []
        self.alpha_history = []  # Guarda la evolución de alpha

    def decay(self):
        """
        Reduce gradualmente `alpha` y lo registra en el historial.
        """
        self.alpha_history.append(self.alpha)
        self.alpha = self.alpha/2
        #self.alpha = max(self.alpha/2, self.min_alpha)
        #self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)

    def update(self, *args):
        """
        Método abstracto para actualizar los parámetros de la función de aproximación.
        """
        raise NotImplementedError("El método update() debe ser implementado en una subclase.")

    def stats(self):
        return {
            "episode_rewards": self.episode_rewards,
            "episodes": self.episodes,
            'alpha_history': self.alpha_history
        }