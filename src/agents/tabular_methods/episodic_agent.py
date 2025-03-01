"""
Módulo: EpisodicAgent
========================
Este módulo define la clase base `EpisodicAgent`, utilizada por algoritmos
que actualizan su política después de completar un episodio, como Monte Carlo.
"""

import numpy as np
from agents.tabular_methods.base import TabularAgent

class EpisodicAgent(TabularAgent):
    """
    Clase base para agentes basados en episodios.
    Se usa en métodos como Monte Carlo, que requieren observar episodios completos
    antes de actualizar su política.
    """

    def __init__(self, env,seed = 32, gamma=0.99):
        """
        Inicializa un agente episódico con una tabla de retornos.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento.
        """
        super().__init__(env,seed=seed, gamma=gamma)
        self.returns_counts = np.zeros_like(self.Q)
    
    def update(self, episode):
        """
        Método abstracto que debe ser implementado por los agentes episódicos.
        """
        raise NotImplementedError("El método update() debe ser implementado en una subclase.")
