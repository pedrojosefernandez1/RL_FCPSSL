
"""
Módulo: TabularAgent
========================
Este módulo define la clase base `TabularAgent`, que proporciona
una estructura común para los agentes de métodos tabulares
como Monte Carlo, Q-Learning y SARSA.
"""

import numpy as np
from agents.base.agent import Agent

class TabularAgent(Agent):
    """
    Clase base para agentes de métodos tabulares.
    Proporciona estructuras comunes como la tabla Q y la gestión del factor de descuento `gamma`.
    """

    def __init__(self, env,seed= 32, gamma=0.99):
        """
        Inicializa el agente con parámetros compartidos.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento.
        """
        super().__init__(env,seed= seed, gamma=gamma)
        self.gamma = gamma
        
        self.Q = np.zeros([env.observation_space.n, self.nA])
        self.episode_rewards = []
        self.episodes = []
    def update(self, *args):
        """
        Método abstracto para actualizar la tabla Q.
        """
        raise NotImplementedError("El método update() debe ser implementado en una subclase.")

    def get_action(self, state, info) -> int:
        """
        Método abstracto para seleccionar una acción.
        """
        raise NotImplementedError("El método get_action() debe ser implementado en una subclase.")
    
    def stats(self):
        """
        Devuelve estadísticas del entrenamiento.
        
        Returns:
            dict: Contiene la tabla Q y recompensas acumuladas por episodio.
        """
        return {
            "Q-table": self.Q,
            "episode_rewards": self.episode_rewards,
        }
    
    @staticmethod
    def get_best_action(Q, state, action_mask=None):
        """
        Devuelve la mejor acción disponible según la tabla Q.
        
        Args:
            Q (np.array): Tabla de valores Q.
            state (int): Estado actual.
            action_mask (np.array, opcional): Máscara de acciones válidas.
        
        Returns:
            int: Mejor acción disponible.
        """
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            return valid_actions[np.argmax(Q[state, valid_actions])]
        return np.argmax(Q[state])
