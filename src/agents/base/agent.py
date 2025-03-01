"""
Módulo: Agent
========================
Este módulo define la clase abstracta `Agent`, que es la base para todas
las implementaciones de agentes en aprendizaje por refuerzo. Proporciona
una estructura común para métodos de decisión, actualización de políticas
y seguimiento de estadísticas.
"""

import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import gymnasium as gym

class Agent(ABC):
    """
    Clase abstracta para agentes de aprendizaje por refuerzo.
    Todos los agentes específicos deben heredar de esta clase e implementar
    los métodos abstractos.
    """

    def __init__(self, env: gym.Env, seed=32, **hyperparameters):
        """
        Inicializa el agente con los parámetros dados.

        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            seed (int, opcional): Semilla para la reproducibilidad. Por defecto 32.
            **hyperparameters: Parámetros adicionales específicos del agente.
        """
        self.env = env
        self.seed = seed
        self.nA = env.action_space.n
        self.hyperparameters = hyperparameters

    @abstractmethod
    def get_action(self, state, info) -> int:
        """
        Devuelve la acción a realizar en un estado dado, siguiendo la política del agente.
        
        Args:
            state: Estado actual del entorno.
            info: Información adicional del entorno.

        Returns:
            int: Acción seleccionada.
        """
        pass
    
    @abstractmethod
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Actualiza la política del agente basado en una transición del entorno.
        
        Args:
            obs: Estado actual.
            action (int): Acción tomada.
            next_obs: Estado siguiente.
            reward (float): Recompensa obtenida.
            terminated (bool): Indica si el episodio ha terminado.
            truncated (bool): Indica si el episodio fue truncado.
            info: Información adicional del entorno.
        """
        pass

    @abstractmethod
    def stats(self):
        """
        Devuelve estadísticas del entrenamiento del agente.

        Returns:
            dict: Diccionario con estadísticas como recompensas y métricas de aprendizaje.
        """
        pass
    
    @abstractmethod
    def decay(self):
        """
        Aplica decay a los parámetros relevantes (por ejemplo, tasa de exploración ε).
        """
        pass

    @abstractmethod
    def pi_star(self):
        """
        Obtiene la política óptima aprendida por el agente.

        Returns:
            dict o estructura adecuada: Representación de la política óptima.
        """
        pass
    
    @abstractmethod
    def __str__(self):
        """
        Devuelve una representación en cadena del agente.
        """
        pass
