import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import gymnasium as gym

class Agent(ABC):
    def __init__(self, env: gym.Env, **hyperparameters):
        """Inicializa todo lo necesario para el aprendizaje"""
        self.env = env
        self.nA = env.action_space.n
        self.hyperparameters = hyperparameters
    
    @abstractmethod
    def get_action(self, state) -> int:
        """
        Indicará qué acción realizar de acuerdo al estado.
        Responde a la política del agente.
        """
        pass
    
    @abstractmethod
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Con la muestra (s, a, s', r) e información complementaria aplicamos el algoritmo.
        """
        pass
    
    @abstractmethod
    def stats(self):
        """Devuelve estadísticas del entrenamiento"""
        pass