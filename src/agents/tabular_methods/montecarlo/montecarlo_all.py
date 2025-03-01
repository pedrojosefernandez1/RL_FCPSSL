
"""
Módulo: MonteCarloAllAgent
========================
Este módulo implementa la clase `MonteCarloAllAgent`, que utiliza el método
Monte Carlo de todas las visitas para actualizar su política de decisión.
"""

import numpy as np
from agents.tabular_methods.episodic_agent import EpisodicAgent
import gymnasium as gym
from tqdm import tqdm

class MonteCarloAllAgent(EpisodicAgent):
    """
    Agente basado en Monte Carlo de todas las visitas.
    Aprende a partir de episodios completos y actualiza la estimación de valores Q.
    """

    def update(self, episode):
        """
        Actualiza los valores de Q usando el método Monte Carlo de todas las visitas.
        
        Args:
            episode (list): Lista de transiciones (estado, acción, recompensa).
        """
        G = 0  # Inicializamos el retorno acumulado
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G  # Aplicar descuento
            self.returns_counts[state, action] += 1
            n = self.returns_counts[state, action]
            self.Q[state, action] += (1/n) * (G - self.Q[state, action])  # Promedio incremental

    def train(self, num_episodes, render_interval=-1, video_path=None):
        """
        Entrena el agente Monte Carlo. Configura la grabación de videos si es necesario.
        
        Args:
            num_episodes (int): Número de episodios de entrenamiento.
            render_interval (int, opcional): Frecuencia de grabación de episodios.
            video_path (str, opcional): Directorio donde almacenar videos.
        """
        super().train(num_episodes, render_interval, video_path)  # Configura video si es necesario
        
        state, info = self.env.reset()
        for episode in tqdm(range(num_episodes)):
            done = False
            episode_reward = 0
            episode_data = []
            while not done:
                action = self.get_action(state, info, self.Q, action_space=self.nA)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_data.append((state, action, reward))
                episode_reward += reward
                state = next_state
                done = terminated or truncated
            
            self.update(episode_data)  
            self.episode_rewards.append(episode_reward)  # Guarda recompensa acumulada
            self.episodes.append(episode_data)  # Guarda historial del episodio
            self.decay()
            state, info = self.env.reset()
            
