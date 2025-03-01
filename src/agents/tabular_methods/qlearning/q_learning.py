"""
Módulo: QLearningAgent
========================
Este módulo implementa la clase `QLearningAgent`, que utiliza el algoritmo
Q-Learning para la toma de decisiones en entornos de aprendizaje por refuerzo.
"""

import numpy as np
from agents.tabular_methods.td_learning_agent import TDLearningAgent
import gymnasium as gym
from tqdm import tqdm

##### agents/tabular_methods/qlearning/q_learning.py #####
"""
Módulo: QLearningAgent
========================
Este módulo implementa la clase `QLearningAgent`, que utiliza el algoritmo
Q-Learning para la toma de decisiones en entornos de aprendizaje por refuerzo.
"""

import numpy as np
from agents.tabular_methods.td_learning_agent import TDLearningAgent
import gymnasium as gym
from tqdm import tqdm

class QLearningAgent(TDLearningAgent):
    """
    Agente basado en el algoritmo Q-Learning.
    Aprende de manera off-policy actualizando la función Q de forma iterativa.
    """

    def update(self, state, action, next_state, reward, done):
        """
        Actualiza la tabla Q usando la ecuación de actualización de Q-Learning.
        
        Args:
            state: Estado actual.
            action (int): Acción tomada.
            next_state: Estado siguiente.
            reward (float): Recompensa obtenida.
            done (bool): Indica si el episodio ha terminado.
        """
        target = reward + self.gamma * np.max(self.Q[next_state]) * (not done)
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
    
    def train(self, num_episodes, render_interval=-1, video_path=None):
        """
        Entrena el agente Q-Learning. Configura la grabación de videos si es necesario.
        
        Args:
            num_episodes (int): Número de episodios de entrenamiento.
            render_interval (int, opcional): Frecuencia de grabación de episodios.
            video_path (str, opcional): Directorio donde almacenar videos.
        """
        super().train(num_episodes, render_interval, video_path)  # Configura video si es necesario
        
        state, info = self.env.reset(seed=self.seed)
        for episode in tqdm(range(num_episodes)):
            done = False
            episode_reward = 0
            episode_data = []
            while not done:
                action = self.get_action(state, info, Q=self.Q, action_space=self.nA)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update(state, action, next_state, reward, terminated)
                episode_data.append((state, action, reward))
                episode_reward += reward
                state = next_state
                done = terminated or truncated
            
            self.episode_rewards.append(episode_reward)  # Guarda recompensa acumulada
            self.episodes.append(episode_data)  # Guarda historial del episodio
            self.decay()  # Aplicar decay después de cada episodio
            state, info = self.env.reset()
