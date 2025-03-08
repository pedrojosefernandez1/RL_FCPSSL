##### agents/tabular_methods/sarsa/sarsa.py #####
"""
Módulo: SarsaAgent
========================
Este módulo implementa la clase `SarsaAgent`, que emplea el algoritmo SARSA para
la toma de decisiones en entornos de aprendizaje por refuerzo.
"""

import numpy as np
from agents.tabular_methods.td_learning_agent import TDLearningAgent
import gymnasium as gym
from tqdm import tqdm

class SarsaAgent(TDLearningAgent):
    """
    Agente basado en el algoritmo SARSA.
    Aprende de manera on-policy utilizando una política que se ajusta a medida
    que aprende la función de valor de acción Q.
    """

    def update(self, state, action, next_state, next_action, reward, done):
        """
        Actualiza la tabla Q usando la ecuación de actualización de SARSA.
        
        Args:
            state: Estado actual.
            action (int): Acción tomada.
            next_state: Estado siguiente.
            next_action (int): Próxima acción seleccionada.
            reward (float): Recompensa obtenida.
            done (bool): Indica si el episodio ha terminado.
        """
        target = reward + self.gamma * self.Q[next_state, next_action] * (not done)
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
    
    def train(self, num_episodes, render_interval=-1, video_path=None):
        """
        Entrena el agente SARSA. Configura la grabación de videos si es necesario.
        
        Args:
            num_episodes (int): Número de episodios de entrenamiento.
            render_interval (int, opcional): Frecuencia de grabación de episodios.
            video_path (str, opcional): Directorio donde almacenar videos.
        """
        super().train(num_episodes, render_interval, video_path)  # Configura video si es necesario
        
        state, info = self.env.reset()
        action = self.get_action(state, info, Q=self.Q, action_space=self.nA)
        for episode in tqdm(range(num_episodes)):
            done = False
            episode_reward = 0
            episode_data = []
            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_action = self.get_action(next_state, info, Q=self.Q, action_space=self.nA)
                self.update(state, action, next_state, next_action, reward, terminated)
                episode_data.append((state, action, reward))
                episode_reward += reward
                state, action = next_state, next_action
                done = terminated or truncated
            
            self.episode_rewards.append(episode_reward)  # Guarda recompensa acumulada
            self.episodes.append(episode_data)  # Guarda historial del episodio
            self.decay()  # Aplicar decay después de cada episodio
            state, info = self.env.reset()
            action = self.get_action(state, info, Q=self.Q, action_space=self.nA)

        self.env.close()
