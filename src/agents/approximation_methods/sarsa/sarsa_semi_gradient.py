##### agents/approximation_methods/sarsa_semi_gradient.py #####
"""
Módulo: SarsaSemiGradientAgent
========================
Este módulo implementa el algoritmo SARSA Semi-Gradiente, que extiende SARSA para manejar
funciones de aproximación en lugar de tablas de valores discretas.
"""

import numpy as np
from agents.approximation_methods.base import ApproximationAgent
import gymnasium as gym
from tqdm import tqdm

class SarsaSemiGradientAgent(ApproximationAgent):
    """
    Agente basado en SARSA Semi-Gradiente.
    Utiliza funciones de aproximación en lugar de tablas Q discretas.
    """

    def __init__(self, env: gym.Env,seed = 32, gamma=0.99, alpha=0.01, feature_extractor=None, seed=42):
        """
        Inicializa el agente SARSA Semi-Gradiente.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            seed (int): Semilla para la reproducibilidad.
            gamma (float): Factor de descuento.
            alpha (float): Tasa de aprendizaje.
            feature_extractor (callable, opcional): Función de extracción de características.
            seed (int): Semilla para la reproducibilidad.
        """
        super().__init__(env,seed=seed, gamma=gamma, alpha=alpha)
        self.feature_extractor = feature_extractor or (lambda s: s)
        self.weights = np.zeros(self.feature_extractor(env.observation_space.sample()).shape)

    def get_action(self, state, info):
        """
        Método abstracto para seleccionar una acción.
        """
        raise NotImplementedError("El método get_action() debe ser implementado en una subclase.")

    def update(self, state, action, next_state, next_action, reward, done):
        """
        Actualiza los pesos de la función de aproximación usando la actualización SARSA Semi-Gradiente.
        
        Args:
            state: Estado actual.
            action (int): Acción tomada.
            next_state: Estado siguiente.
            next_action (int): Próxima acción seleccionada.
            reward (float): Recompensa obtenida.
            done (bool): Indica si el episodio ha terminado.
        """
        phi_s = self.feature_extractor(state)
        phi_next_s = self.feature_extractor(next_state)
        target = reward + self.gamma * np.dot(self.weights, phi_next_s) * (not done)
        prediction = np.dot(self.weights, phi_s)
        self.weights += self.alpha * (target - prediction) * phi_s
    
    def decay(self):
        """
        Reduce `alpha` llamando a la función de decaimiento del agente base.
        """
        super().decay()
    
    def train(self, num_episodes, render_interval=-1, video_path=None):
        """
        Entrena el agente SARSA Semi-Gradiente.
        
        Args:
            num_episodes (int): Número de episodios de entrenamiento.
            render_interval (int, opcional): Frecuencia de grabación de episodios.
            video_path (str, opcional): Directorio donde almacenar videos.
        """
        super().train(num_episodes, render_interval, video_path)  # Configuración de video
        
        state, info = self.env.reset()
        action = self.get_action(state, info)
        for episode in tqdm(range(num_episodes)):
            done = False
            episode_reward = 0
            episode_data = []
            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_action = self.get_action(next_state, info)
                self.update(state, action, next_state, next_action, reward, terminated)
                episode_data.append((state, action, reward))
                episode_reward += reward
                state, action = next_state, next_action
                done = terminated or truncated
            
            self.episode_rewards.append(episode_reward)  # Guarda recompensa acumulada
            self.episodes.append(episode_data)  # Guarda historial del episodio
            self.decay()  # Aplicar decay después de cada episodio
            state, info = self.env.reset()
            action = self.get_action(state, info)
