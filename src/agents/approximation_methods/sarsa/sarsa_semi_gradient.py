##### agents/approximation_methods/sarsa_semi_gradient.py #####
"""
Módulo: SarsaSemiGradientAgent
========================
Este módulo implementa el algoritmo SARSA Semi-Gradiente, que extiende SARSA para manejar
funciones de aproximación en lugar de tablas de valores discretas.
"""

import numpy as np
from agents.approximation_methods.base import ApproximationAgent
from agents.approximation_methods.tile import TileCodingFeatureExtractor
import gymnasium as gym
from tqdm import tqdm


class SarsaSemiGradientAgent(ApproximationAgent):
    """
    Agente basado en SARSA Semi-Gradiente.
    Utiliza funciones de aproximación en lugar de tablas Q discretas.
    """

    def __init__(self, env: gym.Env, seed = 32, gamma=0.99, alpha=0.01, alpha_decay=0.995, min_alpha=0.01,  num_tilings=64, iht_size=131072,feature_extractor=None):
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
        super().__init__(env, seed=seed, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, min_alpha=min_alpha)
        if feature_extractor is None:
            self.feature_extractor = TileCodingFeatureExtractor(num_tilings=num_tilings, low=env.observation_space.low, high=env.observation_space.high, iht_size=iht_size)
        else:
            self.feature_extractor = feature_extractor
        self.weights = np.zeros((self.nA, self.feature_extractor.iht.size))

    def get_action(self, state, info):
        """
        Método abstracto para seleccionar una acción.
        """
        raise NotImplementedError("El método get_action() debe ser implementado en una subclase.")

    def update(self, state, action, reward, next_state, next_action, done):
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
        features = self.feature_extractor(state)
        Q_current = np.dot(self.weights[action], features)
        Q_next = np.dot(self.weights[next_action], self.feature_extractor(next_state)) if not done else 0.0
        td_error = reward + self.gamma * Q_next - Q_current
        # Actualización de los pesos en la dirección del gradiente (multiplicado por features)
        self.weights[action] += self.alpha * td_error * features
    
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
        
        print(self.alpha, self.epsilon, self.min_alpha, self.min_epsilon, self.alpha_decay, self.epsilon_decay, self.feature_extractor.iht.size, self.feature_extractor.num_tilings)
        for episode in tqdm(range(num_episodes)):
            state, info = self.env.reset(seed=self.seed)
            action = self.get_action(state, info)
            episode_reward = 0
            done = False
            episode_data = []
            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_action = self.get_action(next_state, info)
                self.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                episode_data.append((state, action, reward))
                episode_reward += reward
            self.decay()
            self.episode_rewards.append(episode_reward)  # Guarda recompensa acumulada
            self.episodes.append(episode_data)  # Guarda historial del episodio

    def stats(self):
        stats = super().stats()
        stats = stats | {'weights': self.weights}
        return stats