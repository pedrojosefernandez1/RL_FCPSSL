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

    def __init__(self, env: gym.Env, seed = 32, gamma=0.99, alpha=0.01, alpha_decay=0.995, min_alpha=0.01,  num_tilings=16, iht_size=1024,feature_extractor=None):
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
        self.feature_extractor = TileCodingFeatureExtractor(num_tilings=num_tilings, low=env.observation_space.low, high=env.observation_space.high, iht_size=iht_size)
        self.weights = np.zeros((self.nA, iht_size))

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


        phi_s = self._normalize(self.feature_extractor(state))
        phi_next_s = self._normalize(self.feature_extractor(next_state))
        #target = reward + self.gamma * np.dot(self.weights[next_action], phi_next_s) * (not done)
        #prediction = np.dot(self.weights[action], phi_s)
        #self.weights[action] += self.alpha * (target - prediction)

        td_error = (reward + self.gamma * np.dot(self.weights[next_action], phi_next_s) * (not done)) - np.dot(self.weights[action], phi_s)
        self.weights[action] += self.alpha * td_error * phi_s  # Se multiplica por phi_s para aplicar el gradiente correcto


    def _normalize(self, phi):
        norm = np.linalg.norm(phi)
        return phi / (norm + 1e-8)  # Evita división por cero
    
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
            if episode % 1000 == 0:
                print(f"Episode {episode}, weight norm: {np.linalg.norm(self.weights)}")
            max_weight_norm = 25  # Límite para la norma de los pesos
            if np.linalg.norm(self.weights) > max_weight_norm:
                self.weights = self.weights * (max_weight_norm / np.linalg.norm(self.weights))


    def stats(self):
        stats = super().stats()
        stats = stats | {'weights': self.weights}
        return stats