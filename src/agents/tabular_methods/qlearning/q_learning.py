"""
Módulo: QLearningAgent
========================
Este módulo implementa la clase `QLearningAgent`, que utiliza el algoritmo
Q-Learning para la toma de decisiones en entornos de aprendizaje por refuerzo.
"""

import numpy as np
from tqdm import tqdm
from agents.base.agent import Agent
import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo

class QLearningAgent(Agent):
    """
    Agente basado en el algoritmo Q-Learning.
    Aprende de manera off-policy actualizando la función Q de forma iterativa.
    """

    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.1):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento para la recompensa futura.
            alpha (float): Tasa de aprendizaje.
        """
        super().__init__(env, gamma=gamma)
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros([env.observation_space.n, self.nA])
        self.episode_rewards = []
        self.episodes = []

    def get_action(self, state, info) -> int:
        """
        Método abstracto que debe ser implementado en subclases.
        """
        raise NotImplementedError("El método get_action() debe ser implementado por una subclase.")
    
    def decay(self):
        """
        Método abstracto para modificar parámetros como epsilon.
        """
        raise NotImplementedError("El método decay() debe ser implementado por una subclase.")

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

    def stats(self):
        """
        Devuelve estadísticas del entrenamiento.
        
        Returns:
            dict: Contiene la tabla Q y métricas de episodios.
        """
        return {"Q-table": self.Q, "episode_rewards": self.episode_rewards, "episodes": self.episodes}

    def train(self, num_episodes, render_interval=-1, video_path=None):
        """
        Entrena el agente usando Q-Learning.
        
        Args:
            num_episodes (int): Número de episodios de entrenamiento.
            render_interval (int, opcional): Cada cuántos episodios renderizar.
            video_path (str, opcional): Directorio para almacenar videos del entrenamiento.
        """
        if video_path:
            env_name = self.env.spec.id if self.env.spec else "UnknownEnv"
            env_dir = os.path.join(video_path, env_name)
            model_name = str(self).replace("=", "").replace(",", "").replace(" ", "_")
            model_dir = os.path.join(env_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            self.env = RecordVideo(self.env, model_dir, episode_trigger=lambda episode: episode % render_interval == 0)
        
        state, info = self.env.reset(seed=self.seed)
        for t in tqdm(range(num_episodes)):
            done = False
            episode_reward = 0
            episode = []
            while not done:
                action = self.get_action(state, info)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode.append((state, action, reward))
                self.update(state, action, next_state, reward, terminated)
                state = next_state
                episode_reward += reward
                done = terminated or truncated
            self.episode_rewards.append(episode_reward)
            self.decay()
            self.episodes.append(episode)
            state, info = self.env.reset()
    
    def pi_star(self):
        """
        Devuelve la política óptima aprendida por el agente.
        
        Returns:
            tuple: Matriz de política óptima y secuencia de acciones óptimas.
        """
        state, info = self.env.reset(seed=self.seed)
        done = False
        pi_star = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        actions = []
        while not done:
            if 'action_mask' in info:
                valid_actions = np.where(info['action_mask'])[0]
                best_action = valid_actions[np.argmax(self.Q[state, valid_actions])]
            else:
                best_action = np.argmax(self.Q[state])
            pi_star[state, best_action] += 1
            actions.insert(0, best_action)
            state, reward, terminated, truncated, info = self.env.step(best_action)
            done = terminated or truncated
        return pi_star, actions
