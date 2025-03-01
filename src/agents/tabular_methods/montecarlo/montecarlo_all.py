"""
Módulo: MonteCarloAllAgent
========================
Este módulo implementa el agente `MonteCarloAllAgent`, que utiliza el método
Monte Carlo de todas las visitas para actualizar su política de decisión.
"""

import numpy as np
from tqdm import tqdm
from agents.base.agent import Agent
import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo

class MonteCarloAllAgent(Agent):
    """
    Agente basado en Monte Carlo de todas las visitas.
    Aprende a partir de episodios completos y actualiza la estimación de valores Q.
    """

    def __init__(self, env: gym.Env, gamma=0.99):
        """
        Inicializa el agente Monte Carlo.
        
        Args:
            env (gym.Env): Entorno de OpenAI Gym o Gymnasium.
            gamma (float): Factor de descuento para la recompensa futura.
        """
        super().__init__(env, gamma=gamma)
        self.gamma = gamma 
        self.Q = np.zeros([env.observation_space.n, self.nA])
        self.returns_counts = np.zeros([env.observation_space.n, self.nA])
        self.episode_rewards = []
        self.episodes = []
    
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
    
    def stats(self):
        """
        Devuelve estadísticas del entrenamiento.
        
        Returns:
            dict: Contiene la tabla Q, recompensas y episodios.
        """
        return {"Q-table": self.Q, "episode_rewards": self.episode_rewards, "episodes": self.episodes}
    
    def train(self, num_episodes, render_interval=-1, video_path=None):
        """
        Entrena el agente usando Monte Carlo.
        
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
            episode = []
            episode_reward = 0
            while not done:
                action = self.get_action(state, info)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
                episode_reward += reward
                done = terminated or truncated
            self.update(episode)
            self.decay()
            self.episode_rewards.append(episode_reward)
            self.episodes.append(episode)
            state, info = self.env.reset()
    
    def pi_star(self):
        """
        Devuelve la política óptima basada en la tabla Q aprendida.
        
        Returns:
            tuple: Matriz de política óptima y secuencia de acciones óptimas.
        """
        done = False
        pi_star = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        state, info = self.env.reset(seed=self.seed)
        actions = []
        while not done:
            if 'action_mask' in info:
                valid_actions = np.where(info['action_mask'])[0]
                best_action = valid_actions[np.argmax(self.Q[state, valid_actions])]
            else:
                best_action = np.argmax(self.Q[state])
            actions.insert(0, best_action)
            pi_star[state, best_action] = 1
            state, reward, terminated, truncated, info = self.env.step(best_action)
            done = terminated or truncated
        return pi_star, actions
