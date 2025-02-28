import numpy as np
from tqdm import tqdm
from agents.agent import Agent
import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo

class QLearningAgent(Agent):
    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.1):
        """Clase base para algoritmos QLearning"""
        super().__init__(env, gamma=gamma)
        self.gamma = gamma  # Factor de descuento
        self.alpha = alpha  # Tasa de aprendizaje
        self.Q = np.zeros([env.observation_space.n, self.nA])  # Tabla Q
        self.episode_rewards = []  # Almacenar recompensas por episodio
        self.episodes = []


    def get_action(self, state, info) -> int:
        """Método abstracto: debe ser implementado en clases hijas"""
        raise NotImplementedError("El método get_action() debe ser implementado por una subclase.")
    
    def decay(self):
        """Método abstracto para modificar parámetros como epsilon"""
        raise NotImplementedError("El método decay() debe ser implementado por una subclase.")

    def update(self, state, action, next_state, reward, done):
        """Actualiza Q usando QLearning"""
        target = reward + self.gamma * np.max(self.Q[next_state]) * (not done)  # Si done=True, Q(next) es 0
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])  # Regla de actualización

    def stats(self):
        """Devuelve estadísticas del entrenamiento"""
        return {"Q-table": self.Q, "episode_rewards": self.episode_rewards, "episodes": self.episodes}

    def train(self, num_episodes, render_interval=-1, video_path=None):
        """Entrena el agente usando QLearning"""
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
            episode = []  # Guardaremos el historial de estado, acción, recompensa

            while not done:
                action = self.get_action(state, info)  # Obtener acción inicial
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode.append((state, action, reward))  # Guardar cada transición
                
                self.update(state, action, next_state, reward, terminated)

                state = next_state  # Avanzar en la secuencia
                episode_reward += reward
                done = terminated or truncated

            self.episode_rewards.append(episode_reward)
            self.decay()  # Aplicar decay de epsilon

            self.episodes.append(episode)
            state, info = self.env.reset()

    def pi_star(self):
        """Devuelve la política óptima estimada"""
        state, info = self.env.reset(seed=self.seed)  # Estado inicial
        done = False
        pi_star = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        actions = []
        while not done:
            if 'action_mask' in info:
                valid_actions = np.where(info['action_mask'])[0] 
                best_action = valid_actions[np.argmax(self.Q[state, valid_actions])]   
            
            else:       
                best_action = np.argmax(self.Q[state])# Elegir la mejor acción según Q(s,a)
            pi_star[state, best_action] += 1  # Marcar la mejor acción
            actions.insert(0,best_action)  # Registrar el estado y la acción
            state, reward, terminated, truncated, info = self.env.step(best_action)
            done = terminated or truncated  # Detener si se llega a un estado terminal

        return pi_star, actions
