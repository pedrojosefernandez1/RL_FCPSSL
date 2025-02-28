import numpy as np
from tqdm import tqdm
from agents.agent import Agent
import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo

class SarsaAgent(Agent):
    def __init__(self, env: gym.Env, gamma=0.99, alpha=0.1):
        """Clase base para algoritmos SARSA"""
        super().__init__(env, gamma=gamma)
        self.gamma = gamma  # Factor de descuento
        self.alpha = alpha  # Tasa de aprendizaje
        self.Q = np.zeros([env.observation_space.n, self.nA])  # Tabla Q
        self.episode_rewards = []  # Almacenar recompensas por episodio
        self.episodes = []


    def get_action(self, state) -> int:
        """Método abstracto: debe ser implementado en clases hijas"""
        raise NotImplementedError("El método get_action() debe ser implementado por una subclase.")

    def decay(self):
        """Método abstracto para modificar parámetros como epsilon"""
        raise NotImplementedError("El método decay() debe ser implementado por una subclase.")

    def update(self, state, action, next_state, next_action, reward, done):
        """Actualiza Q usando SARSA"""
        target = reward + self.gamma * self.Q[next_state, next_action] * (not done)  # Si done=True, Q(next) es 0
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])  # Regla de actualización

    def stats(self):
        """Devuelve estadísticas del entrenamiento"""
        return {"Q-table": self.Q, "episode_rewards": self.episode_rewards, "episodes": self.episodes}

    def train(self, num_episodes, render_interval=-1, video_path=None):
        """Entrena el agente usando SARSA"""
        if video_path:
            env_name = self.env.spec.id if self.env.spec else "UnknownEnv"
            env_dir = os.path.join(video_path, env_name)
            model_name = str(self).replace("=", "").replace(",", "").replace(" ", "_")
            model_dir = os.path.join(env_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            self.env = RecordVideo(self.env, model_dir, episode_trigger=lambda episode: episode % render_interval == 0)

        for t in tqdm(range(num_episodes)):
            state, _ = self.env.reset(seed=32)
            action = self.get_action(state)  # Obtener acción inicial
            done = False
            episode_reward = 0
            episode = []  # Guardaremos el historial de estado, acción, recompensa


            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode.append((state, action, reward))  # Guardar cada transición

                next_action = self.get_action(next_state)  # Obtener siguiente acción (On-Policy)

                self.update(state, action, next_state, next_action, reward, terminated)

                state, action = next_state, next_action  # Avanzar en la secuencia
                episode_reward += reward
                done = terminated or truncated

            self.episode_rewards.append(episode_reward)
            if not truncated:
                self.decay()  # Aplicar decaimiento de epsilon
            self.episodes.append(episode)


    def pi_star(self):
        """Devuelve la política óptima estimada"""
        state, _ = self.env.reset()  # Estado inicial
        done = False
        pi_star = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        actions = []
        while not done:
            best_action = np.argmax(self.Q[state])# Elegir la mejor acción según Q(s,a)
            pi_star[state, best_action] += 1  # Marcar la mejor acción
            actions.insert(0,best_action)  # Registrar el estado y la acción
            state, reward, terminated, truncated, _ = self.env.step(best_action)
            done = terminated or truncated  # Detener si se llega a un estado terminal

        return pi_star, actions

    def pi_star2(self):
        """Devuelve la política óptima estimada"""
        pi_star = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        for state in range(self.env.observation_space.n):
            best_action = np.argmax(self.Q[state])
            pi_star[state, best_action] = 1  # Marcar la mejor acción

        actions = []
        actions.append(best_action)
    
    def pi_star3(self):
        """Devuelve el camino óptimo aprendido por el agente, partiendo del estado inicial."""
        state, _ = self.env.reset()  # Estado inicial
        done = False
        path = []  # Lista para almacenar (estado, acción) del camino óptimo

        while not done:
            action = np.argmax(self.Q[state, :])  # Elegir la mejor acción según Q(s,a)
            path.append((state, action))  # Registrar el estado y la acción
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated  # Detener si se llega a un estado terminal

        return path  # Devuelve la secuencia de (estado, acción) óptima
'''(array([ [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.]])'''
