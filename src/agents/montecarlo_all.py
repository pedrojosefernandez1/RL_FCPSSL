import numpy as np
from tqdm import tqdm
from agents.agent import Agent
import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo

class MonteCarloAllAgent(Agent):
    def __init__(self, env: gym.Env, gamma=0.99):
        super().__init__(env, gamma=gamma)
        self.gamma = gamma 
        self.Q = np.zeros([env.observation_space.n, self.nA])
        self.returns_counts = np.zeros([env.observation_space.n, self.nA])
        self.episode_rewards = []
    
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """Actualiza los valores de Q usando el método Monte Carlo"""
        G = reward
        self.returns_counts[obs,action] += 1
        n = self.returns_counts[obs,action]
        self.Q[obs,action] += (1/n) * (G - self.Q[obs, action])
    
    def stats(self):
        """Devuelve estadísticas del entrenamiento"""
        return {"Q-table": self.Q, "episode_rewards": self.episode_rewards}
    
    def train(self, num_episodes, render_interval=-1, video_path=None):
        """Entrena el agente usando Monte Carlo"""
        if video_path:
            env_name = self.env.spec.id if self.env.spec else "UnknownEnv"
            env_dir = os.path.join(video_path, env_name)
            model_name = str(self).replace("=", "").replace(",", "").replace(" ", "_") 
            model_dir = os.path.join(env_dir, model_name)
            os.makedirs(model_dir, exist_ok=True) 
            
            # Envolver el entorno con el grabador de video 
            self.env = RecordVideo(self.env, model_dir, episode_trigger=lambda episode: episode % render_interval == 0)

        for t in tqdm(range(num_episodes)):
            state, _ = self.env.reset(seed=32)
            done = False
            episode_reward = 0
            episodes = []
            while not done:
                action = self.get_action(state)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episodes.append((state, action))

                
                state = next_state
                episode_reward += reward
                done = terminated or truncated
            for (state, action) in episodes:
                self.update(state, action, next_state, episode_reward, terminated, truncated, info)   

            
                
            self.decay()
            self.episode_rewards.append(episode_reward)
            