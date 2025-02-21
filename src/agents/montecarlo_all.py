import numpy as np
from tqdm import tqdm
from agents.agent import Agent
import gymnasium as gym

class MonteCarloAllAgent(Agent):
    def __init__(self, env: gym.Env, gamma=0.99):
        super().__init__(env, gamma=gamma)
        self.gamma = gamma 
        self.Q = np.zeros((env.observation_space.n, self.nA))
        self.returns = {s: {a: [] for a in range(self.nA)} for s in range(env.observation_space.n)}
        self.episode_rewards = []
    
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """Actualiza los valores de Q usando el método Monte Carlo"""
        G = reward
        self.returns[obs][action].append(G)
        self.Q[obs][action] = np.mean(self.returns[obs][action])
    
    def stats(self):
        """Devuelve estadísticas del entrenamiento"""
        return {"Q-table": self.Q, "episode_rewards": self.episode_rewards}
    
    def train(self, num_episodes):
        """Entrena el agente usando Monte Carlo"""
        for _ in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update(state, action, next_state, reward, terminated, truncated, info)
                state = next_state
                episode_reward += reward
                done = terminated or truncated
            
            self.episode_rewards.append(episode_reward)