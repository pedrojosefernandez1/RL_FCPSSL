import numpy as np
from tqdm import tqdm
import gymnasium as gym
from agents.agent import Agent

class SarsaAgent(Agent):
    def __init__(self, env: gym.Env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        super().__init__(env, alpha=alpha, gamma=gamma, epsilon=epsilon,
                         epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)
        
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_action(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        next_action = self.get_action(next_obs)  # SARSA usa la acción siguiente real
        td_target = reward + self.gamma * self.Q[next_obs, next_action] * (not terminated)
        td_error = td_target - self.Q[obs, action]
        self.Q[obs, action] += self.alpha * td_error
        return next_action  # Devuelve la acción para continuidad en episodios

    def update(self, episode):
        state, _ = self.env.reset()
        action = self.get_action(state)
        
        for _ in range(len(episode)):
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_action = self.update(state, action, next_state, reward, terminated, truncated, {})
            
            if terminated or truncated:
                break
            
            state, action = next_state, next_action

    def stats(self):
        return {"Q": self.Q.copy(), "epsilon": self.epsilon}

    def decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def pi_star(self):
        return np.argmax(self.Q, axis=1)
