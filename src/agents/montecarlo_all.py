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
        self.episodes = []
    
    def update(self, episode):
        """Actualiza los valores de Q usando Monte Carlo de Todas las Visitas"""
        G = 0  # Inicializamos el retorno acumulado

        # Recorrer el episodio en orden inverso para calcular G
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G  # Aplicar descuento

            # Actualizar la tabla Q usando todas las visitas
            self.returns_counts[state, action] += 1
            n = self.returns_counts[state, action]
            self.Q[state, action] += (1/n) * (G - self.Q[state, action])  # Promedio incremental
    
    def stats(self):
        """Devuelve estadísticas del entrenamiento"""
        return {"Q-table": self.Q, "episode_rewards": self.episode_rewards, "episodes": self.episodes}
    
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
        
        state, info = self.env.reset(seed=self.seed)
        for t in tqdm(range(num_episodes)):
            done = False
            episode = []  # Guardaremos el historial de estado, acción, recompensa
            episode_reward = 0

            while not done:
                action = self.get_action(state, info)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                episode.append((state, action, reward))  # Guardar cada transición
                state = next_state
                episode_reward += reward
                done = terminated or truncated

            # Actualizar Q-table con todas las visitas
            
            self.update(episode)  
            self.decay()  # Decaimiento de epsilon si se usa epsilon-greedy


            # Guardar recompensa total del episodio y el episodio junto al resto de episodios
            self.episode_rewards.append(episode_reward)
            self.episodes.append(episode)
            state, info = self.env.reset()

    def pi_star(self):
        done = False
        pi_star = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        state, info = self.env.reset(seed=self.seed) # start in top-left, = 0
        actions = []
        while not done:
            if 'action_mask' in info:
                valid_actions = np.where(info['action_mask'])[0] 
                best_action = valid_actions[np.argmax(self.Q[state, valid_actions])]   
                
            else:       
                best_action = np.argmax(self.Q[state]) # Elegir la mejor acción según Q(s,a)
            
            actions.insert(0,best_action)
            pi_star[state,best_action] = 1
            state, reward, terminated, truncated, info = self.env.step(best_action)
            done = terminated or truncated
        return pi_star, actions
            