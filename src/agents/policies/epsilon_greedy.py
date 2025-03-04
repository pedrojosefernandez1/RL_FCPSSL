
"""
Módulo: EpsilonGreedy
========================
Este módulo define la clase `EpsilonGreedy`, que agrega la estrategia
ε-greedy a cualquier agente que la necesite. También almacena el historial
de valores de `epsilon` para su análisis.
"""

import numpy as np

class EpsilonGreedy:
    """
    Mixin para agregar comportamiento ε-greedy a cualquier agente.
    Maneja la selección de acciones y la reducción progresiva de `epsilon`.
    """

    def __init__(self, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Inicializa la estrategia ε-greedy.
        
        Args:
            epsilon (float): Probabilidad inicial de exploración.
            epsilon_decay (float): Factor de decaimiento de epsilon.
            min_epsilon (float): Valor mínimo de epsilon.
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon_history = []  
    def get_action(self, state, info, Q=None, Q_function=None, action_space=None):
        """
        Selecciona una acción usando una política ε-greedy, compatible con métodos tabulares y de aproximación.
        
        Args:
            state: Estado actual del entorno.
            info: Información adicional del entorno.
            Q (np.array, opcional): Tabla de valores Q para métodos tabulares.
            Q_function (callable, opcional): Función de aproximación de Q (SARSA Semi-Gradiente, DQN).
            action_space (int): Número total de acciones posibles.
        
        Returns:
            int: Acción seleccionada.
        """
        assert (Q is None) != (Q_function is None), "Se debe proporcionar solo Q para tabulares o solo Q_function para aproximación."
    
        if Q is not None:
            return self._get_tabular_action(state, info, Q, action_space)
        
        if Q_function is not None:
            return self._get_approximation_action(state, Q_function, action_space)
        
        raise ValueError("Se debe proporcionar Q para métodos tabulares o Q_function para métodos de aproximación.")

    def _get_tabular_action(self, state, info, Q, nA):
        """
        Selecciona una acción usando una política ε-greedy.
        
        Args:
            state: Estado actual del entorno.
            info: Información adicional del entorno.
            Q (np.array): Tabla de valores Q.
            nA (int): Número de acciones disponibles.
        
        Returns:
            int: Acción seleccionada.
        """
        if 'action_mask' in info:
            pi_A = info['action_mask'] * self.epsilon / np.sum(info['action_mask'])
            valid_actions = np.where(info['action_mask'])[0]
            best_action = valid_actions[np.argmax(Q[state, valid_actions])]
        else:
            pi_A = np.ones(nA, dtype=float) * self.epsilon / nA
            best_action = np.argmax(Q[state])
        
        pi_A[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(nA), p=pi_A)
    
    def _get_approximation_action(self, state, Q_function, action_space):
        """
        Selecciona una acción usando una política ε-greedy en métodos de aproximación.
        """
        
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        
        Q_values = self.get_Q(state)
        return np.argmax(Q_values)
    
    def decay(self):
        """
        Reduce gradualmente el valor de `epsilon` y lo registra en el historial.
        """
        self.epsilon_history.append(self.epsilon)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def stats(self):
        return {'epsilon_history': self.epsilon_history}