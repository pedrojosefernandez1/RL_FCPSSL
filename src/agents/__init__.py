"""
Módulo: agents
========================
Este módulo agrupa todos los agentes disponibles dentro del paquete, organizándolos
según su método de aprendizaje por refuerzo. Cada submódulo representa una categoría
específica de algoritmos tabulares.
"""
from .base.agent import Agent
from .tabular_methods.montecarlo.montecarlo_all import MonteCarloAllAgent
from .tabular_methods.montecarlo.montecarlo_epsilongreedy import MonteCarloEpsilonGreedyAgent
from .tabular_methods.qlearning.q_learning import QLearningAgent
from .tabular_methods.qlearning.q_learning_epsilongreedy import QLearningEpsilonGreedyAgent
from .tabular_methods.sarsa.sarsa import SarsaAgent
from .tabular_methods.sarsa.sarsa_epsilonGreedy import SarsaEpsilonGreedyAgent
from .approximation_methods.sarsa.sarsa_semi_gradient import SarsaSemiGradientAgent
from .approximation_methods.sarsa.sarsa_semi_gradient_epsilongreedy import SarsaSemiGradientEpsilonGreedyAgent
__all__ = [
    "Agent",
    "MonteCarloAllAgent",
    "MonteCarloEpsilonGreedyAgent",
    "QLearningAgent",
    "QLearningEpsilonGreedyAgent",
    "SarsaAgent",
    "SarsaEpsilonGreedyAgent",
    "SarsaSemiGradientAgent",
    "SarsaSemiGradientEpsilonGreedyAgent"
]