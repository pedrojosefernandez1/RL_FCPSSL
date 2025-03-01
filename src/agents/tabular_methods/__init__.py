"""
Módulo de Métodos Tabulares
========================
Este submódulo agrupa los métodos tabulares de aprendizaje por refuerzo,
que incluyen Monte Carlo, Q-Learning y SARSA. 
"""
from .montecarlo import *
from .qlearning import *
from .sarsa import *

__all__ = montecarlo.__all__ + qlearning.__all__ + sarsa.__all__
