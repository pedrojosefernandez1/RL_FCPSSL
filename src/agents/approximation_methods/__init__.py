"""
Módulo de Métodos Aproximados
========================
Este submódulo agrupa los métodos aproximados de aprendizaje por refuerzo,
que incluyen Deep Q-Learning y SARSA-Gradiente. 
"""
from .sarsa import *
from .tiles3 import IHT, hashcoords, tiles,tileswrap

__all__ = sarsa.__all__ + ['IHT', 'hashcoords', 'tiles', 'tileswrap']
