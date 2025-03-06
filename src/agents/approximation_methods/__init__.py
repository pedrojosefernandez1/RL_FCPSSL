"""
Módulo de Métodos Aproximados
========================
Este submódulo agrupa los métodos aproximados de aprendizaje por refuerzo,
que incluyen Deep Q-Learning y SARSA-Gradiente. 
"""
from .sarsa import *
from .tile import IHT, tiles, TileCodingFeatureExtractor

__all__ = sarsa.__all__ + ['IHT', 'tiles', TileCodingFeatureExtractor]
