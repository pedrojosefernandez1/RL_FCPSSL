"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List, Dict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# @title Funciones para mostrar los resultados

def plot_episode_rewards(list_stats):
  # Creamos una lista de índices para el eje x
  indices = list(range(len(list_stats)))

  # Creamos el gráfico
  plt.figure(figsize=(6, 3))
  plt.plot(indices, list_stats)

  # Añadimos título y etiquetas
  plt.title('Proporción de recompensas')
  plt.xlabel('Episodio')
  plt.ylabel('Proporción')

  # Mostramos el gráfico
  plt.grid(True)
  plt.show()

# Define la función para mostrar el tamaño de los episodios
# Pon aquí tu código.

def plot_len_episodes(episodes):
  # Creamos una lista de índices para el eje x

  # Creamos el gráfico
  plt.figure(figsize=(6, 3))
  plt.plot(    [len(episode) for episode in episodes]
)

  # Añadimos título y etiquetas
  plt.title('Longirud de episodios por T')
  plt.xlabel('Episodio T')
  plt.ylabel('Número de pasos del episodio')

  # Mostramos el gráfico
  plt.grid(True)
  plt.show()

def render_episode(actions_to_take, env, titulo="Camino encontrado"):
    done = False
    env.reset()
    renders = []
    renders.append(env.render())
    while not done or len(actions_to_take) > 0:
        action = actions_to_take.pop()
        next_state, reward, terminated, truncated, info = env.step(action)
        renders.append(env.render())
        done = terminated or truncated
    render_environment(renders, env.render_mode, titulo)

def render_environment(renders, render_mode="rgb_array", titulo="Camino encontrado"):
    if render_mode == "rgb_array":
        show_images_grid(renders, titulo=titulo)
    elif render_mode == "human":
        for render in renders:
            print(render)
        raise ("NOT IMPLEMENTED OSTRASSSSSS")
    elif render_mode == "ansi":
        for render in renders:
            print(render)

def show_images_grid(images, columns=3, titulo="Camino encontrado"):
    """
    Muestra una lista de imágenes en una cuadrícula con un número fijo de columnas.

    Parámetros:
    - images (list): Lista de imágenes en formato NumPy array (altura, ancho, canales).
    - columns (int): Número de columnas en la cuadrícula (por defecto 3).

    Retorna:
    - Muestra la cuadrícula de imágenes.
    """
    num_images = len(images)
    rows = (num_images + columns - 1) // columns  # Calcula el número de filas necesarias
    
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    fig.suptitle(titulo)
    axes = axes.flatten()  # Convierte la matriz de ejes en una lista para fácil acceso

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(f"Paso {i}")
            ax.axis("off")  # Oculta los ejes de la imagen
        else:
            ax.axis("off")  # Oculta los ejes de los espacios vacíos

    plt.tight_layout()
    plt.show()