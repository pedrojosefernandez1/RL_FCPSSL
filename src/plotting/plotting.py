"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Jaime Pujante Sáez
Email: jaime.pujantes@um.es

Author: Ricardo Javier Sendra Lázaro
Email: ricardojavier.sendral@um.es

Author: Pedro José Fernandez Campillo
Email: pedrojose.fernandez1@um.es

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List, Dict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# @title Funciones para mostrar los resultados

import numpy as np
import matplotlib.pyplot as plt

def agrupar_por_media(lista, n):
    # Convertir la lista en un array de numpy para facilitar operaciones
    array = np.array(lista)

    # Calcular la cantidad de grupos
    num_grupos = len(lista) // n

    # Crear la lista de medias usando reshape y mean
    medias = array[:num_grupos * n].reshape(-1, n).mean(axis=1)

    # Crear la lista para el eje X con saltos de `n`
    x_values = list(range(n, num_grupos * n + 1, n))

    return medias.tolist(), x_values

def plot_episode_rewards_subplot(list_stats, n_media=None, ax=None):
  indices = list(range(len(list_stats)))
  if n_media is not None:
     list_stats, indices = agrupar_por_media(list_stats, n=n_media)

  if ax is None:
    # Creamos una lista de índices para el eje x
    
    
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
  else:
    ax.plot(indices, list_stats)

    # Añadimos título y etiquetas
    ax.set_title('Proporción de recompensas')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Proporción')

    # Mostramos el gráfico
    ax.grid(True)

# Define la función para mostrar el tamaño de los episodios
# Pon aquí tu código.

def plot_len_episodes_subplot(episodes, n_media=None, ax=None):
  # Creamos una lista de índices para el eje x

  len_episodes = [len(episode) for episode in episodes]
  indices = range(len(len_episodes))
  if n_media is not None:
     len_episodes, indices = agrupar_por_media(len_episodes, n=n_media)

  if ax is None:
    # Creamos el gráfico
    plt.figure(figsize=(6, 3))
    plt.plot(indices, len_episodes)

    # Añadimos título y etiquetas
    plt.title('Longitud de episodios por T')
    plt.xlabel('Episodio T')
    plt.ylabel('Número de pasos del episodio')

    # Mostramos el gráfico
    plt.grid(True)
    plt.show()
  else:
    ax.plot(indices, len_episodes)

    ax.set_title('Longitud de episodios por T')
    ax.set_xlabel('Episodio T')
    ax.set_ylabel('Número de pasos del episodio')

    # Mostramos el gráfico
    ax.grid(True)

### Refactor

def plot_episode_rewards(list_stats, smoothing_window=50):
    """
    Genera una gráfica de la evolución de recompensas por episodio con opción de suavización.
    
    Args:
        list_stats (list): Lista de recompensas obtenidas por episodio.
        smoothing_window (int, opcional): Ventana para suavizar la curva mediante media móvil.
    """
    indices = np.arange(len(list_stats))
    
    if smoothing_window > 1:
        smoothed_rewards = np.convolve(list_stats, np.ones(smoothing_window) / smoothing_window, mode='valid')
        indices = indices[:len(smoothed_rewards)]
    else:
        smoothed_rewards = list_stats

    plt.figure(figsize=(8, 4))
    plt.plot(indices, smoothed_rewards, label="Recompensas por Episodio", color='b')
    plt.title('Recompensas por Episodio (Suavizado)')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_episode_proporcional_rewards(list_stats, smoothing_window=50):
    """
    Muestra la proporción acumulada de recompensas obtenidas a lo largo del entrenamiento.
    
    Args:
        list_stats (list): Lista de recompensas obtenidas.
        smoothing_window (int, opcional): Ventana de suavización con media móvil.
    """
    indices = np.arange(len(list_stats))
    proporciones = np.cumsum(list_stats) / (indices + 1)

    if smoothing_window > 1:
        smoothed_rewards = np.convolve(proporciones, np.ones(smoothing_window) / smoothing_window, mode='valid')
        indices = indices[:len(smoothed_rewards)]
    else:
        smoothed_rewards = proporciones

    plt.figure(figsize=(8, 4))
    plt.plot(indices, smoothed_rewards, label="Proporción de Recompensas", color='b')
    plt.title('Proporción Acumulada de Recompensas (Suavizado)')
    plt.xlabel('Episodio')
    plt.ylabel('Proporción de éxitos')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_len_episodes(episodes, smoothing_window=50):
    """
    Muestra la evolución de la longitud de los episodios con opción de suavización.
    
    Args:
        episodes (list): Lista donde cada elemento representa un episodio.
        smoothing_window (int, opcional): Ventana de suavización con media móvil.
    """
    episode_lengths = [len(episode) for episode in episodes]
    indices = np.arange(len(episode_lengths))

    if smoothing_window > 1:
        smoothed_lengths = np.convolve(episode_lengths, np.ones(smoothing_window) / smoothing_window, mode='valid')
        indices = indices[:len(smoothed_lengths)]
    else:
        smoothed_lengths = episode_lengths

    plt.figure(figsize=(8, 4))
    plt.plot(indices, smoothed_lengths, label="Longitud de Episodio", color='b')
    plt.title('Longitud de Episodios por T (Suavizado)')
    plt.xlabel('Episodio T')
    plt.ylabel('Número de Pasos')
    plt.legend()
    plt.grid(True)
    plt.show()


def render_episode(actions_to_take, env, titulo="Camino encontrado", seed=32):
    done = False
    env.reset(seed=seed)
    renders = []
    renders.append(env.render())
    while not done and len(actions_to_take) > 0:
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