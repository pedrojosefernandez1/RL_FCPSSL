# 🤖 Aprendizaje en entornos complejos
## ℹ️ Información
- **Alumnos:** Sendra Lázaro, Ricardo Javier; Pujante Saez, Jaime; Fernández Campillo, Pedro José;
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** FCPSSL

## 📖 Descripción

El objetivo del proyecto es analizar y comparar diferentes estrategias de aprendizaje por refuerzo en entornos simulados.  

- **Métodos tabulares**: Se utilizan para resolver problemas de menor dimensión, almacenando valores en tablas de valores de acción.  
  - Monte Carlo  
  - SARSA  
  - Q-Learning  

- **Métodos de aproximación**: Permiten manejar espacios de estado continuos mediante funciones aproximadas.  
  - SARSA Semigradiente  
  - Deep Q-Learning (DQN)  (No contemplado en los estudios)

Este trabajo busca entender cómo varía el desempeño de cada método según la complejidad del entorno y la capacidad de generalización del modelo.  

## 📂 Estructura del Proyecto

```
docs/                           # Archivos correspondientes con la documentacion
src/
│── agents/
│   ├── approximationmethods/   # Métodos de aproximación de funciones
│   ├── base/                   # Clases base para los agentes
│   ├── policies/               # Implementaciones de políticas de exploración/explotación
│   ├── tabularmethods/         # Métodos tabulares como Monte Carlo, SARSA y Q-Learning
│── plotting/                   # Funciones para visualización de resultados

Estudios principales:
├── MetodosAproximados.ipynb    # Análisis detallado de métodos de aproximación (SARSA semigradiente y DQN)
├── MetodosTabulares.ipynb      # Comparación de métodos tabulares (Monte Carlo, SARSA, Q-Learning)
├── GymnasiumMonteCarlo.ipynb   # Estudio específico y detallado del método Monte Carlo
```

Estos notebooks contienen análisis detallados de cada categoría de algoritmos, incluyendo implementación, comparación de rendimiento y observaciones clave sobre su desempeño en distintos entornos.

## ▶️ Ejecución
Para ejecutar los notebooks en Google Colab:
Dirigirse primero de todo al notebook **`main.ipynb`** donde se podra acceder a cualquier notebook que se encuentra en el repositorio de una manera intercativa. Podria seleccionar **`introduccion.ipynb`** para ver una breve introduccion al problema y enlace a los demas estudios.


## 🛠️ Tecnologías Utilizadas

- **Python** - Lenguaje principal del proyecto. 
- **Matplotlib** - Visualización de resultados.  
- **Gymnasium** - Entorno de simulación para entrenamiento de agentes.
- **PyTorch** - Implementación de redes neuronales para Deep Q-Learning.
- **Google Colab** - Para el desarrollo e interacción interactiva.

---

