import numpy as np

class IHT:
    """Tabla hash de índices para tile coding."""
    def __init__(self, size):
        self.size = size
        self.dictionary = {}
        self.overfullCount = 0

    def getindex(self, obj, readonly=False):
        if obj in self.dictionary:
            return self.dictionary[obj]
        elif readonly:
            return None
        else:
            if len(self.dictionary) >= self.size:
                self.overfullCount += 1
                return hash(obj) % self.size
            else:
                index = len(self.dictionary)
                self.dictionary[obj] = index
                return index

def tiles(iht, num_tilings, floats):
    """
    Calcula los índices activados para un vector de entrada "floats" usando 'num_tilings'
    rejillas (tilings). Se aplica un pequeño desplazamiento en cada tiling.
    """
    qfloats = [int(np.floor(f * num_tilings)) for f in floats]
    tiles_indices = []
    for tiling in range(num_tilings):
        coords = [tiling]
        for f in qfloats:
            # Se suma el tiling como offset para desplazar cada rejilla ligeramente.
            coords.append((f + tiling) // num_tilings)
        tiles_indices.append(iht.getindex(tuple(coords)))
    return tiles_indices

class TileCodingFeatureExtractor:
    """
    Convierte un estado continuo en un vector de características disperso usando tile coding.
    
    Parámetros:
      - num_tilings: número de rejillas superpuestas.
      - low, high: límites inferior y superior para cada dimensión del estado.
      - iht_size: tamaño de la tabla hash (debe ser potencia de 2).
    """
    def __init__(self, num_tilings=8, low=None, high=None, iht_size=2048):
        self.num_tilings = num_tilings
        self.low = np.array(low)
        self.high = np.array(high)
        self.iht = IHT(iht_size)
        # Factor de escala para mapear cada dimensión al número de tilings.
        self.scale = self.num_tilings / (self.high - self.low)

    def __call__(self, state):
        state = np.array(state)
        # Escala el estado y lo desplaza para cada tiling.
        scaled_state = (state - self.low) * self.scale
        indices = tiles(self.iht, self.num_tilings, scaled_state)
        feature_vector = np.zeros(self.iht.size)
        feature_vector[indices] = 1  # Activa los índices correspondientes.
        return feature_vector