from abc import ABC, abstractmethod
from typing import Dict

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: list, var: Dict = None):
        pass

class AptitudReinas(Aptitud):
    def evaluar_poblacion(self, poblacion: list, var: Dict = None):
        for cromosoma in poblacion:
            reinas = cromosoma.obtener_reinas()
            conflictos = 0
            
            # Comparar cada par de reinas (columna i y columna j)
            for i in range(len(reinas)):
                for j in range(i + 1, len(reinas)):
                    # Misma fila
                    if reinas[i] == reinas[j]:
                        conflictos += 1
                        
                    # Misma diagonal: |fila_i - fila_j| == |col_i - col_j|
                    elif abs(reinas[i] - reinas[j]) == abs(i - j):
                        conflictos += 1
            
            # f(x) = 28 - C(x)
            cromosoma.score = 28 - conflictos