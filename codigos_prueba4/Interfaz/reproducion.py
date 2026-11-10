import random
import copy
from abc import ABC, abstractmethod
from typing import List

class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, generacion_nueva: List, mating_pool: List, num_remplazo: int) -> List:
        pass

class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes: int):
        self.num_genes = num_genes

    def cruzar(self, padre1, padre2):
        # Cruce de un punto entre dos cromosomas 
        hijo1 = copy.deepcopy(padre1)
        hijo2 = copy.deepcopy(padre2)

        # Seleccionar un punto de corte aleatorio 
        if self.num_genes > 1:
            punto_corte = random.randint(1, self.num_genes - 1)
            
            # Intercambiar los genes a partir del punto de corte
            for i in range(punto_corte + 1, self.num_genes + 1):
                gen_p1 = padre1.get_bits(i)
                gen_p2 = padre2.get_bits(i)
                
                hijo1.set_bits(gen_p2, i)
                hijo2.set_bits(gen_p1, i)

        return hijo1, hijo2

    def reproduccion_poblacion(self, generacion_nueva: List, mating_pool: List, num_remplazo: int) -> List:
        
        # Conservar los mejores individuos (elitismo)
        hijos = copy.deepcopy(generacion_nueva)
        
        # Calcular cuántos descendientes faltan para completar la población
        num_padres = len(mating_pool)
        
        # Cruzar parejas hasta alcanzar el tamaño requerido
        i = 0
        while len(hijos) < num_padres and i < len(mating_pool):
            padre1 = mating_pool[i]
            
            # Seleccionar un segundo padre 
            padre2 = mating_pool[(i + 1) % len(mating_pool)]
            
            # Realizar el cruce
            h1, h2 = self.cruzar(padre1, padre2)
            
            # Agregar los descendientes generados
            hijos.append(h1)
            if len(hijos) < num_padres:
                hijos.append(h2)
                
            i += 2

        return hijos