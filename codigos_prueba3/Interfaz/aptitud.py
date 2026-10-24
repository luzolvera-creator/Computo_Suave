from abc import ABC, abstractmethod
from typing import Dict

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: list, var: Dict):
        pass

class AptitudProcesadores(Aptitud):
    def evaluar_poblacion(self, poblacion: list, var: Dict):
        tiempos = var["tiempos"]  # Tiempos de las tareas 
        
        for cromosoma in poblacion:
            asignacion = cromosoma.obtener_asignacion()
            cargas = [0, 0, 0]  # Tiempos acumulados para los 3 procesos
            
            for idx_tarea, proc in enumerate(asignacion):
                cargas[proc - 1] += tiempos[idx_tarea]
            
            # M(x) es el máximo tiempo de carga entre los procesadores
            max_carga = max(cargas)
            
            # f(x) = 1 / (1 + M(x))
            cromosoma.score = 1.0 / (1.0 + max_carga)
            
            