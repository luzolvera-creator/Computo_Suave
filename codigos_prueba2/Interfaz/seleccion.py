import random
from abc import ABC, abstractmethod

class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: list) -> list:
        pass

# Selección por Torneo 
class SeleccionTorneo(Seleccion):
    def __init__(self, participantes: int = 3):
        # 3 individuos por torneo
        self.participantes = participantes

    def seleccion_poblacion(self, poblacion: list) -> list:
        ganadores = []
        
        # Torneo por cada posición a llenar en la nueva población de padres
        for _ in range(len(poblacion)):
            # Competidores random de la población
            competidores = random.sample(poblacion, self.participantes)
            
            # Ganador = mayor fitness --> Menos conflictos
            ganador = max(competidores, key=lambda c: c.score)
            
            # Ganador al grupo de padres seleccionados
            ganadores.append(ganador)
            
        return ganadores
    
    