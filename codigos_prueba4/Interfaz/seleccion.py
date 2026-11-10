import random
from abc import ABC, abstractmethod

class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: list) -> list:
        pass

# Selección por Ruleta
class SeleccionRuleta(Seleccion):
    def seleccion_poblacion(self, poblacion: list) -> list:
        # Suma las puntuaciones de fitness de toda la población
        suma_fitness = sum(c.score for c in poblacion)
        
        # 0 = individuos random
        if suma_fitness == 0:
            return random.choices(poblacion, k=len(poblacion))

        # Probabilidad según el fitness
        pesos = [c.score / suma_fitness for c in poblacion]
        
        # Ruleta para escoger los ganadores
        ganadores = random.choices(poblacion, weights=pesos, k=len(poblacion))
        
        return ganadores
    
    