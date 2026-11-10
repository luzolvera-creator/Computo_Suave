import random
from abc import ABC, abstractmethod

class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: list):
        pass

class MutacionDecimal(Mutacion):
    def __init__(self, probabilidad: float = 0.05):
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: list):
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                if random.random() < self.prob:
                    nuevo_procesador = random.randint(1, 3)
                    crom.set_bits(nuevo_procesador, i)