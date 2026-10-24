import random
from abc import ABC, abstractmethod

class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: list):
        pass

class MutacionOctal(Mutacion):
    def __init__(self, probabilidad: float = 0.05):
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: list):
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                if random.random() < self.prob:
                    nuevo_valor = random.randint(0, 7)
                    crom.set_bits(nuevo_valor, i)