import random
from abc import ABC, abstractmethod

class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: list):
        pass

class MutacionFlip(Mutacion):
    def __init__(self, probabilidad: float):
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: list):
        max_valor = 2**poblacion[0].gen_width - 1
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                if random.random() < self.prob:
                    valor = crom.get_bits(i)
                    valor = valor ^ max_valor  # Invierte el bit
                    crom.set_bits(valor, i)