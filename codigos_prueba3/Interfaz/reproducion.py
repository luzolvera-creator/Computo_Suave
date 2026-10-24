import random
from abc import ABC, abstractmethod
from CromosomaDecimal import CromosomaDecimal

class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: list, potencial: list, max_poblacion: int) -> None:
        pass

class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes: int = 6):
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion: list, potencial: list, max_poblacion: int) -> None:
        curr_p = len(poblacion)
        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres = 0

        while curr_p < max_poblacion:
            hijoA = CromosomaDecimal(self.num_genes)
            hijoB = CromosomaDecimal(self.num_genes)
            punto_cruce = random.randint(1, self.num_genes - 1)

            padreA = ord_padres[id_padres % len(ord_padres)]
            padreB = ord_padres[(id_padres + 1) % len(ord_padres)]

            for i in range(1, self.num_genes + 1):
                if i <= punto_cruce:
                    hijoA.set_bits(padreA.get_bits(i), i)
                    hijoB.set_bits(padreB.get_bits(i), i)
                else:
                    hijoA.set_bits(padreB.get_bits(i), i)
                    hijoB.set_bits(padreA.get_bits(i), i)

            if curr_p < max_poblacion:
                poblacion.append(hijoA)
                curr_p += 1
            if curr_p < max_poblacion:
                poblacion.append(hijoB)
                curr_p += 1
            id_padres += 1