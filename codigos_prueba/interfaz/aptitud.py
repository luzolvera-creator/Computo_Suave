from abc import ABC, abstractmethod
from typing import List, Dict

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: list, var: Dict):
        pass

class AptitudMochila(Aptitud):
    def evaluar_poblacion(self, poblacion: list, var: Dict):
        for cromosoma in poblacion:
            peso_total = 0
            valor_total = 0
            pesos = var["pesos"]
            valores = var["valores"]
            capacidad_maxima = var["capacidad"]
            factor_penalizacion = 70  # %

            for i in range(1, cromosoma.n_genes + 1):
                if cromosoma.get_bits(i) == 1:
                    peso_total += pesos[i - 1]
                    valor_total += valores[i - 1]

            if peso_total > capacidad_maxima:
                penalizacion = (peso_total - capacidad_maxima) * factor_penalizacion
                score = valor_total - penalizacion
                cromosoma.score = max(0, score)
            else:
                cromosoma.score = valor_total