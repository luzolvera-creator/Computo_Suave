import random
from typing import Dict

class CromosomaDecimal:
    def __init__(self, genes: int = 6):
        self.n_genes = genes
        self.score = -1.0
        self.gen_width = 3  # 3 bits por gen --> 1 por procesador
        self.mascaras: Dict = {}
        self.bits = 0x00000000
        
        max_value = (1 << self.gen_width) - 1
        for g, i in enumerate(reversed(range(0, self.n_genes * self.gen_width, self.gen_width))):
            self.mascaras["p" + str(g + 1)] = i
            self.mascaras["m" + str(g + 1)] = max_value << i

    def llenar_random_data(self):
        for i in range(self.n_genes):
            valor = random.randint(1, 3)  # Procesadores 1, 2 o 3
            self.set_bits(valor, i + 1)

    def get_bits(self, gen: int) -> int:
        return (self.bits & self.mascaras["m" + str(gen)]) >> self.mascaras["p" + str(gen)]

    def set_bits(self, valor: int, gen: int) -> None:
        self.bits = (self.bits & ~self.mascaras["m" + str(gen)]) | (valor << self.mascaras["p" + str(gen)])

    def obtener_asignacion(self) -> list:
        return [self.get_bits(i) for i in range(1, self.n_genes + 1)] #Lsita con el procesador asignado a cada tarea
    
    