import random
from typing import Dict

class CromosomaDecimal:
    def __init__(self, genes: int = 3):  # 3 para 3 capas ocultas
        self.n_genes = genes
        self.score = -1.0
        self.gen_width = 3  # 3 bits por gen 
        self.mascaras: Dict = {}
        self.bits = 0x00000000
        
        max_value = (1 << self.gen_width) - 1
        for g, i in enumerate(reversed(range(0, self.n_genes * self.gen_width, self.gen_width))):
            self.mascaras["p" + str(g + 1)] = i
            self.mascaras["m" + str(g + 1)] = max_value << i

    def llenar_random_data(self):
        for i in range(self.n_genes):
            valor = random.randint(0, 3)  # 0: ReLU, 1: Sigmoid, 2: Tanh, 3: Linear
            self.set_bits(valor, i + 1)

    def get_bits(self, gen: int) -> int:
        return (self.bits & self.mascaras["m" + str(gen)]) >> self.mascaras["p" + str(gen)]

    def set_bits(self, valor: int, gen: int) -> None:
        self.bits = (self.bits & ~self.mascaras["m" + str(gen)]) | (valor << self.mascaras["p" + str(gen)])

    def obtener_asignacion(self) -> list:
        # Lista con la función de activación asignada a cada capa
        return [self.get_bits(i) for i in range(1, self.n_genes + 1)]