import random
from typing import Dict

class CromosomaBinario:
    def __init__(self, genes: int):
        self.n_genes = genes
        self.score = -1
        self.gen_width = 32 // genes  # ancho por gen
        self.mascaras: Dict = {}
        self.bits = 0x00000000
        
        max_value = 2**self.gen_width - 1
        for g, i in enumerate(reversed(range(0, 32, self.gen_width))):
            self.mascaras["p" + str(g + 1)] = i
            self.mascaras["m" + str(g + 1)] = max_value << i

    def llenar_random_data(self):
        max_value = 2**self.gen_width - 1
        for i in range(self.n_genes):
            valor = random.randint(0, max_value)
            self.set_bits(valor, i + 1)

    def get_bits(self, gen: int) -> int:
        return (self.bits & self.mascaras["m" + str(gen)]) >> self.mascaras["p" + str(gen)]

    def set_bits(self, valor: int, gen: int) -> None:
        self.bits = (self.bits & ~self.mascaras["m" + str(gen)]) | (valor << self.mascaras["p" + str(gen)])