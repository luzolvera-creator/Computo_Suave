import random
from typing import Dict

class CromosomaOctal:
    def __init__(self, genes: int = 8):
        self.n_genes = genes
        self.score = -1
        self.gen_width = 3  # 3 bits por gen (0 a 7)
        self.mascaras: Dict = {}
        self.bits = 0x00000000
        
        max_value = (1 << self.gen_width) - 1
        for g, i in enumerate(reversed(range(0, 24, self.gen_width))):
            self.mascaras["p" + str(g + 1)] = i
            self.mascaras["m" + str(g + 1)] = max_value << i

    def llenar_random_data(self):
        max_value = (1 << self.gen_width) - 1
        for i in range(self.n_genes):
            valor = random.randint(0, max_value)
            self.set_bits(valor, i + 1)

    def get_bits(self, gen: int) -> int:
        return (self.bits & self.mascaras["m" + str(gen)]) >> self.mascaras["p" + str(gen)]

    def set_bits(self, valor: int, gen: int) -> None:
        self.bits = (self.bits & ~self.mascaras["m" + str(gen)]) | (valor << self.mascaras["p" + str(gen)])

    def obtener_reinas(self) -> list:
        return [self.get_bits(i) for i in range(1, self.n_genes + 1)]

    def dibujar_tablero_texto(self) -> str:
        """Genera una representación visual en formato texto del tablero 8x8."""
        reinas = self.obtener_reinas()
        lineas = ["  C0 C1 C2 C3 C4 C5 C6 C7"]
        for fila in range(8):
            linea = f"F{fila} "
            for col in range(8):
                if reinas[col] == fila:
                    linea += " R "
                else:
                    linea += " . "
            lineas.append(linea)
        return "\n".join(lineas)
    
    