import random
import copy
from typing import Dict, List
from CromosomaBin import CromosomaBinario
from interfaz import Aptitud, Seleccion, Reproduccion, Mutacion, AptitudMochila, SeleccionTorneo, ReproduccionUnCruce, MutacionFlip

class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaBinario] = [CromosomaBinario(num_genes) for i in range(num_inv)]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar: Aptitud = hyperparams["Aptitud"]
        self.seleccion: Seleccion = hyperparams["Seleccion"]
        self.reproducion: Reproduccion = hyperparams["Reproduccion"]
        self.mutacion: Mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por-remplazo"] * num_inv)
        
        # inicializar
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, tolerancia: float, var: dict):
        fail = False
        fail_count = 0
        fail_score = 0.0
        generacion = 1
        sum_score_prev = 1.0

        while not fail:
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            sum_scores = sum(c.score for c in self.poblacion) / self.num_inv
            avg_tol = abs(sum_scores - sum_score_prev) / sum_scores if sum_scores != 0 else 0
            sum_score_prev = sum_scores

            if (generacion + 1) % 100 == 0:
                print(f"[{generacion:04d}] avg_err: {avg_tol:.5f} avg_score: {sum_scores:.5f}")

            if fail_score == avg_tol:
                fail_count += 1
            if fail_count >= 10 or avg_tol < tolerancia:
                break

            fail_score = avg_tol
            self.poblacion.sort(key=lambda c: c.score, reverse=True)
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)

            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)

def sum_pesos_test(pesos, cromosoma):
    s = 0
    for i in range(cromosoma.n_genes):
        if cromosoma.get_bits(i + 1) == 1:
            s += pesos[i]
    return s

if __name__ == "__main__":
    random.seed(42)  # Para resultados reproducibles
    variables = {
        "pesos": [random.randint(5, 20) for _ in range(32)],    # Pesos de 5 a 20 kg
        "valores": [random.randint(50, 200) for _ in range(32)], # Valores de $50 a $200
        "capacidad": 120  # Capacidad ajustada a 120 kg
    }

    num_genes = 32
    cant_ind = 100

    gaparams = {
        "Aptitud": AptitudMochila(),
        "Seleccion": SeleccionTorneo(3),
        "Reproduccion": ReproduccionUnCruce(num_genes),
        "Mutacion": MutacionFlip(0.05),
        "por-remplazo": 0.5
    }

    algoritmoGA = GA(cant_ind, num_genes, gaparams)
    algoritmoGA.buscar_solucion(1e-4, variables)

    solucion = sorted(algoritmoGA.poblacion, key=lambda c: c.score, reverse=True)
    print("\n--- MEJORES SOLUCIONES ---")
    for i, cromosoma in enumerate(solucion[:10]):
        bits_str = bin(cromosoma.bits)[2:].zfill(32)
        peso_usado = sum_pesos_test(variables["pesos"], cromosoma)
        print(f"[{i+1:02d}] {bits_str} | Score: {cromosoma.score} | Peso: {peso_usado} kg")