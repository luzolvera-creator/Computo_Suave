import random
import copy
from typing import Dict, List
from CromosomaDecimal import CromosomaDecimal
from Interfaz import (Aptitud, Seleccion, Reproduccion, Mutacion, AptitudProcesadores, SeleccionRuleta, ReproduccionUnCruce, MutacionDecimal)

class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaDecimal] = [CromosomaDecimal(num_genes) for _ in range(num_inv)]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar: Aptitud = hyperparams["Aptitud"]
        self.seleccion: Seleccion = hyperparams["Seleccion"]
        self.reproducion: Reproduccion = hyperparams["Reproduccion"]
        self.mutacion: Mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por-remplazo"] * num_inv)
        
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, tolerancia: float, var: dict):
        generacion = 1
        sum_score_prev = 1.0

        while generacion <= 100:
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            sum_scores = sum(c.score for c in self.poblacion) / self.num_inv
            avg_tol = abs(sum_scores - sum_score_prev) / sum_scores if sum_scores != 0 else 0
            sum_score_prev = sum_scores

            if generacion % 10 == 0 or generacion == 1:
                mejor = max(self.poblacion, key=lambda c: c.score)
                max_carga = (1.0 / mejor.score) - 1.0
                print(f"[{generacion:03d}] avg_score: {sum_scores:.5f} | Mejor Fitness: {mejor.score:.5f} | Carga M(x): {max_carga:.0f}")

            if avg_tol < tolerancia and generacion > 10:
                break

            self.poblacion.sort(key=lambda c: c.score, reverse=True)
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)

            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)

def mostrar_resultado(cromosoma: CromosomaDecimal, tiempos: List[int]):
    asignacion = cromosoma.obtener_asignacion()
    cadenas_crom = "".join(str(x) for x in asignacion)
    
    procesadores = {1: [], 2: [], 3: []}
    tiempos_proc = {1: 0, 2: 0, 3: 0}
    
    for idx, proc in enumerate(asignacion):
        tarea = f"T{idx+1}"
        procesadores[proc].append(tarea)
        tiempos_proc[proc] += tiempos[idx]

    print("\n-------- Solución --------")
    print(f"Cromosoma Decimal: {cadenas_crom}")
    print(f"Fitness f(x): {cromosoma.score:.5f}\n")
    print("-" * 50)
    print(f"{'Procesador':<12} | {'Tareas':<15} | {'Tiempo Total'}")
    print("-" * 50)
    
    for p in range(1, 4):
        tareas_str = ", ".join(procesadores[p]) if procesadores[p] else "Ninguna"
        detalles_tiempo = " + ".join(str(tiempos[int(t[1:])-1]) for t in procesadores[p]) if procesadores[p] else "0"
        print(f"P{p:<11} | {tareas_str:<15} | {tiempos_proc[p]} ({detalles_tiempo})")
    
    max_carga = max(tiempos_proc.values())
    print("-" * 50)
    print(f"\nCarga Máxima M(x): {max_carga}")

if __name__ == "__main__":
    random.seed(42)

    variables = {
        "tiempos": [8, 5, 7, 3, 6, 4]
    }

    num_genes = 6
    cant_ind = 50

    gaparams = {
        "Aptitud": AptitudProcesadores(),
        "Seleccion": SeleccionRuleta(),
        "Reproduccion": ReproduccionUnCruce(num_genes),
        "Mutacion": MutacionDecimal(0.05),
        "por-remplazo": 0.5
    }

    algoritmoGA = GA(cant_ind, num_genes, gaparams)
    algoritmoGA.buscar_solucion(1e-5, variables)

    solucion = sorted(algoritmoGA.poblacion, key=lambda c: c.score, reverse=True)[0]
    mostrar_resultado(solucion, variables["tiempos"])