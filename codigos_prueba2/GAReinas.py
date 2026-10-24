import random
import copy
from typing import Dict, List
from CromosomaOctal import CromosomaOctal
from Interfaz import Aptitud, Seleccion, Reproduccion, Mutacion, AptitudReinas, SeleccionTorneo, ReproduccionUnCruce, MutacionOctal

class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaOctal] = [CromosomaOctal(num_genes) for _ in range(num_inv)]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar: Aptitud = hyperparams["Aptitud"]
        self.seleccion: Seleccion = hyperparams["Seleccion"]
        self.reproducion: Reproduccion = hyperparams["Reproduccion"]
        self.mutacion: Mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por-remplazo"] * num_inv)
        
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, max_generaciones: int = 1000):
        generacion = 1

        while generacion <= max_generaciones:
            self.evaluar.evaluar_poblacion(self.poblacion)
            self.poblacion.sort(key=lambda c: c.score, reverse=True)
            
            mejor_individuo = self.poblacion[0]
            mejor_score = mejor_individuo.score
            sum_scores = sum(c.score for c in self.poblacion) / self.num_inv

            # Tablero del mejor individuo cada 10 generaciones o al llegar al mejor
            if generacion % 10 == 0 or mejor_score == 28:
                print(f"\n--------------------- Generación: {generacion:04d} ---------------------")
                print(f"Mejor Fitness: {mejor_score}/28 | Conflictos: {28 - mejor_score} | Promedio Pob: {sum_scores:.2f}\n")
                print(mejor_individuo.dibujar_tablero_texto())

            if mejor_score == 28:
                print(f"\nMejor solución en la generación: {generacion}")
                break

            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)

            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)


if __name__ == "__main__":
    random.seed(42)

    num_genes = 8
    cant_ind = 100

    gaparams = {
        "Aptitud": AptitudReinas(),
        "Seleccion": SeleccionTorneo(participantes=3),
        "Reproduccion": ReproduccionUnCruce(num_genes),
        "Mutacion": MutacionOctal(probabilidad=0.08),
        "por-remplazo": 0.5
    }

    algoritmoGA = GA(cant_ind, num_genes, gaparams)
    algoritmoGA.buscar_solucion(max_generaciones=500)

    solucion = sorted(algoritmoGA.poblacion, key=lambda c: c.score, reverse=True)[0]
    
    print("\n--------------------- Solución Final ---------------------")
    print(f"Fitness Score: {solucion.score}/28 (0 Conflictos)\n")
    print(solucion.dibujar_tablero_texto())