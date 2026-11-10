import random
import copy
from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Importaciones del proyecto 
from CromosomaDecimal import CromosomaDecimal
from Interfaz.aptitud import AptitudRedNeuronal
from Interfaz.seleccion import SeleccionRuleta
from Interfaz.reproducion import ReproduccionUnCruce
from Interfaz.mutacion import MutacionDecimal


class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        # Instanciación de la población con CromosomaDecimal
        self.poblacion: List[CromosomaDecimal] = [
            CromosomaDecimal(num_genes) for i in range(num_inv)
        ]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar: AptitudRedNeuronal = hyperparams["Aptitud"]
        self.seleccion: SeleccionRuleta = hyperparams["Seleccion"]
        self.reproducion: ReproduccionUnCruce = hyperparams["Reproduccion"]
        self.mutacion: MutacionDecimal = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"] * num_inv)

        # Inicializar genes de la población aleatoriamente
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, tolerancia: float, var: Dict, max_generaciones: int = 15):
        avg_tol: float = 99999999.99      # Tolerancia/cambio promedio inicial 
        sum_score_prev: float = 0        # Fitness promedio de la generación anterior para comparar
        sum_scores: float = 0            # Suma y promedio del fitness de la población actual
        generacion: int = 1              # Contador de la generación en curso
        mating_pool = None               # Lista de padres seleccionados que se van a cruzar
        generacion_nueva: List = None    # Lista temporal para ir guardando a los hijos de la nueva generación
        fail: bool = False               # Bandera para detener el algoritmo si se estanca
        fail_count: int = 0              # Contador de cuántas generaciones seguidas lleva sin mejorar
        fail_score: float = 0.0          # Tolerancia anterior para detectar si se quedó congelada
        
        # Variable para registrar al mejor individuo de todas las generaciones
        campeon_historico = None
        
        # Diccionario auxiliar para traducir los genes a nombres entendibles
        malla_nombres = {0: 'ReLU', 1: 'Sigmoide', 2: 'Tanh', 3: 'Lineal'}

        while generacion <= max_generaciones and not fail:
            # Fase de Evaluación
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            sum_scores = float(sum([cromosoma.score for cromosoma in self.poblacion])) / self.num_inv

            # Cálculo de variación/error promedio
            if sum_score_prev > 0:
                avg_tol = float(abs(sum_scores - sum_score_prev) / sum_scores)
            sum_score_prev = sum_scores

            # Avance de cada generación
            mejor = max(self.poblacion, key=lambda c: c.score)
            
            # Guardar el mejor absoluto histórico
            if campeon_historico is None or mejor.score > campeon_historico.score:
                campeon_historico = copy.deepcopy(mejor)

            # Variación de fitness inicial
            variacion_str = " N/A " if generacion == 1 else f"{avg_tol:.5f}"

            # Impresiones por generación
            print(f"[Gen {generacion:02d}] "
                  f"CambioFitness: {variacion_str} | "
                  f"FitnessProm: {sum_scores:.5f} | "
                  f"Mejor Fitness: {campeon_historico.score:.5f}")

            # Control de estancamiento
            if abs(fail_score - avg_tol) < 1e-6 and generacion > 1:
                fail_count += 1
            else:
                fail_count = 0
            
            if fail_count >= 8:
                print("\n--> Detenido por estancamiento en la población.")
                break

            fail_score = avg_tol

            # Ordenar Población
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)

            # Fase de Selección
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)

            # Conservación por Elitismo
            generacion_nueva = copy.deepcopy(self.poblacion[: self.num_remplazo])

            # Fase de Reproducción
            hijos = self.reproducion.reproduccion_poblacion(
                generacion_nueva, mating_pool, self.num_remplazo
            )

            # Fase de Mutación
            self.mutacion.mutacion_poblacion(hijos)

            generacion += 1
            self.poblacion = copy.deepcopy(hijos)

        # Cálculo final de pérdida y precisión del campeón histórico
        loss_final = (1.0 / campeon_historico.score) - 1.0
        accuracy_final = (1.0 - loss_final) * 100.0
        genes_codificados = campeon_historico.obtener_asignacion()
        funciones_nombres = [malla_nombres.get(g, 'ReLU') for g in genes_codificados]

        # Resultado final 
        print("\n--Resultados:--")
        print(f"Códigos de Activaciones : {genes_codificados}")
        print(f"Funciones Recomendadas  : {funciones_nombres}")
        print(f"Mejor Fitness Alcanzado  : {campeon_historico.score:.5f}")
        print(f"Pérdida Mínima (Loss)    : {loss_final:.5f}")
        print(f"Precisión de Validación : {accuracy_final:.2f}%")


# Dataseet
if __name__ == "__main__":
    print("Descargando dataset Breast Cancer...")
    # Cargar y preprocesar dataset de UCI (id=17)
    breast_cancer = fetch_ucirepo(id=17)
    X = breast_cancer.data.features
    y = breast_cancer.data.targets
    
    # Llenar valores nulos si existieran con la media
    X = X.fillna(X.mean())

    # Para datos continuos numéricos usamos StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    y_encoded = LabelEncoder().fit_transform(y.values.ravel())

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    variables = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val
    }

    num_capas = 3        # 3 genes = optimizar 3 capas de activaciones
    cant_ind = 10       # Tamaño de la población
    num_clases = len(set(y_encoded)) # 2 clases (M, B)

    gaparams = {
        "Aptitud": AptitudRedNeuronal(num_clases=num_clases),
        "Seleccion": SeleccionRuleta(),
        "Reproduccion": ReproduccionUnCruce(num_capas),
        "Mutacion": MutacionDecimal(probabilidad=0.11),
        "por_remplazo": 0.5
    }

    # Ejecutar AG
    algoritmoGA = GA(cant_ind, num_capas, gaparams)
    algoritmoGA.buscar_solucion(tolerancia=1e-4, var=variables)