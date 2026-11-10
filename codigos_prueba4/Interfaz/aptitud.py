from abc import ABC, abstractmethod
from typing import Dict
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: list, var: Dict):
        pass

class AptitudRedNeuronal(Aptitud):
    def __init__(self, num_clases: int = 4):
        # Mapeo de activación según el entero del cromosoma
        self.activaciones = {0: 'relu', 1: 'logistic', 2: 'tanh', 3: 'identity'}

    def evaluar_poblacion(self, poblacion: list, var: Dict):
        X_train, y_train = var["X_train"], var["y_train"]
        X_val, y_val = var["X_val"], var["y_val"]

        for cromosoma in poblacion:
            genes = cromosoma.obtener_asignacion()
            
            # Mapeo de la primera activación del cromosoma
            act = self.activaciones.get(genes[0], 'relu')

            # Para evaluar la combinación única de los 3 genes de las 3 capas,
            # usamos los genes como semilla determinista de inicialización de la red
            seed_cromosoma = int("".join(map(str, genes)))

            # Red neuronal ajustada para 30 variables de entrada
            model = MLPClassifier(
                hidden_layer_sizes=(16, 8, 4), # Capas ajustadas
                activation=act,
                max_iter=100,
                random_state=seed_cromosoma
            )
            
            # Entrenamiento
            model.fit(X_train, y_train)

            # Pérdida (Loss) basada en la precisión inversa de validación
            score_val = model.score(X_val, y_val)
            loss = 1.0 - score_val  # Menor pérdida = Mayor precisión

            # Aptitud f(x)
            cromosoma.score = 1.0 / (1.0 + loss)
            
            