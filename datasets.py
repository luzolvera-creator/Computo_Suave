#%% MOTOR DE LA RED NEURONAL

import numpy as np
import pandas as pd

# Funciones de activación y sus derivadas

def sigmoid(x):
    x = np.clip(x, -500, 500) # Para el desbordamiento
    return 1 / (1 + np.exp(-x))   

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)         

def relu(x):
    return np.maximum(0, x)     

def d_relu(x):
    return np.where(x > 0, 1.0, 0.0)  

def leakyrelu(x):
    return np.maximum(0.01 * x, x)    

def d_leakyrelu(x):
    return np.where(x > 0, 1.0, 0.01) 

def softmax(entradas):
    exponenciales = np.exp(entradas)
    suma = np.sum(exponenciales)
    return exponenciales / suma 

def lineal(x):
    return x

def d_lineal(x):
    return np.ones_like(x)

# Clase Capa --> Una sola capa individual
class Capa:

    def __init__(self, entradas, neuronas, activacion, d_activacion, bias):

        self.neuronas = neuronas
        self.activacion = activacion
        self.d_activacion = d_activacion
        self.bias = bias

        # Inicialización de pesos
        if bias:
            self.pesos = np.random.uniform(-0.5, 0.5, (entradas + 1, neuronas))
        else:
            self.pesos = np.random.uniform(-0.5, 0.5, (entradas, neuronas))

        self.entrada = None
        self.a = None

    def FeedForward(self, entrada):
        # Matriz aumentada para el Bias
        if self.bias:
            if entrada.ndim == 1:
                self.entrada = np.append(entrada, 1.0)
            else:
                col_ones = np.ones((entrada.shape[0], 1))
                self.entrada = np.hstack([entrada, col_ones])
        else:
            self.entrada = entrada

        self.a = np.dot(self.entrada, self.pesos) # Entrada * Pesos (X*W)
        resultado = self.activacion(self.a) # Aplicación de la función de activación
        return resultado


# Función FeedForward para recorrer el arreglo de capas
def FeedForward(capas, X):
    a = X
    for capa in capas:
        a = capa.FeedForward(a)  # a . W^c y F(a) 
    return a


# Función Backpropagation para recorrer las capas hacia atras
def Backpropagation(capas, T, YH, alpha):
    deltas = [None] * len(capas)

    for i in reversed(range(len(capas))):
        
        # Si la capa es la última entonces:
        if i == len(capas) - 1:
            de_dyh = -(T - YH)
            deltas[i] = de_dyh * capas[i].d_activacion(capas[i].a)  # delta = (de/dy) * F'(capa)
            
        # de lo contrario:
        else:
            pesos_sig = capas[i + 1].pesos
            if capas[i + 1].bias:
                pesos_sig = pesos_sig[:-1, :]  # Quitamos pesos de bias para propagar el error hacia atrás
            
            deltas[i] = np.dot(deltas[i + 1], pesos_sig.T) * capas[i].d_activacion(capas[i].a)  # delta = delta^(capa+1) . (W^(capa+1))^T * F'(a)

        gradiente_W = np.dot(capas[i].entrada.T, deltas[i]) # gradiente_W = (entrada)^T . delta

        capas[i].pesos = capas[i].pesos - alpha * gradiente_W # W_nueva = W - alpha * gradiente

        
#%% DATASET 1 BANKNOTE AUTHENTICATION
        
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from ucimlrepo import fetch_ucirepo

print("-" * 50)
print("\n--- DATASET 1: BANKNOTE AUTHENTICATION ---")

# Carga de datos 
banknote_authentication = fetch_ucirepo(id=267)

# Convertimos a matrices 
X = banknote_authentication.data.features.values
y = banknote_authentication.data.targets.values.reshape(-1, 1)

# Separación 60% Entrenamiento 40% Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

# Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Arquitectura de la red neuronal
capas_banknote = [
    Capa(entradas=4, neuronas=8, activacion=relu, d_activacion=d_relu, bias=True),
    Capa(entradas=8, neuronas=1, activacion=sigmoid, d_activacion=d_sigmoid, bias=True)
]

# Comprobación de la Pasada Hacia Adelante (Inicial)
YH_inicial = FeedForward(capas_banknote, X_train)
error_inicial = np.mean(np.abs(y_train - YH_inicial))

print("\n-- Pasada hacia delante --")
print("Predicción YH Inicial (primeras 3 muestras):\n", YH_inicial[:3])
print("Error Inicial (MAE):", error_inicial)

# Entrenamiento
alpha = 0.001 
epocas = 10

print("\n-- Entrenando por Épocas --")
for epoca in range(epocas):
    YH = FeedForward(capas_banknote, X_train) # Entradas y predicción calculadas hacia adelante
    Backpropagation(capas_banknote, T=y_train, YH=YH, alpha=alpha) # Comparación predicción/real = error hacía atrás, gradiante y ajuste de pesos
    mae_epoca = np.mean(np.abs(y_train - YH)) # MAE
    print(f"Época {epoca+1} | Error MAE: {mae_epoca:.4f}")

# Comprobación Final (Entrenado)
YH_train_final = FeedForward(capas_banknote, X_train)
error_final = np.mean(np.abs(y_train - YH_train_final))

print("\n-- Pasada hacia atrás --")
print("Predicción YH Final (primeras 3 muestras):\n", YH_train_final[:3])
print("Error Final (MAE):", error_final)

# Evaluación de Métricas

# Pasamos los datos de prueba por la red entrenada
YH_test = FeedForward(capas_banknote, X_test)

# Predicción > 0.5 --> Billete real (1)
# Predicción <= 0.5 --> Billete falso (2)
y_pred = (YH_test > 0.5).astype(int)

print("\n--- RESULTADOS FINALES ---")

# Porcentaje total de billetes clasificados correctamente
print(f"Exactitud: {accuracy_score(y_test, y_pred):.4f}")

# De todos los billetes que la red predijo como Reales cuántos lo eran de verdad
print(f"Precisión: {precision_score(y_test, y_pred):.4f}")

# Capacidad de la red para distinguir entre billetes reales y falsos
print(f"ROC-AUC: {roc_auc_score(y_test, YH_test):.4f}")


#%% DATASET 2 AUTO MPG

from sklearn.metrics import r2_score
from ucimlrepo import fetch_ucirepo

print("-" * 50)
print("\n--- DATASET 2: AUTO MPG ---")

# Carga de datos 
auto_mpg = fetch_ucirepo(id=9)

# Extraemos las características y objetivos como DataFrames de Pandas temporales
X_df = auto_mpg.data.features
y_df = auto_mpg.data.targets

# Unimos para eliminar filas que contengan datos faltantes (NaN)
df = pd.concat([X_df, y_df], axis=1).dropna()

X = df.drop('mpg', axis=1).values
y = df['mpg'].values.reshape(-1, 1)

# Separación 60% Entrenamiento 40% Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

# Normalización
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Arquitectura de la red neuronal 
capas_mpg = [
    Capa(entradas=7, neuronas=16, activacion=relu, d_activacion=d_relu, bias=True),
    Capa(entradas=16, neuronas=1, activacion=lineal, d_activacion=d_lineal, bias=True)
]

# Comprobación de la Pasada Hacia Adelante (Inicial)
YH_inicial = FeedForward(capas_mpg, X_train)
error_inicial = np.mean(np.abs(y_train - YH_inicial))

print("\n-- Pasada hacia delante --")
print("Predicción YH Inicial (primeras 3 muestras):\n", YH_inicial[:3])
print("Error Inicial (MAE):", error_inicial)

# 5. Entrenamiento
alpha = 0.0001 
epocas = 10

print("\n-- Entrenando por Épocas --")
for epoca in range(epocas):
    YH = FeedForward(capas_mpg, X_train)
    Backpropagation(capas_mpg, T=y_train, YH=YH, alpha=alpha)
    mae_epoca = np.mean(np.abs(y_train - YH))
    print(f"Época {epoca+1} | Error MAE: {mae_epoca:.4f}")

# Comprobación Final (Entrenado)
YH_train_final = FeedForward(capas_mpg, X_train)
error_final = np.mean(np.abs(y_train - YH_train_final))

print("\n-- Pasada hacia atrás --")
print("Predicción YH Final (primeras 3 muestras):\n", YH_train_final[:3])
print("Error Final (MAE):", error_final)

# Evaluación de Métricas

# Calculo de las predicciones numéricas para los autos del conjunto de prueba
YH_test = FeedForward(capas_mpg, X_test)

print("\n--- RESULTADOS FINALES ---")

# Cercano a 1.0 --> Predicción casi perfecta de las millas por galón
# Cercano a 0.0 o negativo --> La red no logró aprender la relación
print(f"Coeficiente de determinación (R^2): {r2_score(y_test, YH_test):.4f}")


#%% DATASET 3 STUDENTS DROPOUT
from ucimlrepo import fetch_ucirepo

print("-" * 50)
print("\n--- DATASET 3: STUDENTS DROPOUT ---")

# Carga de datos 
predict_students_dropout = fetch_ucirepo(id=697)

X = predict_students_dropout.data.features.values

# Sacamos las etiquetas y las aplanamos
y_labels = predict_students_dropout.data.targets.values.ravel() 

# Convertimos las clases a 0 y 1
y = pd.get_dummies(y_labels).astype(int).values 

# Separación 60% Entrenamiento 40% Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

# Normalización
scaler_stud = StandardScaler()
X_train = scaler_stud.fit_transform(X_train)
X_test = scaler_stud.transform(X_test)

# Arquitectura de la red neuronal
capas_students = [
    Capa(entradas=X_train.shape[1], neuronas=16, activacion=relu, d_activacion=d_relu, bias=True),
    Capa(entradas=16, neuronas=3, activacion=sigmoid, d_activacion=d_sigmoid, bias=True)
]

# Comprobación de la Pasada Hacia Adelante (Inicial)
YH_inicial = FeedForward(capas_students, X_train)
error_inicial = np.mean(np.abs(y_train - YH_inicial))

print("\n-- Pasada hacia delante --")
print("Predicción YH Inicial (primeras 3 muestras):\n", YH_inicial[:3])
print("Error Inicial (MAE):", error_inicial)

# Entrenamiento
alpha = 0.0005 
epocas = 10

print("\n-- Entrenando por Épocas --")
for epoca in range(epocas):
    YH = FeedForward(capas_students, X_train)
    Backpropagation(capas_students, T=y_train, YH=YH, alpha=alpha)
    mae_epoca = np.mean(np.abs(y_train - YH))
    print(f"Época {epoca+1} | Error MAE: {mae_epoca:.4f}")

# Comprobación Final (Entrenado)
YH_train_final = FeedForward(capas_students, X_train)
error_final = np.mean(np.abs(y_train - YH_train_final))

print("\n-- Pasada hacia atrás --")
print("Predicción YH Final (primeras 3 muestras):\n", YH_train_final[:3])
print("Error Final (MAE):", error_final)

# Evaluación de Métricas

# Calculamos la probabilidad de cada una de las 3 clases para cada estudiante de prueba
YH_test = FeedForward(capas_students, X_test)

# Índice de la neurona con el valor más alto (clase ganadora)
y_pred_classes = np.argmax(YH_test, axis=1) 

# Convertimos las etiquetas para la comparación
y_test_classes = np.argmax(y_test, axis=1)

print("\n--- RESULTADOS FINALES ---")

# Porcentaje total de estudiantes asignados a su categoría correcta
print(f"Exactitud: {accuracy_score(y_test_classes, y_pred_classes):.4f}")

# Promedio de la precisión individual calculada para las 3 clases
print(f"Precisión: {precision_score(y_test_classes, y_pred_classes, average='macro'):.4f}")

# Evaluación del modelo promediando la separación de las 3 clases
print(f"ROC-AUC: {roc_auc_score(y_test, YH_test, average='macro'):.4f}")
