import numpy as np

# Funciones de activación y sus derivadas
def sigmoid(x):
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

        self.a = np.dot(self.entrada, self.pesos) # Entrada * Pesos
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


# Comprobación
X = np.ones((3, 2)) * 5 # Matriz de 3x2 llena de 5s
y = np.zeros((3, 2)) # Matriz de 3x2 llena de 0s

# Capa profunda
capas = [Capa(entradas=2, neuronas=4, activacion=sigmoid, d_activacion=d_sigmoid, bias=True),
    Capa(entradas=4, neuronas=2, activacion=sigmoid, d_activacion=d_sigmoid, bias=True)]

# Comprobación de la Pasada Hacia Adelante 
YH_inicial = FeedForward(capas, X)
error_inicial = np.mean((y - YH_inicial) ** 2)

print("-- Pasada hacia delante --\nPredicción YH Inicial:\n", YH_inicial)
print("\nError Inicial (MAE):", error_inicial)

# Pasadas hacia adelante y Atrás iterativas
alpha = 0.1 # Tasa de aprendizaje
epocas = 2000 # Número de iteraciones de entrenamiento

for epoca in range(epocas):
    YH = FeedForward(capas, X) # Pasada hacia adelante
    Backpropagation(capas, T=y, YH=YH, alpha=alpha) # Pasada hacia atrás y ajuste de pesos

# Comprobación Final
YH_final = FeedForward(capas, X)
error_final = np.mean((y - YH_final) ** 2)

print("\n-- Pasada hacia atrás --\nPredicción YH Final:\n", YH_final)
print("\nError Final (MAE):", error_final)
