import numpy as np


# Funciones de activación

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leakyrelu(x):
    return np.maximum(0.01 * x, x)


def softmax(entradas):
    exponenciales = np.exp(entradas)
    suma = np.sum(exponenciales)
    return exponenciales / suma

# Clase para la capa

class Capa:

    def __init__(self, entradas, neuronas, activacion, bias):

        self.neuronas = neuronas
        self.activacion = activacion
        self.bias = bias

        if bias:
            self.pesos = np.zeros((entradas + 1, neuronas))
        else:
            self.pesos = np.zeros((entradas, neuronas))

    def FeedForward(self, entrada):
        resultado = np.dot(entrada, self.pesos)
        resultado = self.activacion(resultado)
        return resultado


# Crear capa
Capa1 = Capa(3, 4, relu, False)


# Pesos
Capa1.pesos = np.array([
    [0, 0, 0.1, 0.2],
    [0.3, -1, 0.4, 0.5],
    [1, 0.6, 0.7, 0.8]
])


# Entradas
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])


# FeedForward
C = Capa1.FeedForward(A)


print(C)
