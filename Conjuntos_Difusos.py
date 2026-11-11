# VARIABLES LINGÚISTICAS

import numpy as np
import matplotlib.pyplot as plt

# Dominios Rangos
x_temp = np.linspace(0, 40, 100)      # Temperatura 0 a 40 °C
x_vel = np.linspace(0, 100, 100)      # Velocidad 0 a 100 km/h
x_propina = np.linspace(0, 30, 100)  # Propina 0 a 30 %

# Funciones de Membresía 
def trimf(x, a, b, c): # Triangular
    return np.maximum(0, np.minimum((x - a)/(b - a + 1e-6), (c - x)/(c - b + 1e-6)))

def trapmf(x, a, b, c, d): # Trapezoidal
    return np.maximum(0, np.minimum(np.minimum((x - a)/(b - a + 1e-6), 1), (d - x)/(d - c + 1e-6)))

def gaussmf(x, c, sigma): # Gaussiana
    return np.exp(-((x - c)**2) / (2 * sigma**2))

# Temperatura
temp_fria = trapmf(x_temp, 0, 0, 10, 18)
temp_templada = trimf(x_temp, 12, 22, 30)
temp_caliente = trapmf(x_temp, 25, 32, 40, 40)

# Velocidad
vel_lenta = trapmf(x_vel, 0, 0, 20, 40)
vel_moderada = gaussmf(x_vel, 50, 10)
vel_rapida = trapmf(x_vel, 60, 80, 100, 100)

# Propina
prop_baja = trimf(x_propina, 0, 5, 10)
prop_media = trimf(x_propina, 8, 15, 22)
prop_alta = trimf(x_propina, 20, 25, 30)

#%%

# EJEMPLOS DE LAS VARIABLES LINGÜISTICAS

plt.figure(figsize=(12, 12))

# Gráfica Temperatura
plt.subplot(3, 1, 1)
plt.plot(x_temp, temp_fria, label='Fría (Trapezoidal)', color='blue')
plt.plot(x_temp, temp_templada, label='Templada (Triangular)', color='green')
plt.plot(x_temp, temp_caliente, label='Caliente (Trapezoidal)', color='red')
plt.title('Temperatura')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Membresía μ(x)')
plt.legend()
plt.grid(True)

# Gráfica Velocidad
plt.subplot(3, 1, 2)
plt.plot(x_vel, vel_lenta, label='Lenta (Trapezoidal)', color='blue')
plt.plot(x_vel, vel_moderada, label='Moderada (Gaussiana)', color='orange')
plt.plot(x_vel, vel_rapida, label='Rápida (Trapezoidal)', color='red')
plt.title('Velocidad')
plt.xlabel('Velocidad (km/h)')
plt.ylabel('Membresía μ(x)')
plt.legend()
plt.grid(True)

# Gráfica Propina
plt.subplot(3, 1, 3)
plt.plot(x_propina, prop_baja, label='Baja (Triangular)', color='blue')
plt.plot(x_propina, prop_media, label='Media (Triangular)', color='green')
plt.plot(x_propina, prop_alta, label='Alta (Triangular)', color='purple')
plt.title('Propina')
plt.xlabel('Propina (%)')
plt.ylabel('Membresía μ(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%%

# OPERADORES

def union_difusa(A, B): # OR 
    return np.maximum(A, B)

def interseccion_difusa(A, B): # AND 
    return np.minimum(A, B)

def complemento_difuso(A): # NOT 
    return 1 - A

#%%

# EJEMPLOS DE OPERADORES

# Pruebas - Temperatura fría y templada
u = union_difusa(temp_fria, temp_templada)
i = interseccion_difusa(temp_fria, temp_templada)
c = complemento_difuso(temp_fria)

plt.figure(figsize=(10, 9))

# Gráfica Unión
plt.subplot(3, 1, 1)
plt.plot(x_temp, temp_fria, 'b--', alpha=0.6, label='Temp. Fría')
plt.plot(x_temp, temp_templada, 'g--', alpha=0.6, label='Temp. Templada')
plt.plot(x_temp, u, 'r', linewidth=2.5, label='Unión: max(Fría, Templada)')
plt.title('Unión Difusa (Operador OR / Máximo)')
plt.ylabel('Membresía μ(x)')
plt.legend(loc='upper right')
plt.grid(True)

# Gráfica Intersección 
plt.subplot(3, 1, 2)
plt.plot(x_temp, temp_fria, 'b--', alpha=0.6, label='Temp. Fría')
plt.plot(x_temp, temp_templada, 'g--', alpha=0.6, label='Temp. Templada')
plt.plot(x_temp, i, 'purple', linewidth=2.5, label='Intersección: min(Fría, Templada)')
plt.title('Intersección Difusa (Operador AND / Mínimo)')
plt.ylabel('Membresía μ(x)')
plt.legend(loc='upper right')
plt.grid(True)

# Gráfica Complemento 
plt.subplot(3, 1, 3)
plt.plot(x_temp, temp_fria, 'b--', alpha=0.6, label='Temp. Fría (Original)')
plt.plot(x_temp, c, 'orange', linewidth=2.5, label='Complemento: 1 - Fría')
plt.title('Complemento Difuso (Operador NOT)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Membresía μ(x)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

#%%

# COMPOSICIÓN

def composicion_sup_min(R, S):
    R, S = np.array(R), np.array(S)
    T = np.zeros((R.shape[0], S.shape[1]))
    for r in range(R.shape[0]):
        for col in range(S.shape[1]):
            T[r, col] = np.max(np.minimum(R[r, :], S[:, col]))
    return T

R = [[1, 0], [0, 1]]
S = [[1, 0, 0], [1, 0, 1]]

print("\nComposición Sup-Min:")
print(composicion_sup_min(R, S))

#%%

# EJEMPLO T-Norma = Producto

def composicion_sup_producto(R, S):
    R, S = np.array(R), np.array(S)
    T = np.zeros((R.shape[0], S.shape[1]))
    for r in range(R.shape[0]):
        for col in range(S.shape[1]):
            T[r, col] = np.max(R[r, :] * S[:, col])
    return T

mu_C = [[0.9, 0.4, 0.1], [0.1, 0.4, 0.9]]
mu_MB = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.7]]

print("\nComposición Sup-Producto:")
print(composicion_sup_producto(mu_C, mu_MB))