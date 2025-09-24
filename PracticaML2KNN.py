"""
    Pérez Lucio Kevyn Alejandro
"""
import pandas as pd
import numpy as np
import csv

def calcular_distancia_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def calcular_distancia_manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2))

def kNN(X_train, y_train, X_test, k, tipo_distancia):
    predicciones = []

    for i in range(len(X_test)):
        distancias = []

        for j in range(len(X_train)):
            if tipo_distancia == 'euclidiana':
                dist = calcular_distancia_euclidiana(X_train[j], X_test[i])
            elif tipo_distancia == 'manhattan':
                dist = calcular_distancia_manhattan(X_train[j], X_test[i])
            else:
                raise ValueError("Tipo de distancia no válido")

            distancias.append((dist, y_train[j]))

        distancias.sort(key=lambda x: x[0])  # Ordena las distancias de menor a mayor
        k_vecinos = distancias[:k]  # Toma los k vecinos más cercanos

        # Realiza la votación de mayoría
        conteo_clases = {}
        for vecino in k_vecinos:
            clase_vecino = vecino[1]
            if clase_vecino in conteo_clases:
                conteo_clases[clase_vecino] += 1
            else:
                conteo_clases[clase_vecino] = 1

        clase_predicha = max(conteo_clases, key=conteo_clases.get)
        predicciones.append(clase_predicha)

    return predicciones

def cargar_datos_desde_csv(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lector_csv = csv.reader(archivo)
        next(lector_csv)  # Saltar la primera fila si contiene encabezados
        datos = [list(map(float, fila)) for fila in lector_csv]
    return np.array(datos)

# Lee el dataset
dataset = pd.read_csv('2do Parcial/Practica ML 2/dataset_iris.csv')

# Separar las características y las etiquetas
X = dataset.iloc[:, :-1].values  # Características
y = dataset.iloc[:, -1].values   # Etiquetas de clase

# Pregunta al usuario por el valor de K y el tipo de distancia
k = int(input("Ingrese el valor de K para k-NN: "))
tipo_distancia = input("Ingrese el tipo de distancia (euclidiana/manhattan): ").lower()

# Generar índices aleatorios para seleccionar muestras de prueba
indices_prueba = np.random.choice(len(X), size=2, replace=False)  # Cambiar 2 por el número de muestras que desees

# Seleccionar las muestras de prueba aleatorias
X_test = X[indices_prueba]

# Realizar predicciones usando k-NN
knn = kNN(X, y, X_test, k, tipo_distancia)

# Mostrar los datos de entrenamiento
print("Características:\n", X)

# Utiliza los datos de prueba para hacer las pruebas y verificar si pertenece o no a la clase
for i in range(len(X_test)):
    valores_prueba = X_test[i]
    clase_real = y[indices_prueba[i]]

    # Realiza la predicción usando k-NN
    clase_predicha = knn[i]

    print(f"Valores de prueba: {valores_prueba}")
    print(f"Clase real: {clase_real}")
    print(f"Clase predicha para la entrada de prueba: {clase_predicha}")

    # Verificar si la clase predicha coincide con la clase real
    if clase_predicha == clase_real:
        print("+ Si pertenece a la clase")
    else:
        print("- No pertenece a la clase")
    print("")
