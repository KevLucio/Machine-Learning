"""
    Pérez Lucio Kevyn Alejandro
"""
import pandas as pd
import numpy as np

def calcular_distancias(promedios, valores_prueba, tipo_distancia='euclidiana'):
    if tipo_distancia == 'euclidiana':
        return np.sqrt(np.sum((promedios.values - valores_prueba)**2, axis=1))
    elif tipo_distancia == 'manhattan':
        return np.sum(np.abs(promedios.values - valores_prueba), axis=1)
    else:
        raise ValueError("Tipo de distancia no válido")

# Lee el dataset
dataset = pd.read_csv('2do Parcial\Practica ML 2\dataset_iris.csv')

# Calcula el promedio por cada clase y por cada propiedad
promedios = dataset.groupby('clase').mean()

# Imprime los promedios
print("Promedio por clase y propiedad:")
print(promedios)
print("\n")

# Utiliza los datos del dataset para hacer las pruebas y verificar si pertenece o no a la clase
for i in range(len(dataset)):
    # Obtener los valores de prueba y parsearlos a float
    valores_prueba = dataset.iloc[i, 0:4].values
    # Parsear a float
    valores_prueba = valores_prueba.astype(float)
    print("Valores de prueba:", valores_prueba)
    clase_real = dataset.iloc[i, 4]
    print("Clase real:", clase_real)

    # Pregunta al usuario por el tipo de distancia
    tipo_distancia = input("Ingrese el tipo de distancia (euclidiana/manhattan): ").lower()

    # Calcular las distancias según el tipo especificado
    distancias = calcular_distancias(promedios, valores_prueba, tipo_distancia)
    print("Distancias:", distancias)
    clase_predicha = promedios.index[np.argmin(distancias)]
    print("Clase predicha para la entrada de prueba:", clase_predicha, "Clase real:", clase_real)
    if clase_predicha == clase_real:
        print("Si pertenece a la clase")
    else:
        print("No pertenece a la clase")
    print("")
