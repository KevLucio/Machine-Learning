"""
    Pérez Lucio Kevyn Alejandro
"""
import pandas as pd
import numpy as np
#import random

# Lee el archivo CSV y omite la primera fila que contiene los nombres de las columnas
datos = pd.read_csv("2do Parcial/Proyecto ML/BDD/DataSet.csv", skiprows=1, header=None)

# Obtén los nombres de las columnas del archivo original
nombres_columnas = [
    "BallAcceleration", "Time", "DistanceWall", "DistanceCeil", "DistanceBall",
    "PlayerSpeed", "BallSpeed", "up", "accelerate", "slow", "goal", "left",
    "boost", "camera", "down", "right", "slide", "Win"
]

# Asigna los nombres de las columnas al DataFrame
datos.columns = nombres_columnas

# Función para describir atributos
def describir_atributos(datos):
    descripcion = {}
    for columna in datos.columns[:100]:
        if np.issubdtype(datos[columna].dtype, np.number):
            descripcion[columna] = {
                'TipoDato': 'Numerico',
                'Min': datos[columna].min(),
                'Max': datos[columna].max(),
                'Promedio': datos[columna].mean(),
                'DesviacionEstandar': datos[columna].std()
            }
        else:
            descripcion[columna] = {
                'TipoDato': 'Categorico',
                'Categorias': datos[columna].unique().tolist()
            }

    return descripcion

# Función para obtener estadísticas de cada clase
def estadisticas_por_clase(datos, clase):
    subset = datos[datos['Win'] == clase]
    estadisticas = {}

    for columna in datos.columns[:-1]:  # Excluimos la columna de clase
        if np.issubdtype(datos[columna].dtype, np.number):
            estadisticas[columna] = {
                'Min': subset[columna].min(),
                'Max': subset[columna].max(),
                'Promedio': subset[columna].mean(),
                'DesviacionEstandar': subset[columna].std()
            }
        else:
            estadisticas[columna] = {
                'Categorias': subset[columna].unique().tolist()
            }

    return estadisticas

atributos_descripcion = describir_atributos(datos)
print("Descripción de atributos:")
print(atributos_descripcion)

clases = datos['Win'].unique()
for clase in clases:
    print(f"\nEstadísticas para la clase 'Win={clase}':")
    estadisticas_clase = estadisticas_por_clase(datos, clase)
    print(estadisticas_clase)
    
# Define los atributos del vector de entrada X y de salida Y
X = datos.drop('Win', axis=1)  # Elimina la columna 'Win' del DataFrame
Y = datos['Win']

# Función para dividir el conjunto de datos en entrenamiento y prueba
def dividir_entrenamiento_prueba(x, y, porcentaje_entrenamiento):
    total_muestras = len(x)
    indice_corte = int(total_muestras * porcentaje_entrenamiento)

    x_entrenamiento = x.iloc[:indice_corte]
    y_entrenamiento = y.iloc[:indice_corte]
    x_prueba = x.iloc[indice_corte:]
    y_prueba = y.iloc[indice_corte:]

    return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Clasificador de Mínima Distancia
class MinimaDistancia:
    def fit(self, x_entrenamiento, y_entrenamiento):
        clases = y_entrenamiento.unique()
        centroides = {}

        for clase in clases:
            muestras_clase = x_entrenamiento[y_entrenamiento == clase]
            centroide = muestras_clase.mean()
            centroides[clase] = centroide

        self.centroides = centroides

    def predict(self, x_prueba):
        predicciones = []

        for indice, muestra in x_prueba.iterrows():
            distancias = {clase: distancia_euclidiana(muestra, centroide) for clase, centroide in self.centroides.items()}
            clase_predicha = min(distancias, key=distancias.get)
            predicciones.append(clase_predicha)

        return pd.Series(predicciones)

# Función para evaluar el modelo
def evaluar_modelo(modelo, x_prueba, y_prueba):
    predicciones = modelo.predict(x_prueba)
    accuracy = sum(predicciones.reset_index(drop=True) == y_prueba.reset_index(drop=True)) / len(y_prueba)
    error = 1 - accuracy
    return accuracy, error

# Función para realizar K-fold Cross Validation
def k_fold_cross_validation(modelo, X, Y, k):
    num_muestras = len(X)
    tamano_subconjunto = num_muestras // k

    accuracy_resultados = []
    error_resultados = []

    for i in range(k):
        inicio = i * tamano_subconjunto
        fin = (i + 1) * tamano_subconjunto

        x_prueba, y_prueba = X.iloc[inicio:fin, :], Y.iloc[inicio:fin]
        x_entrenamiento, y_entrenamiento = pd.concat([X.iloc[:inicio, :], X.iloc[fin:, :]]), pd.concat([Y.iloc[:inicio], Y.iloc[fin:]])

        modelo.fit(x_entrenamiento, y_entrenamiento)
        accuracy, error = evaluar_modelo(modelo, x_prueba, y_prueba)

        accuracy_resultados.append(accuracy)
        error_resultados.append(error)

    return np.mean(accuracy_resultados), np.mean(error_resultados)

# Función para realizar Bootstrap Validation
def bootstrap_validation(modelo, X, Y, num_bootstrap):
    accuracy_resultados = []
    error_resultados = []

    for _ in range(num_bootstrap):
        indices_bootstrap = np.random.choice(len(X), len(X), replace=True)
        indices_bootstrap = np.intersect1d(indices_bootstrap, X.index)  # Asegurar que los índices existan en el DataFrame

        x_bootstrap, y_bootstrap = X.loc[indices_bootstrap, :], Y.loc[indices_bootstrap]

        x_prueba = X.drop(indices_bootstrap)
        y_prueba = Y.drop(indices_bootstrap)

        modelo.fit(x_bootstrap, y_bootstrap)
        accuracy, error = evaluar_modelo(modelo, x_prueba, y_prueba)

        accuracy_resultados.append(accuracy)
        error_resultados.append(error)

    return np.mean(accuracy_resultados), np.mean(error_resultados)

# Especificar el porcentaje de muestras para el aprendizaje
porcentaje_entrenamiento = float(input("\nIngrese el valor del porcentaje de entrenamiento: "))

# Dividir en conjuntos de entrenamiento y prueba
x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = dividir_entrenamiento_prueba(datos.drop('Win', axis=1), datos['Win'], porcentaje_entrenamiento)

# Crear y entrenar el modelo (Clasificador de Mínima Distancia)
modelo_minima_distancia = MinimaDistancia()
modelo_minima_distancia.fit(x_entrenamiento, y_entrenamiento)

# Evaluar el modelo usando K-fold Cross Validation
k_fold_accuracy, k_fold_error = k_fold_cross_validation(modelo_minima_distancia, x_entrenamiento, y_entrenamiento, k=5)

# Evaluar el modelo usando Bootstrap Validation
bootstrap_accuracy, bootstrap_error = bootstrap_validation(modelo_minima_distancia, x_entrenamiento, y_entrenamiento, num_bootstrap=5)

# Evaluar el modelo usando Train and Test
accuracy_minima_distancia, error_minima_distancia = evaluar_modelo(modelo_minima_distancia, x_prueba, y_prueba)

# Imprime los resultados
print("Resultados para Minima Distancia")
print("\nTrain and Test Validation (Minima Distancia)")
print(f'Accuracy: {accuracy_minima_distancia * 100:.2f}%')
print(f'Error: {error_minima_distancia * 100:.2f}%')

print("\nK-fold Cross Validation (Minima Distancia)")
print(f'Accuracy: {k_fold_accuracy * 100:.2f}%')
print(f'Error: {k_fold_error * 100:.2f}%')

print("\nBootstrap Validation (Minima Distancia)")
print(f'Accuracy: {bootstrap_accuracy * 100:.2f}%')
print(f'Error: {bootstrap_error * 100:.2f}%')

# Clasificador KNN
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x_entrenamiento, y_entrenamiento):
        self.x_entrenamiento = x_entrenamiento
        self.y_entrenamiento = y_entrenamiento

    def predict(self, x_prueba):
        predicciones = []

        for indice, muestra in x_prueba.iterrows():
            distancias = self.x_entrenamiento.apply(lambda row: distancia_euclidiana(muestra, row), axis=1)
            k_indices = distancias.sort_values().head(self.k).index
            k_vecinos = self.y_entrenamiento.loc[k_indices]
            clase_predicha = k_vecinos.mode().iloc[0]
            predicciones.append(clase_predicha)

        return pd.Series(predicciones)

# Funciones para realizar K-fold Cross Validation y Bootstrap Validation
def k_fold_cross_validation_knn(modelo, X, Y, k_fold):
    num_muestras = len(X)
    tamano_subconjunto = num_muestras // k_fold

    accuracy_resultados = []
    error_resultados = []

    for i in range(k_fold):
        inicio = i * tamano_subconjunto
        fin = (i + 1) * tamano_subconjunto

        x_prueba, y_prueba = X.iloc[inicio:fin, :], Y.iloc[inicio:fin]
        x_entrenamiento, y_entrenamiento = pd.concat([X.iloc[:inicio, :], X.iloc[fin:, :]]), pd.concat([Y.iloc[:inicio], Y.iloc[fin:]])

        modelo.fit(x_entrenamiento, y_entrenamiento)
        accuracy, error = evaluar_modelo(modelo, x_prueba, y_prueba)

        accuracy_resultados.append(accuracy)
        error_resultados.append(error)

    return np.mean(accuracy_resultados), np.mean(error_resultados)

# Funciones para realizar Bootstrap Validation
def bootstrap_validation_knn(modelo, X, Y, num_bootstrap):
    accuracy_resultados = []
    error_resultados = []

    for _ in range(num_bootstrap):
        indices_bootstrap = np.random.choice(len(X), len(X), replace=True)
        x_bootstrap, y_bootstrap = X.iloc[indices_bootstrap, :], Y.iloc[indices_bootstrap]

        x_prueba, y_prueba = X.drop(indices_bootstrap), Y.drop(indices_bootstrap)

        modelo.fit(x_bootstrap, y_bootstrap)
        accuracy, error = evaluar_modelo(modelo, x_prueba, y_prueba)

        accuracy_resultados.append(accuracy)
        error_resultados.append(error)

    return np.mean(accuracy_resultados), np.mean(error_resultados)

# Especificar el valor de k para KNN
k_knn = int(input("\nIngrese el valor de k para KNN: "))

# Crear y entrenar el modelo (Clasificador KNN)
modelo_knn = KNN(k_knn)
modelo_knn.fit(x_entrenamiento, y_entrenamiento)

# Evaluar el modelo KNN
accuracy_knn, error_knn = evaluar_modelo(modelo_knn, x_prueba, y_prueba)

# Realizar K-fold cross validation para KNN
accuracy_kfold_knn, error_kfold_knn = k_fold_cross_validation_knn(modelo_knn, X, Y, k_fold=5)

# Realizar Bootstrap validation para KNN
accuracy_bootstrap_knn, error_bootstrap_knn = bootstrap_validation_knn(modelo_knn, X, Y, num_bootstrap=5)

# Imprimir resultados
print("Resultados para KNN:")
print("\nTrain and Test Validation (KNN)")
print(f'Accuracy: {accuracy_knn * 100:.2f}%')
print(f'Error: {error_knn * 100:.2f}%')

print("\nK-fold Cross Validation (KNN)")
print(f'Accuracy: {accuracy_kfold_knn * 100:.2f}%')
print(f'Error: {error_kfold_knn * 100:.2f}%')

print("\nBootstrap Validation (KNN)")
print(f'Accuracy: {accuracy_bootstrap_knn * 100:.2f}%')
print(f'Error: {error_bootstrap_knn * 100:.2f}%')

# Parte 6: Eliminar uno de los elementos
atributos_a_eliminar = ['DistanceBall']

# Definir el porcentaje de entrenamiento para train-test split
porcentaje_entrenamiento = 0.8

# Elimina un atributo y evalúa la eficiencia
for atributo in atributos_a_eliminar:
    print(f"\nEliminando el atributo '{atributo}':")

    # Elimina el atributo
    datos_sin_atributo = datos.drop(atributo, axis=1)

    # Define los atributos del vector de entrada X y de salida Y
    X = datos_sin_atributo.drop('Win', axis=1)  # Elimina la columna 'Win' del DataFrame
    Y = datos_sin_atributo['Win']

    # Train-test split
    n_entrenamiento = int(len(X) * porcentaje_entrenamiento)
    X_entrenamiento, X_prueba = X[:n_entrenamiento], X[n_entrenamiento:]
    Y_entrenamiento, Y_prueba = Y[:n_entrenamiento], Y[n_entrenamiento:]

    # Crear y entrenar el modelo (Clasificador de Mínima Distancia)
    modelo_minima_distancia_sin_atributo = MinimaDistancia()
    modelo_minima_distancia_sin_atributo.fit(X_entrenamiento, Y_entrenamiento)

    # Evaluar el modelo
    accuracy_minima_distancia_sin_atributo, error_minima_distancia_sin_atributo = evaluar_modelo(
        modelo_minima_distancia_sin_atributo, X_prueba, Y_prueba)

    print(f"Train and Test Accuracy (Minima Distancia): {accuracy_minima_distancia_sin_atributo * 100:.2f}%")
    print(f"Train and Test Error (Minima Distancia): {error_minima_distancia_sin_atributo * 100:.2f}%")

    # K-fold cross-validation
    k = 5 
    k_fold_accuracy, k_fold_error = k_fold_cross_validation(modelo_minima_distancia_sin_atributo, X, Y, k)

    print(f"K-Fold Cross-Validation Accuracy (Minima Distancia): {k_fold_accuracy * 100:.2f}%")
    print(f"K-Fold Cross-Validation Error (Minima Distancia): {k_fold_error * 100:.2f}%")

    # Bootstrap
    n_bootstrap_samples = 5
    bootstrap_accuracy, bootstrap_error = bootstrap_validation(modelo_minima_distancia_sin_atributo, X, Y, n_bootstrap_samples)

    print(f"Bootstrap Accuracy (Minima Distancia): {bootstrap_accuracy * 100:.2f}%")
    print(f"Bootstrap Error (Minima Distancia): {bootstrap_error * 100:.2f}%")

# Parte 6: Eliminar otro de los elementos
atributos_a_eliminar = ['BallSpeed']

# Elimina un atributo y evalúa la eficiencia
for atributo in atributos_a_eliminar:
    print(f"\nEliminando el atributo '{atributo}':")

    # Elimina el atributo
    datos_sin_atributo = datos.drop(atributo, axis=1)

    # Define los atributos del vector de entrada X y de salida Y
    X = datos_sin_atributo.drop('Win', axis=1)  # Elimina la columna 'Win' del DataFrame
    Y = datos_sin_atributo['Win']

    # Train-test split
    n_entrenamiento = int(len(X) * porcentaje_entrenamiento)
    X_entrenamiento, X_prueba = X[:n_entrenamiento], X[n_entrenamiento:]
    Y_entrenamiento, Y_prueba = Y[:n_entrenamiento], Y[n_entrenamiento:]

    # Crear y entrenar el modelo (Clasificador de Mínima Distancia)
    modelo_minima_distancia_sin_atributo = MinimaDistancia()
    modelo_minima_distancia_sin_atributo.fit(X_entrenamiento, Y_entrenamiento)

    # Evaluar el modelo
    accuracy_minima_distancia_sin_atributo, error_minima_distancia_sin_atributo = evaluar_modelo(
        modelo_minima_distancia_sin_atributo, X_prueba, Y_prueba)

    print(f"Train and Test Accuracy (Minima Distancia): {accuracy_minima_distancia_sin_atributo * 100:.2f}%")
    print(f"Train and Test Error (Minima Distancia): {error_minima_distancia_sin_atributo * 100:.2f}%")

    # K-fold cross-validation
    k = 5 
    k_fold_accuracy, k_fold_error = k_fold_cross_validation(modelo_minima_distancia_sin_atributo, X, Y, k)

    print(f"K-Fold Cross-Validation Accuracy (Minima Distancia): {k_fold_accuracy * 100:.2f}%")
    print(f"K-Fold Cross-Validation Error (Minima Distancia): {k_fold_error * 100:.2f}%")

    # Bootstrap
    n_bootstrap_samples = 5
    bootstrap_accuracy, bootstrap_error = bootstrap_validation(modelo_minima_distancia_sin_atributo, X, Y, n_bootstrap_samples)

    print(f"Bootstrap Accuracy (Minima Distancia): {bootstrap_accuracy * 100:.2f}%")
    print(f"Bootstrap Error (Minima Distancia): {bootstrap_error * 100:.2f}%")

# Parte 7: Eliminar los dos elementos
atributos_a_eliminar = ['DistanceBall', 'BallSpeed']

print("\nEliminando dos atributos")

# Elimina dos atributos y evalúa la eficiencia
for atributo in atributos_a_eliminar:
    print(f"\nEliminando el atributo '{atributo}':")

    # Elimina el atributo
    datos_sin_atributo = datos.drop(atributo, axis=1)

    # Define los atributos del vector de entrada X y de salida Y
    X = datos_sin_atributo.drop('Win', axis=1)  # Elimina la columna 'Win' del DataFrame
    Y = datos_sin_atributo['Win']

    # Train-test split
    n_entrenamiento = int(len(X) * porcentaje_entrenamiento)
    X_entrenamiento, X_prueba = X[:n_entrenamiento], X[n_entrenamiento:]
    Y_entrenamiento, Y_prueba = Y[:n_entrenamiento], Y[n_entrenamiento:]

    # Crear y entrenar el modelo (Clasificador de Mínima Distancia)
    modelo_minima_distancia_sin_atributo = MinimaDistancia()
    modelo_minima_distancia_sin_atributo.fit(X_entrenamiento, Y_entrenamiento)

    # Evaluar el modelo
    accuracy_minima_distancia_sin_atributo, error_minima_distancia_sin_atributo = evaluar_modelo(
        modelo_minima_distancia_sin_atributo, X_prueba, Y_prueba)

    print(f"Train and Test Accuracy (Minima Distancia): {accuracy_minima_distancia_sin_atributo * 100:.2f}%")
    print(f"Train and Test Error (Minima Distancia): {error_minima_distancia_sin_atributo * 100:.2f}%")

    # K-fold cross-validation
    k = 5 
    k_fold_accuracy, k_fold_error = k_fold_cross_validation(modelo_minima_distancia_sin_atributo, X, Y, k)

    print(f"K-Fold Cross-Validation Accuracy (Minima Distancia): {k_fold_accuracy * 100:.2f}%")
    print(f"K-Fold Cross-Validation Error (Minima Distancia): {k_fold_error * 100:.2f}%")

    # Bootstrap
    n_bootstrap_samples = 5
    bootstrap_accuracy, bootstrap_error = bootstrap_validation(modelo_minima_distancia_sin_atributo, X, Y, n_bootstrap_samples)

    print(f"Bootstrap Accuracy (Minima Distancia): {bootstrap_accuracy * 100:.2f}%")
    print(f"Bootstrap Error (Minima Distancia): {bootstrap_error * 100:.2f}%")

# Parte 8: Eliminar una fracción de muestras
fraccion_eliminar = 0.2
datos_muestra_eliminada = datos.sample(frac=1 - fraccion_eliminar, random_state=42)

# Define los atributos del vector de entrada X y de salida Y
X = datos_muestra_eliminada.drop('Win', axis=1)  # Elimina la columna 'Win' del DataFrame
Y = datos_muestra_eliminada['Win']

# Crear y entrenar el modelo (Clasificador de Mínima Distancia)
modelo_minima_distancia_muestra_eliminada = MinimaDistancia()
modelo_minima_distancia_muestra_eliminada.fit(X, Y)

# Evaluar el modelo con Train and Test
accuracy_minima_distancia_train_test, error_minima_distancia_train_test = evaluar_modelo(
    modelo_minima_distancia_muestra_eliminada, x_prueba, y_prueba)
print("\nResultados Eliminando el 20% de las Muestras - Train and Test")
print(f"Accuracy: {accuracy_minima_distancia_train_test * 100:.2f}%")
print(f"Error: {error_minima_distancia_train_test * 100:.2f}%")

# Evaluar el modelo con K-Fold Cross-Validation
k_fold_accuracy_minima_distancia, k_fold_error_minima_distancia = k_fold_cross_validation(
    modelo_minima_distancia_muestra_eliminada, X, Y, k=5)
print("\nResultados Eliminando el 20% de las Muestras - K-Fold Cross-Validation")
print(f"Accuracy: {k_fold_accuracy_minima_distancia * 100:.2f}%")
print(f"Error: {k_fold_error_minima_distancia * 100:.2f}%")

# Evaluar el modelo con Bootstrap
n_bootstrap_samples = 5
bootstrap_accuracy_minima_distancia, bootstrap_error_minima_distancia = bootstrap_validation(
    modelo_minima_distancia_muestra_eliminada, X, Y, n_bootstrap_samples)
print("\nResultados Eliminando el 20% de las Muestras - Bootstrap")
print(f"Accuracy: {bootstrap_accuracy_minima_distancia * 100:.2f}%")
print(f"Error: {bootstrap_error_minima_distancia * 100:.2f}%")
