"""
    Pérez Lucio Kevyn Alejandro
"""
import pandas as pd
import numpy as np

# Leer el archivo CSV y omitir la primera fila que contiene los nombres de las columnas
datos = pd.read_csv("2do Parcial/Proyecto ML/BDD/DataSet.csv", skiprows=1, header=None)

# Obtener los nombres de las columnas del archivo original
nombres_columnas = [
    "BallAcceleration", "Time", "DistanceWall", "DistanceCeil", "DistanceBall",
    "PlayerSpeed", "BallSpeed", "up", "accelerate", "slow", "goal", "left",
    "boost", "camera", "down", "right", "slide", "Win"
]

# Asignar los nombres de las columnas al DataFrame
datos.columns = nombres_columnas

# Definir los atributos del vector de entrada X y de salida Y
X = datos.drop('Win', axis=1)  # Eliminar la columna 'Win' del DataFrame
Y = datos['Win']

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples


# Crear y entrenar el modelo (aquí se usa regresión lineal)
class RegresionLineal:
    def __init__(self, alpha=1e-5):
        self.coeficientes = None
        self.alpha = alpha

    def fit(self, x, y):
        x_ext = np.c_[np.ones((x.shape[0], 1)), x]  # Agregar un término de sesgo
        identity_matrix = np.eye(x_ext.shape[1])
        self.coeficientes = np.linalg.inv(x_ext.T @ x_ext + self.alpha * identity_matrix) @ x_ext.T @ y

    def predict(self, x):
        x_ext = np.c_[np.ones((x.shape[0], 1)), x]  # Agregar un término de sesgo
        return x_ext @ self.coeficientes

def dividir_entrenamiento_prueba(x, y, porcentaje_entrenamiento):
    n = len(x)
    indices_entrenamiento = np.random.choice(n, int(porcentaje_entrenamiento * n), replace=False)
    indices_prueba = np.setdiff1d(np.arange(n), indices_entrenamiento)

    x_entrenamiento, x_prueba = x.iloc[indices_entrenamiento], x.iloc[indices_prueba]
    y_entrenamiento, y_prueba = y.iloc[indices_entrenamiento], y.iloc[indices_prueba]

    return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba

def evaluar_modelo(modelo, x_prueba, y_prueba):
    predicciones = modelo.predict(x_prueba)

    if isinstance(y_prueba.iloc[0], (int, float)):
        # Regresión (variable continua)
        error = mean_squared_error(y_prueba, predicciones)
        return 1 - error, error
    else:
        # Clasificación (variable categórica)
        eficiencia = accuracy_score(y_prueba, predicciones.round())
        error = 1 - eficiencia
        return eficiencia, error

def kfold_cross_validation(x, y, modelo, k):
    n = len(x)
    indices = np.arange(n)
    np.random.shuffle(indices)

    eficiencias = []
    errores = []

    for i in range(k):
        test_indices = indices[int(i * n / k):int((i + 1) * n / k)]
        train_indices = np.setdiff1d(indices, test_indices)

        x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        modelo.fit(x_train, y_train)
        eficiencia, error = evaluar_modelo(modelo, x_test, y_test)

        eficiencias.append(eficiencia)
        errores.append(error)

    return eficiencias, errores

def estadisticas_generales(eficiencias, errores):
    eficiencia_promedio = np.mean(eficiencias)
    error_promedio = np.mean(errores)
    desviacion_eficiencia = np.std(eficiencias)
    desviacion_error = np.std(errores)

    return eficiencia_promedio, error_promedio, desviacion_eficiencia, desviacion_error

def bootstrap_validation(x, y, modelo, k, train_size, test_size):
    n = len(x)
    eficiencias = []
    errores = []

    for _ in range(k):
        indices_resampled = np.random.choice(n, n, replace=True)
        train_indices = indices_resampled[:int(train_size * n)]
        test_indices = indices_resampled[int(train_size * n):]

        x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        modelo.fit(x_train, y_train)
        eficiencia, error = evaluar_modelo(modelo, x_test, y_test)

        eficiencias.append(eficiencia)
        errores.append(error)

    return eficiencias, errores

# Especificar el porcentaje de muestras para el aprendizaje
porcentaje_entrenamiento = float(input("Ingrese el valor del porcentaje de entrenamiento: "))

# Dividir en conjuntos de entrenamiento y prueba
x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = dividir_entrenamiento_prueba(X, Y, porcentaje_entrenamiento)

# Crear y entrenar el modelo (aquí se usa regresión lineal como ejemplo)
modelo = RegresionLineal()
modelo.fit(x_entrenamiento, y_entrenamiento)

# Evaluar el modelo
eficiencia, error = evaluar_modelo(modelo, x_prueba, y_prueba)

print("Train and Test Validation")
print(f'Eficiencia: {eficiencia * 100:.2f}%')
print(f'Error: {error * 100:.2f}%')

# Especificar la cantidad de grupos (k) para K-fold cross validation
k_groups = int(input("Ingrese la cantidad de grupos para K-fold cross validation: "))

# Crear y entrenar el modelo (aquí se usa regresión lineal como ejemplo)
modelo = RegresionLineal()

# Realizar K-fold cross validation
eficiencias, errores = kfold_cross_validation(X, Y, modelo, k_groups)

# Calcular estadísticas generales
eficiencia_promedio, error_promedio, desviacion_eficiencia, desviacion_error = estadisticas_generales(eficiencias, errores)

print("\nK Fold Cross Validation")
# Imprimir resultados
for i in range(k_groups):
    print(f'Grupo {i+1}: Eficiencia: {eficiencias[i] * 100:.2f}%, Error: {errores[i] * 100:.2f}%')

print(f'\nEficiencia Promedio: {eficiencia_promedio * 100:.2f}%')
print(f'Error Promedio: {error_promedio * 100:.2f}%')
print(f'Desviación Estándar de Eficiencia: {desviacion_eficiencia * 100:.2f}%')
print(f'Desviación Estándar de Error: {desviacion_error * 100:.2f}%')

# Especificar la cantidad de experimentos (k) para Bootstrap
k_bootstrap = int(input("Ingrese la cantidad de experimentos para Bootstrap validation: "))

# Especificar la cantidad de muestras en el conjunto de aprendizaje y prueba
train_size = float(input("Ingrese el porcentaje del conjunto de datos para aprendizaje: "))
test_size = float(input("Ingrese el porcentaje del conjunto de datos para prueba: "))

# Realizar Bootstrap validation
eficiencias, errores = bootstrap_validation(X, Y, modelo, k_bootstrap, train_size, test_size)

# Calcular estadísticas generales
eficiencia_promedio, error_promedio, desviacion_eficiencia, desviacion_error = estadisticas_generales(eficiencias, errores)

print("\nBootstrap validation")
# Imprimir resultados
for i in range(k_bootstrap):
    print(f'Experimento {i+1}: Eficiencia: {eficiencias[i] * 100:.2f}%, Error: {errores[i] * 100:.2f}%')

print(f'\nEficiencia Promedio: {eficiencia_promedio * 100:.2f}%')
print(f'Error Promedio: {error_promedio * 100:.2f}%')
print(f'Desviación Estándar de Eficiencia: {desviacion_eficiencia * 100:.2f}%')
print(f'Desviación Estándar de Error: {desviacion_error * 100:.2f}%')
