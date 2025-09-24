"""
    Pérez Lucio Kevyn Alejandro
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore

def cargar_datos_desde_csv(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lineas = archivo.readlines()

    datos = [linea.strip().split(',') for linea in lineas]
    columnas = ["LargoSepalo", "AnchoSepalo", "LargoPetalo", "AnchoPetalo", "Clase"]
    df = pd.DataFrame(datos, columns=columnas)
    return df

def informacion_general(df):
    print("1. Cantidad de registros por clase:")
    print(df['Clase'].value_counts())
    print("\n2. Distribución de clases:")
    print(df['Clase'].value_counts(normalize=True))
    print("\n3. Valores faltantes:")
    faltantes = df.isnull().sum()
    print(faltantes[faltantes > 0])

def porcentaje_faltantes_por_clase(df):
    print("\n3b. Porcentaje de valores faltantes por atributo-clase:")
    for clase in df['Clase'].unique():
        print(f"\nClase: {clase}")
        for col in df.columns[:-1]:  # Excluimos la columna de la clase
            total_faltantes = df[df['Clase'] == clase][col].isnull().sum()
            total_registros_clase = df[df['Clase'] == clase].shape[0]
            porcentaje_faltantes = (total_faltantes / total_registros_clase) * 100
            print(f"{col}: {total_faltantes} ({porcentaje_faltantes:.2f}%)")

        total_faltantes_clase = df[df['Clase'] == clase].isnull().sum().sum()
        total_registros_clase = df[df['Clase'] == clase].shape[0]
        porcentaje_faltantes_clase = (total_faltantes_clase / (total_registros_clase * df.shape[1])) * 100
        print(f"Total Clase {clase}: {total_faltantes_clase} ({porcentaje_faltantes_clase:.2f}%)")

def normalizar_datos(df):
    print("\n4. Normalización de datos:")
    features = df.columns[:-1]
    df[features] = df[features].apply(pd.to_numeric)
    df[features] = (df[features] - df[features].mean()) / df[features].std()
    print(df)

def detectar_atipicos(df):
    print("\n5. Detección de valores atípicos:")
    z_scores = zscore(df[df.columns[:-1]])
    atipicos = (np.abs(z_scores) > 3).any(axis=1)
    atipicos_indices = df[atipicos].index
    atipicos_info = df.loc[atipicos_indices, :]
    print("Tuplas con valores atípicos:")
    print(atipicos_info)
    print("\nPromedio y desviación estándar por atributo en cada clase:")
    for clase in df['Clase'].unique():
        print(f"\nClase: {clase}")
        for col in df.columns[:-1]:
            promedio = df[df['Clase'] == clase][col].mean()
            desviacion_estandar = df[df['Clase'] == clase][col].std()
            print(f"{col}: Promedio={promedio:.2f}, Desviación estándar={desviacion_estandar:.2f}")

def reducir_datos(df):
    print("\n6. Reducción de datos:")
    opcion = input("¿Desea reducir filas o columnas? (filas/columnas): ").lower()
    if opcion == 'filas':
        n_filas = int(input("Ingrese el número de filas a mantener: "))
        df = df.head(n_filas)
    elif opcion == 'columnas':
        columnas_eliminar = input("Ingrese el nombre de las columnas a eliminar (separadas por coma): ").split(',')
        df = df.drop(columns=columnas_eliminar)
    else:
        print("Opción no válida.")
        return df

    print("Nuevo conjunto de datos:")
    print(df)
    return df
    
def main():
    archivo = "2do Parcial\\Practica ML 3\\textos\\iris2.txt"
    df = cargar_datos_desde_csv(archivo)

    informacion_general(df)
    porcentaje_faltantes_por_clase(df)
    normalizar_datos(df)
    detectar_atipicos(df)
    df_reducido = reducir_datos(df)

    print("\nConjunto de datos reducido:")
    print(df_reducido)

if __name__ == "__main__":
    main()
