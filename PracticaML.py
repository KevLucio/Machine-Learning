"""
    Pérez Lucio Kevyn Alejandro
"""
import numpy as np
import statistics

def cargar_datos_desde_archivo(nombre_archivo, separador):
    try:
        with open(nombre_archivo, 'r') as archivo:
            lineas = archivo.readlines()
            datos = [linea.strip().split(separador) for linea in lineas]
            datos = [[float(valor) if '.' in valor else int(valor) if valor.isdigit() else valor for valor in linea] for linea in datos]
        return datos
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo '{nombre_archivo}'")
        return None

def definir_tipo_y_medida(datos):
    tipos_de_dato = []
    medidas = []
    for atributo in datos[0:len(datos[0])-1]:
        valores = np.array(atributo)

        # Determinar el tipo de dato
        tipo_dato = str(valores.dtype)

        # Determinar la medida (por ejemplo, media)
        medida = np.mean(valores)

        tipos_de_dato.append(tipo_dato)
        medidas.append(medida)

    return tipos_de_dato, medidas

class clasificador:
    def __init__(self,identificador,noAtributos):
        self._identificador = identificador
        self._atributos = []
        self._atributosAnalisis = []
        self._atributosEvaluar = []
        self._atributosEvaluarAnalisis = []
        for i in range(noAtributos):
            self._atributos.append([])
            self._atributosAnalisis.append([])

    def getIdentificador(self):
        return self._identificador
    
    def getNoAtributos(self):
        return len(self._atributos)
    
    def getNoAtributosAnalisis(self):
        return len(self._atributosAnalisis)
    
    def setAtributo(self,noAtributo,valor):
        self._atributos[noAtributo].append(valor)

    def setAtributosEvaluar(self,vector):
        for indice in vector:
            self._atributosEvaluar.append(self._atributos[indice])
            self._atributosEvaluarAnalisis.append(self._atributosAnalisis[indice])       

    def getAtributosEvaluar(self):
        return self._atributosEvaluar  

    def getAtributosEvaluarAnalisis(self):
        return self._atributosEvaluarAnalisis   

    def getAtributos(self):
        return self._atributos
    
    def getNoMuestras(self):
        return len(self._atributos[0])

    def findMMM(self,atributo):
        lista = self._atributos[atributo]
        minT = min(lista)
        maxT = max(lista)
        meanT = statistics.mean(lista)
        self._atributosAnalisis[atributo].append(minT)
        self._atributosAnalisis[atributo].append(maxT)
        self._atributosAnalisis[atributo].append(meanT)

    def findCat(self,atributo):
        lista = self._atributos[atributo]            
        cat = list(set(lista))
        self._atributosAnalisis[atributo] = cat

    def getMin(self, atributo):
        return self._atributosAnalisis[atributo][0]

    def getMax(self, atributo):
        return self._atributosAnalisis[atributo][1]
    
    def getMean(self, atributo):
        return self._atributosAnalisis[atributo][2]
    
    def getMMM(self, atributo):
        return (self.getMin(atributo),self.getMax(atributo),self.getMean(atributo))
    
    def getCat(self,atributo):
        return self._atributosAnalisis[atributo]

def abrirArchivo(path):
    vectores = []
    try:
        with open(path,'r') as archivo:
            for linea in archivo:
                datos = linea.strip()
                vectores.append(datos)
        return vectores    
    except FileNotFoundError:
        print("Archivo no encontrado")
        return vectores
    except IOError:
        print("Archivo no se pudo abrir")
        return vectores

def separar(lista, separador):
    elementos_individuales = []
    for dato in lista:
        elemento = dato.split(separador)
        elementos_individuales.append(elemento)

    return elementos_individuales        

def mismaLongitud(lista):
    tam = len(lista[0])
    for elemento in lista:
        if not(len(elemento) == tam):
            return False
    
    return True
        
def solicitarClase(noCols):
    validarColumna = False
    print(f"El numero de columnas de los registros es {noCols}")
    while not validarColumna:
        columna = int(input("Ingrese la columna que será la clase: ")) - 1

        if(columna >= 0 and columna <= noCols):
            validarColumna = True
        else:
            print("El numero de columna es incorrecto") 

    return columna 

def solicitarIntervalo(noRows):
    validarIntervalo = False
    print(f"El numero de registros es de: {noRows}")    
    while not validarIntervalo:
        intervalo_min = int(input("Ingrese el registro inicial: ")) - 1

        if(intervalo_min >= 0 and intervalo_min <= noRows):
            intervalo_max = int(input("Ingrese el registro final: ")) - 1
            
            if(intervalo_max >= 0 and intervalo_max <= noRows and intervalo_min < intervalo_max):
                validarIntervalo = True

            else:
                print("El intervalo final esta fuera de rango")    

        else:
            print("El indice esta fuera de rango")

    return (intervalo_min,intervalo_max)        

def obtenerDatosInteres(lista,intervalo):
    datosIntervalo = []
    for i in range(intervalo[0],intervalo[1]):
        listaAux = []
        for col in lista[i]:
            if col.replace('.','',1).isdigit():
                listaAux.append(float(col))
            else:
                listaAux.append(col)   
        datosIntervalo.append(listaAux)

    return datosIntervalo  

def clasificar(lista,col):
    clasificaciones = []
    noAtributos = len(lista[0]) - 1
    for datos in lista:
        if len(clasificaciones) == 0:
            clasificacion = clasificador(datos[col],noAtributos)
            clasificaciones.append(clasificacion)
            contador = 0
            for i in range(len(datos)):
                if not(i == col):
                    clasificacion.setAtributo(contador,datos[i])
                    contador += 1
            
        else:
            existeClase = False
            for clase in clasificaciones:
                if clase.getIdentificador() == datos[col]:
                    existeClase = True
                    contador = 0
                    for i in range(len(datos)):
                        if not(i == col):
                            clase.setAtributo(contador,datos[i])
                            contador += 1 

            if not existeClase:
                clasificacion = clasificador(datos[col],noAtributos)
                clasificaciones.append(clasificacion) 
                contador = 0
                for i in range(len(datos)):
                    if not(i == col):
                        clasificacion.setAtributo(contador,datos[i])
                        contador += 1                         

    return clasificaciones

def obtenerAtributos(tam):    
    pasa = False
    while not pasa:
        indices = [int(indice)-1 for indice in input(f"Ingrese los atributos ({tam} max) a tomar en cuenta para analizar (separados por comas): ").split(',')]

        if len(indices) > tam:
            pasa = False
            print(f"Ingreso mas de {tam} atributos")

        else:  
            checkIndex = []
            for i in range(len(indices)):
                if(indices[i] < 0 or indices[i] > tam-1):
                    checkIndex.append(False)
                else:
                    checkIndex.append(True)    

            if all(checkIndex):
                pasa = True
    
    return indices        

def main():
    path = "2do Parcial\\Practica ML\\textos\\" + input("Ingrese el nombre del archivo: ")
    separador = input("Ingrese el separador: ")
    datos = separar(abrirArchivo(path),separador)

    if not mismaLongitud(datos):
        print("Hay un error con el numero de atributos de un renglon del archivo")
        return

    noCols = len(datos[0])
    noRows = len(datos)
    colClase = solicitarClase(noCols)   
    intervalo = solicitarIntervalo(noRows)
    datosInteres = obtenerDatosInteres(datos,intervalo)
    print("\n-----DATOS DEL TXT------")
    print(f" 1. Numero de atributos: {len(datos[0])}")
    print(f" 2. Numero de registros: {len(datos)}")
    datosClasificados = clasificar(datosInteres,colClase)
    print(f"El numero de clasificaciones encontradas son {len(datosClasificados)}:")
    for clase in datosClasificados:
        print(f"\033[1;31mClase: {clase.getIdentificador()}:\033[0m") 
        #print(f"    -No de registros de esta clase: {clase.getNoMuestras()}")
        lista = clase.getAtributos()    

        for i in range(len(lista)):
            print(f"   Atributo {i + 1}")
            if type(lista[i][0]) == float:
                if all(elemento == 1 or elemento == 0 for elemento in lista[i]):
                    clase.findCat(i)
                    print(f"Categories: {clase.getCat(i)}")
                else:    
                    clase.findMMM(i)
                    print(f"Min, Max, Mean: {clase.getMMM(i)}")
            else:
                clase.findCat(i)
                print(f"Categories: {clase.getCat(i)}")

    vector = obtenerAtributos(datosClasificados[0].getNoAtributosAnalisis())
    for clase in datosClasificados:
        clase.setAtributosEvaluar(vector)

    print("Los atributos a evaluar por clase son:")
    for clase in datosClasificados:
        print(f"\033[1;31mClase: {clase.getIdentificador()}:\033[0m") 
        lista = clase.getAtributosEvaluar()
        lista2 = clase.getAtributosEvaluarAnalisis()
        for i in range(len(lista)):
            print(f"   Atributo {i + 1}")
            print("Lista valores")
            print(lista[i])
            print()
            print("Medidas de la lista")
            print(lista2[i])
          
if __name__ == "__main__":
    main()