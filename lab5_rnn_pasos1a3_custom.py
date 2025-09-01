
# --------------------------------------------------
# Laboratorio 5 - Predicción con RNN (LSTM)
# Paso 1 - 3: Carga, división y preparación de datos
# --------------------------------------------------

# Importación de librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

# ========== PASO 1: CARGA Y VISUALIZACIÓN ==========
# Se descarga la serie temporal de FRED con código IPN31152N

# Descargar serie temporal
datos_fred = web.DataReader(name="IPN31152N", data_source="fred")

# Verificar estructura
print("Datos descargados desde FRED:")
print(datos_fred.head())

# Renombrar la columna con un nombre más claro
datos_fred = datos_fred.rename(columns={"IPN31152N": "indice_produccion"})

# Graficar la serie completa
plt.figure(figsize=(14, 5))
plt.plot(datos_fred.index, datos_fred["indice_produccion"], color="steelblue", linewidth=2,
         label="Producción de helados y postres congelados (EE.UU.)")
plt.xlabel("Fecha")
plt.ylabel("Índice de producción (base 2012 = 100)")
plt.title("Serie Temporal Mensual de Producción Industrial")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== PASO 2: DIVISIÓN DE CONJUNTOS ==========
# Tomaremos los últimos 24 meses como prueba, el resto para entrenamiento

total_filas = datos_fred.shape[0]
meses_para_prueba = 24

# Separar conjuntos
conjunto_entrenamiento = datos_fred.iloc[0:total_filas - meses_para_prueba]
conjunto_prueba = datos_fred.iloc[total_filas - meses_para_prueba:]

# Mostrar resultados
print("Cantidad de datos de entrenamiento:", len(conjunto_entrenamiento))
print("Cantidad de datos de prueba:", len(conjunto_prueba))

# ========== PASO 3: ESCALAMIENTO Y GENERADOR ==========
# Se aplica MinMaxScaler para convertir los valores entre 0 y 1

# Crear una instancia del escalador
escalador = MinMaxScaler()

# Ajustar escalador con los datos de entrenamiento
entrenamiento_escalado = escalador.fit_transform(conjunto_entrenamiento)

# Transformar los datos de prueba usando el mismo escalador
prueba_escalada = escalador.transform(conjunto_prueba)

# Parámetros para la generación de secuencias
tamano_ventana = 12   # meses anteriores usados para predecir el siguiente
tamanio_lote = 1

# Crear generadores
generador_train = TimeseriesGenerator(entrenamiento_escalado, entrenamiento_escalado,
                                      length=tamano_ventana, batch_size=tamanio_lote)

generador_test = TimeseriesGenerator(prueba_escalada, prueba_escalada,
                                     length=tamano_ventana, batch_size=tamanio_lote)

# Mostrar un ejemplo de entrada y salida
ejemplo_x, ejemplo_y = generador_train[0]

print("Ejemplo de secuencia de entrada (X):")
print(ejemplo_x.reshape(-1))

print("Valor esperado de salida (y):")
print(ejemplo_y.reshape(-1))
