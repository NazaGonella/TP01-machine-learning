import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from codigo.utils import Utils, RawData

# 1 - Utilizo la función de pandas info() para analizar los datos. ---- *

# casas_dev.csv

RawData.casas_dev.info(verbose = True, memory_usage=False)
print("")
print("Cantidad de filas con uno o más elementos nulos: ", Utils.get_null_rows(RawData.casas_dev))

# OUTPUT
# RangeIndex: 900 entries, 0 to 899
# Data columns (total 9 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   area        900 non-null    float64
#  1   area_units  900 non-null    object 
#  2   is_house    900 non-null    int64  
#  3   has_pool    900 non-null    int64  
#  4   age         770 non-null    float64
#  5   price       900 non-null    float64
#  6   lat         900 non-null    float64
#  7   lon         900 non-null    float64
#  8   rooms       828 non-null    float64
# dtypes: float64(6), int64(2), object(1)
# Cantidad de filas con uno o más elementos nulos:  189

# OBSERVACIONES / POSIBLES CAMBIOS
#   Tipos de datos erróneos:
#   * Las features 'is_house' y 'has_pool' podrían ser booleanos en lugar de enteros.
#   * Las features 'age' y 'rooms' podrían ser enteros en lugar de floats.}
#   Datos incompletos:
#   * La feature 'age' presenta 130 datos nulos.   Podría completarse con el promedio del valor de la feature o remover las filas donde 'age' es nulo.
#   * La feature 'rooms' presenta 72 datos nulos.  Podría completarse con el promedio del valor de la feature o remover las filas donde 'rooms' es nulo.
#   * En total hay 189 filas con elementos nulos.

# --------------------------------------------------------------------- *

# casas_test.csv

RawData.casas_test.info(verbose = True, memory_usage=False)
print("")
print("Cantidad de filas con uno o más elementos nulos: ", Utils.get_null_rows(RawData.casas_test))

# OUTPUT
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 9 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   area        100 non-null    float64
#  1   area_units  100 non-null    object 
#  2   is_house    100 non-null    int64  
#  3   has_pool    100 non-null    int64  
#  4   age         84 non-null     float64
#  5   price       100 non-null    float64
#  6   lat         100 non-null    float64
#  7   lon         100 non-null    float64
#  8   rooms       96 non-null     float64
# dtypes: float64(6), int64(2), object(1)
# Cantidad de filas con uno o más elementos nulos:  19

# OBSERVACIONES / POSIBLES CAMBIOS
#   Tipos de datos erróneos:
#   * Mismos tipos de datos que el dataset anterior.
#   Datos incompletos:
#   * La feature 'age' presenta 16 datos nulos.   Podría completarse con el promedio del valor de la feature o remover las filas donde 'age' es nulo.
#   * La feature 'rooms' presenta 4 datos nulos.  Podría completarse con el promedio del valor de la feature o remover las filas donde 'rooms' es nulo.
#   * En total hay 19 filas con elementos nulos.

# --------------------------------------------------------------------- *

# vivienda_amanda.csv

RawData.vivienda_amanda.info(verbose = True, memory_usage=False)
print("")
print("Cantidad de filas con uno o más elementos nulos: ", Utils.get_null_rows(RawData.vivienda_amanda))

# OUTPUT
# RangeIndex: 1 entries, 0 to 0
# Data columns (total 8 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   area        1 non-null      float64
#  1   area_units  1 non-null      object 
#  2   is_house    1 non-null      int64  
#  3   has_pool    1 non-null      int64  
#  4   age         1 non-null      float64
#  5   lat         1 non-null      float64
#  6   lon         1 non-null      float64
#  7   rooms       1 non-null      int64  
# dtypes: float64(4), int64(3), object(1)
# Cantidad de filas con uno o más elementos nulos:  0

# OBSERVACIONES / POSIBLES CAMBIOS
#   Tipos de datos erróneos:
#   * Mismos tipos de datos que el dataset anterior con excepción de 'rooms' cuyo tipo es int.

# --------------------------------------------------------------------- *