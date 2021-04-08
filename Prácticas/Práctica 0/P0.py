# -*- coding: utf-8 -*-

print_if = lambda boolean, *args: print(*args) if boolean else None
do_print = False   # poner a False si no queremos explicaciones en output

print_if(do_print, """
PRÁCTICA 0
==========

Aprendizaje Automático UGR 2020-2021

Alumno: Miguel Ángel Fernández Gutiérrez <mianfg@correo.ugr.es>

---------------------------------------------------------------
""")

from sklearn import datasets     # scikit-learn
import numpy as np               # numpy
import pandas as pd              # pandas
import matplotlib.pyplot as plt  # matplotlib

print_if(do_print, """
# ====================
# PARTE 1
# ====================
""")

print_if(do_print, "\n=> Leer la base de datos de iris que hay en scikit-learn")
iris = datasets.load_iris()

x = iris.data
y = iris.target

print("\nBase de datos de iris cargada")
print(f"Número de muestras: {x.shape[0]}. Número de características: {x.shape[1]}")

print_if(do_print, "\n=> Obtener las características (x) y la clase (y)")
print(f"\nCaracterísticas: {iris.feature_names}")
print(f"Clases: {iris.target_names}")
print("\nVisualizamos con DataFrame los datos de iris que hemos cargado:")
print(pd.DataFrame(x, columns=iris.feature_names))

print_if(do_print, "\n=> Nos quedamos con las características 1 y 3")
# Nos quedamos con las características 1 y 3
# Podemos quedarnos con estas características en el array de numpy 'data':
print("\nComo array de numpy:")
x_less = x[:, [0,2]]
features_less = [iris.feature_names[i] for i in [0,2]]
print(x[:, [0,2]])
# Podemos visualizarlo cómodamente haciendo uso de un DataFrame
print("\nComo DataFrame:")
print(pd.DataFrame(x, columns=iris.feature_names).iloc[:, [0, 2]])
# print(pd.DataFrame(x_less, columns=features_less))
#  alternativamente podemos pasarle directamente los datos que queremos al df

print_if(do_print, "\n=> Visualizar los datos usando un scatter plot")
# crear un dict con los diferentes valores y los índices donde se encuentran éstos
# ej.: lambda([1,2,3,1]) = {1: [0,3], 2: [1], 3: [2]}
ind = lambda arr: {val: [i for i, x in enumerate(arr) if x == val] for val in list(set(arr))}
indexed = ind(y)
colors = ['orange', 'black', 'green']
for val in indexed:
    # mostramos los puntos de cada variedad
    plt.scatter(x_less[indexed[val],0], x_less[indexed[val],1], c=colors[val-1], \
                s=10, label=iris.target_names[val-1])
plt.xlabel(features_less[0])
plt.ylabel(features_less[1])
plt.title("iris dataset")
plt.legend()
plt.show()
print("\nPlot creado")

print("\n---------------------------------------------------------------")
print_if(do_print, """
# ====================
# PARTE 2
# ====================
""")

print_if(do_print, "\n=> Separar en training (75 % de los datos) y test (25 %) aleatoriamente \
conservando la proporción de elementos en cada clase tanto en training como en test. \
Con esto se pretende evitar que haya clases infra-representadas en entrenamiento o test.")
# Podemos hacerlo directamente usando scikit-learn
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=19, stratify=y)
print("\nMuestra para training (usando sklearn.model_selection):")
print(pd.DataFrame(np.hstack((x_train, y_train.reshape((-1,1)))), columns=(iris.feature_names+["category"])))
print("\nMuestra para test (usando sklearn.model_selection):")
print(pd.DataFrame(np.hstack((x_test, y_test.reshape((-1,1)))), columns=(iris.feature_names+["category"])))
# Otra forma es, por ejemplo, obtener un array de índices generado aleatoriamente:
# los primeros que conforman el 75% del total será training, y el resto test
random_indexes = np.arange(x.shape[0])
np.random.shuffle(random_indexes)
tp = 0.75
# la idea es generar una partición de los índices con el criterio anterior
ri_train, ri_test = random_indexes[:int(random_indexes.shape[0]*tp)], random_indexes[int(random_indexes.shape[0]*tp):]
# usamos la partición anterior para obtener los sets
x_train_2, x_test_2 = x[ri_train,:], x[ri_test,:]
y_train_2, y_test_2 = y[ri_train], y[ri_test]
print("\nMuestra para training (\"manualmente\"):")
print(pd.DataFrame(np.hstack((x_train_2, y_train_2.reshape((-1,1)))), columns=(iris.feature_names+["category"])))
print("\nMuestra para test (\"manualmente\"):")
print(pd.DataFrame(np.hstack((x_test_2, y_test_2.reshape((-1,1)))), columns=(iris.feature_names+["category"])))

print("\n---------------------------------------------------------------")
print_if(do_print, """
# ====================
# PARTE 3
# ====================
""")

print_if(do_print, "\n=> Obtener 100 valores equiespaciados entre 0 y 4π")
values = np.random.uniform(low=0, high=np.pi, size=(1,100))
values = np.sort(values)
print(f"\n{values}")

print_if(do_print, "\n=> Obtener el valor de sin(x), cos(x) y tanh(sin(x)+cos(x)) para los 100 valores \
anteriormente calculados\n")
values = np.random.uniform(low=0, high=np.pi, size=(1,100))
fvalues = {
    'sin(x)': np.sin(values),
    'cos(x)': np.cos(values),
    'tanh(sin(x)+cos(x))': np.tanh(np.sin(values)+np.cos(values))
}
for f in fvalues:
    print(f"{f}: {fvalues[f]}")

print_if(do_print, "\n=> Visualizar las tres curvas simultáneamente en el mismo plot (con líneas discontinuas \
en verde, negro y rojo")
colors = ['green', 'black', 'red']
i = 0
for f in fvalues:
    # ordenamos los valores para mostrar la gráfica correctamente
    ordered = np.hstack((values.T, fvalues[f].T))[values.argsort()][0,:]
    plt.plot(ordered[:,0], ordered[:,1], c=colors[i], label=f)
    i += 1
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('100 valores aleatorios')
plt.legend()
plt.show()
print("\nPlot creado")
