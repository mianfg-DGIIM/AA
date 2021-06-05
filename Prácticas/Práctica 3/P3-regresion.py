# -*- coding: utf-8 -*-

"""
PRÁCTICA 3 - Regresión
======================

Aprendizaje Automático UGR 2020-2021

Alumno: Miguel Ángel Fernández Gutiérrez <mianfg@correo.ugr.es>

"""

import numpy as np
import pandas as pd
from timeit import default_timer

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor

from P3_common import *

SEED = 19122000
MAX_ITER = 2000
np.random.seed(SEED)

do_pause = True  # True para pausar ejecución, False para ejecutar todo
pause = lambda: input("~~~ Pulse Intro para continuar ~~~") if do_pause else None

# =============================================================================

# Cargamos los datos
print("Cargando datos...", end='')
df = df_from_data('./datos/regresion/train.csv')
X, y = xy_from_df(df)
print(" [hecho]")
print("\tNúmero de datos:", df.shape[0])
print("\tNúmero de atributos:", df.shape[1])

# nombres de features y target (target es la última columna)
features, target = names_from_df(df)

pause()

# Vemos cuáles son los atributos más predictivos
importances_sorted = get_importances_sorted(X, y, features, model=DecisionTreeRegressor())
plot_bar(list(list(zip(*importances_sorted)))[0], list(zip(*importances_sorted))[1])
print("\nGráfica: Importancia de variables segúnDecisionTreeRegressor")
most_predictive = [importance[0] for importance in importances_sorted[:2]]
print("Los dos atributos más predictivos son:", most_predictive)
plot_scatter_df(df, most_predictive, target)
print("Gráfica: Importancia de variables segúnDecisionTreeRegressor")

# Mostramos histograma de la distribución del target
plot_hist(y, xlabel=target, ylabel="frecuencia absoluta")
print("Gráfica: Importancia de variables segúnDecisionTreeRegressor")

# Vemos si hay valores perdidos
print("No hay valores perdidos" if np.all(df.notnull()) else "Hay valores perdidos")

pause()

# PREPROCESADO
preprocessing = [
    ("standardization", StandardScaler()),
    ("dimreduction", PCA(0.8)),
    #("standardization3", StandardScaler()),
    ("zspace", PolynomialFeatures(2)),
    ("variancethresh", VarianceThreshold(0.1)),
    #("standardization2", StandardScaler())
]

# REGRESIÓN
# dividimos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# hacemos pipeline de preprocesado
preprocessing_pipeline = Pipeline(preprocessing)

# obtenemos los datos preprocesados
X_train_pre = preprocessing_pipeline.fit_transform(X_train, y_train)
X_test_pre = preprocessing_pipeline.transform(X_test)

# creamos pipeline de entrenamiento
pipeline = Pipeline(preprocessing + [("reg", LinearRegression())])

# especificamos los modelos (espacio de búsqueda)
search_space = [
    {"reg": [SGDRegressor(max_iter=MAX_ITER, random_state=SEED)],
     "reg__alpha": np.logspace(-5, 5, 10),
     "reg__loss": ['squared_loss', 'epsilon_insensitive'],
     "reg__penalty": ['l1', 'l2'],
     "reg__learning_rate": ['optimal', 'adaptive']},
    {"reg": [Ridge(max_iter=MAX_ITER, random_state=SEED)],
     "reg__alpha": np.logspace(-5, 5, 10)},
    {"reg": [Lasso(max_iter=MAX_ITER, random_state=SEED)],
     "reg__alpha": np.logspace(-5, 5, 10)}
]

# cross-validation
best_model = GridSearchCV(pipeline, search_space, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)

plot_correlations(X_train, X_train_pre)
print("\nGráfica: Matriz de correlación antes y después del preprocesado")

pause()

print("\nEntrenando modelo...", end="\n\n")
t0 = default_timer()
best_model.fit(X_train, y_train)
t = default_timer() - t0
print(f"\n[Hecho] Tiempo empleado: {t}s")

best_params = best_model.best_params_
best_estimator = best_model.best_estimator_

print("\nParámetros del mejor modelo:")
print(best_params)

print("\nRaíz de MSE en CV:", (-best_model.best_score_)**0.5)

def sqrt_mean_squared_error(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5

metrics = get_metrics(best_model, [sqrt_mean_squared_error, r2_score], X_train, y_train, X_test, y_test)
print("Raíz de MSE en train:", metrics['train']['sqrt_mean_squared_error'])
print("R² en train:", metrics['train']['r2_score'])
print("Raíz de MSE en test:", metrics['test']['sqrt_mean_squared_error'])
print("R² en test:", metrics['test']['r2_score'])

pause()

plot_learning_curve(best_estimator, X_train, y_train, 'neg_mean_squared_error', cv=5)
print("\nGráfica: Curvas de aprendizaje para el problema de regresión")

pause()

preprocessing_alt = [
    ("standardization", StandardScaler()),
    ("dimreduction", PCA(0.8)),
    ("variancethresh", VarianceThreshold(0.1))
]
pipeline_alt = Pipeline(preprocessing_alt + [("reg", LinearRegression())])
best_model_alt = GridSearchCV(pipeline_alt, search_space, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=0)
dummy_reg = DummyRegressor(strategy='mean')

print("\nEvaluando métricas alternativas...")
best_model_alt.fit(X_train, y_train)
dummy_reg.fit(X_train, y_train)

metrics_alt1=get_metrics(best_model_alt, [sqrt_mean_squared_error, r2_score], X_train, y_train, X_test, y_test)
metrics_alt2=get_metrics(dummy_reg, [sqrt_mean_squared_error, r2_score], X_train, y_train, X_test, y_test)
print("\nSin transformaciones:")
print("\tRaíz de MSE en train:", metrics_alt1['train']['sqrt_mean_squared_error'])
print("\tR² en train:", metrics_alt1['train']['r2_score'])
print("\tRaíz de MSE en test:", metrics_alt1['test']['sqrt_mean_squared_error'])
print("\tR² en test:", metrics_alt1['test']['r2_score'])
print("Regresor dummy:")
print("\tRaíz de MSE en train:", metrics_alt2['train']['sqrt_mean_squared_error'])
print("\tR² en train:", metrics_alt2['train']['r2_score'])
print("\tRaíz de MSE en test:", metrics_alt2['test']['sqrt_mean_squared_error'])
print("\tR² en test:", metrics_alt2['test']['r2_score'])

print("\n¡Esto es todo! :)")