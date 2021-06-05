# -*- coding: utf-8 -*-

"""
PRÁCTICA 3 - Clasificación
==========================

Aprendizaje Automático UGR 2020-2021

Alumno: Miguel Ángel Fernández Gutiérrez <mianfg@correo.ugr.es>

"""

import numpy as np
import pandas as pd
from timeit import default_timer

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, f_classif
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
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
df = df_from_data('./datos/clasificacion/Sensorless_drive_diagnosis.csv', header=None).rename(columns={48:'target'})
X, y = xy_from_df(df)
print(" [hecho]")
print("\tNúmero de datos:", df.shape[0])
print("\tNúmero de atributos:", df.shape[1])

# nombres de features y target (target es la última columna)
features, target = names_from_df(df)

pause()

# Mostramos bar plot del target
plot_bar(df.values[:,-1], xlabel="clase (target)", ylabel="frecuencia absoluta")
print("\nGráfica: Distribución de clases en los datos de clasificación")

# Vemos si hay valores perdidos
print("No hay valores perdidos" if np.all(df.notnull()) else "Hay valores perdidos")

pause()

df_tsne = tsne(df, 'target', standardizer=StandardScaler)
plt.subplots(dpi=200)
sns.scatterplot(x="TSNE1", y="TSNE2", data=df_tsne, hue='target', palette="Accent")
print("\nGráfica: Diagrama t-SNE (en 2D) para problema de clasificación")

# PREPROCESADO
preprocessing1 = [
    ("standardization", StandardScaler()),
    ("dimreduction", RFE(estimator=LinearRegression(), n_features_to_select=24)),
    ("zspace", PolynomialFeatures(2)),
    ("variancethresh", VarianceThreshold(0.1)),
]
preprocessing2 = [
    ("standardization", StandardScaler()),
    ("variancethresh0", VarianceThreshold()),
    ("dimreduction", SelectKBest(f_classif, k=24)),
    ("zspace", PolynomialFeatures(2)),
    ("variancethresh", VarianceThreshold(0.1)),
]

# CLASIFICACIÓN
# dividimos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# hacemos pipeline de preprocesado
preprocessing_pipeline1 = Pipeline(preprocessing1)
preprocessing_pipeline2 = Pipeline(preprocessing2)

# obtenemos los datos preprocesados
X_train_pre1 = preprocessing_pipeline1.fit_transform(X_train, y_train)
X_test_pre1 = preprocessing_pipeline1.transform(X_test)
X_train_pre2 = preprocessing_pipeline2.fit_transform(X_train, y_train)
X_test_pre2 = preprocessing_pipeline2.transform(X_test)

# creamos pipeline de entrenamiento
pipeline1 = Pipeline(preprocessing1 + [("clf", LinearRegression())])
pipeline2 = Pipeline(preprocessing2 + [("clf", LinearRegression())])

# especificamos los modelos (espacio de búsqueda)
search_space = [
    {"clf": [SGDClassifier(loss='hinge', penalty='l2', max_iter=MAX_ITER, random_state=SEED)],
     "clf__alpha": np.logspace(-5, 5, 5)},
    {"clf": [SGDClassifier(loss='log', max_iter=MAX_ITER, random_state=SEED)],
     "clf__penalty": ['l1', 'l2'],
     "clf__alpha": np.logspace(-5, 5, 5)}
]

# cross-validation
best_model1 = GridSearchCV(pipeline1, search_space, scoring='accuracy', cv=5, n_jobs=-1, verbose=10)
best_model2 = GridSearchCV(pipeline2, search_space, scoring='accuracy', cv=5, n_jobs=-1, verbose=10)

plot_correlations(X_train, X_train_pre1)
print("\nGráfica: Matriz de correlación antes y después del preprocesado, RFE")
plot_correlations(X_train, X_train_pre2)
print("Gráfica: Matriz de correlación antes y después del preprocesado, ANOVA")

pause()

print("\nEntrenando modelo, RFE...", end="\n\n")
t0 = default_timer()
best_model1.fit(X_train, y_train)
t = default_timer() - t0
print(f"\n[Hecho] Tiempo empleado: {t}s")

print("\nEntrenando modelo, ANOVA...", end="\n\n")
t0 = default_timer()
best_model2.fit(X_train, y_train)
t = default_timer() - t0
print(f"\n[Hecho] Tiempo empleado: {t}s")

best_params1 = best_model1.best_params_
best_estimator1 = best_model1.best_estimator_
best_params2 = best_model2.best_params_
best_estimator2 = best_model2.best_estimator_

print("\nParámetros del mejor modelo, RFE:")
print(best_params1)
print("\nRaíz de MSE en CV:", best_model1.best_score_)

print("\nParámetros del mejor modelo, ANOVA:")
print(best_params2)
print("\nRaíz de MSE en CV:", best_model2.best_score_)

metrics1 = get_metrics(best_model1, [accuracy_score], X_train, y_train, X_test, y_test)
metrics2 = get_metrics(best_model2, [accuracy_score], X_train, y_train, X_test, y_test)
print("Métricas RFE:")
print("\tAccuracy en train:", metrics1['train']['accuracy_score'])
print("\tAccuracy en test:", metrics1['test']['accuracy_score'])
print("Métricas ANOVA:")
print("\tAccuracy en train:", metrics2['train']['accuracy_score'])
print("\tAccuracy en test:", metrics2['test']['accuracy_score'])

pause()

plot_learning_curve(best_estimator1, X_train, y_train, 'neg_mean_squared_error', cv=5)
print("\nGráfica: Curvas de aprendizaje para el problema de clasificación")

pause()

plot_confusion_matrix(best_estimator1, X_train, X_test, y_train, y_test)
print("\nGráfica: Matriz de confusión para los conjuntos de train y test")

preprocessing_alt = [
    ("standardization", StandardScaler()),
    ("dimreduction", RFE(estimator=LinearRegression(), n_features_to_select=24)),
    ("variancethresh", VarianceThreshold(0.1)),
]
pipeline_alt = Pipeline(preprocessing_alt + [("clf", LinearRegression())])
best_model_alt = GridSearchCV(pipeline_alt, search_space, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=0)
dummy_classif = DummyClassifier(strategy='stratified', random_state=SEED)

print("\nEvaluando métricas alternativas...")
best_model_alt.fit(X_train, y_train)
dummy_classif.fit(X_train, y_train)

metrics_alt1=get_metrics(best_model_alt, [accuracy_score], X_train, y_train, X_test, y_test)
metrics_alt2=get_metrics(dummy_classif, [accuracy_score], X_train, y_train, X_test, y_test)
print("Sin transformaciones:")
print("\tAccuracy en train:", metrics_alt1['train']['accuracy_score'])
print("\tAccuracy en test:", metrics_alt1['test']['accuracy_score'])
print("Regresor dummy:")
print("\tAccuracy en train:", metrics_alt2['train']['accuracy_score'])
print("\tAccuracy en test:", metrics_alt2['test']['accuracy_score'])

print("\n¡Esto es todo! :)")