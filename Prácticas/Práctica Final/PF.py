# -*- coding: utf-8 -*-

"""
PRÁCTICA FINAL
==============

Aprendizaje Automático UGR 2020-2021

Alumnos:
 - Celia Arias Martínez <ariasmartinez@correo.ugr.es>
 - Miguel Ángel Fernández Gutiérrez <mianfg@correo.ugr.es>

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

#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix as _plot_confusion_matrix

SEED = 23
MAX_ITER = 2000
np.random.seed(SEED)

do_pause = True  # True para pausar ejecución, False para ejecutar todo
pause = lambda: input("~~~ Pulse Intro para continuar ~~~") if do_pause else None

# ===========================================================================================================================
# FUNCIONES AUXILIARES
# ===========================================================================================================================

# ==== MANEJO DE DATOS ====

def df_from_data(filename, **kwargs):
    return pd.read_csv(filename, **kwargs)

def xy_from_df(df):
    X = df.values[:,:-1]
    y = df.values[:,-1]
    return X, y

def names_from_df(df):
    target = df.columns.values[-1]
    features = [f for f in df.columns if f != target]
    return features, target

def get_importances(X, y, features, model):
    model.fit(X, y)
    return list(zip(features, model.feature_importances_))

def get_importances_sorted(X, y, features, model):
    importances = get_importances(X, y, features, model)
    importances.sort(key=lambda x: x[1], reverse=True)
    return importances

def tsne(df, target, n_components=2, standardizer=None, **tsne_args):
    features = [f for f in df.columns if f != target]
    x = df.loc[:, features].values
    y = df.loc[:, [target]].values
    if standardizer: x = standardizer().fit_transform(x)
    
    tsne = TSNE(n_components=n_components, **tsne_args)
    tsne_results = tsne.fit_transform(x)
    tsne_df = pd.DataFrame(data=tsne_results, columns=[f'TSNE{i+1}' for i in range(n_components)])
    final_df = pd.concat([tsne_df, df[[target]]], axis=1)
    return final_df

def get_metrics(model, metrics, X_train, y_train, X_test, y_test):
    output = {}
    for tag, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        model_metrics = {}
        for metric in metrics:
            model_metrics[metric.__name__] = metric(y, y_pred)
        output[tag] = model_metrics
    return output

# ==== GRÁFICAS ====

def plottable(function):
    def wrapper(*args, **kwargs):
        plt = function(*args)
        if 'xlabel' in kwargs.keys(): plt.xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs.keys(): plt.ylabel(kwargs['ylabel'])
        if 'title' in kwargs.keys(): plt.title(kwargs['title'])
        plt.show()
    return wrapper

@plottable
def plot_bar(tags, values):
    plt.figure(figsize=(20,6), dpi=200)
    plt.xticks(rotation=90)
    plt.bar(tags, values)
    return plt

@plottable
def plot_scatter_df(df, features, target):
    fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=200)
    sns.scatterplot(x=features[0], y=target, data=df, ax=ax[0])
    sns.scatterplot(x=features[1], y=target, data=df, ax=ax[1])
    fig.tight_layout()
    return plt

@plottable
def plot_hist(data):
    plt.figure(dpi=200)
    plt.hist(data, edgecolor='white')
    return plt

@plottable
def plot_bar_data(data):
    values = []
    tags = list(set(data))
    for tag in tags:
        values.append(list(data).count(tag))
    plt.figure(dpi=200)
    plt.bar(tags, values)
    plt.xticks(tags)
    return plt

@plottable
def plot_correlations(X, X_pre):
    fig, ax = plt.subplots(1,2, figsize=(10, 4), dpi=200)
    # correlación antes de preprocesado
    with np.errstate(invalid='ignore'):  # ignorar inválidos sin procesar
        corr = np.abs(np.corrcoef(X, rowvar=False))
    im = ax[0].matshow(corr, cmap='viridis')
    ax[0].title.set_text("Antes de preprocesado")
    # correlación tras preprocesado
    corr_pre = np.abs(np.corrcoef(X_pre, rowvar=False))
    im = ax[1].matshow(corr_pre, cmap='viridis')
    ax[1].title.set_text("Tras preprocesado")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6)
    return plt

@plottable
def plot_learning_curve(estimator, X, y, scoring, title=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    # Basado en: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=200)
    
    if scoring == 'accuracy':
        score_name = 'Accuracy'
    elif scoring == 'neg_mean_squared_error':
        # si es neg_mean_squared_error, representamos la raíz de MSE
        score_name = 'Raíz de MSE'
    else:
        score_name = scoring

    if title: axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(score_name)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    #if scoring == 'neg_mean_squared_error':
    #    train_scores = np.sqrt(-train_scores)
    #    test_scores = np.sqrt(-test_scores)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

@plottable
def plot_confusion_matrix(model, X_train, X_test, y_train, y_test):
    fig, ax = plt.subplots(1,2, figsize=(10, 4), dpi=200)
    img = _plot_confusion_matrix(model, X_train, y_train, cmap='viridis', values_format='d', ax=ax[0], colorbar=False)
    img.ax_.set_title("Matriz de confusión en train")
    img.ax_.set_xlabel("etiqueta predicha")
    img.ax_.set_ylabel("etiqueta real")
    img = _plot_confusion_matrix(model, X_test, y_test, cmap='viridis', values_format='d', ax=ax[1], colorbar=True)
    img.ax_.set_title("Matriz de confusión en test")
    img.ax_.set_xlabel("etiqueta predicha")
    img.ax_.set_ylabel("etiqueta real")
    return plt


# ===========================================================================================================================

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
