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
import seaborn as sns
from timeit import default_timer

# from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix as _plot_confusion_matrix

#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SEED = 23
MAX_ITER = 10000
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

def pca(df, target, n_components=2, standardizer=None, **pca_args):
    features = [f for f in df.columns if f != target]
    x = df.loc[:, features].values
    y = df.loc[:, [target]].values
    if standardizer: x = standardizer().fit_transform(x)
    
    tsne = PCA(n_components=n_components, **pca_args)
    tsne_results = tsne.fit_transform(x)
    tsne_df = pd.DataFrame(data=tsne_results, columns=[f'PCA{i+1}' for i in range(n_components)])
    final_df = pd.concat([tsne_df, df[[target]].astype(int)], axis=1)
    return final_df

def tsne(df, target, n_components=2, standardizer=None, **tsne_args):
    features = [f for f in df.columns if f != target]
    x = df.loc[:, features].values
    y = df.loc[:, [target]].values
    if standardizer: x = standardizer().fit_transform(x)
    
    tsne = TSNE(n_components=n_components, **tsne_args)
    tsne_results = tsne.fit_transform(x)
    tsne_df = pd.DataFrame(data=tsne_results, columns=[f'TSNE{i+1}' for i in range(n_components)])
    final_df = pd.concat([tsne_df, df[[target]].astype(int)], axis=1)
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

def categories_plot(y_train, y_test):
    fig, ax = plt.subplots(1, 2, figsize=(10,4), dpi=200)
    ax[0].title.set_text("train")
    ax[1].title.set_text("test")
    values = []
    tags = list(set(y_train))
    for tag in tags:
        values.append(list(y_train).count(tag))
    ax[0].bar(tags, values)
    ax[0].set_xticks(tags)
    values = []
    tags = list(set(y_test))
    for tag in tags:
        values.append(list(y_test).count(tag))
    ax[1].bar(tags, values)
    ax[1].set_xticks(tags)
    return plt

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
    plt.xticks(rotation=45)
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
def plot_bar_data(data, labels=None):
    values = []
    tags = list(set(data))
    for tag in tags:
        values.append(list(data).count(tag))
    if labels: tags = [labels[tag] for tag in tags]
    plt.figure(dpi=200)
    plt.bar(tags, values)
    plt.xticks(tags)
    return plt

@plottable
def plot_correlations(X, X_pre):
    fig, ax = plt.subplots(1,2, figsize=(10, 4), dpi=200)
    # correlación antes de preprocesado
    with np.errstate(invalid='ignore'):  # ignorar inválidos sin procesar
        corr = np.abs(np.corrcoef(X.astype(float), rowvar=False))
    im = ax[0].matshow(corr, cmap='viridis')
    ax[0].title.set_text("Antes de preprocesado")
    # correlación tras preprocesado
    corr_pre = np.abs(np.corrcoef(X_pre.astype(float), rowvar=False))
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
    axes[0].set_title("Learning curve")

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
    img = _plot_confusion_matrix(model, X_test, y_test, cmap='viridis', values_format='d', ax=ax[1], colorbar=False)
    img.ax_.set_title("Matriz de confusión en test")
    img.ax_.set_xlabel("etiqueta predicha")
    img.ax_.set_ylabel("etiqueta real")
    return plt

# ===========================================================================================================================

print("""\
PRÁCTICA FINAL
==============

Aprendizaje Automático UGR 2020-2021

Alumnos:
 - Celia Arias Martínez <ariasmartinez@correo.ugr.es>
 - Miguel Ángel Fernández Gutiérrez <mianfg@correo.ugr.es>

""")

# === CARGAR DATOS ===

#leemos los datos del archivo xls y los convertimos en un data frame de pandas
excel_data_df = pd.read_excel('./datos/CTG.xls', sheet_name='Data')
#eliminamos la penúltima columna, correspondiente a las etiquetas de problema que no vamos a solucionar
data = np.concatenate((excel_data_df.values[:-3,10:31], excel_data_df.values[:-3, 45].reshape(-1,1)), axis=1)
#especificamos la fila correspondiente a los nombres de los atributos
df = pd.DataFrame(data[1:,:], columns=data[0,:])
print("Datos cargados")

pause()

# === EXPLORAR DATASET ===

# pasamos los datos del dataframe a las matrices X, y
X, y = xy_from_df(df)
X = X.astype(float)
y = y.astype(int)
features, target = names_from_df(df)

# dividimos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
df_train = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis=1), columns=data[0,:])

# mostramos datos de atributos
print("\nEstadísticos de atributos:")
print(pd.DataFrame({'min': df.min(axis=0).values, 'max': df.max(axis=0).values, 'mean': df.mean(axis=0).values,
    'std': df.std(axis=0).values}, index=df.columns.values))

df_tsne = tsne(df_train, 'NSP', n_components=2, standardizer=StandardScaler)
plt.subplots(dpi=200)
sns.scatterplot(x="TSNE1", y="TSNE2", data=df_tsne, hue='NSP', palette="Accent")
print("\nGráfica: Proyección t-SNE 2D")

df_pca = pca(df_train, 'NSP', n_components=2, standardizer=StandardScaler)
plt.subplots(dpi=200)
sns.scatterplot(x="PCA1", y="PCA2", data=df_pca, hue='NSP', palette="Accent")
print("Gráfica: Proyección PCA 2D")

plot_bar_data(y_train, xlabel="categoría", ylabel="frecuencia absoluta")
print("Gráfica: Frecuencia de target en train")

# importances_sorted = get_importances_sorted(X, y, features, model=DecisionTreeRegressor())
# plot_bar(list(list(zip(*importances_sorted)))[0], list(zip(*importances_sorted))[1])
# print("Gráfica: Importancia de variables segúnDecisionTreeRegressor")

pause()

# === PREPROCESADO ===

preprocessing = [
    ("standardization", StandardScaler()),
    ("dimreduction", PCA(0.99)),
]

# hacemos pipeline de preprocesado
preprocessing_pipeline = Pipeline(preprocessing)

# obtenemos los datos preprocesados
X_train_pre = preprocessing_pipeline.fit_transform(X_train, y_train)
X_test_pre = preprocessing_pipeline.transform(X_test)

# creamos pipeline de entrenamiento
pipeline = Pipeline(preprocessing + [("clf", LinearRegression())])

plot_correlations(X_train, X_train_pre)
print("\nGráfica: Matriz de correlación antes y después del preprocesado")

pause()

# === ENTRENAMIENTO ===

search_spaces = {
    'Regresión Logística':
        {"clf": [SGDClassifier(loss='log', max_iter=MAX_ITER, random_state=SEED)],
         "clf__penalty": ['l1', 'l2'],
         "clf__alpha": np.logspace(-5, -1, 10)},
    'SVM':
        {"clf": [SVC(kernel='rbf', max_iter=MAX_ITER)],
         "clf__gamma": ['auto', 'scale'],
         "clf__C": np.logspace(0, 4, 10)},
    'MLP':
        {"clf": [MLPClassifier(learning_rate="adaptive", activation="tanh", solver="sgd", max_iter=2000)],
         "clf__hidden_layer_sizes": [40, 50, 60],
         "clf__alpha": np.logspace(-3, 1, 5)},
    'Random Forest':
        {"clf": [RandomForestClassifier(random_state = SEED)],
         "clf__max_depth": [5, 10, 15, 20],
         "clf__n_estimators": [50, 100, 200, 400, 600]}
}

best_models = {key: GridSearchCV(pipeline, search_spaces[key], scoring='accuracy', cv=5, n_jobs=-1, verbose=10) for key in search_spaces.keys()}
time = {}

print("\nComenzando cross-validation...\n")

for key in best_models.keys():
    t0 = default_timer()
    best_models[key].fit(X_train, y_train)
    time[key] = default_timer() - t0

print("\nCross-validation terminada")

pause()

# === ANÁLISIS DE RESULTADOS DE CV ===

best_params = {key: best_models[key].best_params_ for key in best_models.keys()}
best_estimators = {key: best_models[key].best_estimator_ for key in best_models.keys()}
best_scores = {key: best_models[key].best_score_ for key in best_models.keys()}
metrics = {key: get_metrics(best_models[key], [accuracy_score], X_train, y_train, X_test, y_test) for key in best_models.keys()}
measures = {}
for alg in search_spaces.keys():
    measures[alg] = {
        'cv'    : 100*best_scores[alg],
        'train' : 100*metrics[alg]['train']['accuracy_score'],
        'test'  : 100*metrics[alg]['test']['accuracy_score'],
        'time'  : time[alg]
    }

print("\nParámetros de los mejores modelos:")
print(best_params)
print("\nAccuracy en cv, test y train (%) y tiempo empleado en cv (s):")
print(pd.DataFrame(measures).transpose())

plot_confusion_matrix(best_estimators['SVM'], X_train, X_test, y_train, y_test)
print("\nGráfica: matriz de confusión en train y en test")

pause()

# === CURVAS DE APRENDIZAJE ===

plot_learning_curve(best_estimators['SVM'], X_train, y_train, 'accuracy', cv=5)
print("\nCurvas de aprendizaje para el mejor modelo (SVM)")

pause()

print("\n¡Esto es todo! :)")
