import pandas as pd
import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix as _plot_confusion_matrix

import seaborn as sns
import random

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
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    
    See
    ---
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    scoring : function or str
        Scoring function.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
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