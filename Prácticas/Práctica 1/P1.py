# -*- coding: utf-8 -*-

print("""
PRÁCTICA 1
==========

Aprendizaje Automático UGR 2020-2021

Alumno: Miguel Ángel Fernández Gutiérrez <mianfg@correo.ugr.es>

** Aquí sólo aparecen datos y gráficas, ver memoria para conclusiones **
""")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

pd.options.display.float_format = '{:1.8f}'.format
np.random.seed(1)

do_pause = True  # True para pausar ejecución, False para ejecutar todo
pause = lambda: input("~~~ Pulse Intro para continuar ~~~") if do_pause else None

##############################################################################

# FUNCIONES AUXILIARES

def read_data(file_x, file_y, c1, c1_label, c2, c2_label):
    """
    Function to read data

    Parameters
    ----------
    file_x : string
        Relative route to data file.
    file_y : string
        Relative route to categories file.
    c1 : any
        Category 1.
    c2 : any
        Category 2.
    c1_label : float
        Numeric label for category 1.
    c2_label : float
        Numeric label for category 2.

    Returns
    -------
    x : np.array
        Data vector, in homogeneous coordinates.
    y : np.array
        Categories vector.

    """
    # Leemos los ficheros   
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []  
    # Solo guardamos los datos cuya clase sea la c1 o la c2
    for i in range(0,datay.size):
         if datay[i] == c1 or datay[i] == c2:
             if datay[i] == c1:
                 y.append(c1_label)
             else:
                 y.append(c2_label)
             x.append(np.array([1, datax[i][0], datax[i][1]]))
    x = np.array(x, np.float64)
    y = np.array(y, np.float64)
    
    return x, y

def plot_gd(f, wl, xrange, yrange, title=None,
            xlabel=None, ylabel=None, zlabel=None, view_init=None):
    """
    Gradient descent plot (only for k=2)

    Parameters
    ----------
    f : function
        Function whose surface is to plot.
    wl : array
        Array of ws, trajectories on GD iteration.
    xrange : tuple
        X-axis range.
    yrange : tuple
        Y-axis range.
    title : string, optional
        Plot title.
    xlabel : string, optional
        X-axis label.
    ylabel : string, optional
        Y-axis label.
    zlabel : string, optional
        Z-axis label.
    view_init : tuple, optional
        view_init on matplotlib3d, (elevation, azimuth).
        Defaults to matplotlib default.
    
    Returns
    -------
    None.

    """
    x = np.linspace(xrange[0], xrange[1], 50)
    y = np.linspace(yrange[0], yrange[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1,
                           cmap='jet', alpha=0.8)
    if wl:
        for k in wl:
            ax.plot([wi[0] for wi in wl[k]],
                    [wi[1] for wi in wl[k]],
                    [f(*wi) for wi in wl[k]],
                    '-o', markersize=3,
                    label=(k if not k.startswith('_') else None))
                        # no se mostrará en la leyenda si empieza por '_'
    if title: ax.set(title=title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if zlabel: ax.set_zlabel(zlabel)
    # mostrar si al menos hay que mostrar algo en la leyenda
    if wl and any((not k.startswith('_')) for k in wl): plt.legend()
    if view_init: ax.view_init(*view_init)
    plt.draw()
    plt.pause(.001)

def plot_line(points, labels, title=None, xlabel=None, ylabel=None,
              style='-o', do_show=True):
    """
    Line plot

    Parameters
    ----------
    points : matrix
        Array of arrays, size n.
    labels : array
        Array of labels, must have size n.
        Labels beggining with _ will be ignored.
    title : string, optional
        Plot title.
    xlabel : string, optional
        X-axis label.
    ylabel : string, optional
        Y-axis label.
    style : string, optional
        Plot style, in matplotlib format. The default is '-o'.
    do_show: boolean, optional
        If True, shows plot. If false, can overlap with more plt statements.

    Returns
    -------
    None.

    """
    plt.gca().set_prop_cycle(None)
    for i in range(len(points)):
        plt.plot(points[i], style, label=labels[i])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if do_show: plt.show()
    
def plot_scatter(X, y=None, y_labels=None, ws=None, ws_labels=None,
                 title=None, xlabel=None, ylabel=None, ax=None):
    """
    Scatter plot

    Parameters
    ----------
    X : np.array
        Features data, in homogenous coordinates.
    y : no.array, optional
        Labels, numeric.
    y_labels : array, optional
        Name for each y-label, one-to-one in order with y.
        If none is specified, it will not be shown in legend.
    ws : np.array, optional
        Weights for linear regression (show line).
    ws_labels : TYPE, optional
        Label for each weight, one-to-one in order with ws.
    title : string, optional
        Plot title.
    xlabel : string, optional
        X-axis label.
    ylabel : string, optional
        Y-axis label.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Plot object. If it is specified, this function will render on top
        of what is already plotted. If not, it begins new plot.

    Returns
    -------
    None.

    """
    if not ax: _, ax = plt.subplots()
    if y is not None:
        for _y in np.unique(y):
            _X = X[y == _y]
            ax.scatter(_X[:,1], _X[:,2], s=10, alpha=0.4, cmap='viridis',
                       label=(y_labels[_y] if y_labels else None))
    else:
        ax.scatter(X[:,1], X[:,2], s=10, alpha=0.4, cmap='viridis')
    X_min0, X_max0 = np.min(X[:,1]), np.max(X[:,1])
    X_min1, X_max1 = np.min(X[:,2]), np.max(X[:,2])
    ax.set_xlim(X_min0, X_max0)
    ax.set_ylim(X_min1, X_max1)
    if ws is not None:
        for i in range(len(ws)):
            w = ws[i]
            r = np.array([X_min0, X_max0])
            ax.plot(r, -1*(w[1]*r+w[0])/w[2],
                    label=(ws_labels[i] if ws_labels else None))
    if y_labels or ws_labels: ax.legend()
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.show()

def simula_unif(N, d, size):
    """
    Uniform distribution sample

    Parameters
    ----------
    N : int
        Points.
    d : int
        Dimension.
    size : float
        Bounds.

    Returns
    -------
    np.array
        N points in [-size,size]^d.
    
    """
    return np.random.uniform(-size,size,(N,d))

##############################################################################

print("""
==============================================================================
EJERCICIO 1 · Búsqueda iterativa de óptimos
==============================================================================
""")

# Ejercicio 1.1
def gradient_descent(w0, f, gf, lr, sc, log=None):
    """
    Gradient descent. General algorithm

    Parameters
    ----------
    w0 : np.array or list
        Start point.
    f : function
        Function to minimize.
    gf : function
        Gradient of f.
    lr : function(i, w, f, gf)
        Learning rate.
    sc : function(i, w, f, gf)
        Stop condition.
    log : function(i, w, f, gf), optional
        Logging function.

    Returns
    -------
    w : list<np.array>
        All points per iteration.
    i : list<int>
        Last iteration number.
    
    """
    i = 0
    w = [w0]
    while not sc(i, w, f, gf):
        i += 1
        gi = gf(*w[i-1])
        w.append(w[i-1] - lr(i, w, f, gf)*gi)
        if log: log(i, w, f, gf)
    return w, i

print("Ejercicio 1.2.\n")

# una forma de usar la función gradient_descent general es pasarle un dict
# con los parámetros de la función usando el operador ** (kwargs)
ejer12 = {
    'w0'  : np.array([1, 1], np.float64),
    'f'   : lambda u, v: (u**3*np.exp(v-2)-2*v**2*np.exp(-u))**2,
    'gf'  : lambda u, v: 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u)) \
                * np.array([
                    3*u**2*np.exp(v-2)+2*v**2*np.exp(-u),
                    u**3*np.exp(v-2)-4*v*np.exp(-u)
                ]),
    'lr'  : lambda i, w, f, gf: 0.1,
    'sc'  : lambda i, w, f, gf: f(*w[i]) <= 10**-14,
    'log' : lambda i, w, f, gf: \
                print(f"\tb) Pausado en iteración: {i}, con E(u,v)={f(*w[i])}\n" \
                      + f"\tc) Coordenadas: (u,v)=({w[i][0]}, {w[i][1]})\n") \
                      if f(*w[i]) < 10**-14 else None
}
w_12, _ = gradient_descent(**ejer12)

plot_gd(ejer12['f'], {'_12': w_12}, (-30, 30), (-30, 30), 'Gráfica 1.2.1', 'u', 'v', 'E(u,v)')
print("\t=> Gráfica 1.2.1: progreso del descenso de gradiente")
plot_gd(ejer12['f'], {'_12': w_12}, (0.9, 1.3), (0.8, 1.2), 'Gráfica 1.2.2', 'u', 'v', 'E(u,v)')
print("""\t=> Gráfica 1.2.2: una especie de \"zoom\" en la gráfica anterior, pues
\t   el algoritmo no se evidencia en la gráfica 1.2.1.""")

pause()

print("\nEjercicio 1.3.\n")

# otra forma (la más "inmediata" es usar una función que devuelve
# gradient_descent, pasándole los parámetros
f_13  = lambda x, y: (x+2)**2+2*(y-2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
gf_13 = lambda x, y: np.array([
                2*(x+2)+4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y),
                4*(y-2)+4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
            ])
def gd_13(x0, y0, eta, max_iter=50):
    w0  = np.array([x0, y0], np.float64)
    f   = f_13
    gf  = gf_13
    lr  = lambda i, w, f, gf: eta
    sc  = lambda i, w, f, gf: i >= max_iter
    return gradient_descent(w0, f, gf, lr, sc)

# lo haremos de forma iterativa (podemos hacerlo con todos los valores de
# eta que queramos con sólo cambiar el siguiente array)
etas = [0.01, 0.1]  # <= ¡prueba a modificarlo! :)
# unos diccionarios para trabajar más fácil con los eta
w_13  = {}  # se almacenará el array de ws por cada iteración, con llave eta
N_13  = {}  # se almacenará el número de iteraciones hasta parar, con llave eta
fw_13 = {}  # se almacenará el f(w) por cada w, con llave eta
for eta in etas:
    w_13[eta], N_13[eta] = gd_13(-1, 1, eta)
    fw_13[eta] = []
    for w in w_13[eta]:
        fw_13[eta].append(f_13(*w))

print("\ta) Obtenemos los siguientes resultados en la última iteración:\n")
for eta in etas:
    print(f"\teta={eta}, i={N_13[eta]}:\n\t\t(x,y)=({w_13[eta][-1][0]}, {w_13[eta][-1][1]})\n\t\tf(x,y)={f_13(*w_13[eta][-1])}")

points_13 = []
labels_13 = []
for eta in etas:
    points_13.append(fw_13[eta])
    labels_13.append(fr"$\eta$={eta}")
plot_line(points_13, labels_13,
          title="Gráfica 1.3.1", xlabel="i", ylabel="f(w(i))")
print(f"\n\t=> Gráfica 1.3.1: Valor de f por iteraciones para cada eta en {etas}")

pause()

print("""\n\tb) Obtenemos los siguientes resultados en la última iteración
\t   (con eta=0.01):\n""")

w0s = [(-1, 1), (-0.5, -0.5), (1, 1), (2.1, -2.1), (-3, 3), (-2, 2)]
w_13b  = {}  # se almacenará el array de ws por cada iteración, con llave w0
N_13b  = {}  # se almacenará el número de iteraciones hasta parar, con llave w0
fw_13b = {}  # se almacenará el f(w) por cada w, con llave w0
for w0 in w0s:
    w0_key = f"$w_0$={w0}"
    w_13b[w0_key], N_13b[w0_key] = gd_13(w0[0], w0[1], 0.01)
    fw_13b[w0_key] = []
    for w in w_13b[w0_key]:
        fw_13b[w0_key].append(f_13(*w))

df_13 = {'w0': w0s, '(x,y)': [], 'f(x,y)': []}
for w0 in w0s:
    w0_key = f"$w_0$={w0}"
    df_13['(x,y)'].append(w_13b[w0_key][-1])
    df_13['f(x,y)'].append(fw_13b[w0_key][-1])
print(pd.DataFrame(data=df_13))

# mostramos las gráficas
plot_gd(f_13, w_13b, (-3.5, 3.5), (-3.5, 3.5),
        'Gráfica 1.3.2', 'x', 'y', 'f(x,y)', (40, 80))
print("""\n\t=> Gráfica 1.3.2: Visualización de las iteraciones del algoritmo
\t   para cada punto inicial (w0)""")

pause()

##############################################################################

print("""
==============================================================================
EJERCICIO 2 · Regresión lineal
==============================================================================
""")

# error para regresión lineal
reg_err = \
    lambda X, y, w: np.linalg.norm(X.dot(w)-y)**2/len(y) #np.sum(np.dot(w[np.newaxis,:],x.T).flatten()-y)/len(y)

# derivada del error para regresión lineal
d_reg_err = \
    lambda X, y, w: 2/len(y)*(X.T.dot(X.dot(w)-y))

# pseudoinversa de Moore-Penrose
pseudoinverse = lambda x: np.linalg.pinv(x)
                # equivalentemente: np.linalg.inv(x.T.dot(x)).dot(x.T)

def stochastic_gradient_descent(X, y, lr, sc, bs=32, log=None):
    """
    Stochastic Gradient Descent (SGD),
    general implementation

    Parameters
    ----------
    X : numpy.ndarray, shape (N, d+1)
        Features array.
    y : numpy.ndarray, shape (N,)
        Labels array.
    lr : function(i, w, e, b)
        Learning rate.
    sc : function(i, w, e, b)
        Stop condition.
    bs : int, optional
        Block size. The default is 32.
    log : function(i, w, e, b), optional
        Logging function.

    Returns
    -------
    w : list, length i
        All points per iteration.
    e : list, length i-1
        All errors per iteration (except start point).

    """
    w0 = np.zeros((X.shape[1],))    # comenzamos en (0, 0, ...)
    w = [w0]                        # array de pesos
    e = []                          # array de errores
    i = 0                           # iteración actual
    ids = np.arange(y.size)         # índices que recorreremos
    b = []                          # batch actual
    bb = 0                          # comienzo del batch (batch begin)
    
    while not sc(i, w, e, b):
        # si comenzamos una era, permutamos los índices
        if bb == 0:
            ids = np.random.permutation(ids)
        # tomamos batch de tamaño bs (batch size) comenzando en bb (batch begin)
        b = ids[bb:bb+bs]
        e.append(reg_err(X[b,:], y[b], w[i]))
        w.append(w[i]-lr(i, w, e, b)*d_reg_err(X[b,:], y[b], w[i]))
        i += 1
        bb += bs                    # cálculo del nuevo bb (batch begin)
        if bb > y.size: bb = 0      # si es el último batch, inicia nueva era
        if log: log(i, w, e, b)
    
    return w, e

def sgd_iter_eta(X, y, eta=0.01, max_iter=1000, bs=32, log=None):
    """
    Stochastic Gradient Descent (SGD),
    with fixed learning rate and maximum of iterations

    Parameters
    ----------
    X : numpy.ndarray, shape (N, d+1)
        Features array.
    y : numpy.ndarray, shape (N,)
        Labels array.
    eta : int, optional
        Learning rate. The default is 0.01.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.
    bs : int, optional
        Block size. The default is 32.
    log : function, optional
        Logging function.

    Returns
    -------
    w : list, length i
        All points per iteration.
    e : list, length i-1
        All errors per iteration (except start point).

    """
    lr = lambda i, w, e, b: eta
    sc = lambda i, w, e, b: i >= max_iter
    return stochastic_gradient_descent(X, y, lr, sc, bs, log)

def regression_pseudoinverse(X, y):
    """
    Regression with pseudoinverse

    Parameters
    ----------
    X : numpy.ndarray, shape (N, d+1)
        Features array.
    y : numpy.ndarray, shape (N,)
        Labels array.

    Returns
    -------
    w : list, length i
        All points per iteration.
    e : list, length i-1
        All errors per iteration (except start point).

    """
    return pseudoinverse(X).dot(y)

print("Ejercicio 2.1")

X_train, y_train = read_data('datos/X_train.npy', 'datos/y_train.npy', 1, -1, 5, 1)
X_test, y_test   = read_data('datos/X_test.npy', 'datos/y_test.npy', 1, -1, 5, 1)

w_21_pinv     = regression_pseudoinverse(X_train, y_train)
w_21_sgd, _   = sgd_iter_eta(X_train, y_train, max_iter=1000)
w_21_sgd_2, _ = sgd_iter_eta(X_train, y_train, max_iter=30000)

ein_21_pinv   = reg_err(X_train, y_train, w_21_pinv)
ein_21_sgd    = reg_err(X_train, y_train, w_21_sgd[-1])
ein_21_sgd_2  = reg_err(X_train, y_train, w_21_sgd_2[-1])
eout_21_pinv  = reg_err(X_test, y_test, w_21_pinv)
eout_21_sgd   = reg_err(X_test, y_test, w_21_sgd[-1])
eout_21_sgd_2 = reg_err(X_test, y_test, w_21_sgd_2[-1])

print("\n\tRegresión lineal por pseudoinversa")
print(f"\t\tE_in = {ein_21_pinv}\n\t\tE_out = {eout_21_pinv}")
print("\n\tRegresión lineal por SGD")
print(f"""\t\tEn 1000 iteraciones:
\t\t\tE_in = {ein_21_sgd}\n\t\t\tE_out = {eout_21_sgd}
\t\tEn 30000 iteraciones:
\t\t\tE_in = {ein_21_sgd_2}\n\t\t\tE_out = {eout_21_sgd_2}""")

plot_scatter(X_train, y_train, y_labels={-1: "1", 1: "5"},
             ws=[w_21_pinv, w_21_sgd[-1], w_21_sgd_2[-1]], ws_labels=["Pseudoinversa", "SGD (1000 iters.)", "SGD (30000 iters.)"],
             title="Gráfica 2.1", xlabel="Intensidad promedio", ylabel="Simetría")
print("\n\t=> Gráfica 2.1: comparación de pseudoinversa y SGD")

pause()

print("\nEjercicio 2.2.1\n")

# siempre trabajamos con x en coordenadas homogéneas (con primera coord. 1)
prepend_1 = lambda x: np.concatenate((np.ones((len(x),1)), x), axis=1)

def generate_experiment(f):
    # generamos los datos
    X = simula_unif(1000, 2, 1)
    # asignamos las etiquetas usando f
    y = np.array([f(*x) for x in X])
    # cambiamos aleatoriamente las etiquetas de un 10% de éstas
    rep_indexes = np.random.choice(1000, size=100, replace=False)
    y[rep_indexes] = -y[rep_indexes]
    
    return prepend_1(X), y

sign = lambda z: 1 if z >= 0 else -1
f_22 = lambda x1, x2: sign((x1-0.2)**2+x2**2-0.6)
# lo hacemos todo para ahorrar código: de este modo hacemos ambos apartados
X_22, y_22 = generate_experiment(f_22)

print("\ta) Muestra de N=1000 puntos en [-1,1]x[-1,1] generada.\n")
plot_scatter(X_22, title="Gráfica 2.2.1")
print("\t=> Gráfica 2.2.1: muestra de N=1000 puntos en [-1,1]x[-1,1]")

pause()

print("\n\tb) Mapa de etiquetas generado.\n")
plot_scatter(X_22, y_22,
             title="Gráfica 2.2.2", xlabel="$x_1$", ylabel="$x_2$")
print("\t=> Gráfica 2.2.2: muestra anterior con etiquetas asignadas")

pause()

w_22, _ = sgd_iter_eta(X_22, y_22)
ein_22 = reg_err(X_22, y_22, w_22[-1])
print("\n\tc) Modelo de regresión lineal ajustado con SGD.")
print(f"\t\tE_in = {ein_22}\n")
plot_scatter(X_22, y_22, ws=[w_22[-1]],
             title="Gráfica 2.2.3", xlabel="$x_1$", ylabel="$x_2$")
print("\t=> Gráfica 2.2.3: recta de regresión obtenida con SGD sobre muestra")

pause()

print("\n\td) Ejecutando experimento anterior 1000 veces...\n")

ein_22d  = []
eout_22d = []
for i in range(1000):
    X_train, y_train = generate_experiment(f_22)
    w, e = sgd_iter_eta(X_train, y_train)
    ein_22d.append(e[-1])
    X_test, y_test = generate_experiment(f_22)
    eout_22d.append(reg_err(X_test, y_test, w[-1]))

ein_22d_avg  = np.average(ein_22d)
eout_22d_avg = np.average(eout_22d)

print("\t   Experimento finalizado. Resultados medios:")
print(f"\t\t   E_in={ein_22d_avg}\n\t\t   E_out={eout_22d_avg}")

pause()

print("\nEjercicio 2.2.2.")

phi_2 = lambda x1, x2: [1, x1, x2, x1*x2, x1**2, x2**2]
X_trans = np.array([phi_2(*x[1:]) for x in X_22])
w_trans, _ = sgd_iter_eta(X_trans, y_22)
ein_trans = reg_err(X_trans, y_22, w_trans[-1])

print(f"""\n\tVector de pesos obtenido (con SGD, no lineal):\n\t\tw={w_trans[-1]}
\n\tComparación de Ein (ambos SGD):
\t\tmodelo lineal:       {ein_22}
\t\tmodelo transformado: {ein_trans}""")

# vamos a modificar plot_scatter para mostrar la curva de regresión no lineal
def plot_scatter_trans(X, y=None, y_labels=None, ws=None, ws_labels=None,
                     title=None, xlabel=None, ylabel=None):
    _, ax = plt.subplots()
    X_min0, X_max0 = np.min(X[:,1]), np.max(X[:,1])
    X_min1, X_max1 = np.min(X[:,2]), np.max(X[:,2])
    x_t, y_t = np.meshgrid(np.linspace(X_min0-0.5, X_max0+0.5, 100),
                           np.linspace(X_min1-0.5, X_max1+0.5, 100))
    w = ws[1]
    f_t = w[0] + w[1]*x_t + w[2]*y_t + w[3]*x_t*y_t + w[4]*x_t*x_t + w[5]*y_t*y_t
    plt.contour(x_t, y_t, f_t, [0])
    plot_scatter(X, y, y_labels, ws[:1], None, title, xlabel, ylabel, ax)

plot_scatter_trans(X_22, y_22, ws=[w_22[-1], w_trans[-1]],
                   title="Gráfica 2.2.4", xlabel="$x_1$", ylabel="$x_2$")
print("\n\t=> Gráfica 2.2.4: recta de regresión y curva de regresión no lineal")

pause()

##############################################################################

print("""
==============================================================================
BONUS · Método de Newton
==============================================================================
""")

def newton(w0, f, gf, hf, lr, sc, log=None):
    """
    Newton Method

    Parameters
    ----------
    w0 : np.array or list
        Start point.
    f : function
        Function to minimize.
    gf : function
        Gradient of f.
    hf : function
        Hessian of f.
    lr : function(i, w, f, gf, hf)
        Learning rate.
    sc : function(i, w, f, gf, hf)
        Stop condition.
    log : function(i, w, f, gf, hf), optional
        Logging function.

    Returns
    -------
    w : list<np.array>
        All points per iteration.
    i : list<int>
        Last iteration number.

    """
    i = 0
    w = [w0]
    while not sc(i, w, f, gf, hf):
        i += 1
        w.append(w[i-1] - lr(i, w, f, gf, hf)*np.linalg.inv(hf(*w[i-1])).dot(gf(*w[i-1])))
        if log: log(i, w, f, gf, hf)
    return w, i

# vamos a implementarlo para un eta fijo y un máximo de iteraciones
def newton_iter_eta(w0, f, gf, hf, eta=0.01, max_iter=1000, log=None):
    """
    

    Parameters
    ----------
    w0 : np.array or list
        Start point.
    f : function
        Function to minimize.
    gf : function
        Gradient of f.
    hf : function
        Hessian of f.
    eta : float, optional
        Learning rate. The default is 0.01.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.
    log : function(i, w, f, gf, hf), optional
        Logging function.

    Returns
    -------
    w : list<np.array>
        All points per iteration.
    i : list<int>
        Last iteration number.

    """
    lr = lambda i, w, f, gf, hf: eta
    sc = lambda i, w, f, gf, hf: i >= max_iter
    return newton(w0, f, gf, hf, lr, sc, log)

# retomamos la función f_13, ahora indicamos aquí su hessiana
hf_13 = lambda x, y: np.array([
                [
                    2-8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
                    8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
                ],
                [
                    8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
                    4-8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
                ]
            ])

# probaremos con dos algoritmos
algoritmos = ["newton", "gd"]
# y variaremos los etas con w0 fijo
etas = [0.01, 0.1]  # <= ¡prueba a modificarlo! :)
eta_fijo = 0.01
# así como los w0 con eta fijo
w0s = [(-1, 1), (-0.5, -0.5), (1, 1), (2.1, -2.1), (-3, 3), (-2, 2)] # <= ídem :)
w0_fijo = (-1, 1)
w_bonus_etas, fw_bonus_etas, gfw_bonus_etas = {}, {}, {}  # eta variable, w0 fijo
w_bonus_w0s, fw_bonus_w0s, gfw_bonus_w0s = {}, {}, {}    # w0 variable, eta fija

print(f"""Compararemos Newton y GD, mediante:\n
\teta variable, w0 fijo
\t\tetas = {etas}
\t\tw0 = {w0_fijo}\n
\tw0 variable, eta fijo
\t\teta = {eta_fijo}
\t\tw0s = {w0s}""")

pause()

for algoritmo in algoritmos:
    for eta in etas:
        w0 = np.array(w0_fijo, np.float64)
        w_bonus_etas[(algoritmo, eta)] = \
            newton_iter_eta(w0, f_13, gf_13, hf_13, eta)[0] if algoritmo == 'newton' \
            else gd_13(w0[0], w0[1], eta, max_iter=1000)[0]
        fw_bonus_etas[(algoritmo, eta)], gfw_bonus_etas[(algoritmo, eta)] = [], []
        for w in w_bonus_etas[(algoritmo, eta)]:
            fw_bonus_etas[(algoritmo, eta)].append(f_13(*w))
            gfw_bonus_etas[(algoritmo, eta)].append(np.linalg.norm(gf_13(*w)))
    for w0 in w0s:
        eta = eta_fijo
        _w0 = np.array(w0, np.float64)
        w_bonus_w0s[(algoritmo, w0)] = \
            newton_iter_eta(_w0, f_13, gf_13, hf_13, eta)[0] if algoritmo == 'newton' \
            else gd_13(w0[0], w0[1], eta, max_iter=1000)[0]
        fw_bonus_w0s[(algoritmo, w0)], gfw_bonus_w0s[(algoritmo, w0)] = [], []
        for w in w_bonus_w0s[(algoritmo, w0)]:
            fw_bonus_w0s[(algoritmo, w0)].append(f_13(*w))
            gfw_bonus_w0s[(algoritmo, w0)].append(np.linalg.norm(gf_13(*w)))

points_bonus_etas, labels_bonus_etas = [], []
points_bonus_w0s, labels_bonus_w0s = [], []
gs_bonus_etas, gs_bonus_w0s = [], []
for algoritmo in algoritmos:
    for eta in etas:
        points_bonus_etas.append(fw_bonus_etas[(algoritmo, eta)])
        gs_bonus_etas.append(gfw_bonus_etas[(algoritmo, eta)])
        labels_bonus_etas.append(fr"{algoritmo}, $\eta$={eta}")
    for w0 in w0s:
        points_bonus_w0s.append(fw_bonus_w0s[(algoritmo, w0)])
        gs_bonus_w0s.append(gfw_bonus_w0s[(algoritmo, w0)])
        labels_bonus_w0s.append(fr"{algoritmo}, $w_0$={w0}")

print("\nCálculos finalizados\n")
plot_line(np.array(points_bonus_etas)[:len(etas),:50], labels_bonus_etas[:len(etas)], style='-',
          do_show=False)
plot_line(np.array(points_bonus_etas)[len(etas):,:50], labels_bonus_etas[len(etas):], style='-.',
          title="Gráfica B.1", xlabel="i", ylabel="f(w(i))")
print("""\t=> Gráfica B.1: comportamiento Newton vs. GD, con eta variable, w0 fijo
\t   Función evaluada en cada punto. Número de iteraciones mostradas: 50\n""")
plot_line(points_bonus_etas[:len(etas)], labels_bonus_etas[:len(etas)], style='-',
          do_show=False)
plot_line(points_bonus_etas[len(etas):], labels_bonus_etas[len(etas):], style='-.',
          title="Gráfica B.2", xlabel="i", ylabel="f(w(i))")
print("""\t=> Gráfica B.2: comportamiento Newton vs. GD, con eta variable, w0 fijo
\t   Función evaluada en cada punto. Número de iteraciones mostradas: 1000\n""")
plot_line(np.array(points_bonus_w0s)[:len(w0s),:50], labels_bonus_w0s[:len(w0s)], style='-',
          do_show=False)
plot_line(np.array(points_bonus_w0s)[len(w0s):,:50], labels_bonus_w0s[len(w0s):], style='-.',
          title="Gráfica B.3", xlabel="i", ylabel="f(w(i))")
print("""\t=> Gráfica B.3: comportamiento Newton vs. GD, con w0 variable, eta fija
\t   Función evaluada en cada punto. Número de iteraciones mostradas: 50\n""")
plot_line(points_bonus_w0s[:len(w0s)], labels_bonus_w0s[:len(w0s)], style='-',
          do_show=False)
plot_line(points_bonus_w0s[len(w0s):], labels_bonus_w0s[len(w0s):], style='-.',
          title="Gráfica B.4", xlabel="i", ylabel="f(w(i))")
print("""\t=> Gráfica B.4: comportamiento Newton vs. GD, con w0 variable, eta fija
\t   Función evaluada en cada punto. Número de iteraciones mostradas: 1000\n""")
plot_line(np.array(gs_bonus_w0s)[:len(w0s),:50], labels_bonus_w0s[:len(w0s)], style='-',
          do_show=False)
plot_line(np.array(gs_bonus_w0s)[len(w0s):,:50], labels_bonus_w0s[len(w0s):], style='-.',
          title="Gráfica B.5", xlabel="i", ylabel=r"$\Vert\nabla$ f(w(i))$\Vert$")
print("""\t=> Gráfica B.5: comportamiento Newton vs. GD, con w0 variable, eta fija
\t   Norma del gradiente en cada punto. Número de iteraciones mostradas: 50\n""")
plot_line(gs_bonus_w0s[:len(w0s)], labels_bonus_w0s[:len(w0s)], style='-',
          do_show=False)
plot_line(gs_bonus_w0s[len(w0s):], labels_bonus_w0s[len(w0s):], style='-.',
          title="Gráfica B.6", xlabel="i", ylabel=r"$\Vert\nabla$ f(w(i))$\Vert$")
print("""\t=> Gráfica B.6: comportamiento Newton vs. GD, con w0 variable, eta fija
\t   Norma del gradiente en cada punto. Número de iteraciones mostradas: 1000""")

pause()

print("\n¡Esto es todo! ;)")
