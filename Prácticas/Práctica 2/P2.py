# -*- coding: utf-8 -*-

print("""
PRÁCTICA 2
==========

Aprendizaje Automático UGR 2020-2021

Alumno: Miguel Ángel Fernández Gutiérrez <mianfg@correo.ugr.es>

** Aquí sólo aparecen datos y gráficas, ver memoria para conclusiones **
""")

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

do_pause = True  # True para pausar ejecución, False para ejecutar todo
pause = lambda: input("~~~ Pulse Intro para continuar ~~~") if do_pause else None

##############################################################################

# FUNCIONES AUXILIARES

# añadimos la columna de unos al principio para seguir el formato de X
prepend_1 = lambda x: np.concatenate((np.ones((len(x),1)), x), axis=1)

def simula_unif(N, dim, rango):
    return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

def plot_fz(X, y=None, fz=None, title=None, xlabel='x', ylabel='y', fz_only=False):
    """
    Scatter plot with implicit function.

    Parameters
    ----------
    X : np.array
        Data array, non-homogenous coordinates.
    y : np.array, optional
        Labels array.
    fz : function, optional
        Implicit function.
    title : string, optional
        Title. The default is None.
    xlabel : string, optional
        X-axis label. The default is 'x'.
    ylabel : string, optional
        Y-axis label. The default is 'y'.
    fz_only : boolean, optional
        If true, display implicit plot and not meshgrid. The default is False.

    Returns
    -------
    None.

    """
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.001
    
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    
    f, ax = plt.subplots(figsize=(8,6))

    if y is not None:
        min_y = y.min(axis=0)
        max_y = y.max(axis=0)
        ticks = np.linspace(min_y, max_y, 3)
        grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    
    if fz:
        if not fz_only:
            pred_y = fz(grid)
            pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
            contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1, alpha=0.8 if fz else 0)
            ax_c = f.colorbar(contour, ax=ax)
            ax_c.set_label('$f(x, y)$')
            ax_c.set_ticks(ticks)
        XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),
                             np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
        positions = np.vstack([XX.ravel(), YY.ravel()])
        ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black' if fz_only else 'white')
    
    im = ax.scatter(X[:,0], X[:,1], c=y if y is not None else np.ones(X.shape[0]), s=50, linewidth=2, 
                    cmap='RdYlBu', edgecolor='white')
    if y is not None:
        plt.gca().add_artist(plt.legend(*im.legend_elements(), title='labels', loc='upper right'))
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    plt.show()

def plot_ws(X, y=None, ws=None, ws_labels=None, title=None, xlabel='x', ylabel='y'):
    """
    Scatter plot with lines from weight vectors.

    Parameters
    ----------
    X : np.array
        Data, in homogenous coordinates.
    y : np.array, optional
        Labels. The default is None.
    ws : list of np.array, optional
        List of weights. The default is None.
    ws_labels : list of string, optional
        List of labels, one-to-one with ws. The default is None.
    title : string, optional
        Title. The default is None.
    xlabel : string, optional
        X-axis label. The default is 'x'.
    ylabel : string, optional
        Y-axis label. The default is 'y'.

    Returns
    -------
    None.

    """
    min_xy = X[:,1:].min(axis=0)
    max_xy = X[:,1:].max(axis=0)
    border_xy = (max_xy-min_xy)*0.001
    
    f, ax = plt.subplots(figsize=(8,6))
    
    im = ax.scatter(X[:,1], X[:,2], c=y if y is not None else np.ones(X.shape[0]), s=50, linewidth=2,
                   cmap='RdYlBu', edgecolor='white')
    
    if y is not None:
        plt.gca().add_artist(plt.legend(*im.legend_elements(), loc='upper right'))
    
    if ws:
        for i in range(len(ws)):
            w = ws[i]
            x = np.array([min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]])
            ax.plot(x, (-w[1]*x - w[0])/w[2], label = ws_labels[i] if ws_labels else None)
    
    if ws_labels:
        ax.legend(loc='upper left')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    plt.show()

def plot_line(points, labels=None, title=None, xlabel=None, ylabel=None,
              style='-o'):
    """
    Line plot.
    
    Parameters
    ----------
    points : list of list
        Array of arrays, size n.
    labels : list
        Array of labels, must have size n, optional
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
    plt.subplots(figsize=(8,6))
    plt.gca().set_prop_cycle(None)
    for i in range(len(points)):
        plt.plot(points[i], style, label=labels[i] if labels else None)
    if labels: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.show()

def read_data(file_x, file_y, c1, c1_label, c2, c2_label):
    """
    Function to read data.
    
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

def perc_ok(X, y, classifier):
    """
    Percentage of well-classified values.

    Parameters
    ----------
    X : np.array
        Data, in homogenous coordinates.
    y : np.array
        Labels.
    classifier : function
        Classifier, in implicit form.

    Returns
    -------
    float
        Percentage of well-classified values, in [0,100].

    """
    # los que sean >= 0 estarán correctamente clasificados
    vals = y*classifier(X)
    return 100*len(vals[vals>=0])/len(vals)

def classifier_w(w):
    """
    Hyperplane classifier from weight vector.

    Parameters
    ----------
    w : np.array
        Weight vector.

    Returns
    -------
    function
        Implicit-type classifier.

    """
    return lambda X: X.dot(w)

##############################################################################

print("""
==============================================================================
EJERCICIO 1 · Complejidad de H y ruido
==============================================================================
""")

# === Ejercicio 1.1 ===
print("Ejercicio 1.1.\n")

# primero obtenemos los puntos en ambas distribuciones
X_unif = simula_unif(50, 2, (-50, +50))
X_gaus = simula_gaus(50, 2, (5, 7))

# los mostramos
plot_fz(X_unif, title="Gráfica 1.1.1")
print("\t=> Gráfica 1.1.1: scatter plot distribución uniforme")

plot_fz(X_gaus, title="Gráfica 1.1.2")
print("\t=> Gráfica 1.1.2: scatter plot distribución gaussiana")

pause()

# === Ejercicio 1.2 ===
print("\nEjercicio 1.2.\n")

# generamos los datos con simula_unif y mostramos los datos
# en efecto, todos los puntos están bien clasificados
X = simula_unif(100, 2, (-50, 50))
r = simula_recta((-50, 50))
f_recta = lambda X: X[:,1] - r[0]*X[:,0] - r[1]
w_recta = [-r[1], -r[0], 1]
sign = np.vectorize(lambda t: 1 if t >= 0 else -1)
y = sign(f_recta(X))

print("\ta) Clasificación por una recta (sin ruido)")

plot_fz(X, y, f_recta, fz_only=True,
        title="Gráfica 1.2.1")
print("\n\t\t=> Gráfica 1.2.1: clasificación por una recta (sin ruido)")
print(f"\t\t\tPorcentaje de bien clasificados: {perc_ok(X, y, f_recta)}")

pause()

# añadimos ruido
y_pert = y.copy()
perc = 10  # 10% de las etiquetas <== ¡cámbialo!
for etiqueta in [-1, 1]:
    indexes = [i for i, v in enumerate(y) if v == etiqueta]
    choice = np.random.choice(indexes, int(perc/100*len(indexes)), replace=False)
    y_pert[choice] = -y_pert[choice]

print("\n\tb) Clasificación por una recta (con ruido)")
print(f"\t\tRuido incorporado: {perc}%")

plot_fz(X, y_pert, f_recta, fz_only=True,
        title="Gráfica 1.2.2")
print("\n\t\t=> Gráfica 1.2.2: clasificación por una recta (con ruido)")
print(f"\t\t\tPorcentaje de bien clasificados: {perc_ok(X, y_pert, f_recta)}")

pause()

classifiers = [
    {
        'name': 'Recta',
        'f':    f_recta
    },
    {
        'name': 'Elipse 1',
        'f':    lambda X: (X[:,0]-10)**2 + (X[:,1]-20)**2 - 400
    },
    {
        'name': 'Elipse 2',
        'f':    lambda X: 0.5*(X[:,0]+10)**2 + (X[:,1]-20)**2 - 400
    },
    {
        'name': 'Hipérbola',
        'f':    lambda X: 0.5*(X[:,0]-10)**2 - (X[:,1]+20)**2 - 400
    },
    {
        'name': 'Parábola',
        'f':    lambda X: X[:,1] - 20*X[:,0]**2 - 5*X[:,0] + 3
    }
]

print("\n\tc) Diversos clasificadores")
print("\n\t\tResultados obtenidos con los diversos clasificadores:\n")
print("\t\tClasificador    Bien clasificados (%)")
print("\t\t------------    ---------------------")
for classifier in classifiers:
    print("\t\t{:<12}    {:>21}".format(classifier['name'], perc_ok(X, y_pert, classifier['f'])))

print("")

for i, classifier in enumerate(classifiers):
    plot_fz(X, y_pert, classifier['f'],
         title=fr'Gráfica 1.2.{i+3}: $f_{i+1}$ ({classifier["name"]})')
    print(f'\t\t=> Gráfica 1.2.{i+3}: clasificación por f_{i+1} ({classifier["name"]})')

pause()

##############################################################################

print("""
==============================================================================
EJERCICIO 2 · Modelos lineales
==============================================================================
""")

# === Ejercicio 2.a) ===
print("Ejercicio 2.a)\n")

def ajusta_pla(datos, label, vini, max_iter=1000,
               show_pb=False):
    """
    Adjust solution hyperplane via PLA.

    Parameters
    ----------
    datos : np.array
        Data, in homogenous coordinates.
    label : np.array
        Labels.
    max_iter : int
        Maximum number of iterations. The default is 1000.
    vini : np.array
        Start vector.
    show_pb : boolean, optional
        If true, show progressbar in console. The default is False.

    Returns
    -------
    w : np.array
        Solution weight vector.
    iters : int
        Number of iterations required.
    ws : list<np.array>
        List of weight vectors (one per iteration).

    """
    
    w = vini.copy()
    iters = 0
    ws = [w.copy()]
    
    while iters < max_iter:
        w_prev = w.copy()
        
        for x, y in zip(datos, label):
            if sign(w.dot(x)) != y:
                w += y*x
        
        iters += 1
        
        # progressbar rudimentario
        if show_pb:
            if iters % int(max_iter/20) == 0: print('.', end='')

        ws.append(w.copy())
        
        if (w == w_prev).all():
            break
    
    if show_pb: print(" [hecho]")
    
    return w, iters, ws

# implementamos el experimento como una función, para poder usarla
# en el apartado siguiente
def experiment_pla(X, y, max_iter=1000, show_pb=False):
    # X debe estar en coordenadas homogéneas
    def print_exp(tag, X, y, ws, iters):
        print("\t\t{:<14}    {:>10}    {:>12.04f}    {:>12.04f}    {:>22.04f}    {:>22.04f}".format(
            tag,
            len(iters),
            np.mean(iters),
            np.std(iters),
            np.mean([perc_ok(X, y, classifier_w(w)) for w in ws]),
            np.std([perc_ok(X, y, classifier_w(w)) for w in ws])
        ))
    
    # primer experimento: vini = 0
    if show_pb: print("\t\tEjecutando experimento  0 ", end='')
    w_0, it_0, ws_0 = ajusta_pla(X, y, np.zeros(3), max_iter=max_iter, show_pb=show_pb)
    
    # segundo experimento: 10 repeticiones vini aleatorios
    w_r_list, it_r_list = [], []
    for i in range(10):  # posible cambiar número de repeticiones
        if show_pb: print(f"\t\tEjecutando experimento {i+1:>2} ", end='')
        w_r, it_r, ws_r = ajusta_pla(X, y, np.random.rand(3), show_pb=show_pb)
        w_r_list.append(w_r)
        it_r_list.append(it_r)
    
    print("\n\t\tVector inicial    Num. reps.    Media iters.    Desv. iters.    Media bien clasif. (%)    Desv. bien clasif. (%)")
    print("\t\t--------------    ----------    ------------    ------------    ----------------------    ----------------------")
    print_exp("Ceros", X, y, [w_0], [it_0])
    print_exp("Aleatorios", X, y, w_r_list, it_r_list)
    
    # devuelve:
    #   w_0  : hiperplano solución con vini=0
    #   ws_0 : progreso de hiperplano solución con vini=0
    #   w_r  : hiperplano solución del último experimento con vini aleatorio
    #   ws_r : progreso de hiperplano solución del último experimento con vini aleatorio
    return w_0, ws_0, w_r, ws_r

print("\t1) Usando datos simulados en apartado 1.2.a)")
w_0_sr, ws_0_sr, w_r_sr, ws_r_sr = experiment_pla(prepend_1(X), y)

plot_ws(prepend_1(X), y, [w_recta, w_0_sr, w_r_sr],
        ["Recta original", r"Recta PLA, $v_{ini}$=0", r"Recta PLA, $v_{ini}$ aleatorio"],
        title="Gráfica 2.a).1",)
print("\n\t\t=> Gráfica 2.a).1: comparación de hiperplanos obtenidos por PLA\n\t\t   (sin ruido)")

percs_r_sr = [perc_ok(prepend_1(X), y, classifier_w(w)) for w in ws_r_sr]

plot_line([percs_r_sr], title="Gráfica 2.a).2", xlabel="iteración", ylabel="bien clasificados (%)", style="-")
print("\t\t=> Gráfica 2.a).2: progreso de porc. bien clasificados por PLA\n\t\t   (sin ruido)")

pause()

print("\n\t2) Usando datos simulados en apartado 1.2.b)\n")
w_0_cr, ws_0_cr, w_r_cr, ws_r_cr = experiment_pla(prepend_1(X), y_pert, show_pb=True)

plot_ws(prepend_1(X), y, [w_recta, w_0_cr, w_r_cr],
        ["Recta original", r"Recta PLA, $v_{ini}$=0", r"Recta PLA, $v_{ini}$ aleatorio"],
        title="Gráfica 2.a).3")
print("\n\t\t=> Gráfica 2.a).3: comparación de hiperplanos obtenidos por PLA\n\t\t   (con ruido)")

percs_r_cr = [perc_ok(prepend_1(X), y, classifier_w(w)) for w in ws_r_cr]

plot_line([percs_r_cr], title="Gráfica 2.a).4", xlabel="iteración", ylabel="bien clasificados (%)", style="-")
print("\t\t=> Gráfica 2.a).4: progreso de porc. bien clasificados por PLA\n\t\t   (con ruido)")

pause()

# === Ejercicio 2.b) ===
print("\nEjercicio 2.b)\n")

def lr_sgd(X, y, eta=0.01):
    """
    Logistic regression, SGD.

    Parameters
    ----------
    X : np.array
        Data, in homogenous coordinates.
    y : np.array
        Labels.
    eta : float, optional
        Learning rate. The default is 0.01.

    Returns
    -------
    w : np.array
        Solution weight vector.
    ep : int
        Number of epochs required.

    """
    N, d = X.shape
    w = np.zeros(d)
    indexes = np.arange(N)
    
    grad = lambda x, y, w: -(y*x)/(1+np.exp(y*w.dot(x)))
    
    ep = 0
    while True:
        ep += 1
        w_prev = w.copy()
        indexes = np.random.permutation(indexes)
        for index in indexes:
            w -= eta*grad(X[index], y[index], w)
        
        if np.linalg.norm(w_prev - w) < 0.01:
            break
    
    return w, ep

# error de regresión logística
def err_lr(X, y, w):
    return np.mean(np.log(1+np.exp(-y*X.dot(w))))

def experimento_lr(num_reps=100):
    E_ins, E_outs, percs_ok_in, percs_ok_out, eps = [], [], [], [], []
    print("\tEjecutando experimentos ", end='')
    for i in range(num_reps):
        X_train = simula_unif(100, 2, (0, 2))
        X_train_hom = prepend_1(X_train)
        X_test = simula_unif(1000, 2, (0, 2))
        X_test_hom = prepend_1(X_test)
        r_gen = simula_recta((0, 2))
        w_gen = [-r_gen[0], -r_gen[1], 1]
        f_gen = lambda X: X[:,1] - r_gen[1]*X[:,0] - r_gen[0]
        y_train = sign(f_gen(X_train))
        y_test = sign(f_gen(X_test))
        w_lr, ep_lr = lr_sgd(X_train_hom, y_train)
        E_ins.append(err_lr(X_train_hom, y_train, w_lr))
        E_outs.append(err_lr(X_test_hom, y_test, w_lr))
        percs_ok_in = perc_ok(X_train_hom, y_train, classifier_w(w_lr))
        percs_ok_out = perc_ok(X_test_hom, y_test, classifier_w(w_lr))
        eps.append(ep_lr)
        if i % int(num_reps/20) == 0: print('.', end='')
    print(" [hecho]\n")
    
    print("\t\tTipo          Media error    Desv. error    Media bien clasif. (%)    Desv. bien clasif. (%)")
    print("\t\t----------    -----------    -----------    ----------------------    ----------------------")
    print("\t\tIn-sample     {:>11.04f}    {:>11.04f}    {:>22.04f}    {:>22.04f}".format(
        np.mean(E_ins), np.std(E_ins), np.mean(percs_ok_in), np.std(percs_ok_in)
    ))
    print("\t\tOut-sample    {:>11.04f}    {:>11.04f}    {:>22.04f}    {:>22.04f}".format(
        np.mean(E_outs), np.std(E_outs), np.mean(percs_ok_out), np.std(percs_ok_out)
    ))
    
    print(f"\n\t\tMedia épocas: {np.mean(eps):.04f}")
    print(f"\t\tDesv. épocas: {np.std(eps):.04f}")
    # devolvemos los resultados del último experimento, para visualización
    return X_train, X_test, y_train, y_test, w_gen, w_lr, ep_lr

X_train, X_test, y_train, y_test, w_gen, w_lr, _ = experimento_lr()

plot_ws(prepend_1(X_train), y_train, [w_gen, w_lr], ["Recta original", "Recta LR SGD"],
        title="Gráfica 2.b).1")
print("\n\t\t=> Gráfica 2.b).1: resultado LR SGD, conjunto de training")

plot_ws(prepend_1(X_test), y_test, [w_gen, w_lr], ["Recta original", "Recta LR SGD"],
        title="Gráfica 2.b).2")
print("\t\t=> Gráfica 2.b).2: resultado LR SGD, conjunto de testing")

pause()

##############################################################################

print("""
==============================================================================
EJERCICIO BONUS · Clasificación de dígitos
==============================================================================
""")

Xd_train, yd_train = read_data('datos/X_train.npy', 'datos/y_train.npy', 4, -1, 8, 1)
Xd_test, yd_test = read_data('datos/X_test.npy', 'datos/y_test.npy', 4, -1, 8, 1)

# error de clasificación
def class_err(X, y, w):
    return np.mean(sign(X.dot(w)) != y)

# regresión lineal por pseudoinversa
def ajusta_pseudoinverse(datos, label):
    pseudoinverse = lambda x: np.linalg.pinv(datos)
    return pseudoinverse(datos).dot(label)
    
def ajusta_pla_pocket(datos, label, vini, max_iter=1000,
                      show_pb=False):
    """
    Adjust solution hyperplane via PLA-Pocket.

    Parameters
    ----------
    datos : np.array
        Data, in homogenous coordinates.
    label : np.array
        Labels.
    max_iter : int
        Maximum number of iterations. The default is 1000.
    vini : np.array
        Start vector.
    show_pb : boolean, optional
        If true, show progressbar in console. The default is False.

    Returns
    -------
    w_best : np.array
        Solution weight vector.
    iters : int
        Number of iterations required.
    ws : list<np.array>
        List of weight vectors (one per iteration).

    """    
    w = vini.copy()
    w_best = w.copy()
    err_best = class_err(datos, label, w_best)
    ws = [w_best.copy()]
    iters = 0
    
    while iters < max_iter:
        w_prev = w.copy()
        
        for x, y in zip(datos, label):
            if sign(w.dot(x)) != y:
                w += y*x
        
        err = class_err(datos, label, w)
        if err < err_best:
            err_best = err
            w_best = w.copy()
        
        ws.append(w_best.copy())
        
        iters += 1
        
        # progressbar rudimentario
        if show_pb:
            if iters % int(max_iter/20) == 0: print('.', end='')
        
        if (w == w_prev).all():
            break
    
    if show_pb: print(" [hecho]")
    
    return w_best, iters, ws

print("\tEjecutando experimentos...\n")
max_iters = 500
w_pinv = ajusta_pseudoinverse(Xd_train, yd_train)
print("\t\tEjecutando experimento 1 ", end='')
w_pocket_cero, _, ws_pocket_cero = ajusta_pla_pocket(Xd_train, yd_train, np.zeros(3), max_iters, show_pb=True)
print("\t\tEjecutando experimento 2 ", end='')
w_pla_random, _, ws_pla_random = ajusta_pla(Xd_train, yd_train, np.random.rand(3), max_iters, show_pb=True)
print("\t\tEjecutando experimento 3 ", end='')
w_pocket_random, _, ws_pocket_random = ajusta_pla_pocket(Xd_train, yd_train, np.random.rand(3), max_iters, show_pb=True)


E_in_pinv = class_err(Xd_train, yd_train, w_pinv)
E_in_pocket_cero = class_err(Xd_train, yd_train, w_pocket_cero)
E_in_pocket_random = class_err(Xd_train, yd_train, w_pocket_random)
E_test_pinv = class_err(Xd_test, yd_test, w_pinv)
E_test_pocket_cero = class_err(Xd_test, yd_test, w_pocket_cero)
E_test_pocket_random = class_err(Xd_test, yd_test, w_pocket_random)

print("\n\t\tTipo de error    Algoritmo                       Valor del error")
print("\t\t-------------    ----------------------------    ---------------")
print(f"\t\tE_in             Pseudoinversa                   {E_in_pinv:>15.04f}")
print(f"\t\tE_in             PLA-Pocket (v_ini cero)         {E_in_pocket_cero:>15.04f}")
print(f"\t\tE_in             PLA-Pocket (v_ini aleatorio)    {E_in_pocket_random:>15.04f}")
print(f"\t\tE_test           Pseudoinversa                   {E_test_pinv:>15.04f}")
print(f"\t\tE_test           PLA-Pocket (v_ini cero)         {E_test_pocket_cero:>15.04f}")
print(f"\t\tE_test           PLA-Pocket (v_ini aleatorio)    {E_test_pocket_random:>15.04f}")

plot_ws(Xd_train, yd_train, [w_pinv, w_pocket_cero, w_pocket_random],
        ["Pseudoinversa", r"PLA-Pocket, $v_{ini}=0$", r"PLA-Pocket, $v_{ini}$ aleatorio"],
        title="Gráfica B.1")
print("\n\t\t=> Gráfica B.1: resultado pinv vs. PLA-Pocket, conjunto de training")

plot_ws(Xd_test, yd_test, [w_pinv, w_pocket_cero, w_pocket_random],
        ["Pseudoinversa", r"PLA-Pocket, $v_{ini}=0$", r"PLA-Pocket, $v_{ini}$ aleatorio"],
        title="Gráfica B.2")
print("\t\t=> Gráfica B.2: resultado pinv vs. PLA-Pocket, conjunto de testing")

perc_ok_pla = [perc_ok(Xd_train, yd_train, classifier_w(w)) for w in ws_pla_random]
perc_ok_pocket = [perc_ok(Xd_train, yd_train, classifier_w(w)) for w in ws_pocket_random]

plot_line([perc_ok_pla, perc_ok_pocket], ["PLA", "PLA-Pocket"], style="-",
          title="Gráfica B.3", xlabel="iteración", ylabel="bien clasificados (%)")
print("\t\t=> Gráfica B.3: comparación entre PLA y PLA-Pocket")

err_b_vc = lambda e, n, d_vc, delta: e+np.sqrt((8/n)*np.log(4*((2*n)**d_vc+1)/delta))
err_b_hoeff = lambda e, n, h, delta: e+np.sqrt((1/(2*n))*np.log((2*h)/delta))

print("\n\t\tError usado    Cota        Algoritmo                        Valor del error")
print("\t\t-----------    ---------    ----------------------------    ---------------")
print(f"\t\tE_in           VC           Pseudoinversa                   {err_b_vc(E_in_pinv, len(Xd_train), 3, 0.05):>15.04f}")
print(f"\t\tE_in           VC           PLA-Pocket (v_ini cero)         {err_b_vc(E_in_pocket_cero, len(Xd_train), 3, 0.05):>15.04f}")
print(f"\t\tE_in           VC           PLA-Pocket (v_ini aleat.)       {err_b_vc(E_in_pocket_random, len(Xd_train), 3, 0.05):>15.04f}")
print(f"\t\tE_in           Hoeffding    Pseudoinversa                   {err_b_hoeff(E_in_pinv, len(Xd_train), 2**(64*3), 0.05):>15.04f}")
print(f"\t\tE_in           Hoeffding    PLA-Pocket (v_ini cero)         {err_b_hoeff(E_in_pocket_cero, len(Xd_train), 2**(64*3), 0.05):>15.04f}")
print(f"\t\tE_in           Hoeffding    PLA-Pocket (v_ini aleat.)       {err_b_hoeff(E_in_pocket_random, len(Xd_train), 2**(64*3), 0.05):>15.04f}")
print(f"\t\tE_test         Hoeffding    Pseudoinversa                   {err_b_hoeff(E_test_pinv, len(Xd_train), 1, 0.05):>15.04f}")
print(f"\t\tE_test         Hoeffding    PLA-Pocket (v_ini cero)         {err_b_hoeff(E_test_pocket_cero, len(Xd_train), 1, 0.05):>15.04f}")
print(f"\t\tE_test         Hoeffding    PLA-Pocket (v_ini aleat.)       {err_b_hoeff(E_test_pocket_random, len(Xd_train), 1, 0.05):>15.04f}")

pause()

print("\n¡Esto es todo! ;)")
