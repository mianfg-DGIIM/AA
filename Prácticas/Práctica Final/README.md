`AA` > Prácticas > **Práctica Final**

# Ajuste del mejor modelo para clasificación de cardiotocografías

> Práctica realizada junto a Celia Arias Martínez ([@ariasmartinez](https://github.com/ariasmartinez))

El objetivo de la práctica final es seleccionar y ajustar el mejor predictor para la base de datos _Cardiotocography_, de [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Cardiotocography). Una **cardiotocografía** es un método de evaluación fetal que registra simultáneamente un conjunto de características tales como la frecuencia cardiaca fetal, los movimientos fetales y las contracciones uterinas. Para ello, compararemos un modelo lineal como **regresión logística**, con otros no lineales como **perceptrón multicapa**, **SVM** (_Support Vector Machine_), y ***random forest***.

La estrategia que seguiremos es la siguiente:

1. Comprender el problema que pretendemos resolver.
2. Preprocesar los datos para poder trabajar adecuadamente con ellos.
3. Fijar un conjunto de modelos y usar validación para escoger el mejor de ellos.
4. Analizar los resultados y estimar el error del modelo.
5. Sacar conclusiones del desempeño de nuestro modelo.

Para la implementación haremos uso principalmente de las funciones de `scikit-learn`, que se especifican en [la memoria](./PF-memoria.pdf). 

---

* Memoria de prácticas: [PF-memoria.pdf](./PF-memoria.pdf)
* Código: [PF.py](./PF.py)
