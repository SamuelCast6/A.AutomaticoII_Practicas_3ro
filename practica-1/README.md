# Práctica 1: Clasificador kNN y Selección de Atributos

**Asignatura:** Aprendizaje Automático II (Curso 2026-27)
**Autor:** Samuel Castaño Farelo

---

## 1. Clasificador kNN desde cero (`src/KNNClassifier.py`)

Se implementó el algoritmo $k$-Nearest Neighbors sin recurrir a clases estimadoras externas:
* **Métricas de distancia:** Distancia Euclidiana y distancia de Minkowski parametrizable por el exponente $p$.
* **Validación con Scikit-Learn:** Comprobación de que las predicciones generadas coinciden al 100% (diferencia media de 0.0000) frente a `KNeighborsClassifier` bajo idénticos parámetros.

---

## 2. Optimización de Hiperparámetros

* **Selección de $k$ Óptimo:** Búsqueda mediante validación cruzada estratificada de 5 pliegues (*5-Fold Stratified CV*) sobre el subconjunto de entrenamiento, fijando $k_{opt} = 7$.
* **Curvas de Aprendizaje:** Representación gráfica de *Accuracy* en entrenamiento y prueba frente a $k \in [1, 10]$.
* **Métrica de Minkowski:** Evaluación del impacto del parámetro $p \in [1, 10]$ con $k$ fijado.

---

## 3. Selección de Atributos

Evaluación comparativa de tres estrategias de reducción de dimensionalidad sobre el conjunto de test:

1. **VarianceThreshold:** Análisis de umbrales $u \in [0, 1]$. Justificación teórica de por qué $u=0$ descarta constantes y $u \ge 0.25$ no tiene sentido en variables normalizadas en $[0, 1]$.
2. **SelectKBest (F-Score):** Selección univariada clasificando las variables por significación estadística individual.
3. **mRMR (`src/mRMR.py`):** Algoritmo codificado desde cero basado en Información Mutua ($I$), maximizando la relevancia con la etiqueta de clase y penalizando la redundancia entre pares de características.

---

## Estructura del Directorio

* `src/`: Código fuente de los estimadores implementados (`KNNClassifier.py` y `mRMR.py`).
* `docs/`: Guión de la práctica (`APAUTII-26_27-P1.pdf`) y Jupyter Notebook con los experimentos y gráficas (`notebook.ipynb`).
* `data/`: Directorio reservado para conjuntos de datos locales.
