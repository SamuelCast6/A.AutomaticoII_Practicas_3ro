import random
import numpy as np
import sklearn.metrics


def minkowski_distance(A, B, p=2):
  """A (N, d) y B (M, d) -> matriz de tamaño (N, M)."""
  A = np.asarray(A)
  B = np.asarray(B)

  diff = (A[:, np.newaxis, :] - B[np.newaxis, :, :])  # restamos los puntos usando broadcasting

  abs_diff_p = np.abs(diff) ** p
  suma = np.sum(abs_diff_p, axis=2)
  distancias = suma ** (1.0 / p)

  return distancias


class KNNClassifier:

  def __init__(self, k=3, distance_metric='euclidean', p=2):
    self.k = k
    self.p = p
    self.X_train = None
    self.Y_train = None

    if isinstance(distance_metric, str):
      if distance_metric == 'euclidean':
        self.metric = sklearn.metrics.pairwise.euclidean_distances
      elif distance_metric == 'manhattan':
        self.metric = sklearn.metrics.pairwise.manhattan_distances
      elif distance_metric == 'minkowski':
        self.metric = lambda X, Y: minkowski_distance(X, Y, p=self.p)
      else:
        raise ValueError('Métrica de distancia desconocida')

    elif callable(distance_metric):

      def metrica_personalizada(X, Y):
        matriz = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
          for j in range(len(Y)):
            matriz[i, j] = distance_metric(X[i], Y[j])
        return matriz

      self.metric = metrica_personalizada

    else:
      raise TypeError('distance_metric debe ser str o callable')

  def fit(self, X, Y):
    self.X_train = np.asarray(X)
    self.Y_train = np.asarray(Y)

  def predict(self, X):
    X = np.asarray(X)
    distances = self.metric(self.X_train, X)
    predictions = []

    for i in range(len(X)):
      indices = np.argsort(distances[:, i])[: self.k]
      neighbors = self.Y_train[indices]

      clases, conteos = np.unique(neighbors, return_counts=True)
      max_votos = np.max(conteos)
      ganadores = clases[conteos == max_votos]

      if len(ganadores) > 1:
        prediction = random.choice(ganadores)
      else:
        prediction = ganadores[0]

      predictions.append(prediction)

    return np.array(predictions)