from sklearn.metrics import mutual_info_score
import numpy as np


class mRMR:
    def __init__(self, n_features: int):

        self.n_features = n_features
        self.selected_indices_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):

        X = np.asarray(X)
        y = np.asarray(y)

        n_total_features = X.shape[1]
        n_to_select = min(self.n_features, n_total_features)

        relevances = np.array([ mutual_info_score(X[:, i], y) for i in range(n_total_features)])

        # 2. Matriz de redundancia I(Xi, Xj)
        redundancy_matrix = np.zeros((n_total_features, n_total_features))
        for i in range(n_total_features):
            for j in range(i, n_total_features):
                val = mutual_info_score(X[:, i], X[:, j])
                redundancy_matrix[i, j] = val
                redundancy_matrix[j, i] = val

        selected = []
        candidates = list(range(n_total_features))

        first_feat = int(np.argmax(relevances))
        selected.append(first_feat)
        candidates.remove(first_feat)

        # los restantes
        for _ in range(1, n_to_select):
            best_score = -np.inf
            best_feat = None

            for i in candidates:
                redundancy = np.mean([redundancy_matrix[i, j] for j in selected])
                score = relevances[i] - redundancy

                if score > best_score:
                    best_score = score
                    best_feat = i

            selected.append(best_feat)
            candidates.remove(best_feat)

        self.selected_indices_ = selected
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
 
        X = np.asarray(X)
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
  
        return self.fit(X, y).transform(X)

