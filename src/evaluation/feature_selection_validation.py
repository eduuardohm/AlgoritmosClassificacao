import numpy as np
from src.config import N_FOLDS, VAR_PERCENTAGES, DATASETS, SEED
from sklearn.model_selection import StratifiedKFold
from src.evaluation.knn_eval import evalute_knn

from methods.laplacian_score import lap_score

def validate_feature_selection(X, y, seed=SEED, n_neighbors=5, n_folds=N_FOLDS, n_filter_rep=50, n_classes=None, filter_method='sum', dataset_name=None, outer_fold_index=0):
    """
    Função para validar a seleção de características usando validação cruzada.
    Args:
        X (numpy.ndarray): Dados de entrada.
        y (numpy.ndarray): Rótulos das classes.
        seed (int): Semente para reprodutibilidade.
        n_neighbors (int): Número de vizinhos para o KNN.
        n_folds (int): Número de dobras para validação cruzada.
        n_filter_rep (int): Número de repetições para o filtro.
        n_classes (int, optional): Número de classes. Se None, será inferido a partir de y.
        filter_method (str): Método de filtro a ser utilizado ('sum_filter', 'variance_filter', 'ls').
        dataset_name (str, optional): Nome do dataset, usado para logging.
        outer_fold_index (int): Índice da dobra externa.
    Returns:
        best_filter_result (numpy.ndarray): Lista de índices das variáveis ordernadas por relevância.
    """

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_score = -1
    best_features_rank = None
    nVar = X.shape[1] // 2
    
    for inner_fold_index, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        
        print(f"Outer Fold {outer_fold_index + 1}, Inner Fold {inner_fold_index + 1}:")

        if filter_method == 'sum_filter':
            print("Selected method: Sum Filter")
        elif filter_method == 'variance_filter':
            print("Selected method: Variance Filter")
        elif filter_method == 'ls':
            l_scores = lap_score(X)
            ranked_indices = np.argsort(l_scores)
            features = [(0, idx) for idx in ranked_indices]
            features = np.argsort(l_scores)[:nVar]
        else:
            raise ValueError("Método de filtro desconhecido.")
        
        # Filtrando datasets
        X_train = X[train_idx][:, features]
        X_test = X[test_idx][:, features]

        f1, accuracy, precision, recall = evalute_knn(X_train, X_test, y[train_idx], y[test_idx], n_neighbors=n_neighbors)

        if f1 > best_score:
            best_score = f1
            best_features_rank = ranked_indices

    return best_features_rank

# Utilizar na função de validação externa:
# selected_indices = ranked_indices[:numVar]
# X_filtered = X[:, selected_indices]