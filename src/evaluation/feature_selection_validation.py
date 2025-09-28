import numpy as np
from src.methods.mf_m import sum_filter
from src.methods.mf_v import variance_filter
from src.methods.mcfs import mcfs
from src.methods.udfs import udfs
from src.methods.laplacian_score import lap_score, feature_ranking
from src.methods.fisher_score import fisher_score
from src.methods.reliefF import reliefF
from src.config import N_FOLDS, VAR_PERCENTAGES, DATASETS, SEED
from sklearn.model_selection import StratifiedKFold
from src.evaluation.knn_eval import evaluate_knn
from src.evaluation.mfcm_eval import exec_mfcm_filter

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

        result = None

        print(f"  Método de filtro: {filter_method}, Selecionando {nVar} variáveis.")


        if filter_method == 'variance_filter' or filter_method == 'sum_filter':
            mfcm = exec_mfcm_filter(X, n_filter_rep, n_classes, y)

        if filter_method == 'sum_filter':
            U = mfcm['bestM']
            sum_scores = sum_filter(X, U, n_classes)
            sum_scores.sort(key=lambda k: k[0], reverse=True)  # menor = melhor
            ranked_indices = [idx for _, idx in sum_scores]
            features = ranked_indices[:nVar]

        elif filter_method == 'baseline':
            ranked_indices = np.arange(X.shape[1])
            features = ranked_indices[:nVar]
            
        elif filter_method == 'variance_filter':
            U = mfcm['bestM']
            var_scores = variance_filter(X, U, n_classes)
            var_scores.sort(key=lambda k: k[0], reverse=True)  # maior = melhor
            ranked_indices = [idx for _, idx in var_scores]
            features = ranked_indices[:nVar]

        elif filter_method == 'ls':
            l_scores = lap_score(X)
            ranked_indices = np.argsort(l_scores)
            features = np.argsort(l_scores)[:nVar]

        elif filter_method == 'mcfs':
            W = mcfs(X, X.shape[1])
            ranked_indices = np.argsort(W.max(axis=1))[::-1]
            features = ranked_indices[:nVar]

        elif filter_method == 'udfs':
            W = udfs(X, n_clusters=n_classes, k=5, gamma=0.1)
            norms = np.linalg.norm(W, axis=1)
            ranked_indices = np.argsort(norms)[::-1]
            features = ranked_indices[:nVar]

        elif filter_method == 'fisher_score':
            f_scores = fisher_score(X, y)
            ranked_indices = np.argsort(f_scores)[::-1]
            features = ranked_indices[:nVar]

        elif filter_method == 'reliefF':
            r_scores = reliefF(X, y)
            ranked_indices = np.argsort(r_scores)[::-1]
            features = ranked_indices[:nVar]

        else:
            raise ValueError("Método de filtro desconhecido.")
        
        # Filtrando datasets
        X_train = X[train_idx][:, features]
        X_test = X[test_idx][:, features]

        f1, accuracy, precision, recall, execution_time = evaluate_knn(X_train, X_test, y[train_idx], y[test_idx], n_neighbors=n_neighbors)

        if f1 > best_score:
            best_score = f1
            best_features_rank = ranked_indices

    return best_features_rank