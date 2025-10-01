import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from src.evaluation.feature_selection_validation import validate_feature_selection
from src.evaluation.knn_eval import evaluate_knn
from src.config import N_FOLDS, SEED, N_REP_MFCM, N_NEIGHBORS, VAR_PERCENTAGES
from datasets import selectDataset
from src.utils.save_summary_csv import save_summary

random.seed(SEED)
np.random.seed(SEED)

def run_experiment(data_index, filter_method='sum_filter'):
    """
    Executa nested cross-validation para avaliação de métodos de feature selection.

    Args:
        data_index (int): Índice do dataset em selectDataset().
        filter_method (str): Método de seleção de variáveis ('sum_filter', 'variance_filter', 'ls').

    Returns:
        dict: métricas médias e desvios padrão dos folds externos.
    """

    data, target, nClasses, data_name = selectDataset(data_index)

    outer_kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    external_results = {
        p: {"f1": [], "accuracy": [], "precision": [], "recall": [], "time": []}
        for p in VAR_PERCENTAGES
    }

    for extern_index, (train_index, test_index) in enumerate(outer_kfold.split(data, target)):

        print(f"Fold {extern_index + 1}/{N_FOLDS}")

        var_rank, filter_time = validate_feature_selection(
            data[train_index], target[train_index],
            seed=SEED,
            n_neighbors=N_NEIGHBORS,
            n_folds=N_FOLDS,
            n_filter_rep=N_REP_MFCM,
            n_classes=nClasses,
            filter_method=filter_method,
            dataset_name=data_name,
            outer_fold_index=extern_index
        )

        for p in VAR_PERCENTAGES:
            if filter_method == 'baseline':
                n_selected = int(data.shape[1])
            else:
                n_selected = int(data.shape[1] * (p / 100))
            selected_features = var_rank[:n_selected]

            X_train = data[train_index][:, selected_features]
            X_test = data[test_index][:, selected_features]

            f1, acc, prec, rec, exec_time = evaluate_knn(
                X_train, X_test, target[train_index], target[test_index], n_neighbors=N_NEIGHBORS
            )

            external_results[p]["f1"].append(f1)
            external_results[p]["accuracy"].append(acc)
            external_results[p]["precision"].append(prec)
            external_results[p]["recall"].append(rec)
            external_results[p]["time"].append(filter_time)

            print(f" {p}% features → F1: {f1:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, Time: {filter_time:.2f}s")

    summary = {}
    for p in VAR_PERCENTAGES:
        summary[p] = {
            metric: {
                "mean": np.mean(values),
                "std": np.std(values, ddof=1)
            }
            for metric, values in external_results[p].items()
        }

    print("\n=== Final Results ===")
    for p in VAR_PERCENTAGES:
        print(f"\n[{p}% Features]")
        for metric, stats in summary[p].items():
            print(f"  {metric.capitalize():<10} → {stats['mean']:.4f} ± {stats['std']:.4f}")

    save_summary(summary, data_name, filter_method)

    return summary

if __name__ == "__main__":
    # data_list = [3, 4, 5, 6, 8, 19, 20, 21, 22]
    # data_list = [3, 22, 21, 4, 19, 20, 8, 5, 6]    # Ordenado por tempo de execução (menor para maior)
    data_list = [6]  # rodar com mf-m e mf-v
    # data_list = [5] 
    # filter_methods = ['ls', 'udfs', 'mcfs', 'fisher_score', 'reliefF']
    filter_methods = ['variance_filter', 'sum_filter']
    # filter_methods = ['sum_filter']
    # filter_methods = ['variance_filter']
    # filter_methods = ['reliefF']

    # filter_methods = ['baseline']

    for dataset_index in data_list:
        for method in filter_methods:
            run_experiment(dataset_index, filter_method=method)