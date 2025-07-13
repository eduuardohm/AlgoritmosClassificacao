# src/main.py

import os
import random
import argparse
from pathlib import Path

# Importa o loader de dados
#from data.loader import select_dataset

# Importa o pipeline de filtro + classificação
#from evaluation.experiment import FilteredKNNExperiment

# Importa configurações globais (paths, seeds, etc)
from src import config


def parse_args():
    parser = argparse.ArgumentParser(description="Roda experimentos de seleção de variáveis + k-NN")
    parser.add_argument(
        "--datasets", nargs="+", type=int, default=[3],
        help="IDs dos datasets a serem executados"
    )
    parser.add_argument(
        "--neighbors", type=int, default=5,
        help="Número de vizinhos do k-NN"
    )
    parser.add_argument(
        "--mfcm-reps", type=int, default=50,
        help="Número de repetições do MFCM para inicializações"
    )
    return parser.parse_args()


def main():
    # O parse_args() deve ler os argumentos da linha de comando antes de qualquer outra cois
    args = parse_args()
    random.seed(config.SEED)

    # Cria pasta de resultados, se necessário
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)

    # Para cada dataset solicitado...
    for ds_id in args.datasets:
        # 1) Carrega dados (data, labels, número de classes, nome)
        data, target, n_classes, name = select_dataset(ds_id)

        # 2) Inicializa o objeto-experimento, alimentando parâmetros
        experiment = FilteredKNNExperiment(
            data=data,
            target=target,
            n_classes=n_classes,
            dataset_name=name,
            n_neighbors=args.neighbors,
            mfcm_reps=args.mfcm_reps,
            filter_percentages=config.FILTER_PERCENTAGES,
            n_folds=config.N_FOLDS,
            out_dir=config.RESULTS_DIR
        )

        # 3) Roda o experimento (interno e externo)
        experiment.run()

        # 4) Salva relatórios e métricas aggregateadas
        experiment.save_results()

        # 5) (opcional) exibe sumário no terminal
        experiment.report_summary()


if __name__ == "__main__":
    main()