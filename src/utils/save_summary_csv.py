

import os
import pandas as pd

def save_summary(summary, data_name, filter_method, base_dir="resultados"):
    """
    Salva o resumo final (médias por porcentagem de features) em CSV.

    Args:
        summary (dict): Saída final do run_experiment (com mean/std).
        data_name (str): Nome do dataset.
        filter_method (str): Método de seleção de variáveis.
        base_dir (str): Diretório base para salvar resultados.
    """
    # Cria diretório resultados/<dataset>/
    output_dir = os.path.join(base_dir, data_name)
    os.makedirs(output_dir, exist_ok=True)

    # Monta dataframe: cada linha é um percentual
    rows = []
    for p, metrics in summary.items():
        row = {
            "f1": metrics["f1"]["mean"],
            "accuracy": metrics["accuracy"]["mean"],
            "precision": metrics["precision"]["mean"],
            "recall": metrics["recall"]["mean"],
            "time": metrics["time"]["mean"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Nome do arquivo: <metodo>_summary.csv
    filename = f"{filter_method}_summary.csv"
    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False)
    print(f"[OK] Summary salvo em {filepath}")