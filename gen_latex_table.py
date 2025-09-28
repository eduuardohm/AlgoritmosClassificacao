import pandas as pd
from scipy.stats import rankdata
import os

percentage_map = {
    0.1: 1,
    0.2: 2,
    0.3: 3,
    0.4: 4,
    0.5: 5,
    0.6: 6,
    0.7: 7,
    0.8: 8,
    0.9: 9
}

# methods = ['BASELINE', 'LS', 'MCFS', 'UDFS', 'DUFS', 'VCSDFS', 'FMIUFS', 'SRCFS', 'MF_M', 'MF_V']
methods = ["baseline", "ls", "mcfs", "udfs", "fisher_score", "reliefF", "sum_filter", "variance_filter"]

def get_metrics_value(dataset_name, method, metric_name, percentage):
    file_path = f'resultados/{dataset_name}/{method}_summary.csv'
    df = pd.read_csv(file_path)
    percentage_value = percentage_map[percentage]
    value = df.iloc[percentage_value - 1][metric_name]

    return value

def format_latex_table_line(dataset_name, metric_name, percentage):
    values = []

    reverse = True  # padrão: maior é melhor

    # Usamos o reverso para métricas onde menor é melhor
    # if metric_name == 'db':
    #     reverse = False  # para DB, menor é melhor

    for method in methods:
        val = get_metrics_value(dataset_name, method, metric_name, percentage)
        values.append(val)

    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)

    # min: ranking olimpico, dense: raking denso, average: ranking médio
    values_for_ranking = [-v if reverse else v for v in values]
    ranks = rankdata(values_for_ranking, method="average")

    # Formata os valores com 4 casas decimais e rank
    formatted_values = [f"{v:.4f} ({int(ranks[i]) if ranks[i].is_integer() else ranks[i]})" for i, v in enumerate(values)]

    # Monta a linha LaTeX
    latex_line = f"{dataset_name} & " + " & ".join(formatted_values) + " \\\\"

    return latex_line

if __name__ == "__main__":
    metric_name = 'recall'

    dataset_name = 'Breast Cancer'
    percentage = 0.9
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)

    dataset_name = 'Heart Statlog Dataset'
    percentage = 0.2
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)

    dataset_name = 'Ionosphere'
    percentage = 0.3
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)

    dataset_name = 'Madelon'
    percentage = 0.1
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)

    dataset_name = 'Musk Version 1'
    percentage = 0.3
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)

    dataset_name = 'Scene'
    percentage = 0.9
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)

    dataset_name = 'Sonar Dataset'
    percentage = 0.6
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)

    dataset_name = 'Wine'
    percentage = 0.5
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)

    dataset_name = 'Zoo Dataset'
    percentage = 0.9
    latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    print(latex_row)



    # metric_name = 'db'
    # percentage = 0.1
    # latex_row = format_latex_table_line(dataset_name, metric_name, percentage)
    # print('DB Metric:')
    # print(latex_row)
