import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns # type: ignore
import scienceplots
from itertools import cycle

# Estilizando os gráficos
# sns.set(style="whitegrid")
plt.rcParams['text.usetex'] = False
plt.style.use(['science'])
colors = plt.get_cmap('tab10').colors
markers = ['s'] * 9

# Variáveis
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y_labels = [
    'F1-Score',
    'Accuracy',
    'Precision',
    'Recall'
]

label_map = {
    'MF_M': 'MF-M', 'MF_V': 'MF-V', 'LS': 'Laplacian Score', 'MCFS': 'MCFS', 'UDFS': 'UDFS',
    'DUFS': 'DUFS', 'VCSDFS': 'VCSDFS', 'FMIUFS': 'FMIUFS', 'SRCFS': 'SRCFS', 'Baseline': 'Baseline', 
    'Fisher': 'Fisher Score', 'ReliefF': 'ReliefF'
}

def load_data(dataset_name):
    path = f'resultados/{dataset_name}'
    return {
        'MF_M': pd.read_csv(f'{path}/sum_filter_summary.csv'),
        'MF_V': pd.read_csv(f'{path}/variance_filter_summary.csv'),
        'LS': pd.read_csv(f'{path}/ls_summary.csv'),
        'MCFS': pd.read_csv(f'{path}/mcfs_summary.csv'),
        'UDFS': pd.read_csv(f'{path}/udfs_summary.csv'),
        'Fisher': pd.read_csv(f'{path}/fisher_score_summary.csv'),
        'ReliefF': pd.read_csv(f'{path}/reliefF_summary.csv')
    }

def plot_metrics_separately(x, dataset_name, dataset_label):
    methods = load_data(dataset_name)

    for i in range(4):
        plt.figure(figsize=(8, 6))

        for idx, (label, data) in enumerate(methods.items()):
            style = {'linestyle': '--', 'marker': '', 'color': 'black'} if label == 'Baseline' else {
                'marker': markers[idx], 'color': colors[idx]
            }

            plt.plot(
                x, data.iloc[:, i], label=label_map[label],
                markersize=6, linewidth=2, **style
            )

        plt.xlabel('Percentage (\%) of features selected', fontsize=12, weight='bold')
        plt.ylabel(y_labels[i], fontsize=12, weight='bold')
        plt.title(f'{dataset_label} Dataset', fontsize=14, weight='bold')
        plt.xticks(ticks=x, labels=[f'{int(v*100)}%' for v in x], fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.show()
    
def plot_metrics_together(x, dataset_name, dataset_label):
    methods = load_data(dataset_name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]

        for idx, (label, data) in enumerate(methods.items()):
            style = {'linestyle': '--', 'marker': '', 'color': 'black'} if label == 'Baseline' else {
                'marker': markers[idx], 'color': colors[idx]
            }

            ax.plot(
                x, data.iloc[:, i], label=label_map[label],
                markersize=5, linewidth=2, **style
            )

        ax.set_xlabel('Percentage (\%) of features selected', fontsize=14)
        ax.set_ylabel(y_labels[i], fontsize=14)
        # ax.set_title(y_labels[i], fontsize=11, weight='bold')
        ax.set_title(f'{y_labels[i]} performance', fontsize=18, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(v*100)}%' for v in x], fontsize=12)
        ax.set_yticks(ax.get_yticks())
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Ajustar a legenda fora do gráfico
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=14, frameon=False)
    fig.suptitle(f'{dataset_label} Dataset', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # espaço para legenda e título
    plt.show()

if __name__ == "__main__":
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    dataset_map = {
        'Breast Cancer': 'Breast Cancer',
        'Heart Statlog Dataset': 'Heart Statlog',
        'Ionosphere': 'Ionosphere',
        'Madelon': 'Madelon',
        'Musk Version 1': 'Musk (Version 1)',
        'Scene': 'Scene',
        'Sonar': 'Sonar',
        'Wine': 'Wine',
        'Zoo': 'Zoo'
    }

    for dataset_name, dataset_label in dataset_map.items():
        print(f"Metrics results for dataset: {dataset_label}")
        plot_metrics_together(x, dataset_name, dataset_label)

    # plot_metrics_separately(x, dataset_label, dataset_name)
    # plot_metrics_together(x, dataset_label, dataset_name)