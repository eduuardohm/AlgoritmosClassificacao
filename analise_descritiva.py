import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import scienceplots
plt.rcParams['text.usetex'] = False
plt.style.use(['science'])

# Cores - vermelho: #f8cecc e #dae8fc

scaler = StandardScaler()

# --- Correlação média ---
def mean_correlation(df):
    corr = df.corr().abs()
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return upper_tri.stack().mean()

# --- Estatísticas descritivas ---
def describe_dataset(df, name):
    print(f"\n=== {name} ===")
    print("Shape:", df.shape)
    print("Classes:", df['target'].value_counts(normalize=True))
    print(df.drop(columns='target').describe().T[['mean','std','min','max']])

# --- Função de análise ---

def analyze_dataset(name, X, y):
    df = pd.DataFrame(X)
    df['Target'] = y

    print(f"\n=== {name} ===")
    print("Shape:", df.shape)
    print("Classes:", np.unique(y, return_counts=True))
    print("Correlação média:", mean_correlation(df.drop(columns='Target')))
    print("Variância média:", np.var(X, axis=0).mean())
    
    # PCA plot
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    pca_df['Target'] = y
    
    n_classes = len(pca_df['Target'].unique())
    palette = sns.color_palette("husl", n_classes)

    # Plot
    plt.figure(figsize=(5,4))
    # custom_palette = ['#f8cecc', '#dae8fc']
    # edge_palette = ['#cc8784', '#6c8ebf']

    # for target, color, edge in zip(sorted(pca_df['Target'].unique()), custom_palette, edge_palette):
    #     subset = pca_df[pca_df['Target'] == target]
    #     plt.scatter(
    #         subset['PC1'], subset['PC2'],
    #         label=f'{target}',
    #         color=color,
    #         edgecolor=edge,
    #         # alpha=0.6,
    #         linewidth=0.8
    #     )

    for target, color in zip(sorted(pca_df['Target'].unique()), palette):
        subset = pca_df[pca_df['Target'] == target]
        plt.scatter(
            subset['PC1'], subset['PC2'],
            label=f'{target}',
            color=color,
            edgecolor='black',
            linewidth=0.8
        )

    plt.title(name)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Target')
    plt.tight_layout()
    plt.show()  

# Análise de todos os nove datasets

def plot_all_datasets(datasets):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, X, y) in enumerate(datasets):
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X)
        pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
        pca_df['Target'] = y

        n_classes = len(pca_df['Target'].unique())
        palette = sns.color_palette("pastel", n_classes)

        ax = axes[i]
        for target, color in zip(sorted(pca_df['Target'].unique()), palette):
            subset = pca_df[pca_df['Target'] == target]
            ax.scatter(
                subset['PC1'], subset['PC2'],
                label=f'{target}', color=color, edgecolor='black',
                linewidth=0.6, alpha=0.7
            )

        ax.set_title(name, fontsize=16)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11, loc='best', title='Target', title_fontsize=12)

    # Esconde eixos extras se tiver menos de 9 datasets
    for j in range(i+1, 9):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# --- Conjuntos de dados ---

datasets = []

# Breast Cancer | MF-M ganha (nada disparidade)
X, y = load_breast_cancer(return_X_y=True)
X = scaler.fit_transform(X)
# analyze_dataset("Breast Cancer", X, y)

datasets.append(("Breast Cancer", X, y))


# Heart Statlog
df = pd.read_csv("datasets/heart-statlog.dat", sep=" ", header=None)
y = (df.iloc[:,-1] - 1).to_numpy()
X = scaler.fit_transform(df.drop(df.columns[-1], axis=1))
# analyze_dataset("Heart Statlog", X, y)

datasets.append(("Heart Statlog", X, y))


# Musk v1 | MF-M ganha (nada disparidade)
dataset = pd.read_csv('datasets/musk1.data', header=None)
dataset = dataset.drop(dataset.columns[[0, 1]], axis=1)
y = dataset.iloc[:,-1].to_numpy()
X = scaler.fit_transform(dataset.iloc[:,:-1])
# analyze_dataset("Musk (Version 1)", X, y)

datasets.append(("Musk (Version 1)", X, y))


# Madelon
data = arff.loadarff('datasets/madelon.arff')
df = pd.DataFrame(data[0])
df[df.columns[-1]] = df.iloc[:,-1].astype(int)
y = df.iloc[:,-1].to_numpy()
X = scaler.fit_transform(df.drop(df.columns[-1], axis=1))
# analyze_dataset("Madelon", X, y)

datasets.append(("Madelon", X, y))


# Scene
data = arff.loadarff('datasets/scene.arff')
df = pd.DataFrame(data[0])
df[df.columns[-1]] = df.iloc[:,-1].astype(int)
df = df.drop(df.columns[294:299], axis=1)
y = df.iloc[:,-1].to_numpy()
X = scaler.fit_transform(df.drop(df.columns[-1], axis=1))
# analyze_dataset("Scene", X, y)

datasets.append(("Scene", X, y))


# Zoo
df = pd.read_csv("datasets/zoo.data", sep=",", header=None)
y = (df.iloc[:,-1] - 1).to_numpy()
X = scaler.fit_transform(df.drop(df.columns[[0, -1]], axis=1))
# analyze_dataset("Zoo", X, y)

datasets.append(("Zoo", X, y))


# Ionosphere
data = pd.read_csv('datasets/ionosphere.data', header=None)		# Ionosphere | UCI Machine Learning Repository | 34 features | 351 instances
y = data.iloc[:, -1].map({'g': 1, 'b': 0}).to_numpy()
data = data.drop(data.columns[-1], axis=1)
X = scaler.fit_transform(data)
# analyze_dataset("Ionosphere", X, y)

datasets.append(("Ionosphere", X, y))


# Sonar
data = pd.read_csv("datasets/sonar.data", sep=",", header=None)
y = data.iloc[:, -1].map({'R': 1, 'M': 0}).to_numpy()
data = data.drop(data.columns[-1], axis=1).to_numpy()
X = scaler.fit_transform(data)
# analyze_dataset("Sonar", X, y)

datasets.append(("Sonar", X, y))


# Wine
X, y = load_wine(return_X_y=True)		# Wine | OPENML: ID 187 | 13 features | 178 instances
X = scaler.fit_transform(X)
# analyze_dataset("Wine", X, y)

datasets.append(("Wine", X, y))


plot_all_datasets(datasets)
