import pandas as pd
import numpy as np
import random
import time
import math
from MFCM import MFCM
from filters import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from datasets import selectDataset

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import pickle
import os
import warnings

# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

random.seed(42)

scaler = StandardScaler()

def execute(nRep, dataset, centersAll):
    Jmin = 2147483647	# int_max do R
    bestL = 0			# melhor valor de L_resp
    bestM = 0			# melhor valor de M_resp
    best_centers = 0

    for r in range(nRep):
        # print(f'MFCM rep: {r}')
        centers = list(map(int, centersAll[r,].tolist()))

        resp = MFCM(dataset, centers, 2)

        J = resp[0]
        L_resp = resp[1]
        M_resp = resp[2]

        if (Jmin > J):
            Jmin = J
            bestL = L_resp
            bestM = M_resp
            best_centers = centers

    dict = {'Jmin': Jmin, 'bestL': bestL, 'bestM': bestM, 'best_centers': centersAll}
    # Retorna os centers de todas as iterações para o KMeans (mudar para criar uma nova lista exclusiva para o KMeans)

    return dict

def exec_mfcm_filter(data, nRep, nClusters):
    ## Inicializando variáveis
    result = {}
    Jmin = 2147483647
    centers = 0

    nObj = len(data)

    centersMC = np.zeros((nRep, nClusters))

    for c in range(nRep):
        centersMC[c] = random.sample(range(1, nObj), nClusters)

    clustering = execute(nRep, data, centersMC)

    if clustering['Jmin'] < Jmin:
        Jmin = clustering['Jmin']
        result = clustering
    centers = clustering['best_centers']

    return result

def run_filter(dataset, result, numVar, numClasses):
	
    data = np.vstack((dataset[0], dataset[1]))
    target = np.hstack((dataset[2], dataset[3]))

    resultado_filtro = variance_filter(data, result['bestM'], numClasses)
    resultado_filtro[0].sort(key=lambda k : k[0])

    data = apply_filter(data, resultado_filtro, numVar)

    return (data, target)

def filter(data, result, numVar, numClasses):

    resultado_filtro = variance_filter(data, result['bestM'], numClasses)
    resultado_filtro[0].sort(key=lambda k : k[0])

    data = apply_filter(data, resultado_filtro, numVar)

    return data

def return_rank(data, result, numClasses):

    resultado_filtro = variance_filter(data, result['bestM'], numClasses)
    resultado_filtro[0].sort(key=lambda k : k[0])

    return resultado_filtro

def exec_knn(data_train, data_test, target_train, target_test, n_neighbors):

    start = time.time()

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(data_train, target_train)

    target_pred = clf.predict(data_test)

    end = time.time()

    f1 = f1_score(target_test, target_pred, average="macro")
    accuracy = accuracy_score(target_test, target_pred)
    precision = precision_score(target_test, target_pred, average="macro")
    recall = recall_score(target_test, target_pred, average="macro")
    
    # print(classification_report(target_test, target_pred))
    # print(f'F1 Score: {score}')
    return f1, accuracy, precision, recall, (end - start)

def atualizaTxt(nome, lista):
	arquivo = open(nome, 'a')
	arquivo.write(lista)
	arquivo.write('\n')
	arquivo.close()

def save_list(list, dataset_name, i_externo, i_interno):
    file_name = f'matrices/{dataset_name}/lista_{i_externo}{i_interno}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(list, f)

def load_list(dataset_name, i_externo, i_interno):
    file_name = f'matrices/{dataset_name}/lista_{i_externo}{i_interno}.pkl'
    with open(file_name, 'rb') as f:
        mfcm = pickle.load(f)
    return mfcm

def filtro_mutual_info(X, y, numVar):

    resultado_filtro = mutual_info_regression(X, y)

    resultado_filtro = [(pontuacao, indice) for indice, pontuacao in enumerate(resultado_filtro)]
    resultado_filtro.sort(key=lambda x: x[0])

    resultado_filtro = (resultado_filtro, 'Filtro por Mutual Info')

    X = apply_filter(X, resultado_filtro, numVar)

    return X

def media_desvio_padrao(lista):
    f1_avg = []
    accuracy_avg = []
    precision_avg = []
    recall_avg = []
    time_avg = []

    f1_std = []
    accuracy_std = []
    precision_std = []
    recall_std = []
    time_std = []

    for percentage in range(len(lista[0])):
        f1_values = []
        accuracy_values = []
        precision_values = []
        recall_values = []
        time_values = []

        for fold in lista:
            f1_values.append(fold[percentage][0])
            accuracy_values.append(fold[percentage][1])
            precision_values.append(fold[percentage][2])
            recall_values.append(fold[percentage][3])
            time_values.append(fold[percentage][4])

        f1_avg.append(np.mean(f1_values))
        accuracy_avg.append(np.mean(accuracy_values))
        precision_avg.append(np.mean(precision_values))
        recall_avg.append(np.mean(recall_values))
        time_avg.append(np.mean(time_values))
        
        f1_std.append(np.std(f1_values))
        accuracy_std.append(np.std(accuracy_values))
        precision_std.append(np.std(precision_values))
        recall_std.append(np.std(recall_values))
        time_std.append(np.std(time_values))

    print("Médias:")
    print("F1-Score:", f1_avg)
    print("Acurácia:", accuracy_avg)
    print("Precisão:", precision_avg)
    print("Recall:", recall_avg)

    print("\nDesvios Padrão:")
    print("F1-Score:", f1_std)
    print("Acurácia:", accuracy_std)
    print("Precisão:", precision_std)
    print("Recall:", recall_std)

    return f1_avg, accuracy_avg, precision_avg, recall_avg, time_avg, f1_std, accuracy_std, precision_std, recall_std, time_std

def cross_validation(data, target, seed, n_neighbors, n_folds, nFilterRep, nClasses, porcentagemVar, filter_name, data_name, i_externo):

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    best_result = -1
    best_mfcm = []
    best_set = []

    for i_interno, (train, test) in enumerate(kfold.split(data, target)):

        # print(f'Fold interno: {i_interno}')

        if filter_name == 'MFCM':
            if not os.path.exists(f'matrices/{data_name}'):
                os.makedirs(f'matrices/{data_name}')
            if os.path.exists(f'matrices/{data_name}/lista_{i_externo}{i_interno}.pkl'):
                mfcm = load_list(data_name, i_externo, i_interno)
            else:
                mfcm = exec_mfcm_filter(data[train], nFilterRep, nClasses)
                save_list(mfcm, data_name, i_externo, i_interno)

        numVar = (data[test].shape[1] // 2)

        if filter_name == 'MFCM':
            filtered_train = filter(data[train], mfcm, numVar, nClasses)
            filtered_test = filter(data[test], mfcm, numVar, nClasses)
        elif filter_name == 'MUTUAL':
            filtered_train = filtro_mutual_info(data[train], target[train], numVar)
            filtered_test = filtro_mutual_info(data[test], target[test], numVar)

        f1, accuracy, precision, recall, tempo = exec_knn(filtered_train, filtered_test, target[train], target[test], n_neighbors)

        if f1 > best_result:
            best_result = f1
            if filter_name == 'MFCM':
                best_mfcm = mfcm
                best_set = data[train]

    var_rank = return_rank(best_set, best_mfcm, nClasses)

    return var_rank

def experimento(indexData, n_neighbors, nFilterRep):

    SEED = 42

    dataset = selectDataset(indexData)

    porcentagemVar = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    n_folds = 5
    # porcentagemVar = [0, 20, 50]

    data, target, nClasses, data_name = dataset
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    lista_resultados_mfcm = []

    for i_externo, (train, test) in enumerate(kfold.split(data, target)):
        # print(f'Fold externo [{i_externo}]')

        var_rank = cross_validation(data[train], target[train], SEED, n_neighbors, 5, nFilterRep, nClasses, porcentagemVar, 'MFCM', data_name, i_externo)
        scores_porcentagem = {}

        for i in porcentagemVar:
            numVar = int(data.shape[1] * (i/100))
            # print(numVar)
            # print(f'Porcentagem de variáveis cortadas: {i}%')
            # print(f'Número de variáveis apos filtro: {data.shape[1] - numVar}')

            filtered_train = apply_filter(data[train], var_rank, numVar)
            filtered_test = apply_filter(data[test], var_rank, numVar)

            f1, accuracy, precision, recall, tempo = exec_knn(filtered_train, filtered_test, target[train], target[test], n_neighbors)
            # print(f'Fold externo [{i_externo}] - F1-Score: {f1}')

            scores_porcentagem[i] = {
                'f1': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'tempo': tempo
            }

            # scores_porcentagem.append((f1, accuracy, precision, recall, tempo))

        lista_resultados_mfcm.append(scores_porcentagem)

    media_resultados = {p: {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'tempo': 0} for p in porcentagemVar}
    desvio_resultados = {p: {'f1': [], 'accuracy': [], 'precision': [], 'recall': [], 'tempo': []} for p in porcentagemVar}

    for p in porcentagemVar:
        for fold_result in lista_resultados_mfcm:
            for metric in media_resultados[p]:
                media_resultados[p][metric] += fold_result[p][metric]
                desvio_resultados[p][metric].append(fold_result[p][metric])
        # Tirar a média (dividir pelo número de folds)
        media_resultados[p] = {k: v / n_folds for k, v in media_resultados[p].items()}
        desvio_resultados[p] = {k: np.std(v, ddof=1) for k, v in desvio_resultados[p].items()}

    data = {
        'F1-Score (Avg)': [media_resultados[p]['f1'] for p in porcentagemVar],
        'Acurácia (Avg)': [media_resultados[p]['accuracy'] for p in porcentagemVar],
        'Precisão (Avg)': [media_resultados[p]['precision'] for p in porcentagemVar],
        'Recall (Avg)': [media_resultados[p]['recall'] for p in porcentagemVar],
        'Tempo (Avg)': [media_resultados[p]['tempo'] for p in porcentagemVar],
        'F1-Score (Std)': [desvio_resultados[p]['f1'] for p in porcentagemVar],
        'Acurácia (Std)': [desvio_resultados[p]['accuracy'] for p in porcentagemVar],
        'Precisão (Std)': [desvio_resultados[p]['precision'] for p in porcentagemVar],
        'Recall (Std)': [desvio_resultados[p]['recall'] for p in porcentagemVar],
        'Tempo (Std)': [desvio_resultados[p]['tempo'] for p in porcentagemVar]
    }

    path = f'resultados/{data_name}'
    if not os.path.exists(path):
        os.makedirs(path)

    result_dir = f'resultados/{data_name}/resultado_{len(os.listdir(path)) + 1}'
    os.makedirs(result_dir)

    # Salvar parâmetros
    basics_info = (f'Seed: {SEED} | Dataset: {data_name} | K: {n_neighbors} | MFCM Reps: {nFilterRep} | Neighbors: {n_neighbors}')
    variables_cut_info = f'Porcentagens de variaveis cortadas: {porcentagemVar}'
    atualizaTxt(f'{result_dir}/parameters.txt', basics_info)
    atualizaTxt(f'{result_dir}/parameters.txt', variables_cut_info)

    # Salvar resultados detalhados
    atualizaTxt(f'{result_dir}/resultados.txt', basics_info)
    for p in porcentagemVar:
        var_info = f'Porcentagem de variaveis cortadas: {p}%'
        metrics_mfcm = f'Com filtro MFCM - F1 Score: {media_resultados[p]["f1"]:.4f} ({desvio_resultados[p]["f1"]:.4f}) | Tempo: {media_resultados[p]["tempo"]:.4f} ({desvio_resultados[p]["tempo"]:.4f})'
        atualizaTxt(f'{result_dir}/resultados.txt', var_info)
        atualizaTxt(f'{result_dir}/resultados.txt', metrics_mfcm)
        atualizaTxt(f'{result_dir}/resultados.txt', '')

    # Exibir resultados médios
    for p, metrics in media_resultados.items():
        print(f"Porcentagem: {p}%")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":

    datasets = [3]
    n_neighbors = 5
    nRepMFCM = 5

    # experimento(3, n_neighbors, nRepMFCM)

    for _, id in enumerate(datasets):
        
        experimento(id, n_neighbors, nRepMFCM)
        