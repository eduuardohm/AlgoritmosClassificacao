import random
import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.clustering.MFCM_otimizado import MFCM_otimizado
from src.clustering.MFCM_bruno import MFCM

def exec_mfcm_filter(data, nRep, nClusters, labels=None, seed=42):
    """
    Executa o algoritmo MFCM múltiplas vezes e retorna o melhor resultado.

    Args:
        data (ndarray): dataset (n_samples, n_features).
        nRep (int): número de repetições com diferentes centros iniciais.
        nClusters (int): número de clusters.
        labels (array, opcional): rótulos verdadeiros para avaliar ARI (se fornecido).

    Returns:
        dict: contém a melhor matriz de pertinência (bestM) e metadados.
            {
                'bestM': ndarray (matriz de pertinência),
                'ARI': float (se labels fornecido),
                'bestL': labels preditos do clustering,
                'bestR': vetor de heterogeneidade (se disponível em resp)
            }
    """
    random.seed(seed)
    np.random.seed(seed)
    nObj = len(data)
    centersMC = np.zeros((nRep, nClusters))

    # inicializa diferentes conjuntos de centros
    for c in range(nRep):
        centersMC[c] = random.sample(range(1, nObj), nClusters)

    best_result = {
        "ARI": -1,
        "bestM": None,
        "bestL": None,
        "bestR": None
    }

    for r in range(nRep):
        centers = list(map(int, centersMC[r, :].tolist()))
        resp = MFCM(data, centers, 2)
        # resp = MFCM(data, centers, 2)

        J = resp[0]
        L_resp = resp[1]   # labels
        M_resp = resp[2]   # matriz U (pertinência)
        R = resp[6]        # heterogeneidade (opcional)

        ari = adjusted_rand_score(labels, L_resp) if labels is not None else None

        if labels is None or ari > best_result["ARI"]:
            best_result["ARI"] = ari if labels is not None else None
            best_result["bestM"] = M_resp
            best_result["bestL"] = L_resp
            best_result["bestR"] = R

    return best_result