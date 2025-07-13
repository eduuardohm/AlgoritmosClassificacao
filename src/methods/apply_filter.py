import numpy as np

def apply_filter(dataset, result, n, method=1):

    # print(f'Resultado sem ordenação: {result}')
    if method == 'var': # Variância
        result[0].sort(key=lambda k : k[0], reverse=True)
    else: # Somatório
        result[0].sort(key=lambda k : k[0])

    listaCorte = [result[0][i][1] for i in range(n)]
    dataset = np.delete(dataset, listaCorte, axis = 1)
        
    return dataset