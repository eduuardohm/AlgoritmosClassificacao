import numpy as np

def apply_filter(dataset, result, n):

    result[0].sort(key=lambda k : k[0])

    listaCorte = [result[0][i][1] for i in range(n)]
    dataset = np.delete(dataset, listaCorte, axis = 1)
        
    return dataset