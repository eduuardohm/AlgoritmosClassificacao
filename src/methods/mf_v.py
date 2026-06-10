import numpy as np

def variance_filter(data, U, nClusters):
    V = []
    nVar = data.shape[1]
    nObj = data.shape[0]

    for i in range(0, nVar):
        aTotal = [] # relevancia total

        for j in range(0, nClusters):  
            a = []
            for k in range(0, nObj):
                u = U[i][k][j] # acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a.append(u)
        
            a = np.asarray(a)
            var = np.var(a)
            aTotal.append(var)

        aTotal = np.asarray(aTotal)
        media = np.mean(aTotal)

        V.append((round(media * 100, 5), i))
    
    return (V, 'Filtro por Variância')