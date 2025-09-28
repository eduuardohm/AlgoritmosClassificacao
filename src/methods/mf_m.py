import numpy as np

def sum_filter(data, U, nClusters):
    V = []
    nVar = data.shape[1]

    for i in range(0, nVar):
        aTotal = [] # Relevância total
        nObj = data.shape[0]

        for j in range(0, nClusters):  
            a = 0
            for k in range(0, nObj):
                u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a += u
        
            a /= nObj # Relevância em relação ao cluster j
            aTotal += a

        aTotal = np.mean(aTotal)
        aTotal = round(aTotal, 5)
        V.append((aTotal, i))
    
    return V