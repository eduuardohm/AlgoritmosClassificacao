import numpy as np
import pandas as pd

def fuzzy_memberships(X, centroids, m=2):
    """
    Calcula as pertinências fuzzy de cada ponto X_i para cada centróide v_j.
    Fórmula do FCM: u_ij = 1 / sum_k (d_ij / d_ik)^(2/(m-1))
    """
    X = np.array(X, dtype=float)
    centroids = np.array(centroids, dtype=float)
    c = len(centroids)
    n = len(X)
    
    # Matriz de distâncias (n x c)
    dist = np.abs(X.reshape(-1, 1) - centroids.reshape(1, -1))
    
    # Evitar divisão por zero (se algum ponto coincide com centróide)
    dist = np.where(dist == 0, 1e-10, dist)
    
    # Cálculo das pertinências
    exponent = 2 / (m - 1)
    u = np.zeros((n, c))
    
    for i in range(n):
        for j in range(c):
            denom = np.sum((dist[i, j] / dist[i, :]) ** exponent)
            u[i, j] = 1 / denom
    
    # Normaliza (só por segurança)
    u = u / np.sum(u, axis=1, keepdims=True)
    return u

# ---------------------
# Exemplo de uso
# ---------------------

# Dados da variável X (como no seu exemplo)
X = [1.5, 2.4, 3.35, 4.6, 5.6,      2,  2.4,   3, 3.7,   4]
centroids_X = [2.5 , 4.5]

# Dados da variável Y (distribuição perfeita)   
Y = [ 2, 2.35, 2.1, 2.05, 2.6,      5, 5.75, 5.1, 5.8, 5.3]
centroids_Y = [2, 5.5]

# Calcular pertinências
U_X = fuzzy_memberships(X, centroids_X, m=2)
U_Y = fuzzy_memberships(Y, centroids_Y, m=2)

# Mostrar resultados em tabelas
df_X = pd.DataFrame(U_X, columns=["u1 (v=2)", "u2 (v=4)"])
df_X["X"] = X
df_Y = pd.DataFrame(U_Y, columns=["u1 (v=2)", "u2 (v=5)"])
df_Y["Y"] = Y

print("Pertinências para variável X:")
print(df_X.round(3))
print("\nPertinências para variável Y:")
print(df_Y.round(3))
