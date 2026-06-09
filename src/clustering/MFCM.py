# A multivariate fuzzy c-means method
# https://www.sciencedirect.com/science/article/abs/pii/S1568494612005686


import numpy as np
from timeit import default_timer as timer

def MFCM(data, centers, parM):

    start = timer()
    
    max_iteration = 100
    J = np.iinfo(np.int32).max
    count = 0
    P = initialize_prototypes(data, centers)
    Ubefore = None
    Jbefore = J + 1.0
    
    while Jbefore - J > 0.0001 and count < max_iteration:
        count += 1
        D = update_distances(data, P)
        U = update_membership(D, parM)
        P = update_prototypes(data, U, parM)
        Jbefore = J
        J = update_criterion(U, D, parM)
        Ubefore = U
    
    M = np.ones((len(centers), data.shape[1]))
    memb = aggregate_matrix(Ubefore, M)
    L = get_partition(memb)

    end = timer()

    # -------------- Calcular Z, B, T, R --------------
    z = overallCentroid(data)
    B = computeBj(Ubefore, data, P, z, parM)
    T = computeTj(Ubefore, data, z, parM)
    R = computeRj(B, T)
    # -------------------------------------------------

    result = [J, L, Ubefore, count, end - start, memb, R]

    return result

def initialize_prototypes(data, centers):
    return np.array([data[c] for c in centers])

# def update_prototypes(data, memberships, parM):
#     # nObj = data.shape[0]  # Number of objects
#     nVar = data.shape[1]  # Number of variables
#     nProt = memberships[0].shape[1]  # Number of prototypes
#     P = np.zeros((nProt, nVar))
#     for i in range(nVar):
#         membership = memberships[i]  # Shape: (nObj, nProt)
#         weighted_memberships = membership ** parM  # Shape: (nObj, nProt)
#         numerator = np.sum(data[:, i, np.newaxis] * weighted_memberships, axis=0)  # Shape: (nProt,)
#         denominator = np.sum(weighted_memberships, axis=0)  # Shape: (nProt,)
#         P[:, i] = numerator / denominator
#     return P

def update_prototypes(data, memberships, parM):

    U = np.stack(memberships)               # → (nVar, nObj, nProt)
    W = U ** parM                           # → pesos elevados a m

    data_T = data.T[:, :, None]             # → (nVar, nObj, 1)

    numer = np.sum(W * data_T, axis=1)      # → (nVar, nProt)
    denom = np.sum(W, axis=1)               # → (nVar, nProt)
    P = (numer / denom).T                   # → (nProt, nVar)

    return P

def update_distances(data, prototypes):
    diff = data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]  # (nObj, nProt, nVar)
    D_full = diff ** 2  

    D = D_full.transpose(2, 0, 1)

    return D

def update_membership(distances, parM):
    # distances: array shape (nVar, nObj, nProt)
    eps = 1e-7
    m = parM
    exponent = 1.0/(m-1.0)

    # Queremos U[v,i,k] = [ Σ_{vv,kk} ((d_{v,i,k}+ε)/(d_{vv,i,kk}+ε))^exponent ]^(-1)

    # 1) Expanda distâncias para termos de numerador e denominador simultâneos:
    #    d_num shape: (nVar, nObj, nProt, 1, 1)
    #    d_den shape: (1,    nObj, 1,    nVar, nProt)
    d_num = distances[:,:,:,None,None]       # → (nVar, nObj, nProt, 1, 1)
    # d_den = distances[None,:,:,:, :] + eps      # → (1,    nObj,   nVar, nProt)
    d_den = distances.transpose(1, 0, 2)[None, :, None, :, :] + eps

    # 2) Calcule razão + potência
    ratio = (d_num + eps) / d_den               # → (nVar, nObj, nProt, nVar, nProt)
    ratio_pow = ratio ** exponent

    # 3) Some sobre variáveis vv e prot kk (eixos 3 e 4)
    soma = np.sum(ratio_pow, axis=(3,4))      # → (nVar, nObj, nProt)

    # 4) Inverso para obter U
    U = soma ** -1.0

    return U

def update_criterion(memberships, distances, parM):
    U = np.stack(memberships)               # → (nVar, nObj, nProt)
    return np.sum((U ** parM) * distances)

def aggregate_matrix(memberships, M):
    nObj, nProt = memberships[0].shape
    nVar = len(memberships)
    memb = np.zeros((nObj, nProt))
    
    for j in range(nObj):
        soma0 = sum(sum(M[k, i] * memberships[i][j, k] for i in range(nVar)) for k in range(nProt))
        for k in range(nProt):
            memb[j, k] = sum(M[k, i] * memberships[i][j, k] for i in range(nVar)) / soma0
    
    return memb

def compute_aij(memberships):
    memberships = np.stack(memberships)  # Stack list of arrays into a 3D array
    soma = np.sum(memberships, axis=(0, 1))  # Sum over objects and variables for each prototype
    M = np.sum(memberships, axis=1) / soma  # Normalize
    return M.T  # Transpose to match the original shape

def computeBj(U, data, P, z, parM):
    nVar = data.shape[1]
    nProt = P.shape[0]
    Bj = np.zeros(nVar, dtype=np.float64)

    for j in range(nVar):
        for i in range(nProt):
            y_ij = P[i, j]  # centróide do cluster i para a variável j
            Bj[j] += np.sum((U[j][:, i] ** parM) * ((y_ij - z[j]) ** 2))

    return Bj

def computeTj(U, data, z, parM):
    
    nObj, nVar = data.shape
    nProt = U[0].shape[1]
    Tj = np.zeros(nVar, dtype=np.float64)

    for j in range(nVar):
        for i in range(nObj):
            for k in range(nProt):
                Tj[j] += (U[j][i, k] ** parM) * ((data[i, j] - z[j]) ** 2)

    return Tj
  
def computeRj(Bj, Tj):
    epsilon = 1e-9
    return Bj / (Tj + epsilon)    # Cuidado divisão por zero

def overallCentroid(data):
    return np.mean(data, axis=0)

def get_partition(memb):
    return np.argmax(memb, axis=1)
