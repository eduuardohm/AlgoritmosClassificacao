# A multivariate fuzzy c-means method
# https://www.sciencedirect.com/science/article/abs/pii/S1568494612005686


import numpy as np


def MFCM(data, centers, parM):
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
    
    result = [J, L, memb, count, 0]
    return result

def initialize_prototypes(data, centers):
    return np.array([data[c] for c in centers])

# def update_prototypes(data, memberships, parM):
#     nObj, nVar = data.shape
#     nProt = memberships[0].shape[1]
#     P = np.zeros((nProt, nVar))
    
#     for i in range(nVar):
#         for k in range(nProt):
#             s = sum((memberships[i][:, k] ** parM) * data[:, i])
#             ss = sum(memberships[i][:, k] ** parM)
#             P[k, i] = s / ss
    
#     return P
def update_prototypes(data, memberships, parM):
    # nObj = data.shape[0]  # Number of objects
    nVar = data.shape[1]  # Number of variables
    nProt = memberships[0].shape[1]  # Number of prototypes
    P = np.zeros((nProt, nVar))
    for i in range(nVar):
        membership = memberships[i]  # Shape: (nObj, nProt)
        weighted_memberships = membership ** parM  # Shape: (nObj, nProt)
        numerator = np.sum(data[:, i, np.newaxis] * weighted_memberships, axis=0)  # Shape: (nProt,)
        denominator = np.sum(weighted_memberships, axis=0)  # Shape: (nProt,)
        P[:, i] = numerator / denominator
    return P

# def update_distances(data, prototypes):
#     nObj, nVar = data.shape
#     nProt = prototypes.shape[0]
#     D = []
    
#     for i in range(nVar):
#         Dvar = np.zeros((nObj, nProt))
#         for j in range(nObj):
#             for k in range(nProt):
#                 Dvar[j, k] = (data[j, i] - prototypes[k, i]) ** 2
#         D.append(Dvar)
    
#     return D
def update_distances(data, prototypes):
    nObj = data.shape[0]  # Number of objects
    nProt = prototypes.shape[0]  # Number of prototypes
    nVar = data.shape[1]  # Number of variables

    # Initialize the distance array
    D = np.zeros((nVar, nObj, nProt))
    for i in range(nVar):
        # Compute the squared differences between data and prototypes for the i-th variable
        diff = data[:, i, np.newaxis] - prototypes[np.newaxis, :, i]
        distance = diff ** 2.0

        # Combine the distance and weighted sum
        D[i] = distance

    return D

# def update_membership(distances, parM):
#     nObj, nProt = distances[0].shape
#     nVar = len(distances)
#     U = []
    
#     for v in range(nVar):
#         Uvar = np.zeros((nObj, nProt))
#         for i in range(nObj):
#             for k in range(nProt):
#                 d = distances[v][i, k]
#                 soma = sum(
#                     ((d + 1e-7) / (distances[vv][i, kk] + 1e-7)) ** (1 / (parM - 1))
#                     for vv in range(nVar) for kk in range(nProt)
#                 )
#                 Uvar[i, k] = soma ** -1
#         U.append(Uvar)
    
#     return U
def update_membership(distances, parM):
    nObj = distances[0].shape[0]
    nProt = distances[0].shape[1]
    nVar = len(distances)
    U = []
    
    # Stack all distances into a single 3D array for efficient computation
    distances_stacked = np.stack(distances)  # Shape: (nVar, nObj, nProt)
    
    # Precompute the exponent term
    exponent = 1.0 / (parM - 1.0)
    
    for v in range(nVar):
        Uvar = np.zeros((nObj, nProt))
        for i in range(nObj):
            # Extract the distance for the current object i and variable v
            d = distances_stacked[v, i, :]  # Shape: (nProt,)
            
            # Reshape d to (nProt, 1, 1) for broadcasting
            d_reshaped = d[:, None, None]  # Shape: (nProt, 1, 1)
            
            # Reshape distances_stacked[:, i, :] to (1, nVar, nProt) for broadcasting
            dd_reshaped = distances_stacked[:, i, :][None, :, :]  # Shape: (1, nVar, nProt)
            
            # Compute the ratio (d + 1e-7) / (dd + 1e-7) for all vv and kk
            ratio = (d_reshaped + 1e-7) / (dd_reshaped + 1e-7)  # Shape: (nProt, nVar, nProt)
            
            # Raise the ratio to the power of the exponent
            ratio_pow = ratio ** exponent  # Shape: (nProt, nVar, nProt)
            
            # Sum over vv and kk
            soma = np.sum(ratio_pow, axis=(1, 2))  # Shape: (nProt,)
            
            # Compute the final membership value
            Uvar[i, :] = soma ** (-1.0)
        
        U.append(Uvar)
    
    return U

# def update_criterion(memberships, distances, parM):
#     J = 0
#     for i in range(len(distances)):
#         J += sum((memberships[i][j, k] ** parM) * distances[i][j, k]
#                  for j in range(distances[i].shape[0])
#                  for k in range(distances[i].shape[1]))
#     return J
def update_criterion(memberships, distances, parM):
    J = 0
    nVar = len(distances)
    
    for i in range(nVar):
        J += np.sum((memberships[i] ** parM) * distances[i])
    
    return J

def aggregate_matrix(memberships, M):
    nObj, nProt = memberships[0].shape
    nVar = len(memberships)
    memb = np.zeros((nObj, nProt))
    
    for j in range(nObj):
        soma0 = sum(sum(M[k, i] * memberships[i][j, k] for i in range(nVar)) for k in range(nProt))
        for k in range(nProt):
            memb[j, k] = sum(M[k, i] * memberships[i][j, k] for i in range(nVar)) / soma0
    
    return memb

# def compute_Aij(memberships):
#     nObj, nProt = memberships[0].shape
#     nVar = len(memberships)
#     M = np.ones((nProt, nVar))
    
#     for j in range(nProt):
#         for k in range(nVar):
#             M[j, k] = sum(memberships[k][:, j]) / sum(sum(memberships[kk][:, j]) for kk in range(nVar))
    
#     return M
def compute_aij(memberships):
    memberships = np.stack(memberships)  # Stack list of arrays into a 3D array
    soma = np.sum(memberships, axis=(0, 1))  # Sum over objects and variables for each prototype
    M = np.sum(memberships, axis=1) / soma  # Normalize
    return M.T  # Transpose to match the original shape

def get_partition(memb):
    return np.argmax(memb, axis=1)
