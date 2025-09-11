
import numpy as np
from scipy.sparse.linalg import eigs

def compute_scaled_laplacian(adj_matrix):

    D = np.diag(np.sum(adj_matrix, axis=1))

    L = D - adj_matrix

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(adj_matrix.shape[0])

def generate_adj_matrix(distance_matrix, n_verticies):
    
    adj_matrix = np.zeros(shape=(n_verticies, n_verticies), dtype=np.int32)
    connections = [(int(row[0]), int(row[1])) for row in distance_matrix]

    for i, j in connections:
        adj_matrix[i,j] = 1

    return adj_matrix

def normalise_data(dataset):
    
    mean = dataset.mean(axis=0, keepdims=True)
    std = dataset.std(axis=0, keepdims=True)

    return (dataset - mean)/std 

def compute_chebyshev_polynomials(L_scaled, polynomial_num):

    N = L_scaled.shape[0]

    cheb_polynomials = [np.identity(N), L_scaled.copy()]

    for i in range(2, polynomial_num):
        cheb_polynomials.append(
            2 * L_scaled * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

