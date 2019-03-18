import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return np.dot(A, B) - C

def problem3 (A, B, C):
    return A * B - np.transpose(C)

def problem4 (x, y):
    return np.dot(np.transpose(x), y)

def problem5 (A):
    return np.zeros(A.shape)

def problem6 (A):
    return np.ones(A.shape)

def problem7 (A, alpha):
    return A + alpha*np.eye(A.shape[0])

def problem8 (A, i, j):
    return A[i][j]

def problem9 (A, i):
    return np.sum(A,axis=1)[i]

def problem10 (A, c, d):
    A = A[np.nonzero(A >= c)]
    A = A[np.nonzero(A<=d)]
    return np.mean(A)

def problem11 (A, k):
    # v is eigenvalues, E is eigenvectors
    v, E = np.linalg.eig(A)
    all_indices = v.argsort()[::-1]
    indices = []
    for i in range(0, k):
        indices.append(all_indices[i])
    P = E[:, indices]
    return P

def problem12 (A, x):
    return np.linalg.solve(A, x)

def problem13 (A, x):
    return np.transpose(np.linalg.solve(np.transpose(A), np.transpose(x)))
