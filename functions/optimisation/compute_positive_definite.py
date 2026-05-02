import torch

def compute(A):
    jitter = 1e-6
    while torch.det(A) <= 0:  # Check determinant until matrix is invertible
        A = A + jitter * torch.eye(A.size(0))  # Add jitter to diagonal
        jitter *= 10  # Increase jitter geometrically
    return A