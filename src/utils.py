import numpy as np


def least_squares_with_equality(A: np.ndarray, c: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solves a constrained least squares problem:
        minimize ||Ax - c||^2 subject to Bx = 0

    This method uses the Karush-Kuhn-Tucker (KKT) conditions to solve the problem
    by forming and solving a linear system involving Lagrange multipliers.

    Args:
        A (np.ndarray): The matrix of shape (m, n) in the least squares term.
        c (np.ndarray): The target vector of shape (m,).
        B (np.ndarray): The constraint matrix of shape (k, n) such that Bx = 0.

    Returns:
        np.ndarray: The solution vector `x` that minimizes ||Ax - c||^2
                    under the constraint Bx = 0.
    """
    num_vars = A.shape[1]
    num_constraints = B.shape[0]

    # Construct the KKT matrix
    kkt_matrix = np.block([
        [2 * A.T @ A, B.T],
        [B, np.zeros((num_constraints, num_constraints))]
    ])

    # Construct the right-hand side
    rhs = np.concatenate([2 * A.T @ c, np.zeros(num_constraints)])

    # Solve the KKT system
    solution = np.linalg.solve(kkt_matrix, rhs)

    # Extract the optimal x
    x = solution[:num_vars]

    return x
