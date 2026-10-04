import sys
import json
import numpy as np


# ---------- Ввод ----------
def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    A = np.array([[float(x) for x in row] for row in data["A"]])
    b = np.array([float(x) for x in data["b"]])
    n = A.shape[0]
    return n, A, b


# ---------- LU-разложение по учебнику ----------
def lu_decompose(A, n):
    U = A.copy().astype(float)
    L = np.eye(n)
    perm = list(range(n))
    swaps = 0

    for k in range(n - 1):
        # частичный выбор ведущего элемента
        p = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[p, k]):
                p = i

        if abs(U[p, k]) < 1e-12:
            raise ValueError("Матрица вырождена")

        if p != k:
            U[[k, p], :] = U[[p, k], :]
            perm[k], perm[p] = perm[p], perm[k]
            swaps += 1
            L[[k, p], :k] = L[[p, k], :k]

        # матрица M_k: единичная с -m_i в столбце k
        M = np.eye(n)
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            M[i, k] = -m

        U = M @ U

        # L <- L * M_k^{-1}
        Minv = np.eye(n)
        for i in range(k + 1, n):
            Minv[i, k] = -M[i, k]
        L = L @ Minv

    P = np.zeros((n, n))
    for i in range(n):
        P[i, perm[i]] = 1.0

    return L, U, P, perm, swaps


# ---------- Подстановки ----------
def forward_sub(L, b, n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i] - L[i, :i] @ x[:i]
    return x


def back_sub(U, b, n):
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


def solve(L, U, P, b, n):
    pb = P @ b
    y = forward_sub(L, pb, n)
    x = back_sub(U, y, n)
    return x


# ---------- Вывод ----------
def print_matrix(M, name):
    print(name)
    for row in M:
        print("  ".join(f"{x:12.6f}" for x in row))
    print()


def print_vector(v, name):
    print(name)
    print("  ".join(f"{x:12.6f}" for x in v))
    print()


# ---------- Главная программа ----------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test1.json"
    n, A, b = read_input(path)

    print(f"n = {n}\n")
    print_matrix(A, "Матрица A:")
    print_vector(b, "Вектор b:")

    L, U, P, perm, swaps = lu_decompose(A, n)

    print_matrix(L, "Матрица L:")
    print_matrix(U, "Матрица U:")

    LU = L @ U
    print_matrix(LU, "L * U:")

    print_matrix(P, "Матрица перестановок P:")

    PA = P @ A
    print_matrix(PA, "P * A:")

    # --- решение системы ---
    x = solve(L, U, P, b, n)
    print_vector(x, "Решение x:")

    # --- обратная матрица ---
    Ainv = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        Ainv[:, j] = solve(L, U, P, e, n)

    print_matrix(Ainv, "Обратная матрица A^-1:")

    E = np.eye(n)
    check = A @ Ainv
    print_matrix(check, "A * A^-1:")

    # --- определитель ---
    det = np.prod(np.diag(U))
    if swaps % 2 == 1:
        det = -det
    print(f"det(A) = {det:.10g}")


if __name__ == "__main__":
    main()