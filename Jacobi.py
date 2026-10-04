import sys
import json
import math
import numpy as np

MAX_ITER = 10000


# ---------- Ввод ----------
def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    A = np.array([[float(x) for x in row] for row in data["A"]])
    eps = float(data["eps"])
    n = A.shape[0]
    return n, A, eps


# ---------- Проверка симметричности ----------
def is_symmetric(A, n, tol=1e-10):
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j] - A[j, i]) > tol:
                return False
    return True


# ---------- Норма внедиагональной части ----------
def off_diag_norm(A, n):
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += A[i, j] ** 2
    return math.sqrt(s)


# ---------- Поиск максимального внедиагонального элемента ----------
def find_max_off_diag(A, n):
    mx = -1.0
    pi, pj = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > mx:
                mx = abs(A[i, j])
                pi, pj = i, j
    return pi, pj


# ---------- Матрица вращения ----------
def rotation_matrix(i, j, phi, n):
    U = np.eye(n)
    c = math.cos(phi)
    s = math.sin(phi)
    U[i, i] = c
    U[j, j] = c
    U[i, j] = -s
    U[j, i] = s
    return U


# ---------- Метод вращений Якоби ----------
def jacobi_rotation(A, n, eps):
    A_curr = A.copy()
    V = np.eye(n)
    iters = 0

    while True:
        t = off_diag_norm(A_curr, n)
        if t < eps:
            break
        if iters >= MAX_ITER:
            break

        i, j = find_max_off_diag(A_curr, n)

        if A_curr[i, i] == A_curr[j, j]:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan(2 * A_curr[i, j] / (A_curr[i, i] - A_curr[j, j]))

        U = rotation_matrix(i, j, phi, n)
        A_curr = U.T @ A_curr @ U
        V = V @ U
        iters += 1

    return A_curr, V, iters


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
    path = sys.argv[1] if len(sys.argv) > 1 else "test4.json"
    n, A, eps = read_input(path)

    if not is_symmetric(A, n):
        print("Ошибка: матрица не симметрична.")
        return

    print(f"n = {n}")
    print(f"eps = {eps}\n")

    print_matrix(A, "Матрица A:")

    A_final, V, iters = jacobi_rotation(A, n, eps)

    # собственные значения — диагональ финальной матрицы
    eigenvalues = np.zeros(n)
    for i in range(n):
        eigenvalues[i] = A_final[i, i]

    print_matrix(A_final, "Финальная матрица (после вращений):")
    print_vector(eigenvalues, "Собственные значения:")

    print_matrix(V, "Матрица собственных векторов V:")
    for j in range(n):
        print_vector(V[:, j], f"Собственный вектор x^{j+1}:")

    # проверка A * V = V * Lambda
    Lambda = np.diag(eigenvalues)
    left = A @ V
    right = V @ Lambda

    print_matrix(left, "A * V:")
    print_matrix(right, "V * Lambda:")
    print(f"Число итераций: {iters}")


if __name__ == "__main__":
    main()