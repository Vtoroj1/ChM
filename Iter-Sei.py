import sys
import json
import numpy as np

from LU import lu_decompose, solve

MAX_ITER = 10000


# ---------- Ввод ----------
def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    A = np.array([[float(x) for x in row] for row in data["A"]])
    b = np.array([float(x) for x in data["b"]])
    eps = float(data.get("eps", 0.01))

    n = len(b)
    return n, A, b, eps


# ---------- Построение alpha и beta ----------
def build_alpha_beta(A, b, n):
    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]
    return alpha, beta


# ---------- Нормы ----------
def norm_c(M):
    n = M.shape[0]
    mx = 0.0
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += abs(M[i, j])
        if s > mx:
            mx = s
    return mx


def norm_inf(v):
    mx = 0.0
    for x in v:
        if abs(x) > mx:
            mx = abs(x)
    return mx


# ---------- Якоби ----------
def iteration(n, eps, alpha, beta):
    x_old = beta.copy()
    x_new = np.zeros(n)

    norm_alpha = norm_c(alpha)

    for k in range(1, MAX_ITER + 1):
        for i in range(n):
            s = beta[i]
            for j in range(n):
                if i != j:
                    s += alpha[i, j] * x_old[j]
            x_new[i] = s

        diff = norm_inf(x_new - x_old)

        if norm_alpha < 1.0:
            eps_k = norm_alpha / (1.0 - norm_alpha) * diff
        else:
            eps_k = diff

        if eps_k <= eps:
            return x_new, k

        x_old = x_new.copy()

    return x_new, MAX_ITER


# ---------- Зейдель ----------
def seidel(n, eps, alpha, beta):
    x = beta.copy()

    norm_alpha = norm_c(alpha)

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = alpha[i, j]
    norm_C = norm_c(C)

    for k in range(1, MAX_ITER + 1):
        x_old = x.copy()
        for i in range(n):
            s = beta[i]
            for j in range(n):
                if i != j:
                    s += alpha[i, j] * x[j]
            x[i] = s

        diff = norm_inf(x - x_old)

        if norm_alpha < 1.0:
            eps_k = norm_C / (1.0 - norm_alpha) * diff
        else:
            eps_k = diff

        if eps_k <= eps:
            return x, k

    return x, MAX_ITER


# ---------- Вывод ----------
def print_vector(v, name):
    print(name)
    print("  ".join(f"{x:10.5f}" for x in v))
    print()


# ---------- Главная программа ----------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test3.json"
    n, A, b, eps = read_input(path)

    print(f"eps = {eps}\n")

    alpha, beta = build_alpha_beta(A, b, n)

    # эталон — LU из первой лабы
    L, U, P, perm, swaps = lu_decompose(A, n)
    x_exact = solve(L, U, P, b, n)
    print_vector(x_exact, "Эталонное решение (LU):")

    # --- Якоби ---
    x_mpi, k_mpi = iteration(n, eps, alpha, beta)
    print("Метод простых итераций")
    print(f"Количество итераций: {k_mpi}")
    print_vector(x_mpi, "Решение:")

    # --- Зейдель ---
    x_sei, k_sei = seidel(n, eps, alpha, beta)
    print("Метод Зейделя")
    print(f"Количество итераций: {k_sei}")
    print_vector(x_sei, "Решение:")

    # --- Сравнение ---
    print("Сравнение методов")
    print(f"МПИ:   {k_mpi} итераций")
    print(f"Зейдель: {k_sei} итераций")
    if k_sei < k_mpi:
        print("Зейдель сходится быстрее.")
    elif k_mpi < k_sei:
        print("МПИ сходится быстрее.")
    else:
        print("Оба метода сошлись за одинаковое число итераций.")


if __name__ == "__main__":
    main()