import sys
import json
import numpy as np

# ---------- Ввод ----------
def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    a = np.array([float(x) for x in data["a"]])
    b = np.array([float(x) for x in data["b"]])
    c = np.array([float(x) for x in data["c"]])
    d = np.array([float(x) for x in data["d"]])

    n = len(b)
    return n, a, b, c, d

# ---------- Прямой ход ----------
def forward_sweep(a, b, c, d, n):
    P = np.zeros(n)
    Q = np.zeros(n)

    # i = 0 (первый индекс в Python, у нас a[0] = 0)
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    # остальные i = 1..n-1
    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

    return P, Q

# ---------- Обратный ход ----------
def backward_sweep(P, Q, n):
    x = np.zeros(n)

    # последнее неизвестное: x[n-1] = Q[n-1] (поскольку P[n-1] = 0)
    x[n - 1] = Q[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x

# ---------- Вывод ----------
def print_vector(v, name):
    print(name)
    print("  ".join(f"{val:10.5f}" for val in v))
    print()

# ---------- Главная программа ----------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test2.json"
    n, a, b, c, d = read_input(path)

    print(f"n = {n}\n")

    print_vector(a, "Массив a (поддиагональ):")
    print_vector(b, "Массив b (главная диагональ):")
    print_vector(c, "Массив c (наддиагональ):")
    print_vector(d, "Массив d (правая часть):")

    # прямой ход
    P, Q = forward_sweep(a, b, c, d, n)
    print_vector(P, "Прогоночные коэффициенты P:")
    print_vector(Q, "Прогоночные коэффициенты Q:")

    # обратный ход
    x = backward_sweep(P, Q, n)
    print_vector(x, "Решение x:")

if __name__ == "__main__":
    main()