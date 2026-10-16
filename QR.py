import sys
import json
import math
import numpy as np

MAX_ITER = 10000


def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    A = np.array([[float(x) for x in row] for row in data["A"]])
    eps = float(data.get("eps", 1e-6))
    return A, eps


def qr_decompose(A):
    n = A.shape[0]
    R = A.copy()
    Q = np.eye(n)

    for k in range(n - 1):
        b = R[k:, k].copy()
        norm_b = math.sqrt(np.dot(b, b))
        if norm_b < 1e-14:
            continue

        sign = 1.0 if b[0] >= 0 else -1.0
        v = b.copy()
        v[0] = v[0] + sign * norm_b
        coef = 2.0 / np.dot(v, v)

        vt_R = v @ R[k:, :]
        R[k:, :] = R[k:, :] - coef * np.outer(v, vt_R)

        Q_v = Q[:, k:] @ v
        Q[:, k:] = Q[:, k:] - coef * np.outer(Q_v, v)

    return Q, R


def qr_algorithm(A, eps, max_iter=MAX_ITER):
    A_k = A.copy()
    sv_prev = None
    it = 0

    for it in range(1, max_iter + 1):
        Q, R = qr_decompose(A_k)
        A_k = R @ Q

        sv_curr = extract_svalues(A_k, eps)

        if sv_prev is not None and len(sv_curr) == len(sv_prev):
            max_diff = 0.0
            for e1, e2 in zip(sv_curr, sv_prev):
                d = abs(e1 - e2)
                if d > max_diff:
                    max_diff = d
            if max_diff < eps:
                break

        sv_prev = sv_curr

    return A_k, it


def extract_svalues(A_k, eps):
    n = A_k.shape[0]
    svalues = []
    i = 0
    while i < n:
        if i == n - 1 or abs(A_k[i + 1, i]) < eps:
            svalues.append(complex(A_k[i, i]))
            i += 1
        else:
            a = A_k[i, i]
            b = A_k[i, i + 1]
            c = A_k[i + 1, i]
            d = A_k[i + 1, i + 1]
            tr = a + d
            det = a * d - b * c
            disc = tr * tr - 4 * det
            if disc >= 0:
                sq = math.sqrt(disc)
                svalues.append(complex((tr + sq) / 2))
                svalues.append(complex((tr - sq) / 2))
            else:
                sq = math.sqrt(-disc)
                svalues.append(complex(tr / 2, sq / 2))
                svalues.append(complex(tr / 2, -sq / 2))
            i += 2
    return svalues


def print_matrix(M, name):
    print(name)
    for row in M:
        print("  ".join(f"{x:14.6f}" for x in row))
    print()


def print_svalues(svalues, name="Собственные значения:"):
    print(name)
    for sv in svalues:
        if abs(sv.imag) < 1e-9:
            print(f"  {sv.real:.6f}")
        else:
            sign = "+" if sv.imag >= 0 else "-"
            print(f"  {sv.real:.6f} {sign} {abs(sv.imag):.6f}i")
    print()


def process(A, eps, with_numpy_check=False):
    print_matrix(A, "Исходная матрица A:")

    Q, R = qr_decompose(A)
    print_matrix(Q, "Матрица Q:")
    print_matrix(R, "Матрица R:")

    QR = Q @ R
    print_matrix(QR, "Q * R:")

    A_final, iters = qr_algorithm(A, eps)
    svalues = extract_svalues(A_final, eps)

    print(f"Число итераций QR-алгоритма: {iters}")
    print_svalues(svalues)

    if with_numpy_check:
        print("Проверка через np.linalg.eigvals:")
        np_eigs = np.linalg.eigvals(A)
        for ev in np_eigs:
            if abs(ev.imag) < 1e-9:
                print(f"  {ev.real:.6f}")
            else:
                sign = "+" if ev.imag >= 0 else "-"
                print(f"  {ev.real:.6f} {sign} {abs(ev.imag):.6f}i")
        print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test5.json"
    A, eps = read_input(path)

    print(f"eps = {eps}\n")
    process(A, eps)

    print("Пример с комплексно-сопряжёнными собственными значениями")
    A_complex = np.array([
        [2.0, -1.0, 0.0],
        [1.0,  2.0, 0.0],
        [0.0,  0.0, 5.0]
    ])
    process(A_complex, eps, with_numpy_check=True)


if __name__ == "__main__":
    main()