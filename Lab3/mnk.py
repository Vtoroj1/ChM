import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["x"], data["y"]

def solve_gauss(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    for k in range(n):
        p = k
        for i in range(k+1, n):
            if abs(M[i][k]) > abs(M[p][k]):
                p = i
        M[k], M[p] = M[p], M[k]
        
        for i in range(k+1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n+1):
                M[i][j] -= factor * M[k][j]
                
    a = [0.0] * n
    for i in range(n-1, -1, -1):
        s = sum(M[i][j] * a[j] for j in range(i+1, n))
        a[i] = (M[i][n] - s) / M[i][i]
    return a


def mnk_fit(x, y, degree):
    n = len(x)
    m = degree + 1
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    
    for i in range(m):
        for j in range(m):
            A[i][j] = sum(x[k]**(i+j) for k in range(n))
        b[i] = sum(y[k] * (x[k]**i) for k in range(n))
        
    return solve_gauss(A, b)


def poly_eval(coefs, x_val):
    return sum(a * (x_val**i) for i, a in enumerate(coefs))


def poly_to_string(coefs):
    parts = []
    for i in range(len(coefs) - 1, -1, -1):
        a = coefs[i]
        term = f"{abs(a):.4f}"
        if i > 0:
            term += "x" if i == 1 else f"x^{i}"
        
        if not parts:
            parts.append(("-" if a < 0 else "") + term)
        else:
            parts.append((" - " if a < 0 else " + ") + term)
    return "".join(parts) if parts else "0"


def compute_mse(x, y, coefs):
    n = len(x)
    err = sum((poly_eval(coefs, x[i]) - y[i])**2 for i in range(n))
    return err / n

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "3.json"
    x, y = read_input(path)

    print("1. Исходные данные")
    print(f"{'i':>2} | {'x_i':>8} | {'y_i':>8}")
    print("-" * 24)
    for i in range(len(x)):
        print(f"{i:2d} | {x[i]:8.4f} | {y[i]:8.4f}")
    print()

    degrees = [1, 2, 3, 5]
    coefs_list = []
    mse_list = []

    print("2. Результаты МНК")
    for d in degrees:
        coefs = mnk_fit(x, y, d)
        coefs_list.append(coefs)
        mse = compute_mse(x, y, coefs)
        mse_list.append(mse)

        print(f"\nМногочлен {d}-й степени")
        print("Коэффициенты (a_0, a_1, ...):")
        for i, c in enumerate(coefs):
            print(f"  a_{i} = {c: .6f}")
        
        print(f"Многочлен P_{d}(x):")
        print(f"P_{d}(x) = {poly_to_string(coefs)}")
        print(f"MSE_{d} = {mse:.6e}")

    print("\n3. Сравнение")
    print(f"MSE_1 = {mse_list[0]:.6e}")
    print(f"MSE_2 = {mse_list[1]:.6e}")
    print(f"MSE_3 = {mse_list[2]:.6e}")
    
    best_degree = degrees[np.argmin(mse_list)]
    print(f"Лучший результат показал многочлен {best_degree}-й степени")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='black', zorder=5, s=50, label='Исходные точки')
    
    x_plot = np.linspace(min(x)-0.5, max(x)+0.5, 400)
    colors = ['green', 'blue', 'red', 'purple']
    styles = ['--', '-.', ':', '-']
    
    for idx, d in enumerate(degrees):
        y_plot = [poly_eval(coefs_list[idx], xi) for xi in x_plot]
        plt.plot(x_plot, y_plot, color=colors[idx], linestyle=styles[idx], 
                 linewidth=2, label=f'P_{d}(x), MSE={mse_list[idx]:.2e}')

    plt.title('Метод наименьших квадратов')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()