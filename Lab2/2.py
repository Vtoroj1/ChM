import sys
import json
import math
import matplotlib.pyplot as plt
import numpy as np

def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    x0 = [float(data["x0_1"]), float(data["x0_2"])]
    x1_min = float(data["x1_min"])
    x1_max = float(data["x1_max"])
    x2_min = float(data["x2_min"])
    x2_max = float(data["x2_max"])
    eps = float(data.get("eps", 1e-4))
    
    return x0, (x1_min, x1_max), (x2_min, x2_max), eps

def f(x):
    x1, x2 = x
    f1 = 4 * x1 - math.cos(x2)
    f2 = 4 * x2 - math.exp(x1)
    return [f1, f2]

def J(x):
    x1, x2 = x
    return [[4, math.sin(x2)], 
            [-math.exp(x1), 4]]

def phi(x):
    x1, x2 = x
    phi1 = math.cos(x2) / 4
    phi2 = math.exp(x1) / 4
    return [phi1, phi2]

def dphi(x):
    x1, x2 = x
    return [[0, -math.sin(x2) / 4], 
            [math.exp(x1) / 4, 0]]

def diff_norm(x_new, x_old):
    return max(abs(x_new[0] - x_old[0]), abs(x_new[1] - x_old[1]))

def estimate_q(x1_range, x2_range):
    x1_vals = np.linspace(x1_range[0], x1_range[1], 50)
    x2_vals = np.linspace(x2_range[0], x2_range[1], 50)
    max_q = 0.0
    for x1 in x1_vals:
        for x2 in x2_vals:
            jac = dphi([x1, x2])
            row1 = abs(jac[0][0]) + abs(jac[0][1])
            row2 = abs(jac[1][0]) + abs(jac[1][1])
            max_q = max(max_q, row1, row2)
    return max_q

def simple_iteration_system(x0, eps, x1_range, x2_range, max_iter=100):
    print("\nМетод простой итерации для системы")
    
    q = estimate_q(x1_range, x2_range)
    print(f"Оценка q на заданной области: {q:.6f}")
    if q >= 1.0:
        print("Условие сходимости q < 1 не выполнено")
        
    print(f"{'k':>3} | {'x1':>10} | {'x2':>10} | {'|dx1|':>10} | {'|dx2|':>10}")
    
    x = list(x0)
    for k in range(max_iter):
        x_new = phi(x)
        diff = diff_norm(x_new, x)
        
        print(f"{k:3d} | {x_new[0]:10.6f} | {x_new[1]:10.6f} | {abs(x_new[0]-x[0]):10.2e} | {abs(x_new[1]-x[1]):10.2e}")
        
        if (q / (1 - q)) * diff <= eps:
            return x_new, k + 1
        x = x_new
        
    return x, max_iter

def newton_system(x0, eps, x_exact=None, max_iter=100):
    print("\nМетод Ньютона для системы")
    print(f"{'k':>3} | {'x1':>10} | {'x2':>10} | {'|dx1|':>10} | {'|dx2|':>10}")
    
    x = list(x0)
    for k in range(max_iter):
        fx = f(x)
        jac = J(x)
        det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
        if abs(det) < 1e-12:
            print("Вырожденная матрица Якоби")
            return x, k
        
        dx1 = (-fx[0] * jac[1][1] + fx[1] * jac[0][1]) / det
        dx2 = (-jac[0][0] * fx[1] + jac[1][0] * fx[0]) / det
        
        x_new = [x[0] + dx1, x[1] + dx2]
        
        print(f"{k:3d} | {x_new[0]:10.6f} | {x_new[1]:10.6f} | {abs(dx1):10.2e} | {abs(dx2):10.2e}")
        
        if diff_norm(x_new, x) <= eps:
            return x_new, k + 1
        x = x_new
        
    return x, max_iter

def plot_system(x1_range, x0, x_exact):
    x1_vals = np.linspace(x1_range[0], x1_range[1], 400)
    x2_curve1 = np.arccos(4 * x1_vals)
    x2_curve2 = np.exp(x1_vals) / 4

    plt.figure(figsize=(8, 6))

    plt.plot(x1_vals, x2_curve1, 'r', label='x_2 = arccos(4 * x_1)')
    plt.plot(x1_vals, x2_curve2, 'b', label='x_2 = e^x_1 / 4')
    
    plt.plot(x0[0], x0[1], 'ko', markersize=8, label=f'Начало {tuple(x0)}')
    if x_exact:
        plt.plot(x_exact[0], x_exact[1], 'g*', markersize=12, label='Точное решение')
    
    plt.title('Графическое решение системы')
    plt.xlabel(r'(x_1)')
    plt.ylabel(r'(x_2)')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "2.json"
    x0, x1_range, x2_range, eps = read_input(path)

    print(f"Область поиска: x1 in {x1_range}, x2 in {x2_range}")
    print(f"Начальное приближение x0 = {x0}")
    print(f"Точность eps = {eps}")

    x_exact, _ = newton_system(x0, 1e-12)
    print(f"\nЭталонное решение: x1 = {x_exact[0]:.10f}, x2 = {x_exact[1]:.10f}")

    plot_system(x1_range, x0, x_exact)

    x_si, k_si = simple_iteration_system(x0, eps, x1_range, x2_range)
    print(f"\nПростая итерация: {k_si} итераций, x1 = {x_si[0]:.6f}, x2 = {x_si[1]:.6f}")

    x_n, k_n = newton_system(x0, eps, x_exact)
    print(f"\nМетод Ньютона: {k_n} итераций, x1 = {x_n[0]:.6f}, x2 = {x_n[1]:.6f}")

if __name__ == "__main__":
    main()