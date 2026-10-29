import sys
import json
import math
import matplotlib.pyplot as plt
import numpy as np

def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    a = float(data["a"])
    b = float(data["b"])
    x0 = float(data["x0"])
    eps = float(data.get("eps", 1e-4))
    return a, b, x0, eps

def f(x):
    return math.log(x + 1) - x**3 + 1

def df(x):
    return 1.0 / (x + 1) - 3 * x**2

def phi(x):
    return (math.log(x + 1) + 1) ** (1.0 / 3.0)

def dphi(x):
    return (1.0 / 3.0) * (math.log(x + 1) + 1) ** (-2.0 / 3.0) / (x + 1)

def simple_iteration(x0, eps, x_exact, a, b, max_iter=1000):
    print("\nМетод простой итерации")

    steps = 100
    q = max(abs(dphi(a + (b - a) * i / steps)) for i in range(steps + 1))
    print(f"Оценка q на отрезке [{a}, {b}]: {q:.6f}")
    
    if q >= 1.0:
        print("Условие сходимости |phi'(x)| < 1 может не выполняться")

    print(f"{'k':>3} | {'x_k':>12} | {'phi(x_k)':>12} | {'|x_{k+1} - x_k|':>18} | {'Погрешность':>14}")
    
    x = x0
    for k in range(max_iter):
        x_new = phi(x)
        diff = abs(x_new - x)
        err = abs(x_new - x_exact)
        
        print(f"{k:3d} | {x:12.8f} | {x_new:12.8f} | {diff:18.8e} | {err:12.8e}")
        
        if q / (1 - q) * diff <= eps:
            return x_new, k + 1
        x = x_new
        
    return x, max_iter

def newton(x0, eps, x_exact, max_iter=1000):
    print("\nМетод Ньютона")
    print(f"{'k':>3} | {'x_k':>12} | {'f(x_k)':>15} | {'f\'(x_k)':>12} | {'|x_{k+1} - x_k|':>18} | {'Погрешность':>14}")
    
    x = x0
    for k in range(max_iter):
        fx = f(x)
        dfx = df(x)
        x_new = x - fx / dfx
        diff = abs(x_new - x)
        err = abs(x_new - x_exact)
        
        print(f"{k:3d} | {x:12.8f} | {fx:12.8e} | {dfx:12.8f} | {diff:18.8e} | {err:12.8e}")
        
        if diff <= eps:
            return x_new, k + 1
        x = x_new
        
    return x, max_iter

def plot_equation(a, b, x0):
    x_vals = np.linspace(a - 0.5, b + 0.5, 400)
    y_vals = [f(x) for x in x_vals]
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label=r'f(x) = ln(x+1) - x^3 + 1', color='blue', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.axvspan(a, b, color='green', alpha=0.2, label=f'Отрезок [{a}, {b}]')
    
    plt.plot(x0, f(x0), 'ro', label=f'x0 = {x0}')
    
    plt.title('Графическое отделение корня уравнения')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "1.json"
    a, b, x0, eps = read_input(path)

    print(f"Отрезок локализации: [{a}, {b}]")
    print(f"Начальное приближение x0 = {x0}")
    print(f"Точность eps = {eps}")

    x_exact, _ = newton(x0, 1e-12, x0)
    print(f"\nЭталонное решение (высокая точность): {x_exact:.12f}")

    plot_equation(a, b, x0)

    x_si, k_si = simple_iteration(x0, eps, x_exact, a, b)
    print(f"\nПростая итерация: {k_si} итераций, x = {x_si:.8f}, погрешность = {abs(x_si - x_exact):.2e}")

    x_n, k_n = newton(x0, eps, x_exact)
    print(f"\nМетод Ньютона: {k_n} итераций, x = {x_n:.8f}, погрешность = {abs(x_n - x_exact):.2e}")

if __name__ == "__main__":
    main()