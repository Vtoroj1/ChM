import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt

def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    x_nodes = [float(x) for x in data["x_nodes"]]
    x_star = float(data["x_star"])
    return x_nodes, x_star

def f(x):
    return math.sqrt(x) + x

def lagrange_eval(x, nodes, values):
    result = 0.0
    n = len(nodes)
    for i in range(n):
        term = values[i]
        for j in range(n):
            if i != j:
                term *= (x - nodes[j]) / (nodes[i] - nodes[j])
        result += term
    return result

def get_lagrange_string(nodes, values):
    n = len(nodes)
    parts = []
    for i in range(n):
        denom = 1.0
        for j in range(n):
            if i != j:
                denom *= (nodes[i] - nodes[j])
        coef = values[i] / denom
        num_str = ""
        for j in range(n):
            if i != j:
                if nodes[j] == 0:
                    num_str += "x"
                elif nodes[j] > 0:
                    num_str += f"(x - {nodes[j]})"
                else:
                    num_str += f"(x + {abs(nodes[j])})"
        
        parts.append(f"{coef:.4f} * {num_str}")
    
    return " + ".join(parts).replace("+ -", "- ")

def get_divided_differences(nodes, values):
    n = len(nodes)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = values[i]
    
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (nodes[i+j] - nodes[i])
    return table

def newton_eval(x, nodes, table):
    n = len(nodes)
    result = table[0][0]
    product = 1.0
    for i in range(1, n):
        product *= (x - nodes[i-1])
        result += table[0][i] * product
    return result

def get_newton_string(nodes, table):
    n = len(nodes)
    parts = [f"{table[0][0]:.4f}"]
    product_str = ""
    
    for i in range(1, n):
        if nodes[i-1] == 0:
            product_str += "x"
        elif nodes[i-1] > 0:
            product_str += f"(x - {nodes[i-1]})"
        else:
            product_str += f"(x + {abs(nodes[i-1])})"
            
        parts.append(f"{table[0][i]:.4f} * {product_str}")
            
    return " + ".join(parts).replace("+ -", "- ")

def plot_results(nodes, values, x_star, f_exact, lagrange_eval_func, newton_eval_func):
    x_min = min(nodes)
    x_max = max(nodes) + 0.5
    
    start_x = max(0.0, x_min - 0.5)
    x_plot = np.linspace(start_x, x_max, 400)
    
    y_exact = [f_exact(x) for x in x_plot]
    y_lagrange = [lagrange_eval_func(x) for x in x_plot]
    y_newton = [newton_eval_func(x) for x in x_plot]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_plot, y_exact, 'k-', linewidth=2, label=r'Исходная функция (f(x) = sqrt{x} + x)')
    plt.plot(x_plot, y_lagrange, 'r--', linewidth=2, label='Многочлен Лагранжа')
    plt.plot(x_plot, y_newton, 'b:', linewidth=2, label='Многочлен Ньютона')
    
    plt.scatter(nodes, values, color='black', zorder=5, s=50, label='Узлы интерполяции')
    plt.axvline(x_star, color='green', linestyle='-.', alpha=0.7, label=f'Точка (X^* = {x_star})')
    plt.scatter([x_star], [f_exact(x_star)], color='green', zorder=5, s=80, marker='*', label='(f(X^*))')
    
    plt.title('Интерполяция функции: Лагранж / Ньютон')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "1.json"
    x_nodes, x_star = read_input(path)
    y_nodes = [f(x) for x in x_nodes]
    n = len(x_nodes)

    print("1. Исходные узлы и значения")
    print(f"{'i':>2} | {'x_i':>10} | {'y_i':>10}")
    print("-" * 28)
    for i in range(n):
        print(f"{i:2d} | {x_nodes[i]:10.4f} | {y_nodes[i]:10.4f}")
    print()

    print("2. Интерполяционный многочлен Лагранжа")
    print(get_lagrange_string(x_nodes, y_nodes))
    print()

    print("3. Таблица разделенных разностей")
    table = get_divided_differences(x_nodes, y_nodes)
    
    print(f"{'x_i':>6} | {'f(x_i)':>10} | {'1-я разн.':>10} | {'2-я разн.':>10} | {'3-я разн.':>10}")
    print("-" * 58)
    for i in range(n):
        row = f"{x_nodes[i]:6.2f} | {table[i][0]:10.4f}"
        for j in range(1, n - i):
            row += f" | {table[i][j]:10.4f}"
        print(row)
    print()

    print("4. Интерполяционный многочлен Ньютона")
    print(get_newton_string(x_nodes, table))
    print()

    L_star = lagrange_eval(x_star, x_nodes, y_nodes)
    N_star = newton_eval(x_star, x_nodes, table)
    f_star = f(x_star)
    
    print("5. Результаты в точке X*")
    print(f"Значение Лагранжа L(X*): {L_star:.6f}")
    print(f"Значение Ньютона P(X*): {N_star:.6f}")
    print(f"Точное значение f(X*): {f_star:.6f}")
    print(f"Абсолютная погрешность Лагранжа: {abs(L_star - f_star):.6e}")
    print(f"Абсолютная погрешность Ньютона: {abs(N_star - f_star):.6e}")
    print(f"Разность методов |L(X*) - P(X*)|: {abs(L_star - N_star):.6e}")

    plot_results(
        x_nodes, y_nodes, x_star, f, 
        lambda x: lagrange_eval(x, x_nodes, y_nodes),
        lambda x: newton_eval(x, x_nodes, table)
    )


if __name__ == "__main__":
    main()