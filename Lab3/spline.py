import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["x"], data["f"], data["x_star"]

def progon(a, b, c, d):
    n = len(b)
    P = [0.0] * n
    Q = [0.0] * n
    x = [0.0] * n

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * P[i-1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i-1]) / denom

    x[n-1] = Q[n-1]
    for i in range(n-2, -1, -1):
        x[i] = P[i] * x[i+1] + Q[i]

    return x

def build_spline(x, f):
    n = len(x) - 1
    h = [x[i+1] - x[i] for i in range(n)]
    m = n - 1
    
    low = [0.0] * m
    mid = [0.0] * m
    up = [0.0] * m
    r = [0.0] * m

    for i in range(m):
        if i == 0:
            mid[i] = 2 * (h[0] + h[1])
            up[i] = h[1]
            r[i] = 3 * ((f[2] - f[1]) / h[1] - (f[1] - f[0]) / h[0])
        elif i == m - 1:
            low[i] = h[i]
            mid[i] = 2 * (h[i] + h[i+1])
            r[i] = 3 * ((f[i+2] - f[i+1]) / h[i+1] - (f[i+1] - f[i]) / h[i])
        else:
            low[i] = h[i]
            mid[i] = 2 * (h[i] + h[i+1])
            up[i] = h[i+1]
            r[i] = 3 * ((f[i+2] - f[i+1]) / h[i+1] - (f[i+1] - f[i]) / h[i])

    c_mid = progon(low, mid, up, r)
    
    c = [0.0] + c_mid + [0.0]
    a = [0.0] * n
    b = [0.0] * n
    d = [0.0] * n

    for i in range(n):
        a[i] = f[i]
        b[i] = (f[i+1] - f[i]) / h[i] - (h[i] / 3) * (c[i+1] + 2 * c[i])
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    return a, b, c, d

def eval_spline(x_star, x, a, b, c, d):
    for i in range(len(x) - 1):
        if x[i] <= x_star <= x[i+1]:
            xx = x_star - x[i]
            val = a[i] + b[i] * xx + c[i] * xx**2 + d[i] * xx**3
            return val, i
    return None, None

def spline_second_derivative(x_val, x, c, d):
    for i in range(len(x) - 1):
        if x[i] <= x_val <= x[i+1]:
            xx = x_val - x[i]
            return 2 * c[i] + 6 * d[i] * xx
    return None

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "2.json"
    x_nodes, f_nodes, x_star = read_input(path)

    print("1. Исходные данные")
    print(f"{'i':>2} | {'x_i':>10} | {'f_i':>10}")
    print("-" * 28)
    for i in range(len(x_nodes)):
        print(f"{i:2d} | {x_nodes[i]:10.4f} | {f_nodes[i]:10.4f}")
    print()

    a, b, c, d = build_spline(x_nodes, f_nodes)

    print("2. Коэффициенты сплайна")
    for i in range(len(x_nodes) - 1):
        print(f"Интервал [{x_nodes[i]:.2f}, {x_nodes[i+1]:.2f}]:")
        print(f"  a_{i} = {a[i]: .6f}, b_{i} = {b[i]: .6f}, c_{i} = {c[i]: .6f}, d_{i} = {d[i]: .6f}")
    print()

    S_star, interval_idx = eval_spline(x_star, x_nodes, a, b, c, d)
    print("3. Значение в точке X*")
    print(f"x* = {x_star}")
    print(f"Точка принадлежит интервалу i = {interval_idx} ([{x_nodes[interval_idx]:.2f}, {x_nodes[interval_idx+1]:.2f}])")
    print(f"S(x*) = {S_star:.6f}\n")

    print("4. Проверка прохождения через узлы")
    print(f"{'Узел':>5} | {'S(x_i)':>12} | {'f_i':>12} | {'Разность':>12}")
    print("-" * 50)
    for i in range(len(x_nodes)):
        val, _ = eval_spline(x_nodes[i], x_nodes, a, b, c, d)
        diff = abs(val - f_nodes[i])
        print(f"{i:5d} | {val:12.6f} | {f_nodes[i]:12.6f} | {diff:12.2e}")
    print()

    print("5. Проверка граничных условий")
    left_sec_deriv = spline_second_derivative(x_nodes[0], x_nodes, c, d)
    right_sec_deriv = spline_second_derivative(x_nodes[-1], x_nodes, c, d)
    print(f"Вторая производная в x_0 ({x_nodes[0]}): {left_sec_deriv:.6e}")
    print(f"Вторая производная в x_n ({x_nodes[-1]}): {right_sec_deriv:.6e}")
    print()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_nodes, f_nodes, color='black', zorder=5, s=50, label='Исходные узлы')

    for i in range(len(x_nodes) - 1):
        x_plot = np.linspace(x_nodes[i], x_nodes[i+1], 50)
        y_plot = [eval_spline(xi, x_nodes, a, b, c, d)[0] for xi in x_plot]
        plt.plot(x_plot, y_plot, linewidth=2, 
                 label=f'Сплайн на [{x_nodes[i]:.1f}, {x_nodes[i+1]:.1f}]')
        
    plt.scatter([x_star], [S_star], color='red', zorder=5, s=80, marker='*', label=f'x* = {x_star}')
    
    plt.title('Кубический сплайн')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()