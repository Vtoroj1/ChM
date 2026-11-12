import sys
import json
import math

def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["x0"], data["xk"], data["h1"], data["h2"]

def f(x):
    return x / (16 - x**4)

def exact_integral(x0, xk):
    def F(x):
        return (1.0 / 16.0) * math.log(abs((4 + x**2) / (4 - x**2)))
    return F(xk) - F(x0)

def left_rect(f, a, b, h):
    n = int(round((b - a) / h))
    s = 0.0
    for i in range(n):
        s += f(a + i * h)
    return s * h

def right_rect(f, a, b, h):
    n = int(round((b - a) / h))
    s = 0.0
    for i in range(1, n + 1):
        s += f(a + i * h)
    return s * h

def mid_rect(f, a, b, h):
    n = int(round((b - a) / h))
    s = 0.0
    for i in range(n):
        s += f(a + i * h + h / 2.0)
    return s * h

def trapezoid(f, a, b, h):
    n = int(round((b - a) / h))
    s = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

def simpson(f, a, b, h):
    n = int(round((b - a) / h))
    if n % 2 != 0:
        raise ValueError("Число интервалов для метода Симпсона должно быть четным")
    
    s = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 0:
            s += 2 * f(a + i * h)
        else:
            s += 4 * f(a + i * h)
    return s * (h / 3.0)

def runge_romberg(I_h1, I_h2, p):
    return I_h2 + (I_h2 - I_h1) / (2**p - 1)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "5.json"
    x0, xk, h1, h2 = read_input(path)

    exact = exact_integral(x0, xk)
    print(f"Интервал: [{x0}, {xk}]")
    print(f"Шаги: h1 = {h1}, h2 = {h2}")
    print(f"Точное значение интеграла: {exact:.8f}\n")

    methods = [
        ("Левые прямоугольники", left_rect, 2),
        ("Правые прямоугольники", right_rect, 2),
        ("Средние прямоугольники", mid_rect, 2),
        ("Трапеции", trapezoid, 2),
        ("Симпсон", simpson, 4)
    ]

    print(f"{'Метод':<25} | {'I(h1)':>10} | {'I(h2)':>10} | {'p':>2} | {'Уточн.':>10} | {'Оценка погр.':>12} | {'Точн. погр.':>12}")
    print("-" * 105)

    best_method = None
    min_error = float('inf')

    for name, func, p in methods:
        I_h1 = func(f, x0, xk, h1)
        I_h2 = func(f, x0, xk, h2)
        
        I_refined = runge_romberg(I_h1, I_h2, p)
        err_est = abs(I_h2 - I_h1) / (2**p - 1)
        abs_err = abs(I_h2 - exact)

        if abs_err < min_error:
            min_error = abs_err
            best_method = name

        print(f"{name:<25} | {I_h1:10.6f} | {I_h2:10.6f} | {p:2d} | {I_refined:10.6f} | {err_est:12.2e} | {abs_err:12.2e}")

    print("\n" + "=" * 50)
    print(f"Наиболее точный метод (с шагом h2): {best_method}")
    print(f"Его абсолютная погрешность: {min_error:.2e}")

if __name__ == "__main__":
    main()