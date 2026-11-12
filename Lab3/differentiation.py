import sys
import json

def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["x"], data["y"], data["x_star"]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "4.json"
    x, y, x_star = read_input(path)

    print("1. Исходная таблица")
    print(f"{'i':>2} | {'x_i':>8} | {'y_i':>8}")
    print("-" * 23)
    for i in range(len(x)):
        print(f"{i:2d} | {x[i]:8.4f} | {y[i]:8.4f}")
    print(f"\nТочка вычисления X* = {x_star}\n")

    idx = x.index(x_star)
    i = idx - 1

    h1 = x[i+1] - x[i]
    h2 = x[i+2] - x[i+1]

    print("2. Использованные узлы")
    print(f"Узел i={i}: x = {x[i]}, y = {y[i]}")
    print(f"Узел i={i+1}: x = {x[i+1]}, y = {y[i+1]} (X*)")
    print(f"Узел i={i+2}: x = {x[i+2]}, y = {y[i+2]}")
    print(f"Шаги: h1 = {h1}, h2 = {h2}\n")

    deriv_left = (y[i+1] - y[i]) / h1

    deriv_right = (y[i+2] - y[i+1]) / h2

    term1 = deriv_left
    term2 = (deriv_right - deriv_left) / (x[i+2] - x[i])
    term3 = 2 * x_star - x[i] - x[i+1]
    deriv1 = term1 + term2 * term3

    deriv2 = 2 * term2

    print("3. Результаты")
    print(f"Первая производная (левосторонняя, 1-й пор.): {deriv_left:.6f}")
    print(f"Первая производная (правосторонняя, 1-й пор.): {deriv_right:.6f}")
    print(f"Первая производная (центральная, 2-й пор.): {deriv1:.6f}")
    print(f"Вторая производная: {deriv2:.6f}")

if __name__ == "__main__":
    main()