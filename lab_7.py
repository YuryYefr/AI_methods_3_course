import numpy as np
from scipy.optimize import minimize

# Функція для мінімізації
def func(x):
    # Уникаємо ділення на 0, обмежимо x від 0.1 для уникнення проблеми ділення на 0
    if x == 0:
        return np.inf  # Повертаємо безкінечність для x = 0
    return (np.cos(x) / 2) - (np.sin(x) / x**2)

# Початкова точка
x0 = np.array([1])

# Мінімізація функції
result = minimize(func, x0)

# Виведення результату
print(f"Мінімум функції: {result.fun:.2f} на x = {result.x}")

# Функція для максимізації (включає y)
def func_2d(vars):
    x = vars[0]
    # Обчислюємо y та z
    if x == 0:
        return np.inf  # Повертаємо безкінечність для x = 0
    y = (np.cos(x) / 2) - (np.sin(x) / x**2)
    z = (np.sin(x) / 2) + y * np.sin(x)
    return -z  # Мінімізуємо від'ємну функцію для максимізації

# Початкова точка
x0 = np.array([1])

# Мінімізація функції (для максимізації від'ємної функції)
result = minimize(func_2d, x0)

# Виведення результату
print(f"Максимум функції: {-result.fun:.2f} на x = {result.x[0]:.2f}")
