import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Ось значень
x = np.linspace(0, 10, 100)

# Трикутна функція належності
triangular = fuzz.trimf(x, [2, 5, 8])

# Трапецієподібна функція належності
trapezoidal = fuzz.trapmf(x, [2, 4, 6, 8])

# Візуалізація
plt.figure()
plt.plot(x, triangular, label="Трикутна функція")
plt.plot(x, trapezoidal, label="Трапецієподібна функція")
plt.title("Трикутна і трапецієподібна функції належності")
plt.legend()
plt.show()

# GAUSS
# Проста функція Гаусса
gaussian = fuzz.gaussmf(x, mean=5, sigma=1.5)

# Двостороння функція Гаусса
gaussian2 = fuzz.gauss2mf(x, mean1=3, sigma1=1, mean2=7, sigma2=1.5)

plt.figure()
plt.plot(x, gaussian, label="Проста Гауссова функція")
plt.plot(x, gaussian2, label="Двостороння Гауссова функція")
plt.title("Функції Гаусса")
plt.legend()
plt.show()

# BELL
general_bell = fuzz.gbellmf(x, a=2, b=3, c=5)

plt.figure()
plt.plot(x, general_bell, label="Узагальнена дзвонова функція")
plt.title("Узагальнена дзвонова функція")
plt.legend()
plt.show()

# SIGMOID
# Односторонні сігмоїдні функції
sigmoid_left = fuzz.sigmf(x, c=3, b=-2)  # Відкрита справа
sigmoid_right = fuzz.sigmf(x, c=7, b=2)  # Відкрита зліва

# Двостороння функція
sigmoid_double = sigmoid_left * sigmoid_right

plt.figure()
plt.plot(x, sigmoid_left, label="Одностороння (відкрита справа)")
plt.plot(x, sigmoid_right, label="Одностороння (відкрита зліва)")
plt.plot(x, sigmoid_double, label="Двостороння сігмоїдна")
plt.title("Сігмоїдні функції")
plt.legend()
plt.show()

# POLYNOMIAL
# Z-функція
z_func = fuzz.zmf(x, 3, 7)

# S-функція
s_func = fuzz.smf(x, 3, 7)

# PI-функція (поєднання Z і S)
pi_func = fuzz.pimf(x, 3, 5, 7, 9)

plt.figure()
plt.plot(x, z_func, label="Z-функція")
plt.plot(x, s_func, label="S-функція")
plt.plot(x, pi_func, label="PI-функція")
plt.title("Поліноміальні функції належності")
plt.legend()
plt.show()

# LOGIC OPERATORS
# Приклад двох нечітких множин
set1 = fuzz.trimf(x, [2, 5, 8])
set2 = fuzz.trimf(x, [4, 6, 9])

# Операції
intersection = np.fmin(set1, set2)  # Мінімум
union = np.fmax(set1, set2)  # Максимум

plt.figure()
plt.plot(x, set1, label="Множина 1")
plt.plot(x, set2, label="Множина 2")
plt.plot(x, intersection, label="Перетин (мінімум)")
plt.plot(x, union, label="Об'єднання (максимум)")
plt.title("Мінімаксна інтерпретація операторів")
plt.legend()
plt.show()

# CONC/DISC
# Кон'юнкція (добуток)
conjunction = set1 * set2

# Диз'юнкція (сума мінус перетин)
disjunction = set1 + set2 - (set1 * set2)

plt.figure()
plt.plot(x, conjunction, label="Кон'юнкція (добуток)")
plt.plot(x, disjunction, label="Диз'юнкція (сума)")
plt.title("Вірогідна інтерпретація операторів")
plt.legend()
plt.show()

# MULTIPLE
complement = 1 - set1

plt.figure()
plt.plot(x, set1, label="Нечітка множина")
plt.plot(x, complement, label="Доповнення")
plt.title("Доповнення нечіткої множини")
plt.legend()
plt.show()
