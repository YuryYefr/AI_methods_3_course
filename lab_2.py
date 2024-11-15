import numpy as np
import skfuzzy as fuzz
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation
import matplotlib.pyplot as plt


# Діапазони змінних
x1_range = np.linspace(0, 10, 100)  # Вхід 1
x2_range = np.linspace(-5, 5, 100)  # Вхід 2
y_range = np.linspace(0, 100, 100)  # Вихід

# Створення змінних
x1 = Antecedent(x1_range, 'x1')
x2 = Antecedent(x2_range, 'x2')
y = Consequent(y_range, 'y')

# Вхідні функції належності
x1.automf(names=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])
x2.automf(names=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])

# Вихідні функції належності
y['very_low'] = fuzz.trimf(y_range, [0, 0, 20])
y['low'] = fuzz.trimf(y_range, [0, 20, 40])
y['medium_low'] = fuzz.trimf(y_range, [20, 40, 60])
y['medium'] = fuzz.trimf(y_range, [40, 60, 80])
y['medium_high'] = fuzz.trimf(y_range, [60, 80, 100])
y['high'] = fuzz.trimf(y_range, [80, 100, 120])
y['very_high'] = fuzz.trimf(y_range, [100, 120, 140])
y['extreme_low'] = fuzz.trimf(y_range, [-20, 0, 20])
y['extreme_high'] = fuzz.trimf(y_range, [80, 100, 120])
# Приклад заповнення бази знань
rules = []
output_labels = list(y.terms.keys())

# Генерація 36 правил
for i, x1_label in enumerate(x1.terms.keys()):
    for j, x2_label in enumerate(x2.terms.keys()):
        output_label = output_labels[(i + j) % len(output_labels)]  # Обираємо вихід циклічно
        rule = Rule(antecedent=(x1[x1_label] & x2[x2_label]), consequent=y[output_label])
        rules.append(rule)

# Побудова системи керування
control_system = ControlSystem(rules)
simulation = ControlSystemSimulation(control_system)

# Вхідні значення
input_x1 = 6.0
input_x2 = 2.0

# Моделювання
simulation.input['x1'] = input_x1
simulation.input['x2'] = input_x2
simulation.compute()

# Результат
output_y = simulation.output['y']
print(f"Для x1 = {input_x1}, x2 = {input_x2} вихід y = {output_y:.2f}")

# Новий набір правил (лише діагональ)
reduced_rules = []
for i, label in enumerate(x1.terms.keys()):
    rule = Rule(antecedent=(x1[label] & x2[label]), consequent=y[output_labels[i]])
    reduced_rules.append(rule)

# Побудова нової системи керування
reduced_control_system = ControlSystem(reduced_rules)
reduced_simulation = ControlSystemSimulation(reduced_control_system)

# Моделювання з меншою кількістю правил
reduced_simulation.input['x1'] = input_x1
reduced_simulation.input['x2'] = input_x2
reduced_simulation.compute()

# Результат
reduced_output_y = reduced_simulation.output['y']
print(f"Результат для скороченої моделі: y = {reduced_output_y:.2f}")

# Візуалізація функцій належності вихідної змінної
y.view()

# Моделювання для набору значень
x1_values = np.linspace(0, 10, 10)
x2_values = np.linspace(-5, 5, 10)
results = []

for x1_val in x1_values:
    for x2_val in x2_values:
        simulation.input['x1'] = x1_val
        simulation.input['x2'] = x2_val
        simulation.compute()
        results.append(simulation.output['y'])

# Вивід результатів
plt.figure()
plt.plot(results, label="Модель з повною базою правил")
plt.legend()
plt.show()
