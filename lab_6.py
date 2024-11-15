import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Завантажуємо CSV файл
data = pd.read_csv('nasdaq_data.csv')

# Формуємо вибірку для навчання
features = data[['Open', 'High', 'Low', 'Close']].values
target = data['Close'].shift(-1).dropna()  # Прогноз на наступний день

# Розбиваємо дані на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(features[:-1], target, test_size=0.1, random_state=42)

# Параметри нечіткої логіки (створюємо функції приналежності)
x = np.arange(0, 20000, 1)  # Збільшуємо діапазон для покриття більших значень
low = fuzz.trapmf(x, [1000, 2000, 5000, 7000])  # Низьке значення
medium = fuzz.trimf(x, [5000, 10000, 15000])  # Середнє значення
high = fuzz.trapmf(x, [12000, 15000, 18000, 20000]) # Високе значення

# Візуалізація функцій приналежності
plt.figure(figsize=(10, 6))
plt.plot(x, low, label='Low')
plt.plot(x, medium, label='Medium')
plt.plot(x, high, label='High')
plt.title("Функції приналежності для нечіткої логіки")
plt.legend()
plt.show()

# Нейронна мережа (Можна змінювати параметри та кількість нейронів)
model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, activation='relu')

# Навчання моделі
model.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred = model.predict(X_test)

# Оцінка точності
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted', color='red')
plt.title("Прогнозування ціни закриття акцій")
plt.xlabel("Дата")
plt.ylabel("Ціна")
plt.legend()
plt.show()

# Створення гібридної моделі (поєднання нейронної мережі та нечіткої логіки)
# Приклад використання нечіткої логіки для корекції виходу мережі
fuzzy_output = []

for val in y_pred:
    # Перевіряємо до якого діапазону відноситься передбачене значення
    if val <= 7000:
        fuzzy_output.append(fuzz.interp_membership(x, low, val))
    elif 7000 < val <= 15000:
        fuzzy_output.append(fuzz.interp_membership(x, medium, val))
    else:
        fuzzy_output.append(fuzz.interp_membership(x, high, val))

# Display the fuzzy output with 2 decimals
fuzzy_output_str = ["{:.2f}".format(val) for val in fuzzy_output]

# Виведемо результати для перевірки
print("Нечітке значення для прогнозу:", fuzzy_output_str[:10])
