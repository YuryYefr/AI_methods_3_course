import numpy as np

# Параметри
learning_rate = 1  # Коефіцієнт навчання
letters = {
    "A": np.array([1, 0, 1, 0, 1,  # Зображення літери А
                   1, 1, 1, 1, 1,
                   1, 0, 1, 0, 1]).reshape(-1, 1),
    "H": np.array([1, 0, 1, 1, 0,  # Зображення літери H
                   1, 1, 1, 1, 1,
                   1, 0, 1, 1, 0]).reshape(-1, 1),
    "C": np.array([1, 1, 1, 0, 0,  # Зображення літери C
                   1, 0, 0, 0, 0,
                   1, 1, 1, 0, 0]).reshape(-1, 1),
    "B": np.array([1, 1, 1, 1, 0,  # Зображення літери B
                   1, 0, 1, 1, 0,
                   1, 1, 1, 1, 0]).reshape(-1, 1),
    "S": np.array([1, 1, 1, 1, 1,  # Зображення літери S
                   0, 0, 1, 1, 1,
                   1, 1, 1, 1, 0]).reshape(-1, 1),
    "M": np.array([1, 0, 0, 0, 1,  # Зображення літери M
                   1, 1, 0, 1, 1,
                   1, 0, 1, 0, 1]).reshape(-1, 1),
    "O": np.array([1, 1, 1, 1, 0,  # Зображення літери O
                   1, 0, 1, 0, 1,
                   1, 1, 1, 1, 0]).reshape(-1, 1),
    "K": np.array([1, 0, 1, 0, 0,  # Зображення літери K
                   1, 1, 1, 1, 0,
                   1, 0, 1, 0, 1]).reshape(-1, 1),
    "E": np.array([1, 1, 1, 1, 1,  # Зображення літери E
                   1, 1, 1, 0, 0,
                   1, 1, 1, 1, 1]).reshape(-1, 1)
}

# Ініціалізація ваг
input_size = letters["A"].shape[0]  # Розмір вхідного вектора
output_size = len(letters)          # Кількість літер
weights = np.zeros((output_size, input_size))

# Навчання
for idx, (label, pattern) in enumerate(letters.items()):
    weights[idx] += learning_rate * pattern.T.flatten()

# Функція розпізнавання
def recognize(input_pattern):
    output = np.dot(weights, input_pattern.flatten())
    recognized_idx = np.argmax(output)
    return list(letters.keys())[recognized_idx]

# Демонстрація роботи
test_patterns = [
    letters["A"],  # Правильне зображення
    letters["C"] * 0.9,  # Зображення з шумом
    letters["S"],  # Правильне зображення
    letters["M"] * 0.8,  # Зображення з шумом
    letters["O"],  # Правильне зображення
    letters["K"] * 0.7,  # Зображення з шумом
    letters["E"]  # Правильне зображення
]

for i, pattern in enumerate(test_patterns):
    result = recognize(pattern)
    print(f"Тест {i + 1}: Розпізнано як '{result}'")
