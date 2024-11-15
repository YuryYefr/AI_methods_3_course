import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Генеруємо дані (наприклад, вік і середня сума покупки)
np.random.seed(42)
n_samples = 100
ages = np.random.normal(35, 10, n_samples)  # Вік
purchase = np.random.normal(500, 100, n_samples)  # Сума покупки

# Формуємо матрицю ознак
data = np.vstack((ages, purchase))

# Задаємо кількість кластерів
n_clusters = 3

# Нечітка кластеризація методом FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, n_clusters, m=2, error=0.005, maxiter=1000, init=None
)
# Центри кластерів
print("Центри кластерів:")
for i, center in enumerate(cntr):
    print(f"Кластер {i+1}: {center}")

# Графік зміни цільової функції
plt.figure()
plt.plot(jm, marker='o')
plt.title("Зміна цільової функції")
plt.xlabel("Ітерація")
plt.ylabel("Значення цільової функції")
plt.grid(True)
plt.show()

# Призначення точок до кластерів за максимальним ступенем належності
cluster_membership = np.argmax(u, axis=0)

# Візуалізація
plt.figure()
colors = ['r', 'g', 'b']

for i in range(n_clusters):
    cluster_points = data[:, cluster_membership == i]
    plt.scatter(cluster_points[0], cluster_points[1], c=colors[i], label=f'Кластер {i+1}')

# Центри кластерів
plt.scatter(cntr[:, 0], cntr[:, 1], c='black', marker='x', s=100, label='Центри кластерів')

plt.title("Результати кластеризації FCM")
plt.xlabel("Вік")
plt.ylabel("Середня сума покупки")
plt.legend()
plt.grid(True)
plt.show()
