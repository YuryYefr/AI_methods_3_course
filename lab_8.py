import random


# Функція для мінімізації (наприклад, квадратична функція)
def fitness_function(x):
    return x ** 2 - 4 * x + 4  # y = (x-2)^2, мінімум у x=2


# Ініціалізація популяції
def initialize_population(pop_size, lower_bound, upper_bound):
    return [random.uniform(lower_bound, upper_bound) for _ in range(pop_size)]


# Вибір двох батьків для схрещування
def select_parents(population):
    sorted_population = sorted(population, key=fitness_function)
    return sorted_population[0], sorted_population[1]  # Мінімізуємо функцію, тому вибираємо найкращих


# Схрещування (перехрестення батьків)
def crossover(parent1, parent2):
    return (parent1 + parent2) / 2  # Простий одновимірний схрест (середнє арифметичне)


# Мутація (додаємо невеликий випадковий шум)
def mutate(child, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        return child + random.uniform(-1, 1)
    return child


# Основна функція генетичного алгоритму
def genetic_algorithm(pop_size, generations, lower_bound, upper_bound, mutation_rate):
    population = initialize_population(pop_size, lower_bound, upper_bound)
    best_solution = None

    for generation in range(generations):
        # Вибір батьків
        parent1, parent2 = select_parents(population)

        # Створення нового покоління
        new_population = []
        for _ in range(pop_size):
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, lower_bound, upper_bound)
            new_population.append(child)

        population = new_population

        # Оцінка рішення
        best_in_generation = min(population, key=fitness_function)
        if best_solution is None or fitness_function(best_in_generation) < fitness_function(best_solution):
            best_solution = best_in_generation

        print(
            f"Generation {generation + 1}: Best Solution = {best_solution:.2f}, Fitness = {fitness_function(best_solution):.7f}")

    return best_solution


# Параметри генетичного алгоритму
pop_size = 10
generations = 50
lower_bound = -10
upper_bound = 10
mutation_rate = 0.1

best_solution = genetic_algorithm(pop_size, generations, lower_bound, upper_bound, mutation_rate)
print(f"Оптимальне рішення: x = {best_solution:.2f}, Fitness = {fitness_function(best_solution):.2f}")
