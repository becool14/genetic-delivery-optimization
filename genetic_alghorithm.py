import numpy as np
import random
import matplotlib.pyplot as plt

# Ustawienia problemu
NUM_CLIENTS = 10  # liczba klientów
POPULATION_SIZE = 100  # wielkość populacji
MUTATION_RATE = 0.01  # współczynnik mutacji
CROSSOVER_RATE = 0.9  # współczynnik krzyżowania
NUM_GENERATIONS = 20  # liczba pokoleń

# Współrzędne klientów (szerokość, długość geograficzna)
client_locations = np.random.rand(NUM_CLIENTS, 2) * 100  # Losowe współrzędne klientów


# Macierz kosztów (odległości między klientami)
def calculate_distance_matrix(locations):
    num_clients = locations.shape[0]
    distance_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(locations[i] - locations[j])
    return distance_matrix


distance_matrix = calculate_distance_matrix(client_locations)


# Kodowanie genotypu (chromosomu): permutacja klientów
def create_chromosome():
    return random.sample(range(NUM_CLIENTS), NUM_CLIENTS)


# Inicjalizacja populacji
def create_initial_population():
    return [create_chromosome() for _ in range(POPULATION_SIZE)]


# Funkcja celu: suma kosztów (odległości) na trasie
def evaluate_fitness(chromosome, distance_matrix):
    total_distance = 0
    for i in range(len(chromosome) - 1):
        total_distance += distance_matrix[chromosome[i]][chromosome[i + 1]]
    total_distance += distance_matrix[chromosome[-1]][chromosome[0]]  # powrót do punktu startu
    return total_distance


# Selekcja: turniejowa
def tournament_selection(population, fitnesses):
    tournament_size = 5
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    return min(selected, key=lambda x: x[1])[0]


# Krzyżowanie jednopunktowe (OX1 - ordered crossover)
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end] = parent1[start:end]
        pointer = end
        for gene in parent2:
            if gene not in child:
                if pointer >= size:
                    pointer = 0
                child[pointer] = gene
                pointer += 1
        return child
    else:
        return parent1


# Mutacja: zamiana miejscami dwóch losowych klientów
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


# Proces ewolucji
def evolve_population(population, distance_matrix):
    fitnesses = [evaluate_fitness(chromosome, distance_matrix) for chromosome in population]
    new_population = []

    for _ in range(len(population)):
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)

    return new_population, fitnesses


# Warunki zatrzymania: maksymalna liczba pokoleń
def genetic_algorithm():
    population = create_initial_population()
    best_fitness_history = []

    for generation in range(NUM_GENERATIONS):
        population, fitnesses = evolve_population(population, distance_matrix)

        best_fitness = min(fitnesses)
        best_fitness_history.append(best_fitness)

        if generation % 10 == 0:
            print(f"Pokolenie {generation}, Najlepszy koszt trasy: {best_fitness}")

    # Zwrócenie najlepszej trasy i jej kosztu
    best_index = np.argmin(fitnesses)
    best_route = population[best_index]
    return best_route, best_fitness_history


# Wizualizacja trasy
def plot_route(route, locations):
    route_locations = locations[route + [route[0]]]
    plt.plot(route_locations[:, 0], route_locations[:, 1], 'b-', marker='o')
    plt.title("Najlepsza trasa")
    plt.show()


# Uruchomienie algorytmu genetycznego
best_route, best_fitness_history = genetic_algorithm()

# Wizualizacja najlepszego rozwiązania
print("Najlepsza trasa:", best_route)
plot_route(best_route, client_locations)

# Wizualizacja zbieżności
plt.plot(best_fitness_history)
plt.title("Zbieżność funkcji celu")
plt.xlabel("Pokolenia")
plt.ylabel("Koszt trasy")
plt.show()
