import random
import math
import matplotlib.pyplot as plt

def route_length(path, city_distances):
    return sum(city_distances[path[i-1]][path[i]] for i in range(len(path)))

def create_random_path(city_count):
    path = list(range(city_count))
    random.shuffle(path)
    return path

def generate_neighbor(path):
    a, b = random.sample(range(len(path)), 2)
    path[a], path[b] = path[b], path[a]
    return path

def tsp_simulated_annealing(city_distances, iterations, start_temp, cooling_factor):
    cities = len(city_distances)
    current_path = create_random_path(cities)
    current_length = route_length(current_path, city_distances)
    best_path = current_path.copy()
    best_length = current_length
    temp = start_temp

    for _ in range(iterations):
        new_path = generate_neighbor(current_path.copy())
        new_length = route_length(new_path, city_distances)
        
        if new_length < current_length or random.random() < math.exp((current_length - new_length) / temp):
            current_path = new_path
            current_length = new_length
            
            if current_length < best_length:
                best_path = current_path.copy()
                best_length = current_length
        
        temp *= cooling_factor

    return best_path, best_length

# Example usage
city_distances = [
    [0, 12, 18, 22],
    [12, 0, 38, 28],
    [18, 38, 0, 32],
    [22, 28, 32, 0]
]

max_iterations = 15000
initial_temp = 150.0
cooling_rate = 0.95

optimal_path, optimal_length = tsp_simulated_annealing(
    city_distances, max_iterations, initial_temp, cooling_rate)

# Visualization
num_cities = len(city_distances)
city_coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

plt.figure(figsize=(10, 8))
x, y = zip(*city_coords)
plt.scatter(x, y, c='green', s=120)

for i, (x, y) in enumerate(city_coords):
    plt.annotate(f'City {i}', (x, y), xytext=(5, 5), textcoords='offset points')

for i in range(num_cities):
    start = city_coords[optimal_path[i]]
    end = city_coords[optimal_path[(i + 1) % num_cities]]
    plt.plot([start[0], end[0]], [start[1], end[1]], c='blue')

plt.title("Optimal TSP Route via Simulated Annealing")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()

print(f"Optimal Route: {' -> '.join(map(str, optimal_path))}")
print(f"Optimal Route Length: {optimal_length}")