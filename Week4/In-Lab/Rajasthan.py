import random
import math
import matplotlib.pyplot as plt

# Rajasthan tourist destinations with coordinates
rajasthan_sites = {
    "Amber Fort": (26.9855, 75.8513),
    "City Palace": (24.5764, 73.6844),
    "Mehrangarh Fort": (26.2983, 73.0193),
    "Brahma Temple": (26.4872, 74.5503),
    "Golden Fort": (26.9115, 70.9177),
    "Dargah Sharif": (26.4519, 74.6275),
    "Dilwara Temples": (24.6001, 72.7145),
    "Junagarh Fort": (28.0121, 73.3187),
    "Ranthambore National Park": (26.0173, 76.5026),
    "Chittorgarh Fort": (24.8879, 74.6454),
    "Taragarh Fort": (25.4359, 75.6473),
    "Bhangarh Fort": (27.0964, 76.2850),
    "Keoladeo National Park": (27.1672, 77.5222),
    "Seven Wonders Park": (25.1602, 75.8510),
    "Ranthambore Fort": (26.0208, 76.4569),
    "Nawalgarh": (27.8513, 75.2739),
    "Juna Mahal": (23.8359, 73.7148),
    "Shrinathji Temple": (24.9268, 73.8315),
    "Mandawa Castle": (28.0559, 75.1545),
    "Sachiya Mata Temple": (26.9128, 72.3941)
}

def calculate_distance(site1, site2):
    """Calculate distance between two sites using Haversine formula."""
    lat1, lon1 = site1
    lat2, lon2 = site2
    earth_radius = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2) * math.sin(dlon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return earth_radius * c

def route_distance(route):
    """Calculate total distance of a route."""
    return sum(calculate_distance(rajasthan_sites[route[i]], rajasthan_sites[route[(i+1) % len(route)]]) 
               for i in range(len(route)))

def optimize_tour(sites, initial_temp=2000, cooling_rate=0.995, iterations=150000):
    """Optimize tour using Simulated Annealing."""
    current_route = list(sites.keys())
    random.shuffle(current_route)
    current_distance = route_distance(current_route)
    best_route = current_route.copy()
    best_distance = current_distance
    temp = initial_temp
    distance_history = []

    for _ in range(iterations):
        new_route = current_route.copy()
        a, b = random.sample(range(len(new_route)), 2)
        new_route[a], new_route[b] = new_route[b], new_route[a]
        new_distance = route_distance(new_route)

        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temp):
            current_route = new_route
            current_distance = new_distance
            if new_distance < best_distance:
                best_route = new_route.copy()
                best_distance = new_distance

        distance_history.append(current_distance)
        temp *= cooling_rate

    return best_route, best_distance, distance_history

# Run the optimization
optimal_tour, optimal_distance, distance_log = optimize_tour(rajasthan_sites)

# Calculate distances between consecutive sites
site_to_site_distances = [calculate_distance(rajasthan_sites[optimal_tour[i]], 
                                             rajasthan_sites[optimal_tour[(i+1) % len(optimal_tour)]])
                          for i in range(len(optimal_tour))]

# Output results
print("Optimized Rajasthan Tour:")
for i, site in enumerate(optimal_tour):
    print(f"{i+1}. {site}")
print(f"\nTotal tour distance: {optimal_distance:.2f} km")
print("\nDistances between consecutive sites:")
for i, distance in enumerate(site_to_site_distances):
    print(f"{optimal_tour[i]} to {optimal_tour[(i+1) % len(optimal_tour)]}: {distance:.2f} km")

# Visualize the tour
plt.figure(figsize=(12, 10))
for site, (lat, lon) in rajasthan_sites.items():
    plt.plot(lon, lat, 'bo', markersize=7)
    plt.text(lon, lat, site, fontsize=8, ha='right', va='bottom')

tour_lons = [rajasthan_sites[site][1] for site in optimal_tour] + [rajasthan_sites[optimal_tour[0]][1]]
tour_lats = [rajasthan_sites[site][0] for site in optimal_tour] + [rajasthan_sites[optimal_tour[0]][0]]
plt.plot(tour_lons, tour_lats, 'r-')

plt.title('Optimized Rajasthan Tourist Circuit')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Plot distance improvement over iterations
plt.figure(figsize=(10, 6))
plt.plot(distance_log)
plt.title('Tour Distance Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Tour Distance (km)')
plt.grid(True)
plt.show()