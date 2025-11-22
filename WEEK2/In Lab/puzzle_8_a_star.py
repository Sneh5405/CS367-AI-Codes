import heapq
import random


class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g  # cost so far
        self.h = h  # heuristic estimate to goal
        self.f = g + h  # evaluation function f = g + h

    def __lt__(self, other):
        return self.f < other.f  # priority on f for A*

def heuristic(state, goal_state):
    """Manhattan distance heuristic"""
    h = 0
    for i, tile in enumerate(state):
        if tile != 0:  # don't count the blank
            goal_index = goal_state.index(tile)
            h += abs(i // 3 - goal_index // 3) + abs(i % 3 - goal_index % 3)
    return h

def get_successors(node, goal_state):
    successors = []
    index = node.state.index(0)  # blank index
    quotient = index // 3
    remainder = index % 3

    moves = []
    # Row constrained moves
    if quotient == 0:
        moves += [3]
    if quotient == 1:
        moves += [-3, 3]
    if quotient == 2:
        moves += [-3]

    # Column constrained moves
    if remainder == 0:
        moves += [1]
    if remainder == 1:
        moves += [-1, 1]
    if remainder == 2:
        moves += [-1]

    for move in moves:
        im = index + move
        if 0 <= im < 9:
            new_state = list(node.state)
            new_state[index], new_state[im] = new_state[im], new_state[index]
            h = heuristic(new_state, goal_state)
            successor = Node(new_state, node, node.g + 1, h)
            successors.append(successor)
    return successors

def search_agent(start_state, goal_state):
    start_node = Node(start_state, g=0, h=heuristic(start_state, goal_state))
    frontier = []
    heapq.heappush(frontier, (start_node.f, start_node))
    visited = set()
    nodes_explored = 0

    while frontier:
        _, node = heapq.heappop(frontier)

        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))

        nodes_explored += 1

        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("Total nodes explored:", nodes_explored)
            return path[::-1]

        for successor in get_successors(node, goal_state):
            heapq.heappush(frontier, (successor.f, successor))

    print("No solution found. Total nodes explored:", nodes_explored)
    return None

# -------------------------------
# Example run
# -------------------------------
start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # solved puzzle
s_node = Node(start_state)

# Generate a random solvable goal state
D = 20
d = 0
while d <= D:
    goal_state = random.choice(get_successors(s_node, start_state)).state
    s_node = Node(goal_state)
    d += 1

print("Start State:", start_state)
print("Goal State:", goal_state)

solution = search_agent(start_state, goal_state)
if solution:
    print("\nSolution Path:")
    for step in solution:
        for i in range(0, 9, 3):
            print(step[i:i+3])
        print()
else:
    print("No solution found.")
