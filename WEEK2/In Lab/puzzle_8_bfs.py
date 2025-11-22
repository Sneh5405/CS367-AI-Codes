from collections import deque
import random


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

def get_successors(node):
    successors = []
    index = node.state.index(0)  # position of blank
    quotient = index // 3        # row
    remainder = index % 3        # column

    moves = []
    # Row constrained moves
    if quotient == 0:
        moves += [3]      # move down
    if quotient == 1:
        moves += [-3, 3]  # move up and down
    if quotient == 2:
        moves += [-3]     # move up

    # Column constrained moves
    if remainder == 0:
        moves += [1]      # move right
    if remainder == 1:
        moves += [-1, 1]  # move left and right
    if remainder == 2:
        moves += [-1]     # move left

    for move in moves:
        im = index + move
        if 0 <= im < 9:
            new_state = list(node.state)
            # swap blank with neighbor
            new_state[index], new_state[im] = new_state[im], new_state[index]
            successors.append(Node(new_state, node))
    return successors

def bfs(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    queue = deque([start_node])
    visited = set()
    nodes_explored = 0

    while queue:
        node = queue.popleft()
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))

        nodes_explored += 1

        if node.state == goal_node.state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored:', nodes_explored)
            return path[::-1]

        for successor in get_successors(node):
            queue.append(successor)

    print('No solution found. Total nodes explored:', nodes_explored)
    return None

# -------------------------------
# Example run
# -------------------------------
start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # solved puzzle
s_node = Node(start_state)

# generate a random solvable goal state
D = 20
d = 0
while d <= D:
    goal_state = random.choice(get_successors(s_node)).state
    s_node = Node(goal_state)
    d += 1

print("Start State:", start_state)
print("Goal State:", goal_state)

solution = bfs(start_state, goal_state)
if solution:
    print("\nSolution Path:")
    for step in solution:
        # pretty print as 3x3 grid
        for i in range(0, 9, 3):
            print(step[i:i+3])
        print()
else:
    print("No solution found.")
