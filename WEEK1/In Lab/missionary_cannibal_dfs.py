def is_valid(state):
    M_L, C_L, M_R, C_R, B = state
    # Check for valid numbers
    if M_L < 0 or C_L < 0 or M_R < 0 or C_R < 0:
        return False
    if M_L > 0 and M_L < C_L:
        return False
    if M_R > 0 and M_R < C_R:
        return False
    if M_L > 3 or C_L > 3 or M_R > 3 or C_R > 3:
        return False
    return True

def get_successors(state):
    M_L, C_L, M_R, C_R, B = state
    successors = []
    moves = [(2,0), (0,2), (1,1), (1,0), (0,1)]
    if B == 1:  # Boat on left bank
        for m, c in moves:
            new_state = (M_L - m, C_L - c, M_R + m, C_R + c, 0)
            if is_valid(new_state):
                successors.append(new_state)
    else:  # Boat on right bank
        for m, c in moves:
            new_state = (M_L + m, C_L + c, M_R - m, C_R - c, 1)
            if is_valid(new_state):
                successors.append(new_state)
    return successors

def dfs(state, goal, visited, path):
    if state == goal:
        return path + [state]
    
    visited.add(state)
    path = path + [state]

    for successor in get_successors(state):
        if successor not in visited:
            result = dfs(successor, goal, visited, path)
            if result:
                return result
    return None

# Initial and goal states
start_state = (3, 3, 0, 0, 1)  # 3 missionaries, 3 cannibals on left, boat on left
goal_state = (0, 0, 3, 3, 0)   # All moved to right, boat on right

solution = dfs(start_state, goal_state, set(), [])

if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
