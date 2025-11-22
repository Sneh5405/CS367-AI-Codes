import numpy as np
import heapq
import itertools
import time


class Node:
    def __init__(self, parent, state, pcost, hcost, action=None):
        self.parent = parent
        self.state = np.array(state, dtype=np.int8)
        self.action = action
        self.pcost = pcost
        self.hcost = hcost
        self.cost = pcost + hcost

    def __hash__(self):
        return hash(self.state.tobytes())

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __str__(self):
        return str(self.state)


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entryFinder = {}
        self.REMOVED = '<removed>'
        self.counter = itertools.count()

    def push(self, node):
        key = node.state.tobytes()
        existing = self.entryFinder.get(key)
        if existing:
            existingCost = existing[0]
            if node.cost >= existingCost:
                return
            existing[2] = self.REMOVED

        count = next(self.counter)
        entry = [node.cost, count, node]
        self.entryFinder[key] = entry
        heapq.heappush(self.heap, entry)

    def pop(self):
        while self.heap:
            cost, count, node = heapq.heappop(self.heap)
            if node is not self.REMOVED:
                key = node.state.tobytes()
                self.entryFinder.pop(key, None)
                return node
        raise KeyError("pop from empty PriorityQueue")

    def isEmpty(self):
        return len(self.entryFinder) == 0

    def __len__(self):
        return len(self.entryFinder)


class Environment:
    def __init__(self, startState=None, goalState=None):
        self.actions = [1, 2, 3, 4]
        self.goalState = goalState if goalState is not None else self.generateGoalState()
        self.startState = startState if startState is not None else self.generateStartState()

    def generateStartState(self):
        start = np.zeros((7, 7), dtype=np.int8)
        x = (0, 1, 5, 6)
        y = (0, 1, 5, 6)
        for i in x:
            for j in y:
                start[i][j] = -1
        x = (2, 3, 4)
        y = range(7)
        for i in x:
            for j in y:
                start[i][j] = 1
                start[j][i] = 1
        start[3][3] = 0
        return start

    def generateGoalState(self):
        goal = np.zeros((7, 7), dtype=np.int8)
        x = (0, 1, 5, 6)
        y = (0, 1, 5, 6)
        for i in x:
            for j in y:
                goal[i][j] = -1
        x = (2, 3, 4)
        y = range(7)
        for i in x:
            for j in y:
                goal[i][j] = 0
                goal[j][i] = 0
        goal[3][3] = 1
        return goal

    def getStartState(self):
        return self.startState

    def getGoalState(self):
        return self.goalState

    def getNextStates(self, state):
        newStates = []
        spaces = [(i, j) for i in range(7) for j in range(7) if state[i][j] == 0]
        for x, y in spaces:
            # up
            if x > 1 and state[x - 1][y] == 1 and state[x - 2][y] == 1:
                newState = state.copy()
                newState[x][y] = 1
                newState[x - 2][y] = 0
                newState[x - 1][y] = 0
                newStates.append((newState, f'({x - 2}, {y}) -> ({x}, {y})'))

            # down
            if x < 5 and state[x + 1][y] == 1 and state[x + 2][y] == 1:
                newState = state.copy()
                newState[x][y] = 1
                newState[x + 2][y] = 0
                newState[x + 1][y] = 0
                newStates.append((newState, f'({x + 2}, {y}) -> ({x}, {y})'))

            # left
            if y > 1 and state[x][y - 1] == 1 and state[x][y - 2] == 1:
                newState = state.copy()
                newState[x][y] = 1
                newState[x][y - 2] = 0
                newState[x][y - 1] = 0
                newStates.append((newState, f'({x}, {y - 2}) -> ({x}, {y})'))

            # right
            if y < 5 and state[x][y + 1] == 1 and state[x][y + 2] == 1:
                newState = state.copy()
                newState[x][y] = 1
                newState[x][y + 2] = 0
                newState[x][y + 1] = 0
                newStates.append((newState, f'({x}, {y + 2}) -> ({x}, {y})'))
        return newStates

    def reachedGoal(self, state):
        return np.array_equal(state, self.goalState)


def heuristic0(currState):
    return 0

def heuristic1(currState):
    cost = 0
    for i in range(7):
        for j in range(7):
            if currState[i][j] == 1:
                cost += abs(i - 3) + abs(j - 3)
    return cost

def heuristic2(currState):
    cost = 0
    for i in range(7):
        for j in range(7):
            if currState[i][j] == 1:
                cost += 2 ** (max(abs(i - 3), abs(j - 3)))
    return cost


class Agent:
    def __init__(self, env, heuristic, useAstar=True):
        self.frontier = PriorityQueue()
        self.explored = {}
        self.startState = env.getStartState()
        self.goalState = env.getGoalState()
        self.env = env
        self.goalNode = None
        self.heuristic = heuristic
        self.useAstar = useAstar

    def run(self):
        initNode = Node(None, self.startState, pcost=0, hcost=self.heuristic(self.startState))
        self.frontier.push(initNode)
        gScores = {initNode.state.tobytes(): 0}
        start = time.time()

        while not self.frontier.isEmpty():
            currNode = self.frontier.pop()
            key = currNode.state.tobytes()

            if self.env.reachedGoal(currNode.state):
                self.goalNode = currNode
                end = time.time()
                print("Reached goal!")
                return end - start

            self.explored[key] = currNode

            for newState, action in self.env.getNextStates(currNode.state):
                g = currNode.pcost + 1 if self.useAstar else 0
                h = self.heuristic(newState)
                child = Node(currNode, newState, g, h, action)
                childKey = newState.tobytes()

                if childKey not in gScores or g < gScores[childKey]:
                    gScores[childKey] = g
                    self.frontier.push(child)

        return None  # no solution found

    def printSolution(self):
        node = self.goalNode
        path = []
        while node:
            path.append(node)
            node = node.parent
        for step, n in enumerate(reversed(path), 1):
            print(f"Step {step}: {n.action}")
        print("Solution length:", len(path))


# ---------------- Example usage ----------------
if __name__ == "__main__":
    env = Environment()

    print("=== A* with heuristic1 (Manhattan) ===")
    agent = Agent(env, heuristic1, useAstar=True)
    t = agent.run()
    print("Time:", t, "Explored:", len(agent.explored), "Frontier:", len(agent.frontier))

    print("\n=== Best-First with heuristic2 (Exponential) ===")
    agent2 = Agent(env, heuristic2, useAstar=False)
    t2 = agent2.run()
    print("Time:", t2, "Explored:", len(agent2.explored), "Frontier:", len(agent2.frontier))
