import random
from typing import List, Dict, Tuple

class KSatSolver:
    def __init__(self, n_vars: int, k: int, m: int):
        self.n_vars = n_vars
        self.k = k
        self.m = m
        self.formula = self._make_formula()
        self.assignment = self._random_assignment()

    def _make_formula(self) -> List[List[Tuple[int, bool]]]:
        """Generate a random k-SAT formula with m clauses."""
        clauses = []
        for _ in range(self.m):
            clause = []
            for _ in range(self.k):
                var = random.randint(1, self.n_vars)
                sign = bool(random.getrandbits(1))  
                clause.append((var, sign))
            clauses.append(clause)
        return clauses

    def _random_assignment(self) -> Dict[int, bool]:
        """Initial random truth assignment."""
        return {var: bool(random.getrandbits(1)) for var in range(1, self.n_vars + 1)}

    def _score(self, assignment: Dict[int, bool]) -> int:
        """Count satisfied clauses."""
        satisfied = 0
        for clause in self.formula:
            if any(assignment[v] == sign for v, sign in clause):
                satisfied += 1
        return satisfied

    def _perturb(self, assignment: Dict[int, bool], flips: int) -> Dict[int, bool]:
        """Flip a set of variables to create neighbor."""
        neighbor = assignment.copy()
        for var in random.sample(list(neighbor.keys()), flips):
            neighbor[var] = not neighbor[var]
        return neighbor

    def search(self, iterations: int = 500) -> Dict[int, bool]:
        """Neighborhood-based search (like VND but rewritten)."""
        best = self.assignment
        best_val = self._score(best)

        for _ in range(iterations):
            for radius in range(1, max(2, self.n_vars // 2)):
                candidate = self._perturb(best, radius)
                val = self._score(candidate)

                if val > best_val:
                    best, best_val = candidate, val
                    break  # restart neighborhoods if improvement found
            if best_val == self.m:
                break  # all clauses satisfied
        return best

    def solve(self) -> Tuple[Dict[int, bool], int]:
        solution = self.search()
        return solution, self._score(solution)


# Example run
if __name__ == "__main__":
    solver = KSatSolver(n_vars=5, k=3, m=10)
    sol, val = solver.solve()
    print("Best assignment:", sol)
    print(f"Satisfied {val} of {solver.m} clauses")
