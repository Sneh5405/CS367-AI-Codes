import random
import numpy as np
import string

def make_ksat_problem(n_vars, k_size, n_clauses):
    """Create a uniform random k-SAT instance with distinct literals in each clause."""
    base_vars = list(string.ascii_lowercase[:n_vars])
    literals = base_vars + [v.upper() for v in base_vars]

    formula = []
    while len(formula) < n_clauses:
        clause = random.sample(literals, k_size)
        if clause not in formula:   # avoid duplicates
            formula.append(clause)
    return literals, formula


def random_assignment(literals, n_vars):
    """Assign random True/False values to all literals (positive and negated)."""
    values = np.random.choice([0, 1], size=n_vars)
    neg_values = 1 - values
    truth_table = np.concatenate((values, neg_values))
    return dict(zip(literals, truth_table))


def check_clause(clause, assignment):
    """Check whether a clause is satisfied under given assignment."""
    return any(assignment[l] for l in clause)


def evaluate_formula(clauses, assignment):
    """Return the number of satisfied clauses for this assignment."""
    return sum(check_clause(c, assignment) for c in clauses)



if __name__ == "__main__":
    n, k, m = 6, 3, 4   # variables, clause length, clauses

    literals, formula = make_ksat_problem(n, k, m)
    print("Literals:", literals)
    print("\nGenerated Formula:")
    for i, c in enumerate(formula, 1):
        print(f"Clause {i}: {c}")

    assignment = random_assignment(literals, n)
    print("\nRandom Truth Assignment:", assignment)

    satisfied = evaluate_formula(formula, assignment)
    print(f"\nSatisfied {satisfied} out of {len(formula)} clauses.")
