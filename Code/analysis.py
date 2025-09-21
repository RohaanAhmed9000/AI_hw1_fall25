import time
import matplotlib.pyplot as plt
from sand import SandProblem
from route import RoutePlanningProblem
import util

# --------------------------
# A* Search
# --------------------------
def aStarSearch(problem, stats=False):
    start_state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((start_state, [], 0), 0)
    explored = set()
    nodes_expanded = 0

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()

        if problem.isGoalState(state):
            if stats:
                return path, cost, nodes_expanded
            return path

        if state not in explored:
            explored.add(state)
            nodes_expanded += 1

            for successor, action, stepCost in problem.getSuccessors(state):
                new_cost = cost + stepCost
                heuristic = problem.getHeuristic(successor)
                priority = new_cost + heuristic
                frontier.push((successor, path + [action], new_cost), priority)

    if stats:
        return [], 0, nodes_expanded
    return []

# --------------------------
# Dijkstra Search
# --------------------------
def dijkstraSearch(problem, stats=False):
    start_state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((start_state, [], 0), 0)
    explored = set()
    nodes_expanded = 0

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()

        if problem.isGoalState(state):
            if stats:
                return path, cost, nodes_expanded
            return path

        if state not in explored:
            explored.add(state)
            nodes_expanded += 1

            for successor, action, stepCost in problem.getSuccessors(state):
                new_cost = cost + stepCost
                frontier.push((successor, path + [action], new_cost), new_cost)

    if stats:
        return [], 0, nodes_expanded
    return []

def compareSand():
    print("\n=== Sand Problem Comparison ===")
    problems = [
        (4, 3, 2),
        (5, 7, 4),
        (6, 9, 3),
        (7, 11, 5)
    ]

    results = []

    for X, Y, Z in problems:
        problem = SandProblem(X, Y, Z)

        # Run A*
        path, cost, expanded = aStarSearch(problem, stats=True)
        results.append(("A*", f"{X},{Y},{Z}", len(path), cost, expanded))

        # Run Dijkstra
        path, cost, expanded = dijkstraSearch(problem, stats=True)
        results.append(("Dijkstra", f"{X},{Y},{Z}", len(path), cost, expanded))

    # Print table
    print("{:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Alg", "Problem", "PathLen", "Cost", "NodesExpanded"))
    for r in results:
        print("{:<12} {:<12} {:<12} {:<12} {:<12}".format(*r))

    # Plot analysis
    problems_labels = [f"{X},{Y},{Z}" for (X, Y, Z) in problems]
    a_star_nodes = [r[4] for r in results if r[0] == "A*"]
    dijkstra_nodes = [r[4] for r in results if r[0] == "Dijkstra"]

    plt.figure(figsize=(8, 5))
    plt.plot(problems_labels, a_star_nodes, marker="o", label="A*")
    plt.plot(problems_labels, dijkstra_nodes, marker="s", label="Dijkstra")
    plt.xlabel("Problem (X,Y,Z)")
    plt.ylabel("Nodes Expanded")
    plt.title("Sand Problem: Nodes Expanded (A* vs Dijkstra)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compareRoute():
    print("\n=== Route Planning Problem Comparison ===")
    start_city = "Islamabad"
    goal_city = "Hunza"

    connection_file = "./CSV/Connections.csv"
    heuristic_file = "./CSV/Heuristics.csv"
    tracktype_file = "./CSV/TrackType.csv"

    problem = RoutePlanningProblem(start_city, goal_city, connection_file, heuristic_file, tracktype_file)

    results = []

    # Run A*
    path, cost, expanded = aStarSearch(problem, stats=True)
    results.append(("A*", len(path), cost, expanded))

    # Run Dijkstra
    path, cost, expanded = dijkstraSearch(problem, stats=True)
    results.append(("Dijkstra", len(path), cost, expanded))

    # Print table
    print("{:<12} {:<12} {:<12} {:<12}".format("Alg", "PathLen", "Cost", "NodesExpanded"))
    for r in results:
        print("{:<12} {:<12} {:<12} {:<12}".format(*r))

    # Bar chart comparison
    labels = [r[0] for r in results]
    nodes = [r[3] for r in results]
    pathlens = [r[1] for r in results]

    x = range(len(labels))
    plt.figure(figsize=(8, 5))

    plt.bar(x, nodes, width=0.4, label="Nodes Expanded")
    plt.bar([i + 0.4 for i in x], pathlens, width=0.4, label="Path Length")

    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Count")
    plt.title("Route Problem: A* vs Dijkstra")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compareSand()
    compareRoute()