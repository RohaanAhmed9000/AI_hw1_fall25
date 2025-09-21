from search import SearchProblem as sp
import csv
import util


def aStarSearch(problem):
    start_state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((start_state, [], 0), 0)  # (state, path, cost), priority
    explored = set()

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()

        if problem.isGoalState(state):
            return path

        if state not in explored:
            explored.add(state)

            for successor, action, stepCost in problem.getSuccessors(state):
                new_cost = cost + stepCost
                heuristic = problem.getHeuristic(successor)
                priority = new_cost + heuristic
                frontier.push((successor, path + [action], new_cost), priority)

    return []  # if no path found


def dijkstraSearch(problem):
    """
    Dijkstra's Algorithm (Uniform Cost Search).
    Expands the node with the lowest path cost g(n).
    """
    start_state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((start_state, [], 0), 0)  # (state, path, cost), priority
    explored = set()

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()

        if problem.isGoalState(state):
            return path

        if state not in explored:
            explored.add(state)

            for successor, action, stepCost in problem.getSuccessors(state):
                new_cost = cost + stepCost
                frontier.push((successor, path + [action], new_cost), new_cost)

    return []  # No solution found



class RoutePlanningProblem(sp):
    def __init__(self, start_city, goal_city, connection_file, heuristic_file, tracktype_file):
        self.start = start_city
        self.goal = goal_city

        # Load CSV files into dictionaries
        self.connections = self._load_csv(connection_file)
        self.heuristics = self._load_csv(heuristic_file)
        self.tracktypes = self._load_csv(tracktype_file)

        # Extract city list for reference
        self.cities = list(self.connections.keys())

    def _load_csv(self, filename):
        """
        Reads a CSV file into a dict-of-dicts {city1: {city2: value}}
        """
        data = {}
        with open(filename, "r") as f:
            reader = csv.reader(f)
            header = next(reader)[1:]  # city names in header
            for row in reader:
                city = row[0]
                values = row[1:]
                data[city] = {header[i]: self._parse_value(values[i]) for i in range(len(header))}
        return data

    def _parse_value(self, val):
        try:
            return float(val)
        except:
            return val  # e.g., "M", "J", "S"

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state == self.goal

    def getSuccessors(self, state):
        successors = []
        for city, dist in self.connections[state].items():
            if dist == -1:  # no connection
                continue

            track_type = self.tracktypes[state][city]
            step_cost = dist

            # Penalize jeepable tracks (J) slightly
            if track_type == "J":
                step_cost += 5
            elif track_type == "S":
                step_cost += 1
            # Motorway (M) is preferred, so no extra cost

            successors.append((city, f"Go {city} via {track_type}", step_cost))

        return successors

    def getCostOfActions(self, actions):
        # Each action encodes its step cost
        return sum([step[2] for step in actions])

    def getHeuristic(self, state):
        return self.heuristics[state][self.goal]


if __name__ == "__main__":
    # Example: test RoutePlanningProblem with given CSVs
    start_city = "Islamabad"
    goal_city = "Hunza"

    connection_file = "./CSV/Connections.csv"
    heuristic_file = "./CSV/Heuristics.csv"
    tracktype_file = "./CSV/TrackType.csv"

    # Create problem instance
    problem = RoutePlanningProblem(start_city, goal_city,
                                   connection_file, heuristic_file, tracktype_file)

    print("=== Route Planning with A* Search ===")
    print(f"Start city: {start_city}")
    print(f"Goal city: {goal_city}\n")

    # Run A* Search
    solution = aStarSearch(problem)

    if solution:
        print("Optimal route found:")
        for step in solution:
            print("  ", step)
        print(f"\nTotal steps: {len(solution)}")
    else:
        print("No route found between the given cities.")

