from search import SearchProblem as sp
from typing import List, Tuple, Any
import util

def aStarSearch(problem):
    """
    Search the node that has the lowest combined cost and heuristic.
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

class SandProblem(sp):
    def __init__(self, X: int, Y: int, Z: int):
        self.x = X
        self.y = Y
        self.z = Z

        self.state: Tuple[int, int] = (0, 0)

    def getStartState(self):
        return (0, 0)

    def isGoalState(self, state):
        if state[0] == self.z or state[1] == self.z:
            return True
        return False
    
    
    def getSuccessors(self, state):
        pos: List[Tuple[Tuple[int, int], str, int]] = []
        cur_x, cur_y = state
        
        # Empty cur_x
        if cur_x > 0:
            pos.append(((0, cur_y), "empty x", 1))

        # Empty cur_y
        if cur_y > 0:
            pos.append(((cur_x, 0), "empty y", 1))

        # Fill X
        if cur_x < self.x:
            pos.append(((self.x, cur_y), "fill x", 1))

        # Fill Y
        if cur_y < self.y:
            pos.append(((cur_x, self.y), "fill y", 1))

        # Pour Y -> X
        if cur_y > 0 and cur_x < self.x:
            pour = min(cur_y, self.x - cur_x)
            pos.append(((cur_x + pour, cur_y - pour), "y to x", 1))

        # Pour X -> Y
        if cur_x > 0 and cur_y < self.y:
            pour = min(cur_x, self.y - cur_y)
            pos.append(((cur_x - pour, cur_y + pour), "x to y", 1))

        return pos

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """

        return len(actions)

        
    def getHeuristic(self,state):
        """
         state: the current state of agent

         THis function returns the heuristic of current state of the agent which will be the 
         estimated distance from goal.
        """
        return min(abs(state[0] - self.z), abs(state[1] - self.z)) / max(self.x, self.y)
    

if __name__ == "__main__":
    # Example values for X, Y, Z
    X = 4  # Capacity of bucket X
    Y = 3  # Capacity of bucket Y
    Z = 2  # Goal amount

    problem = SandProblem(X, Y, Z)

    print("=== Sand Problem with A* Search ===")
    print(f"Bucket capacities: X = {X}, Y = {Y}, Goal = {Z}\n")

    # Run A* Search
    solution = aStarSearch(problem)

    if solution:
        print("Solution found!")
        print(" -> ".join(solution))
        print(f"Total actions: {len(solution)}")
        print(f"Cost of solution: {problem.getCostOfActions(solution)}")
    else:
        print("No solution found.")
