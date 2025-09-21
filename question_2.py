import math
import random
import matplotlib.pyplot as plt

def simulated_annealing(function, initial_temp, n_iter, n_temp, factor, step_size):
    """ 
    Performs Simulated Annealing to minimize a given objective function

    Parameters:
    function     : Objective function to minimize (takes (x,y) tuple as input)
    initial_temp : Starting temperature 
    n_iter       : Number of iterations per temperature level
    n_temp       : Number of temperature updates 
    factor       : Factor with which temperature will be decreased
    step_size    : Maximum step size for generating a random neighbor

    Returns:
    best     : tuple (x, y) -> Best coordinate found
    f(best)  : Objective function value at best point
    record_f : History of function values per iteration
    record_x : History of x values per iteration
    record_y : History of y values per iteration
    """
    # Initializing current by randomly picking a point from the domain
    min_x, max_x, min_y, max_y = get_domain(function)
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    current = (x, y) 
    # print("Current:",(x,y))
    best = current

    # Lists to store record for plotting
    record_f = []
    record_x = []
    record_y = []

    temperature = initial_temp
    for i in range(n_temp):
        for j in range(n_iter):
            trial = rand_neighbor(current, step_size ,function)
            delta = function(trial) - function(current)
            
            if delta < 0:        # Accept if trial is smaller/better : since we are minimising
                current = trial
            else:
                m = math.exp( - delta / temperature )
                p  = rand_prob()
                if p < m:
                    current = trial

            if function(current) < function(best): # For keeping record of the lowest value recorded so far
                best = current

            # Save record
            record_f.append(function(current))
            record_x.append(current[0])
            record_y.append(current[1])
        
        temperature = factor * temperature # Decrease the temperature
        print(f"Current temperature = {temperature}")
    return best, function(best), record_f, record_x, record_y

def get_domain(function):
    """
    Returns domain bounds (min_x, max_x, min_y, max_y) depending on the input function
    """
    if function == sphere:
        min_x, max_x, min_y, max_y = -5, 5, -5, 5
        return min_x, max_x, min_y, max_y
    elif function == rosenbrock:
        min_x, max_x, min_y, max_y = -2, 2, -1, 3
        return min_x, max_x, min_y, max_y
    elif function == griewank:
        min_x, max_x, min_y, max_y = -30, 30, -30, 30
        return  min_x, max_x, min_y, max_y 
    else:
        raise ValueError("Unknown function passed. Cannot determine its domain")
    
def rand_prob():
    """
    Generates a random probability between 0 and 1
    """
    probability = random.uniform(0, 1) 
    print("Probability:", probability)
    return probability

def rand_neighbor(coordinate, step_size, function):
    """
    Generates and returns neighbor coordinates by moving randomly 
    within step_size around current point

    Input parameters:
    coordinate : Current (x, y) -> tuple
    step_size : Maximum step size in each direction
    function : The objective function, used to get the domain

    Returns:
    (x_new, y_new) : New coordinate clamped to domain -> tuple
    """
    x, y = coordinate # Current coordinates/point
    min_x, max_x, min_y, max_y = get_domain(function)

    x_new = x + random.uniform(-step_size, step_size)
    y_new = y + random.uniform(-step_size, step_size)

    # Clamp values to domain in case points go beyond the given domain
    x_new = max(min(x_new, max_x), min_x)
    y_new = max(min(y_new, max_y), min_y)
    # print(f"Clamped values,{x_new},{y_new}")
    # print(f"Neighbors of current: {x}, {y}, are: {x_new}, {y_new}\n")
    return (x_new, y_new)

def sphere(coordinate):
    """
    Sphere objective function (min at (0,0), f = 0)
    f(x,y) = x^2 + y^2
    """
    x, y = coordinate
    f = x**2 + y**2 # Sphere equation
    # print(f"Value of sphere func at x = {x} and y = {y} is {f}")
    return f

def rosenbrock(coordinate):
    """
    Rosenbrock objective function (min at (1,1), f = 0).
    f(x,y) = (1 - x)^2 + 100*(y - x^2)^2
    """
    x, y = coordinate
    f = (1 - x)**2 + (100 * (y - x**2))**2 # Rosenbrock equation
    # print(f"Value of rosenbrock func at x = {x} and y = {y} is {f}")
    return f

def griewank(coordinate):
    """
    Griewank objective function (min at (0,0), f = 0) 
    f(x,y) = 1 + (x^2 + y^2)/4000 - cos(x)*cos(y/sqrt(2))
    """
    x, y = coordinate
    f = 1 + ((x**2 + y**2)/4000) - (math.cos(x) * math.cos(y/math.sqrt(2))) # Griewank equation
    return f


def plotGraph(record_f, record_x, record_y):
    """
    Plots objective function values and variable values over iterations
    """
    iterations = range(len(record_f))

    plt.figure(figsize=(10,6))

    # Objective function values
    plt.subplot(2,1,1)
    plt.plot(iterations, record_f, 'r-', label="Objective function")
    plt.xlabel("Iteration")
    plt.ylabel("f(x,y)")
    plt.legend()

    # x and y values
    plt.subplot(2,1,2)
    plt.plot(iterations, record_x, 'b-', label="x")
    plt.plot(iterations, record_y, 'g--', label="y")
    plt.xlabel("Iteration")
    plt.ylabel("Values")
    plt.legend()

    plt.tight_layout()
    plt.show()
    # Top graph = How well the algorithm is performing (objective function improvement).
    # Bottom graph = How the algorithm is exploring the search space (movement of x and y values).


def main():
    
    # Parameters: objective function, initial_temp, n_iter, n_temp, factor, step_size

    # Run on Sphere
    best_point, best_value, record_f, record_x, record_y = simulated_annealing(sphere, 1, 200, 200, 0.91, 0.4)
    print(f"Best solution for Sphere Function is: {best_value} at {best_point}")
    plotGraph(record_f, record_x, record_y)
    # Best solution for Sphere Function is: 1.684294151783897e-06 at (0.0011746510171369573, -0.0005518053458630212)


    # Run on Rosenbrock
    # best_point, best_value, record_f, record_x, record_y = simulated_annealing(rosenbrock, 2, 110, 100, 0.8, 0.2)
    # print(f"Best solution for Rosenbrock Function is: {best_value} at {best_point}")
    # plotGraph(record_f, record_x, record_y)
    # Best solution for Rosenbrock Function is: 0.00013015892573465596 at (1.0014513581274092, 1.002791662417166)
    

    # Run on Griewank
    # best_point, best_value, record_f, record_x, record_y = simulated_annealing(griewank, 5, 200, 200, 0.95, 1)
    # print(f"Best solution for Griewank Function is: {best_value} at {best_point}")
    # plotGraph(record_f, record_x, record_y)
    # Best solution for Griewank Function is: 1.6333857781436478e-05 at (0.003864516747292601, -0.005951172548345651)

if __name__ == "__main__":
    main()