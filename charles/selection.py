from random import uniform, sample
from operator import attrgetter

def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    if population.optim == "max":
        # Sum total fitnesses
        total_fitness = sum([i.fitness for i in population])
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual
    elif population.optim == "min":
        raise NotImplementedError

    else:
        raise Exception("No optimiziation specified (min or max).")

def tournament(population, size=20):
    # Select individuals based on tournament size
    tournament = sample(population.individuals, size)
    # Check if the problem is max or min
    if population.optim == 'max':
        return max(tournament, key=attrgetter("fitness"))
    elif population.optim == 'min':
        return min(tournament, key=attrgetter("fitness"))
    else:
        raise Exception("No optimiziation specified (min or max).")
