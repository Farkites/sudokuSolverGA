from random import uniform, sample
from operator import attrgetter
import numpy as np

def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """
    if population.optim == "max":
        def calc_ind_fitness(individual):
            return individual.fitness
    elif population.optim == "min":
        worst = max(population, key=lambda x: x.fitness)
        def calc_ind_fitness(individual):
            return worst.fitness - individual.fitness
    else:
        raise Exception("No optimiziation specified (min or max).")

    # Sum total fitnesses
    total_fitness = sum([i.fitness for i in population])
    # Get a 'position' on the wheel
    spin = uniform(0, total_fitness)
    position = 0
    # Find individual in the position of the spin
    for individual in population:
        position += calc_ind_fitness(individual)
        if position > spin:
            return individual
        # https: // rocreguant.com / roulette - wheel - selection - python / 2019 /
        

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
