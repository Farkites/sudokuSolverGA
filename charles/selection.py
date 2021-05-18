from random import uniform, sample
from operator import attrgetter
import numpy as np
from numpy.random import choice

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
        # code adapted from https://rocreguant.com/roulette-wheel-selection-python/2019/
        # Computes the totality of the population fitness
        population_fitness = sum([chromosome.fitness for chromosome in population])

        # Computes for each individual the probability
        individual_probabilities = [individual.fitness / population_fitness for individual in population]

        # Making the probabilities for a minimization problem
        individual_probabilities = 1 - np.array(individual_probabilities)
        individual_probabilities = individual_probabilities / sum(individual_probabilities)

        #
        idx = list(range(len(population.individuals)))

        # test
        # check = sorted([(idx, p) for idx, p in enumerate(individual_probabilities)], key=lambda x: x[1], reverse=True)
        # Selects one individual based on the computed probabilities
        selected_idx = choice(idx, p=individual_probabilities)
        return population[selected_idx]

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
