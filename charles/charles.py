from random import shuffle, choice, sample, random, randint
from operator import attrgetter
from copy import deepcopy
from scipy.spatial import distance
import numpy as np
import types
from sklearn.preprocessing import normalize


class Individual:
    def __init__(
            self,
            representation=None
            #,
            #size=None,
            #replacement=True,
            #valid_set=[i for i in range(13)],
    ):
        """if representation == None:
            if replacement == True:
                self.representation = [choice(valid_set) for i in range(size)]
            elif replacement == False:
                self.representation = sample(valid_set, size)
        else:
            self.representation = representation"""
        if isinstance(self.create_representation, types.MethodType):
            self.representation = self.create_representation()
        else:
            self.representation = self.create_representation[randint(0,2)](self)
        self.fitness = self.evaluate()

    def evaluate(self):
        raise Exception("You need to monkey patch the fitness path.")

    def create_representation(self):
        raise Exception("You need to monkey patch the representation path.")

    def get_neighbours(self, func, **kwargs):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual(size={len(self.representation)}); Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        self.history = None
        self.diversity_hist = None
        self.stopped_early = False
        for _ in range(size):
            self.individuals.append(
                Individual(
                    #size=kwargs["sol_size"],
                    #replacement=kwargs["replacement"],
                    #valid_set=kwargs["valid_set"],
                    # representation=kwargs["representation"](board)
                )
            )


    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism, diversity_measure, fitness_sharing=False, early_stopping_patience=None):
        history = []
        diversity_hist = []
        # used for early stopping
        n_gens_not_improved = 0
        if self.optim == "max":
            best_before = -np.inf
        elif self.optim == "min":
            best_before = +np.inf

        for gen in range(gens):
            new_pop = []
            if elitism == True:
                if self.optim == "max":
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))
            if fitness_sharing == True:
                self.apply_fitness_sharing()
            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)
                # Crossover
                if random() < co_p:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                # Mutation
                if random() < mu_p:
                    offspring1 = mutate(offspring1)
                if random() < mu_p:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

            if elitism == True:
                if self.optim == "max":
                    least = min(new_pop, key=attrgetter("fitness"))
                elif self.optim == "min":
                    least = max(new_pop, key=attrgetter("fitness"))
                new_pop.pop(new_pop.index(least))
                new_pop.append(elite)

            self.individuals = new_pop

            if self.optim == "max":
                best = max(self, key=attrgetter("fitness"))
                history.append((gen, best.fitness))
                print(f'Best Individual: {best}')
                if diversity_measure:
                    diversity_hist.append((gen,self.get_entropy()))
                # early stopping
                best_fitness = best.fitness
                if best_fitness <= best_before:
                    n_gens_not_improved +=1
                else:
                    n_gens_not_improved = 0
                best_before = best_fitness

            elif self.optim == "min":
                best = min(self, key=attrgetter("fitness"))
                history.append((gen, best.fitness))
                print(f'Best Individual: {best}')
                if diversity_measure:
                    diversity_hist.append((gen,self.get_entropy()))
                # early stopping
                best_fitness = best.fitness
                if best_fitness >= best_before:
                    n_gens_not_improved +=1
                else:
                    n_gens_not_improved = 0
                best_before = best_fitness

            # apply early stopping
            if early_stopping_patience is not None:
                if n_gens_not_improved >= early_stopping_patience:
                    self.stopped_early = True
                    break

        self.history = history
        self.diversity_hist = diversity_hist

    def get_entropy(self, entropy_type='phenotypic'):

        distinct_individuals = {}
        for individual in self.individuals:
            if entropy_type == 'genotypic_v1':
                if str(individual.representation) in distinct_individuals.keys():
                    distinct_individuals[str(individual.representation)] += 1
                else:
                    distinct_individuals[str(individual.representation)] = 1

            if entropy_type == 'phenotypic':
                if str(individual.fitness) in distinct_individuals.keys():
                    distinct_individuals[str(individual.fitness)] += 1
                else:
                    distinct_individuals[str(individual.fitness)] = 1

        entropy = 0
        for f in distinct_individuals.values():
            f /= len(distinct_individuals.values())
            entropy += f * np.log10(f)
        return entropy

    def apply_fitness_sharing(self):
        distance_matrix = self.get_pairwise_distance()
        sharing_coefficients = []
        for row_idx, row in enumerate(distance_matrix):
            coefficient = 0
            for idx, distance in enumerate(row):
                if row_idx != idx:
                    coefficient += 1 - distance
            sharing_coefficients.append(coefficient)
        if self.optim == "max":
            for i, individual in enumerate(self.individuals):
                individual.fitness *= sharing_coefficients[i]
        elif self.optim == "min":
            for i, individual in enumerate(self.individuals):
                individual.fitness /= sharing_coefficients[i]
        return

    def get_pairwise_distance(self):
        distance_matrix = []
        for individual_1 in self.individuals:
            distance_row = []
            for individual_2 in self.individuals:
                    distance_row.append(distance.euclidean(individual_2.representation, individual_1.representation))
            distance_matrix.append(distance_row)

        normed_matrix = normalize(distance_matrix, axis=1, norm='l1')
        return normed_matrix

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"
