from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy


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
        self.representation = self.create_representation()
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
        for _ in range(size):
            self.individuals.append(
                Individual(
                    #size=kwargs["sol_size"],
                    #replacement=kwargs["replacement"],
                    #valid_set=kwargs["valid_set"],
                    # representation=kwargs["representation"](board)
                )
            )

    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism):
        history = []

        for gen in range(gens):
            new_pop = []

            if elitism == True:
                if self.optim == "max":
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))

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
            elif self.optim == "min":
                best = min(self, key=attrgetter("fitness"))
                history.append((gen, best.fitness))
                print(f'Best Individual: {best}')

        self.history = history

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"
