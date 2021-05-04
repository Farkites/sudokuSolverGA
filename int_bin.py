from charles.charles import Population, Individual
from charles.search import hill_climb, sim_annealing
from copy import deepcopy


def evaluate(self):
    """A fitness function that takes a base-10 number,
    transforms it into base-2 (binary), and returns the
    number of 1's occuring in the binary representation.

    Args:
        number (int): the integer you want to get the
        fitness for.

    Returns:
        int: the number of 1's in the binary representation.
    """
    return self.representation.count(1)


def get_neighbours(self):
    n = [deepcopy(self.representation) for i in range(len(self.representation))]

    for count, i in enumerate(n):
        if i[count] == 1:
            i[count] = 0
        elif i[count] == 0:
            i[count] = 1

    n = [Individual(i) for i in n]
    return n


Individual.evaluate = evaluate
Individual.get_neighbours = get_neighbours

pop = Population(size=20, optim="max", sol_size=4, valid_set=[0, 1], replacement=True)

hill_climb(pop)
sim_annealing(pop)
