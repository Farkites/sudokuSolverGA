from charles.charles import Population, Individual
from copy import deepcopy
from data import sudoku_data_generator
from charles.selection import fps, tournament
from charles.mutation import binary_mutation, swap_mutation
from charles.crossover import single_point_co
from random import random, sample, randint
from operator import  attrgetter

import numpy as np

def get_box_indices(base=3):
    """
    :param base: length of inner box
    :return: row wise list of box indices
    """
    # total length of outer box
    side=base*base
    box_idx = [None for _ in range(side)]
    # flat index of length i*j
    pos = 0
    for i in range(base):
        for j in range(base):
            # define starting value for each box
            start = i*base*side + j*base
            # c: starting point of each row inside box
            # r: add rows inside box
            box_idx[pos] = [start + c*side + r for c in range(base) for r in range(base)]
            pos += 1
    return box_idx

def get_row_indices(base):
    side = base*base
    return [list(range(i, i + side)) for i in np.arange(0, side*side, side)]

def get_col_indices(base):
    side = base*base
    return [np.arange(i, side*side, side) for i in list(range(side))]


def get_indices(base):
    row_idx = get_row_indices(base)
    col_idx = get_col_indices(base)
    box_idx = get_box_indices(base)
    return row_idx, col_idx, box_idx

def count_duplicates(seq):
    '''takes as argument a sequence and
    returns the number of duplicate elements'''

    counter = 0
    seen = set()
    for elm in seq:
        if elm in seen:
            counter += 1
        else:
            seen.add(elm)
    return counter


puzzle = sudoku_data_generator.Sudoku(difficulty=3)
data = puzzle.puzzle_flat

row_idx, col_idx, box_idx = get_indices(base=puzzle.base)

def evaluate(self):
    repres = self.representation
    #repres = [randint(1,9) for _ in range(81)]
    n_error = 0
    # count errors in rows
    for row in row_idx:
        n_error += count_duplicates([repres[r] for r in row])
    # count errors in cols
    for col in col_idx:
        n_error += count_duplicates([repres[c] for c in col])
    # count errors in squares
    for box in box_idx:
        n_error += count_duplicates([repres[b] for b in box])

    return n_error

def get_neighbours(self):
    pass


# Monkey Patching
Individual.evaluate = evaluate
Individual.get_neighbours = get_neighbours

if __name__ == '__main__':

    pop = Population(
        size=100, optim="min", sol_size=len(data), valid_set=list(range(1,10)), replacement=True)

    pop.evolve(
        gens=300,
        select= tournament,
        crossover= single_point_co,
        mutate=swap_mutation,
        co_p=0.7,
        mu_p=0.2,
        elitism=False
    )

    print('ok')