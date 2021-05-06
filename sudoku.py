from charles.charles import Population, Individual
from copy import deepcopy
from data import sudoku_data_generator
from charles.selection import fps, tournament
from charles.mutation import binary_mutation, swap_mutation
from charles.crossover import single_point_co
from random import random, sample, randint
from operator import  attrgetter
from charles.sudoku_utils import get_indices, count_duplicates

puzzle = sudoku_data_generator.Sudoku(difficulty=3)
puzzle_ref = puzzle.puzzle_flat
side = puzzle.side
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
    # count errors in box
    for box in box_idx:
        n_error += count_duplicates([repres[b] for b in box])

    # penalize deviation from initial puzzle
    if False:
        for pos, v in enumerate(puzzle_ref):
            if repres[pos] != v:
                n_error += 9

    return n_error


def get_neighbours(self):
    pass


# Monkey Patching
Individual.evaluate = evaluate
Individual.get_neighbours = get_neighbours



if __name__ == '__main__':

    pop = Population(
        size=300, optim="min", sol_size=side*side, valid_set=None, replacement=None)

    pop.evolve(
        gens=100,
        select= tournament,
        crossover= single_point_co,
        mutate=swap_mutation,
        co_p=0.7,
        mu_p=0.2,
        elitism=False
    )

    print('ok')