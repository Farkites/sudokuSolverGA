from charles.charles import Population, Individual
from copy import deepcopy
from data.sudoku_data_generator import Sudoku
from data import sudoku_data
from charles.selection import fps, tournament
from charles.mutation import binary_mutation, swap_mutation, inversion_mutation
from charles.crossover import single_point_co
from random import random, sample, randint
from operator import attrgetter
from charles.sudoku_utils import get_indices, count_duplicates, find_init_positions, drop_init_positions, \
    include_init_positions, flatten_board
from random import choice

puzzle = Sudoku()
puzzle.add_board(sudoku_data.board_diff_3)
puzzle.add_puzzle(sudoku_data.puzzle_diff_3)

#puzzle.build_puzzle(difficulty=1)
puzzle_ref = puzzle.puzzle_flat
side = puzzle.side
row_idx, col_idx, box_idx = get_indices(base=puzzle.base)
init_positions = find_init_positions(puzzle_ref)


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
    if True:
        for pos, v in enumerate(puzzle_ref):
            if repres[pos] != v and v != 0:
                n_error += 9

    return n_error


def to_array(matrix):
    array = [n for row in matrix for n in row]
    return array


def create_representation(self):
    matrix = deepcopy(puzzle.puzzle)
    for row in matrix:
        while 0 in row:
            insert_digit = choice(list(set(range(1, 10))-set(row)))
            if insert_digit not in row:
                row[row.index(0)] = insert_digit
    return to_array(matrix)


def mutate(individual):
    indv_without_init = drop_init_positions(individual, init_positions)
    i = swap_mutation(indv_without_init)
    return include_init_positions(i, init_positions)


def crossover(p1, p2):

    p1_w = drop_init_positions(p1, init_positions)
    p2_w = drop_init_positions(p2, init_positions)

    offspring1, offspring2 = single_point_co(p1_w, p2_w)

    offspring1 = include_init_positions(offspring1, init_positions)
    offspring2 = include_init_positions(offspring2, init_positions)

    return offspring1, offspring2


# Monkey Patching
Individual.evaluate = evaluate
Individual.get_neighbours = None
Individual.create_representation = create_representation

if __name__ == '__main__':

    pop = Population(
        size=100, optim="min", sol_size=None, valid_set=None, replacement=None)


    pop.evolve(
        gens=100,
        select=tournament,
        crossover=crossover, # define operator in function above
        mutate=mutate, # define operator in function above
        co_p=0.7,
        mu_p=0.2,
        elitism=True
    )

    # sol_board = Sudoku()
    best = min(pop.individuals, key=attrgetter("fitness"))
    # sol_board.board = board
    # sol_board.pretty_print_solution()
    print(f'Puzzle: \n{puzzle_ref}')
    print(f'Best solution (Fitness = {best.fitness}): \n{best.representation}')
    puzzle.pretty_print_puzzle()

    puzzle.fitness = best.fitness
    puzzle.add_solution(best.representation)
    puzzle.pretty_print_solution()
    print('ok')

