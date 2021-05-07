from charles.charles import Population, Individual
from copy import deepcopy
from data.sudoku_data_generator import Sudoku
from charles.selection import fps, tournament
from charles.mutation import binary_mutation, swap_mutation
from charles.crossover import single_point_co
from random import random, sample, randint
from operator import attrgetter
from charles.sudoku_utils import get_indices, count_duplicates
from random import choice

def find_init_positions(flat_puzzle):
    init_positions = []
    for idx, v in enumerate(flat_puzzle):
        if v != 0:
            init_positions.append((idx, v))
    return init_positions

def drop_init_positions(flat_board, init_positions):
    init_idx = [i[0] for i in init_positions]
    return [v for pos, v in enumerate(flat_board) if pos not in init_idx]

def include_init_positions(flat_board_without_init, init_postitions):
    flat_board_inserted = deepcopy(flat_board_without_init)
    for pos in init_postitions:
        flat_board_inserted.insert(pos[0], pos[1])
    return flat_board_inserted

def mutation_wrapper(individual, mutation, init_positions):
    indv_without_init = drop_init_positions(individual, init_positions)
    i = mutation(indv_without_init)
    return include_init_positions(i, init_positions)

def bla(mutation):
    def mutation_wrapper(individual, mutation, init_positions):
        indv_without_init = drop_init_positions(individual, init_positions)
        i = mutation(indv_without_init)
        return include_init_positions(i, init_positions)
    return mutation_wrapper


puzzle = Sudoku(difficulty=1)
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
    if False:
        for pos, v in enumerate(puzzle_ref):
            if repres[pos] != v:
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


def get_neighbours(self):
    pass


# Monkey Patching
Individual.evaluate = evaluate
Individual.get_neighbours = get_neighbours
Individual.create_representation = create_representation

if __name__ == '__main__':

    pop = Population(
        size=300, optim="min", sol_size=None, valid_set=None, replacement=None)


    #mutation_wrapper(pop.individuals[0], swap_mutation, init_positions)

    pop.evolve(
        gens=100,
        select=tournament,
        crossover=single_point_co,
        mutate=bla(swap_mutation),
        co_p=0.7,
        mu_p=0.2,
        elitism=True
    )

    # sol_board = Sudoku()
    board = min(pop.individuals, key=attrgetter("fitness")).representation
    # sol_board.board = board
    # sol_board.pretty_print_solution()
    print(board)
