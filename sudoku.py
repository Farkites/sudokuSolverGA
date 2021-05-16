from charles.charles import Population, Individual
from copy import deepcopy
from data.sudoku_data_generator import Sudoku
from data import sudoku_data
from charles.selection import fps, tournament
from charles.mutation import swap_mutation, inversion_mutation
from charles.crossover import single_point_co, cycle_co, arithmetic_co
from random import random, sample, randint
from operator import attrgetter
from charles.sudoku_utils import get_indices, count_duplicates, find_init_positions, drop_init_positions, \
    include_init_positions, flatten_board
from random import choice


config = {
    'epochs': 4,
    'pop_size': 100,
    'optim': 'min',
    'difficulty': 3, # [3,2,1]
    'representation': 'maintain_init_puzzle', # [with_replacement, without_replacement, maintain_init_puzzle]
    'gens': 100,
    'co_p': .5,
    'mu_p': .1,
    'elitism': True,
    'selection': 'tournament', # [tournament, fps]
    'mutation': 'swap_mutation', # [swap_mutation, inversion_mutation]
    'crossover': 'single_point_co' # [single_point_co, cycle_co, arithmetic_co]
}


# init
puzzle = Sudoku()

# collect data
if config['difficulty'] == 3:
    puzzle.add_board(sudoku_data.board_diff_3)
    puzzle.add_puzzle(sudoku_data.puzzle_diff_3)
    puzzle.difficulty = 3
elif config['difficulty'] == 2:
    puzzle.add_board(sudoku_data.board_diff_2)
    puzzle.add_puzzle(sudoku_data.puzzle_diff_2)
    puzzle.difficulty = 2
elif config['difficulty'] == 1:
    puzzle.add_board(sudoku_data.board_diff_1)
    puzzle.add_puzzle(sudoku_data.puzzle_diff_1)
    puzzle.difficulty = 1
else:
    raise NotImplementedError(config['difficulty'])

# compute metadata on board shapes
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


if config['representation'] == 'maintain_init_puzzle':
    def create_representation(self):
        matrix = deepcopy(puzzle.puzzle)
        for row in matrix:
            while 0 in row:
                insert_digit = choice(list(set(range(1, 10))-set(row)))
                if insert_digit not in row:
                    row[row.index(0)] = insert_digit
        return flatten_board(matrix)

    Individual.create_representation = create_representation
    sol_size = None
    valid_set = None
    replacement = None
elif config['representation'] == 'with_replacement':
    sol_size = 81
    valid_set = list(range(1,side+1))

    def create_representation(self):
        return [choice(valid_set) for i in range(sol_size)]

    Individual.create_representation = create_representation
elif config['representation'] == 'without_replacement':
    sol_size = 81
    valid_set = [i for _ in range(side) for i in range(1, side +1)]

    def create_representation(self):
        return sample(valid_set, sol_size)

    Individual.create_representation = create_representation


"""# seletion
if config['selection'] == 'fps':
    selection = fps
elif config['selection'] == 'tournament':
    selection = tournament"""


# mutation
if config['mutation'] in ['swap_mutation', 'inversion_mutation']:
    mut = globals()[config['mutation']]
    def mutate(individual):
        indv_without_init = drop_init_positions(individual, init_positions)
        i = mut(indv_without_init)
        return include_init_positions(i, init_positions)
    mutation = mutate
elif config['mutation'] in []:
    mutation = globals()[config['mutation']]


if config['crossover'] in ['single_point_co', 'cycle_co', 'arithmetic_co']:
    co = globals()[config['crossover']]
    def crossover(p1, p2):

        p1_w = drop_init_positions(p1, init_positions)
        p2_w = drop_init_positions(p2, init_positions)

        offspring1, offspring2 = co(p1_w, p2_w)

        offspring1 = include_init_positions(offspring1, init_positions)
        offspring2 = include_init_positions(offspring2, init_positions)

        return offspring1, offspring2

    crossover_fct = crossover
elif config['crossover'] in []:
    crossover_fct = globals()[config['crossover']]



# Monkey Patching
Individual.evaluate = evaluate
Individual.get_neighbours = None

if __name__ == '__main__':

    for _ in range(config['epochs']):
        pop = Population(
            size=config['pop_size'],
            optim=config['optim']
        )

        pop.evolve(
            gens=config['gens'],
            select=globals()[config['selection']],
            crossover=crossover_fct, # define operator in function above
            mutate=mutation, # define operator in function above
            co_p=config['co_p'],
            mu_p=config['mu_p'],
            elitism=config['elitism']
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

