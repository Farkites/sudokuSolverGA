from charles.charles import Population, Individual
from copy import deepcopy
from data.sudoku_data_generator import Sudoku
from data import sudoku_data
from charles.selection import fps, tournament
from charles.mutation import swap_mutation, inversion_mutation, swap_by_row_mutation
from charles.crossover import single_point_co, cycle_co, arithmetic_co, cycle_by_row_co,\
    partially_match_co, partially_match_by_row_co
from random import random, sample, randint
from operator import attrgetter
from charles.sudoku_utils import get_indices, count_duplicates, find_init_positions, drop_init_positions, \
    include_init_positions, flatten_board
from random import choice
from time import time
import numpy as np
import pandas as pd
from definitions import *
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt
import seaborn as sns
import json

config_grid = {
    'difficulty': [3],  # [3,2,1]
    'epochs': [30],
    'pop_size': list(np.arange(100, 1000, 250)),
    'gens': [1000],
    'optim': ['min'],
    'representation': ['with_replacement', 'without_replacement', 'maintain_init_puzzle'],
    'selection': ['tournament'], # [tournament, fps]
    'mutation': ['swap_mutation', 'inversion_mutation', 'swap_by_row_mutation'],
    'crossover': ['single_point_co', 'cycle_co', 'arithmetic_co', 'partially_match_co', 'cycle_by_row_co', 'partially_match_by_row_co'],
    'co_p': list(np.arange(.7, 1.05, .15)),
    'mu_p': list(np.arange(.05, 0.35, .05)),
    'elitism': [True],
    'fitness_sharing': [False],
    'early_stopping_patience': [100]
}

config_grid = {
    'difficulty': [3],  # [3,2,1]
    'epochs': [10],
    'pop_size': [300],
    'gens': [1000],
    'optim': ['min'],
    'representation': ['with_replacement'], # [with_replacement, without_replacement, maintain_init_puzzle]
    'selection': ['tournament'], # [tournament, fps]
    'mutation': ['swap_mutation'], # [swap_mutation, inversion_mutation, swap_by_row_mutation]
    'crossover': ['single_point_co'], # [single_point_co, cycle_co, arithmetic_co, partially_match_co, cycle_by_row_co, partially_match_by_row_co]
    'co_p': [.9],
    'mu_p': [.01],
    'elitism': [True],
    'fitness_sharing': [False],
    'early_stopping_patience': [50]
}

grid = ParameterGrid(config_grid)

if __name__ == '__main__':
    # load experiments overview
    if os.path.isfile(OVERVIEW_FILE_ABS):
        overview = pd.read_csv(OVERVIEW_FILE_ABS, sep=';')
        if len(overview.run_id) == 0:
            run_id = 0
        else:
            run_id = overview.run_id.max() + 1
    else:
        overview = pd.DataFrame()
        run_id = 0


    for gs_id, config in enumerate(grid):
        start = time()
        # init
        best_fitness = []
        history = {}
        run_name = f'{run_id}_{gs_id}'
        stopped_early = []

        # create dir for experiment details
        details_dir = os.path.join(RESULTS_PATH_ABS, run_name)
        if not os.path.exists(details_dir):
            os.mkdir(details_dir)

        for epoch in range(config['epochs']):
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
                # repres = [randint(1,9) for _ in range(81)]
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
                            n_error += 1

                return n_error


            if config['representation'] == 'maintain_init_puzzle':
                def create_representation(self):
                    matrix = deepcopy(puzzle.puzzle)
                    for row in matrix:
                        while 0 in row:
                            insert_digit = choice(list(set(range(1, 10)) - set(row)))
                            if insert_digit not in row:
                                row[row.index(0)] = insert_digit
                    return flatten_board(matrix)


                Individual.create_representation = create_representation
                sol_size = None
                valid_set = None
                replacement = None
            elif config['representation'] == 'with_replacement':
                sol_size = 81
                valid_set = list(range(1, side + 1))


                def create_representation(self):
                    return [choice(valid_set) for i in range(sol_size)]


                Individual.create_representation = create_representation
            elif config['representation'] == 'without_replacement':
                sol_size = 81
                valid_set = [i for _ in range(side) for i in range(1, side + 1)]


                def create_representation(self):
                    return sample(valid_set, sol_size)


                Individual.create_representation = create_representation

            """# seletion
            if config['selection'] == 'fps':
                selection = fps
            elif config['selection'] == 'tournament':
                selection = tournament"""

            # mutation
            if (config['mutation'] in ['swap_mutation', 'inversion_mutation'])\
                    & (config['representation'] == 'maintain_init_puzzle'):
                mut = globals()[config['mutation']]

                def mutate(individual):
                    indv_without_init = drop_init_positions(individual, init_positions)
                    i = mut(indv_without_init)
                    return include_init_positions(i, init_positions)

                mutation = mutate
            else:
                mutation = globals()[config['mutation']]

            if (config['crossover'] in ['single_point_co', 'cycle_co', 'arithmetic_co', 'partially_match_co']) \
                    & (config['representation'] == 'maintain_init_puzzle'):
                co = globals()[config['crossover']]

                def crossover(p1, p2):

                    p1_w = drop_init_positions(p1, init_positions)
                    p2_w = drop_init_positions(p2, init_positions)

                    offspring1, offspring2 = co(p1_w, p2_w)

                    offspring1 = include_init_positions(offspring1, init_positions)
                    offspring2 = include_init_positions(offspring2, init_positions)

                    return offspring1, offspring2

                crossover_fct = crossover
            else:
                co = globals()[config['crossover']]

                def crossover_by_row(p1, p2):
                    offspring1, offspring2 = co(p1.representation, p2.representation)
                    return offspring1, offspring2

                crossover_fct = crossover_by_row

            # Monkey Patching
            Individual.evaluate = evaluate
            Individual.get_neighbours = None

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
                elitism=config['elitism'],
                fitness_sharing=config['fitness_sharing'],
                early_stopping_patience=config['early_stopping_patience']
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
            best_fitness.append(best.fitness)

            #
            history[epoch] = pop.history
            stopped_early.append(pop.stopped_early)

        end = time()
        duration = np.round(end - start, 2)

        # save config
        with open(os.path.join(details_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=5)
        # save results to csv
        tmp_results = pd.DataFrame({
            'run_id': [run_id],
            'gs_id': [int(gs_id)],
            'duration': duration,
            'fitness_mean': np.round(np.mean(best_fitness), 2),
            'fitness_sd': np.round(np.std(best_fitness), 2),
            'stopped_early': sum(stopped_early)
        })

        tmp_add = pd.DataFrame({
            'user_id': USER,
            'comments': [None]
        })
        tmp_config_df = pd.DataFrame(config, index=[0])
        tmp_overview = pd.concat([tmp_results, tmp_config_df, tmp_add], axis=1)
        overview = pd.concat([overview, tmp_overview], axis=0)
        overview.reset_index(inplace=True, drop=True)
        overview.to_csv(OVERVIEW_FILE_ABS, sep=';', index=False)

        # plot history
        fig, ax = plt.subplots(1,1)
        history_df = pd.DataFrame()
        for k,v in history.items():

            tmp = pd.DataFrame(v, columns=['generation','fitness'])
            tmp['epoch'] = k
            history_df = pd.concat([history_df, tmp], axis=0)

        # compute avg
        mean_df = history_df.groupby(['generation']).fitness.mean().reset_index(drop=False)
        mean_df['epoch'] = 'mean'
        history_df = pd.concat([history_df, mean_df], axis=0)
        history_df.to_csv(os.path.join(details_dir, 'history.csv'), sep=';')

        palette = {e: 'red' if e == 'mean' else 'grey' for e in history_df.epoch.unique()}
        sns.lineplot(data=history_df,
                     x='generation',
                     y='fitness',
                     hue='epoch',
                     palette=palette,
                     legend=False,
                     ax=ax)
        fig.suptitle(f'Fitness history for run: {run_name}')
        fig.savefig(os.path.join(details_dir, 'history.pdf'))

    print('ok')



