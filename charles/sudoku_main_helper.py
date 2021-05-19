import numpy as np
import pandas as pd
import seaborn as sns

from copy import deepcopy
from random import sample, choice
from matplotlib import pyplot as plt

from definitions import *

from data.sudoku_data_generator import Sudoku
from data import sudoku_data

from charles.sudoku_utils import flatten_board, get_indices, count_duplicates, drop_init_positions, include_init_positions

from charles.selection import fps, tournament
from charles.mutation import swap_mutation, inversion_mutation, swap_by_row_mutation
from charles.crossover import single_point_co, cycle_co, arithmetic_co, cycle_by_row_co,\
    partially_match_co, partially_match_by_row_co


def resolve_create_representation(config, puzzle):
    side = puzzle.side
    if config['representation'] == 'maintain_init_puzzle':
        def create_representation(self):
            matrix = deepcopy(puzzle.puzzle)
            for row in matrix:
                while 0 in row:
                    insert_digit = choice(list(set(range(1, 10)) - set(row)))
                    if insert_digit not in row:
                        row[row.index(0)] = insert_digit
            return flatten_board(matrix)

        # Individual.create_representation = create_representation
        sol_size = None
        valid_set = None
        replacement = None
    elif config['representation'] == 'with_replacement':
        sol_size = 81
        valid_set = list(range(1, side + 1))

        def create_representation(self):
            return [choice(valid_set) for i in range(sol_size)]

        # Individual.create_representation = create_representation
    elif config['representation'] == 'without_replacement':
        sol_size = 81
        valid_set = [i for _ in range(side) for i in range(1, side + 1)]

        def create_representation(self):
            return sample(valid_set, sol_size)

        # Individual.create_representation = create_representation
    return create_representation


def resolve_evaluate_fct(puzzle):
    row_idx, col_idx, box_idx = get_indices(base=puzzle.base)

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
            for pos, v in enumerate(puzzle.puzzle_flat):
                if repres[pos] != v and v != 0:
                    n_error += 10

        return n_error

    return evaluate


def resolve_mutation(config, init_positions):
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
    return mutation


def resolve_crossover(config, init_positions):
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
    return crossover_fct



def load_experiments_overview():
    if os.path.isfile(OVERVIEW_FILE_ABS):
        overview = pd.read_csv(OVERVIEW_FILE_ABS, sep=';')
        if len(overview.run_id) == 0:
            run_id = 0
        else:
            run_id = overview.run_id.max() + 1
    else:
        overview = pd.DataFrame()
        run_id = 0
    return run_id, overview


def manage_details_dir(run_name):
    details_dir = os.path.join(RESULTS_PATH_ABS, run_name)
    if not os.path.exists(details_dir):
        os.mkdir(details_dir)
    return details_dir


def prepare_history_plot(history, run_name):
    fig, ax = plt.subplots(1, 1)
    history_df = pd.DataFrame()
    for k, v in history.items():
        tmp = pd.DataFrame(v, columns=['generation', 'fitness'])
        tmp['epoch'] = k
        history_df = pd.concat([history_df, tmp], axis=0)

    # compute avg
    mean_df = history_df.groupby(['generation']).fitness.mean().reset_index(drop=False)
    mean_df['epoch'] = 'mean'
    history_df = pd.concat([history_df, mean_df], axis=0)

    palette = {e: 'red' if e == 'mean' else 'grey' for e in history_df.epoch.unique()}
    sns.lineplot(data=history_df,
                 x='generation',
                 y='fitness',
                 hue='epoch',
                 palette=palette,
                 legend=False,
                 ax=ax)
    fig.suptitle(f'Fitness history for run: {run_name}')
    plt.close()
    return history_df, fig


def prepare_diversity_hist_plot(diversity_hist, run_name):
    fig, ax = plt.subplots(1, 1)
    diversity_hist_df = pd.DataFrame()
    for k, v in diversity_hist.items():
        tmp = pd.DataFrame(v, columns=['generation', 'entropy'])
        tmp['epoch'] = k
        diversity_hist_df = pd.concat([diversity_hist_df, tmp], axis=0)

    # compute avg
    mean_df = diversity_hist_df.groupby(['generation']).entropy.mean().reset_index(drop=False)
    mean_df['epoch'] = 'mean'
    diversity_hist_df = pd.concat([diversity_hist_df, mean_df], axis=0)

    palette = {e: 'red' if e == 'mean' else 'grey' for e in diversity_hist_df.epoch.unique()}
    sns.lineplot(data=diversity_hist_df,
                 x='generation',
                 y='entropy',
                 hue='epoch',
                 palette=palette,
                 legend=False,
                 ax=ax)
    fig.suptitle(f'Diversity measure history for run: {run_name}')
    plt.close()
    return diversity_hist_df, fig


def write_results_to_overview(overview, config, run_id, gs_id, duration, best_fitness, stopped_early):
    tmp_add = pd.DataFrame({
        'user_id': USER,
        'comments': [None]
    })

    # save results to csv
    tmp_results = pd.DataFrame({
        'run_id': [run_id],
        'gs_id': [int(gs_id)],
        'duration': duration,
        'fitness_mean': np.round(np.mean(best_fitness), 2),
        'fitness_sd': np.round(np.std(best_fitness), 2),
        'stopped_early': sum(stopped_early)
    })
    tmp_config_df = pd.DataFrame(config, index=[0])
    tmp_overview = pd.concat([tmp_results, tmp_config_df, tmp_add], axis=1)
    overview = pd.concat([overview, tmp_overview], axis=0)
    overview.reset_index(inplace=True, drop=True)
    return overview


def collect_puzzle_data(config):
    # init
    puzzle = Sudoku()
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
    return puzzle


def manage_epoch_results(best, puzzle):
    print(f'Puzzle: \n{puzzle.puzzle_flat}')
    print(f'Best solution (Fitness = {best.fitness}): \n{best.representation}')
    puzzle.pretty_print_puzzle()
    puzzle.fitness = best.fitness
    puzzle.add_solution(best.representation)
    puzzle.pretty_print_solution()
    print('ok')
    return puzzle