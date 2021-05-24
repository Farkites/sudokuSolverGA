import numpy as np
import pandas as pd
import seaborn as sns

from copy import deepcopy
from random import sample, choice
from matplotlib import pyplot as plt

from definitions import *

from data.sudoku_data_generator import Sudoku
from charles.utils import color
from data import sudoku_data

from charles.sudoku_utils import flatten_board, get_indices, count_duplicates, drop_init_positions, include_init_positions

from charles.selection import fps, tournament
from charles.mutation import swap_mutation, inversion_mutation, swap_by_row_mutation
from charles.crossover import single_point_co, cycle_co, arithmetic_co, cycle_by_row_co,\
    partially_match_co, partially_match_by_row_co

def repr_maintain_init_puzzle(puzzle):
    def create_representation(self):
        matrix = deepcopy(puzzle.puzzle)
        for row in matrix:
            while 0 in row:
                insert_digit = choice(list(set(range(1, 10)) - set(row)))
                if insert_digit not in row:
                    row[row.index(0)] = insert_digit
        return flatten_board(matrix)
    return create_representation

def repr_with_replacement(side):
    def create_representation(self):
        sol_size = 81
        valid_set = list(range(1, side + 1))
        return [choice(valid_set) for i in range(sol_size)]
    return create_representation

def repr_without_replacement(side):
    def create_representation(self):
        sol_size = 81
        valid_set = [i for _ in range(side) for i in range(1, side + 1)]
        return sample(valid_set, sol_size)
    return create_representation


def resolve_create_representation(config, puzzle):
    side = puzzle.side
    if config['representation'] == 'maintain_init_puzzle':
        return repr_maintain_init_puzzle(puzzle)
    elif config['representation'] == 'with_replacement':
        return repr_with_replacement(side)
    elif config['representation'] == 'without_replacement':
        return repr_without_replacement(side)
    elif config['representation'] == 'random_mix':
        res = [
            repr_maintain_init_puzzle(puzzle),
            repr_with_replacement(side),
            repr_without_replacement(side)
        ]
        return res
    else:
        raise NotImplementedError


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
    """
    based on the config, build the mutation function, that should be passed to the evolve method
    if condition matched: wrap the maintain init positions functionality around the actual mutation function
    else: fetch and pass the selected mutation function from globals as is
    :param config: main sudoku config
    :param init_positions: init positions, specific to puzzle
    :return: selected mutation function according to config
    """
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
    """
    based on the config, build the crossover function, that should be passed to the evolve method
    if condition matched: wrap the maintain init positions functionality around the actual co function
    else: fetch and pass the selected co function from globals as is
    :param config: main sudoku config
    :param init_positions: init positions, specific to puzzle
    :return: selected co function according to config
    """
    if (config['crossover'] in ['single_point_co', 'cycle_co', 'arithmetic_co', 'partially_match_co']) \
            & (config['representation'] == 'maintain_init_puzzle'):
        co = globals()[config['crossover']]

        def crossover(p1, p2):
            # drop init positions of both  parents
            p1_w = drop_init_positions(p1, init_positions)
            p2_w = drop_init_positions(p2, init_positions)

            # apply actual co to modified parents
            offspring1, offspring2 = co(p1_w, p2_w)

            # insert the init positions to both offspring
            offspring1 = include_init_positions(offspring1, init_positions)
            offspring2 = include_init_positions(offspring2, init_positions)

            return offspring1, offspring2

        crossover_fct = crossover
    else:
        co = globals()[config['crossover']]

        def crossover_by_row(p1, p2):
            # only fetch parents representation
            offspring1, offspring2 = co(p1.representation, p2.representation)
            return offspring1, offspring2

        crossover_fct = crossover_by_row
    return crossover_fct



def load_experiments_overview():
    """
    loads the overview csv file, computes the increment of the run id
    :return: tuple of current run_id and overview DF
    """
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
    """
    defines and creates path to store details of the current run
    :param run_name: name of the current run
    :return: abs path to store details of the current run
    """
    details_dir = os.path.join(RESULTS_PATH_ABS, run_name)
    if not os.path.exists(details_dir):
        os.mkdir(details_dir)
    return details_dir


def prepare_history_plot(history, run_name):
    """
    Creates the fitness history plot over all epochs and addes the mean line
    :param history: fintess history as stored in pop attributes
    :param run_name: name of the current run
    :return: tuple of DF and plot of the fitness history
    """
    fig, ax = plt.subplots(1, 1)

    # reshape history to long DF
    history_df = pd.DataFrame()
    for k, v in history.items():
        tmp = pd.DataFrame(v, columns=['generation', 'fitness'])
        tmp['epoch'] = k
        history_df = pd.concat([history_df, tmp], axis=0)

    # compute avg
    mean_df = history_df.groupby(['generation']).fitness.mean().reset_index(drop=False)
    # append average rows to long DF
    mean_df['epoch'] = 'mean'
    history_df = pd.concat([history_df, mean_df], axis=0)

    # all individual runs per epoch grey, mean: red
    palette = {e: 'red' if e == 'mean' else 'grey' for e in history_df.epoch.unique()}
    # create plot
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
    """
    Same as history of fitness, but for diversity (see details in "prepare_history_plot")
    :param diversity_hist:
    :param run_name:
    :return:
    """
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
    """
    Replaces overview.txt by appending a new row based on the current result.
    Collect all individually computed infos of the run and create a new row.
    :param overview:
    :param config:
    :param run_id:
    :param gs_id:
    :param duration:
    :param best_fitness:
    :param stopped_early:
    :return: updated overview DF
    """
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
    """
    Collects puzzle raw data based on difficulty selected in config
    :param config: main sudoku config
    :return: instance of Sudoku class with board and puzzle already added
    """
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
    """
    Adds GA generated solution to Sudoku class and uses its pretty print method to compare solution with puzzle and
    deterministic solution
    :param best: instance of class Individual
    :param puzzle: instance of class Sudoku
    :return: Sudoku instance with added GA generated solution
    """
    print(f'Puzzle: \n{puzzle.puzzle_flat}')
    print(f'Best solution (Fitness = {best.fitness}): \n{best.representation}')
    puzzle.pretty_print_puzzle()
    puzzle.fitness = best.fitness
    puzzle.add_solution(best.representation)
    puzzle.pretty_print_solution()
    print('ok')
    return puzzle