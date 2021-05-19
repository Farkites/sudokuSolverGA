import json
import numpy as np

from time import time
from operator import attrgetter
from sklearn.model_selection import ParameterGrid

from definitions import *

from charles.charles import Population, Individual
from charles.utils import color
from charles.sudoku_utils import find_init_positions
from charles.sudoku_main_helper import load_experiments_overview, manage_details_dir, collect_puzzle_data, \
    manage_epoch_results, write_results_to_overview, prepare_diversity_hist_plot, prepare_history_plot
from charles.sudoku_main_helper import resolve_evaluate_fct, resolve_create_representation, resolve_crossover, resolve_mutation

from charles.selection import fps, tournament
from charles.mutation import swap_mutation, inversion_mutation, swap_by_row_mutation
from charles.crossover import single_point_co, cycle_co, arithmetic_co, cycle_by_row_co,\
    partially_match_co, partially_match_by_row_co

from configs_available import config_grid_2nd_run_test as config_grid

grid = ParameterGrid(config_grid)
len(grid)
if __name__ == '__main__':
    # load experiments overview
    run_id, overview = load_experiments_overview()

    for gs_id, config in enumerate(grid):
        start = time()
        print(f'{color.GREEN}grid search run {gs_id} / {len(grid)}{color.END}')
        print(config)
        run_name = f'{run_id}_{gs_id}'
        # create dir for experiment details
        details_dir = manage_details_dir(run_name)

        # save config
        with open(os.path.join(details_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=5)

        # init
        best_fitness = []
        history = {}
        diversity_hist = {}
        stopped_early = []

        for epoch in range(config['epochs']):
            # collect data
            puzzle = collect_puzzle_data(config)

            # compute metadata on board shapes
            init_positions = find_init_positions(puzzle.puzzle_flat)

            # Monkey Patching
            Individual.create_representation = resolve_create_representation(config, puzzle)
            Individual.evaluate = resolve_evaluate_fct(puzzle)
            Individual.get_neighbours = None

            pop = Population(
                size=config['pop_size'],
                optim=config['optim']
            )

            pop.evolve(
                gens=config['gens'],
                select=globals()[config['selection']],
                crossover=resolve_crossover(config, init_positions), # define operator in function above
                mutate=resolve_mutation(config, init_positions), # define operator in function above
                co_p=config['co_p'],
                mu_p=config['mu_p'],
                elitism=config['elitism'],
                fitness_sharing=config['fitness_sharing'],
                diversity_measure=config['diversity_measure'],
                early_stopping_patience=config['early_stopping_patience']

            )

            best = min(pop.individuals, key=attrgetter("fitness"))
            puzzle = manage_epoch_results(best, puzzle)

            # append results after every epoch
            best_fitness.append(best.fitness)
            history[epoch] = pop.history
            diversity_hist[epoch] = pop.diversity_hist
            stopped_early.append(pop.stopped_early)

        end = time()
        duration = np.round(end - start, 2)

        # append run results to overview
        overview = write_results_to_overview(overview, config, run_id, gs_id, duration, best_fitness, stopped_early)
        overview.to_csv(OVERVIEW_FILE_ABS, sep=';', index=False)

        # plot history
        history_df, history_fig = prepare_history_plot(history, run_name)
        history_df.to_csv(os.path.join(details_dir, 'history.csv'), sep=';')
        history_fig.savefig(os.path.join(details_dir, 'history.pdf'))

        if config['diversity_measure']:
            # plot diversity history
            diversity_hist_df, diversity_hist_fig = prepare_diversity_hist_plot(diversity_hist, run_name)
            diversity_hist_df.to_csv(os.path.join(details_dir, 'diversity_hist.csv'), sep=';')
            diversity_hist_fig.savefig(os.path.join(details_dir, 'diversity_hist.pdf'))


    print('ok')



