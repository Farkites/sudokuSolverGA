import numpy as np

config_grid_all_options = {
    'difficulty': [3],  # [3,2,1]
    'epochs': [30],
    'pop_size': np.arange(100, 1000, 250).tolist(),
    'gens': [1000],
    'optim': ['min'],
    'representation': ['with_replacement', 'without_replacement', 'maintain_init_puzzle'],
    'selection': ['tournament', 'fps'], # [tournament, fps]
    'mutation': ['swap_mutation', 'inversion_mutation', 'swap_by_row_mutation'],
    'crossover': ['single_point_co', 'cycle_co', 'arithmetic_co', 'partially_match_co', 'cycle_by_row_co', 'partially_match_by_row_co'],
    'co_p': np.arange(.7, 1.05, .15).tolist(),
    'mu_p': np.arange(.05, 0.35, .05).tolist(),
    'elitism': [True],
    'fitness_sharing': [False],
    'diversity_measure': [True],
    'early_stopping_patience': [100]
}

config_grid_2nd_run = {
    'difficulty': [3],  # [3,2,1]
    'epochs': [30],
    'pop_size': [50],
    'gens': [1000],
    'optim': ['min'],
    'representation': ['with_replacement', 'without_replacement', 'maintain_init_puzzle'],
    'selection': ['tournament'],# 'fps'], # [tournament, fps]
    'mutation': ['swap_mutation', 'inversion_mutation', 'swap_by_row_mutation'],
    'crossover': ['single_point_co', 'arithmetic_co', 'partially_match_co', 'partially_match_by_row_co'],
    'co_p': [.95, .85, .5, .1],#np.arange(.7, 1.05, .15).tolist(),
    'mu_p': [0.05, .2, .8, .9], #np.arange(.05, 1, .25).tolist(),
    'elitism': [True],
    'fitness_sharing': [False],
    'diversity_measure': [True],
    'early_stopping_patience': [100]
}

config_grid_testing = {
    'difficulty': [3],  # [3,2,1]
    'epochs': [3],
    'pop_size': [50],
    'gens': [1000],
    'optim': ['min'],
    'representation': ['with_replacement'], # [with_replacement, without_replacement, maintain_init_puzzle]
    'selection': ['tournament'], # [tournament, fps]
    'mutation': ['swap_mutation'], # [swap_mutation, inversion_mutation, swap_by_row_mutation]
    'crossover': ['cycle_co'], # [single_point_co, cycle_co, arithmetic_co, partially_match_co, cycle_by_row_co, partially_match_by_row_co]
    'co_p': [.9],
    'mu_p': [.01],
    'elitism': [True],
    'fitness_sharing': [False],
    'diversity_measure': [True],
    'early_stopping_patience': [50]
}