import os
import getpass
joinpath = os.path.join

USER = [getpass.getuser()]

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(BASE_PATH)

print(f'set base dir to {BASE_PATH}')

# saving models
RESULTS_PATH_ABS = joinpath(BASE_PATH, 'results')

if not os.path.exists(RESULTS_PATH_ABS):
    os.mkdir(RESULTS_PATH_ABS)

OVERVIEW_FILE_REL = 'overview.csv'
OVERVIEW_FILE_ABS = joinpath(RESULTS_PATH_ABS, OVERVIEW_FILE_REL)