from random import randint, sample
from data.sudoku_data_generator import to_array, to_matrix


def template_mutation(individual):
    """[summary]

    Args:
        individual ([type]): [description]

    Returns:
        [type]: [description]
    """
    return individual


def binary_mutation(individual):
    """Binary muation for a GA individual

    Args:
        individual (Individual): A GA individual from charles libray.py

    Raises:
        Exception: When individual is not binary encoded.py

    Returns:
        Individual: Mutated Individual
    """
    mut_point = randint(0, len(individual) - 1)

    if individual[mut_point] == 0:
        individual[mut_point] = 1
    elif individual[mut_point] == 1:
        individual[mut_point] = 0
    else:
        raise Exception(
            f"Trying to do binary mutation on {individual}. But it's not binary."
        )

    return individual


def swap_mutation(individual):
    # Get two mutation points
    mut_points = sample(range(len(individual)), 2)
    # Rename to shorten variable name
    i = individual

    i[mut_points[0]], i[mut_points[1]] = i[mut_points[1]], i[mut_points[0]]

    return i


def swap_by_row_mutation(individual):
    i_matrix = to_matrix(individual)

    for i in range(len(i_matrix)):
        i_matrix[i] = swap_mutation(i_matrix[i])

    return i_matrix


def inversion_mutation(individual):
    i = individual
    # Position of the start and end of substring
    mut_points = sample(range(len(i)), 2)
    # This method assumes that the second point is after (on the right of) the first one
    # Sort the list
    mut_points.sort()
    # Invert for the mutation
    i[mut_points[0]:mut_points[1]] = i[mut_points[0]:mut_points[1]][::-1]

    return i
    
if __name__ == '__main__':
    i1 = [1,2,3,4,5,6]

    print(inversion_mutation(i1))