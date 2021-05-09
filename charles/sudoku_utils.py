import numpy as np
from copy import deepcopy

def get_box_indices(base=3):
    """
    :param base: length of inner box
    :return: row wise list of box indices
    """
    # total length of outer box
    side=base*base
    box_idx = [None for _ in range(side)]
    # flat index of length i*j
    pos = 0
    for i in range(base):
        for j in range(base):
            # define starting value for each box
            start = i*base*side + j*base
            # c: starting point of each row inside box
            # r: add rows inside box
            box_idx[pos] = [start + c*side + r for c in range(base) for r in range(base)]
            pos += 1
    return box_idx


def get_row_indices(base):
    side = base*base
    return [list(range(i, i + side)) for i in np.arange(0, side*side, side)]


def get_col_indices(base):
    side = base*base
    return [np.arange(i, side*side, side) for i in list(range(side))]


def get_indices(base):
    row_idx = get_row_indices(base)
    col_idx = get_col_indices(base)
    box_idx = get_box_indices(base)
    return row_idx, col_idx, box_idx


def count_duplicates(seq):
    '''takes as argument a sequence and
    returns the number of duplicate elements'''

    counter = 0
    seen = set()
    for elm in seq:
        if elm in seen:
            counter += 1
        else:
            seen.add(elm)
    return counter


def get_pop_specs(option, side):
    if option == 1:
        replacement = False
        valid_set = [i for _ in range(side) for i in range(1, side +1)]

    elif option == 2:
        replacement = True
        valid_set = list(range(1,side+1))


    return replacement, valid_set


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
