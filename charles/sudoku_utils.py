import numpy as np
from copy import deepcopy
from charles.utils import color
from charles.charles import Individual


def get_box_indices(base=3):
    """
    Computes box-wise list of indices, where every inner list represents a box
    :param base: length of inner box
    :return: row wise list of box indices
    """
    # total length of outer box
    side=base*base
    # init with list of Nones
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
    """
    Computes row-wise list of indices, where every inner list represents a row
    :param base: length of inner box
    :return: row-wise list of row indices
    """
    side = base*base
    return [list(range(i, i + side)) for i in np.arange(0, side*side, side)]


def get_col_indices(base):
    """
    Computes col-wise list of indices, where every inner list represents a col
    :param base: length of inner box
    :return: row-wise list of col indices
    """
    side = base*base
    return [np.arange(i, side*side, side).tolist() for i in list(range(side))]


def get_indices(base):
    """
    collects list of all 3 types of list index representations of a Sudoku board
    :param base:
    :return: list of all 3 types of list index representations of a Sudoku board
    """
    row_idx = get_row_indices(base)
    col_idx = get_col_indices(base)
    box_idx = get_box_indices(base)
    return row_idx, col_idx, box_idx


def count_duplicates(seq):
    """
    based on https://stackoverflow.com/questions/52090212/counting-the-number-of-duplicates-in-a-list/52090695
    takes as argument a sequence and
    returns the number of duplicate elements
    :param seq: list of numbers
    :return: int of duplicate numbers inside list
    """

    counter = 0
    seen = set()
    for elm in seq:
        if elm in seen:
            counter += 1
        else:
            seen.add(elm)
    return counter


def find_init_positions(flat_puzzle):
    init_positions = []
    for idx, v in enumerate(flat_puzzle):
        if v != 0:
            init_positions.append((idx, v))
    return init_positions


def drop_init_positions(flat_board, init_positions):
    """
    Drop initial positions
    :param flat_board: vector representation of a Sudoku board
    :param init_positions: list of tuples with (idx, value) of the puzzles initial positions
    :return: vector representation of the supplied board without the values at the initial positions
    """
    # extract only the indexes from the tuple
    init_idx = [i[0] for i in init_positions]
    # loop over flatboard and only return values at positions != init positions
    return [v for pos, v in enumerate(flat_board) if pos not in init_idx]


def include_init_positions(flat_board_without_init, init_postitions):
    """
    Insert inital positions
    :param flat_board_without_init: vector representation of Sudoku board,
    where init positions previously have been dropped through drop_init_positions
    :param init_positions: list of tuples with (idx, value) of the puzzles initial positions
    :return: vector representation of the supplied board with the values at the initial positions inserted
    """

    flat_board_inserted = deepcopy(flat_board_without_init)
    # loop over inital positions and insert value back into the flatboard one by one
    for pos in init_postitions:
        flat_board_inserted.insert(pos[0], pos[1])
    return flat_board_inserted


def flatten_board(board):
    """
    Transform matrix representation of a Sudoku board into a row-wise flat representation
    :param board: matrix representation of a Sudoku board
    :return: vector, row-wise flat representation of a Sudoku board
    """
    return [n for row in board for n in row]


def build_board_from_vector(flattboard, base=3):
    """
    Reconstructs matrix representation of a Sudoku board, assuming a row-wise transformation
    :param flattboard: vector representation
    :param base: size of the inner box of a Sudoku board
    :return: lists in list emulating matrix representation of a Sudoku board
    """

    # verify that shape of flattboard matches the expected length
    if len(flattboard) != base**4:
        raise ValueError('')

    # Extract vector from Individual if supllied board it of that type
    if isinstance(flattboard, Individual):
        flattboard = flattboard.representation

    # assert that flattboard is of type list
    if not isinstance(flattboard, list):
        raise ValueError('flattboard needs to be of type list')

    # fetch row-wise indexes of board representation
    idx = get_row_indices(base)
    # loop over rows (outer) and positions insed row (inner) to reconstruct row-wise list of lists
    return [[flattboard[i] for i in row] for row in idx]


def pretty_print(board, base=3, puzzle=None, solution=None, mark=False):
    """
    refactored from Sudoku class, currently not used. Use class method instead
    Pretty prints the board
    :param board: Matrix representation of the sudoku board, which will be printed.
    :param base: size of the inner grid
    :param puzzle: (optional) board with init positions, used the highlight those positions in the print
    :param solution: (optional) board with actual solution, used to highlight solving errors
    :param mark: whether to mark init positions and solving errors
    :return: None
    """
    side = base**2

    def expandLine(line):
        return line[0] + line[5:9].join([line[1:5] * (base - 1)] * base) + line[9:13]

    line0 = expandLine("╔═══╤═══╦═══╗")
    line1 = expandLine("║ . │ . ║ . ║")
    line2 = expandLine("╟───┼───╫───╢")
    line3 = expandLine("╠═══╪═══╬═══╣")
    line4 = expandLine("╚═══╧═══╩═══╝")

    symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if mark:
        nums = []
        for row_idx, row in enumerate(board):
            line = ['']
            for col_idx, n in enumerate(row):
                pos_fmt = ['', '']
                # check if was init pos
                if puzzle[row_idx][col_idx] != 0: #n == 2:
                    #col = [color.BOLD, color.END]
                    #line.append(color.BOLD + symbol[n] + color.END)
                    pos_fmt[0] = color.BOLD
                    pos_fmt[1] = color.END
                if solution[row_idx][col_idx] != n:
                    pos_fmt[0] = pos_fmt[0] + color.RED
                    pos_fmt[1] = color.END
                    #line.append(color.RED + symbol[n] + color.END)
                line.append(pos_fmt[0] + symbol[n] + pos_fmt[1])
            nums.append(line)
    else:
        nums = [[""] + [symbol[n] for n in row] for row in board]

    print(line0)
    for r in range(1, side + 1):
        print("".join(n + s for n, s in zip(nums[r - 1], line1.split("."))))
        print([line2, line3, line4][(r % side == 0) + (r % base == 0)])
