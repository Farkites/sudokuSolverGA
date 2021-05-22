
from charles.sudoku_utils import get_indices, build_board_from_vector, flatten_board
from data.sudoku_data_generator import Sudoku

if __name__ == '__main__':
    base = 2
    idx = get_indices(base)
    flat = list(range(base**4))
    board = build_board_from_vector(flat, base=base)

    print(f'flat: {flat}')
    print(f'board: {board}')
    print(f'by_row: {idx[0]}')
    print(f'by_col: {idx[1]}')
    print(f'by_block: {idx[2]}')


    """tmp = Sudoku()
    tmp.add_board(board_flat=flatten_board(board))
    tmp.pretty_print_board()"""


    print('ok')