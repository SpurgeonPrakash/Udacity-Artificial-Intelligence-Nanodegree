# Course: Udacity AI Nanodegree
# Assignment: Sudoku
# Author: Ching-Hsiang Hsu
# email: chhsu@nyu.edu
# update: 012919

from utils import *


row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
diagonal_units = [[r+c for r,c in zip(rows, cols)], [r+c for r,c in zip(rows[::-1], cols)]]
unitlist = row_units + column_units + square_units + diagonal_units

# Must be called after all units (including diagonals) are added to the unitlist
units = extract_units(unitlist, boxes)
peers = extract_peers(units, boxes)

def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    The naked twins strategy says that if you have two or more unallocated boxes
    in a unit and there are only two digits that can go in those two boxes, then
    those two digits can be eliminated from the possible assignments of all other
    boxes in the same unit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).

    See Also
    --------
    Pseudocode for this algorithm on github:
    https://github.com/udacity/artificial-intelligence/blob/master/Projects/1_Sudoku/pseudocode.md
    """
    candidate_twins = [box for box in values.keys() if len(values[box]) == 2]
    # for candidate_twin in candidate_twins:
    #     print(values[candidate_twin]) # check candidate twins
    twins = [[box1, box2] for box1 in candidate_twins for box2 in peers[box1] if set(values[box1]) == set(values[box2])]
    # for twin in twins:
    #     print(values[twin[0]] + ' ' + values[twin[1]]) # check twins
    for twin in twins:
        box1 = twin[0]
        box2 = twin[1]
        peers1 = set(peers[box1])
        peers2 = set(peers[box2])
        intersection_peers = peers1.intersection(peers2)
        # delete the twin in each box
        for peer_box in intersection_peers:
            for twin_val in values[box1]:
                values = assign_value(values, peer_box, values[peer_box].replace(twin_val, ''))
    return values


def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """
    known_boxes = [box for box in values.keys() if len(values[box]) == 1]
    for box in known_boxes:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values

def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    for unit in unitlist:
        digits = [0]*10 # initialize a sized 10 list with zeros
        place = [""]*10 # initialize a sized 10 list with empty strings
        for box in unit:
            for i in str(values[box]):
                num = int(i)
                digits[num] = digits[num] + 1
                place[num] = box
        for idx, val in enumerate(digits):
            if val == 1:
                values[place[idx]] = str(idx)
    return values


def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable 
    """
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        # Eliminate Strategy
        values = eliminate(values)
        # Only Choice Strategy
        values = only_choice(values)
        # Naked Twins Strategy
        values = naked_twins(values);
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed
    find = True
    for box in boxes:
        if len(values[box]) > 1 :
            find = False
            break
    if find : 
        return values ## Sudoku Solved!
    
    # Choose one of the unfilled squares with the fewest possibilities
    cnt = 9 # find the minimum count
    box = ''
    for idx, val in enumerate(boxes):
        if len(values[val]) > 1 & len(values[val]) < cnt:
            cnt = len(values[val])
            box = val
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for digit in values[box]:
        new_sudoku = values.copy()
        new_sudoku[box] = digit
        solved = search(new_sudoku)
        if solved :
            return solved


def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.
        
        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    # print(unitlist) # test diagonal units
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    display(result)

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
