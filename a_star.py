import sys
import numpy as np
from operator import itemgetter

# Check if a puzzle is solvable
def is_solvable(puzzle):
    p = puzzle[puzzle != 0]
    inversions = 0
    for i, x in enumerate(p):
        for y in p[i+1:]:
            if x > y:
                inversions += 1
    return inversions % 2 == 0


# Generate random arrays until getting a solvable one
def generate_puzzle(size):
    array_size = size*size
    while True:
        puzzle = np.random.choice(np.arange(array_size), size=array_size, replace=False)
        if is_solvable(puzzle):
            break
    return puzzle.reshape(size,size).tolist()


# Generate the goal state
def generate_goal(size):
    array_size = size*size
    goal = np.arange(start=1, stop=array_size, step=1, dtype=int)
    goal = np.append(goal, [0])
    return goal.reshape(size,size).tolist()


def print_puzzle(puzzle):
    for row in puzzle:
        for col in row:
            print("%6d"%col, end ='')
        print("\n")


def create_node(size, data, g_score, h_score, parent):
    node = dict({
        'size': size,
        'data': data,
        'g_score': g_score,
        'h_score': h_score,
        'f_score': g_score + h_score,
        'parent': parent
        }
    )
    return node


# Get "good" position of a tile
def get_good_position(tile, size):
    x = (tile-1) // size # Floor division
    y = (tile-1) % size # Modulo
    return [x, y]


# Calculate Hamming distance / Misplaced tiles
def get_h_hamming(data, size):
    count = 0
    for i in range(0, size):
        for j in range(0, size):
            if data[i][j] != 0:
                [x, y] = get_good_position(data[i][j], size)
                if (x != i) or (y != j):
                    count += 1
    return count

l = [[1,2,3], [4,5,6], [7,8,9]]

print(get_h_hamming(l, 3))




# Find the blank element
def find_blank(node):
    size = node["size"]
    for i in range(0, size):
        for j in range(0, size):
            if node["data"][i][j] == 0:
                return i, j


# A "copy" function to workaround the not-released local memory call stack
def copy(node):
    temp = []
    for row in node["data"]:
        t = []
        for col in row:
            t.append(col)
        temp.append(t)
    return temp


def move_blank(node, row, col, new_row, new_col):
    # Boundary check
    temp_node = copy(node)
    size = node["size"]
    if (new_row >= 0) and (new_row < size) and (new_col >= 0) and (new_col < size):
        # Swap values
        # Temp value
        data_new = temp_node
        temp = data_new[row][col]
        data_new[row][col] = data_new[new_row][new_col]
        data_new[new_row][new_col] = temp
    else:
        data_new = None
    return data_new


# Create children nodes from current Node
# There should be a "copy" function to prevent the "node" variable not being released after each blank_move call
def transition(node):
    row_blank, col_blank = find_blank(node)
    move_list = [[row_blank-1, col_blank], [row_blank, col_blank-1], [row_blank+1, col_blank], [row_blank, col_blank+1]]
    nodes_child = []
    for [new_row, new_col] in move_list:
        data_new = move_blank(node, row_blank, col_blank, new_row, new_col)
        if data_new is not None:
            child_new = create_node(
                size=node["size"],
                data=data_new,
                g_score=node["g_score"]+1,
                h_score=get_h_hamming(data=data_new, size=node["size"]),
                parent=node
            )
            nodes_child.append(child_new)
    return nodes_child


# Compare two puzzle configurations
def compare(data1, data2, size):
    for i in range(0, size):
        for j in range(0, size):
            if (data1[i][j] != data2[i][j]):
                return False
    return True


# Find a dictionary in a list of dictionary


# Main functions
if __name__ == "__main__":

    # print("INPUT DATA\n")
    # try:
    #     size = int(input("Puzzle size: "))
    # except ValueError:
    #     print("Puzzle size must be an integer.")
    #     sys.exit(0)


    size = 2
    # Print generated puzzle
    start_data = generate_puzzle(size)
    print("\nGenerated puzzle:")
    print_puzzle(start_data)
    
    # Print goal puzzle
    goal = generate_goal(size)
    print("\nGoal puzzle:")
    print_puzzle(goal)

    
    # Initialize root node (start node)
    start_node = create_node(
        size=size,
        data=start_data,
        g_score=0,
        h_score=0,
        parent=None
    )

    # Initialize Open and Close
    open_list = []
    closed_list = []
    
    open_list.append(start_node)

    # Main loop
    while (1):
        if len(open_list) == 0:
            print("Open is empty. Failed.")
            sys.exit(0)
        
        # Get node as the first element in Open
        node = open_list[0]

        # Remove node from Open and add it to Close
        open_list.remove(node)
        closed_list.append(node)

        # Compare node to the goal
        if compare(node["data"], goal, size):
            print("Success. Exiting...")
            break
        
        nodes_child = []
        # Get child nodes by moving the blank space
        nodes_child = transition(node) # Get node childs with g_score and h_score, parent

        for child in nodes_child:
            open_closed_list = open_list + closed_list

            # Find child configuration if it is present in close and open
            for node_open_closed in open_closed_list:
                # If catch the same configuration
                # BUG: The comparison return True even if the two arrays are different
                if compare(child["data"], node_open_closed["data"], size):
                    print("Child:\n" + str(child["data"]))
                    print("Node:\n" + str(node["data"]))
                    print("\n")
                    if child["f_score"] <= node_open_closed["f_score"]:
                        if node_open_closed in open_list:
                            open_list.append(child)
                            open_list.remove(node_open_closed)
                        else:
                            open_list.append(child)
                            closed_list.remove(node_open_closed)
                else:
                    open_list.append(child)

            # Sort Open
            f = itemgetter('f_score')
            open_list.sort(key=f)
