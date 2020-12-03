import sys
import numpy as np
from operator import itemgetter
import time

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
            print("%5d"%col, end ='')
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


# Get Manhattan distance / Taxicab Geometry
def get_h_mannhatan(data, size):
    count = 0
    for i in range(0, size):
        for j in range(0, size):
            if data[i][j] != 0:
                [x, y] = get_good_position(data[i][j], size)
                if (x != i) or (y != j):
                    count += abs(x-i) + abs(y-j)
    return count

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
                h_score=get_h_mannhatan(data=data_new, size=node["size"]),
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


# Check if a configuration exists in node list
def checkExist(node, nodelist):
    for node_temp in nodelist:
        if compare(node["data"], node_temp["data"], size):
            return node_temp,True
    return [],False


def printResult(node, size):
    space = size * 5 // 2
    if node["parent"]  != None:
        printResult(node["parent"],space)
    print("\n")
    print_puzzle(node['data'])


# Insert child to OpenList and sort
def addChildToOpenList(list, child):
    index = 0
    for cur in list:
        if child['f_score'] > cur['f_score']:
            index += 1
        else:
            break
    list.insert(index, child)

# Main functions
# https://www.geeksforgeeks.org/a-search-algorithm/
if __name__ == "__main__":

    # Timer
    start_time = time.time()

    # Metrics
    discovered_node = 1
    visited_node = 1
    
    size = 3
    # Print generated puzzle
    start_data = generate_puzzle(size)
    print("\nGenerated puzzle:")
    print_puzzle(start_data)
    
    start_data = [[3, 6, 4], [8, 2, 0], [1, 7, 5]]
    # start_data = [[0,9,7,4],[10,15,3,6],[2,13,1,8],[11,5,14,12]]

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
    
    # Put start_node in Open list
    open_list.append(start_node)

    # Main loop
    while (len(open_list) != 0):
        # Sort Open List
        f = itemgetter('f_score')
        open_list.sort(key=f)

        
        # Get node as the first element in Open
        visited_node += 1
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
        discovered_node += len(nodes_child)

        for child in nodes_child:
            child_found_closed, is_exist_in_close = checkExist(child, closed_list)

            # If child in Closed list, ignore this child
            if is_exist_in_close:
                continue

            # If child in Open list, but this child not better than the one in Open, ignore this child
            child_found_open, is_exist_in_open = checkExist(child, open_list)
            if is_exist_in_open:
                if child["f_score"] > child_found_open["f_score"]:
                    continue
                # else:
                #     open_list.append(child)
                #     open_list.remove(child_found_open)

            # Else, append the child in Open
            addChildToOpenList(open_list, child)


    print("\n--- Execution time: %s seconds ---" % (time.time() - start_time))
    printResult(node, size)

# 1    7    0
# 6    8    2
# 4    3    5



# 4    5    6
# 7    8    1
# 0    3    2