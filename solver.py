import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
from generator import generate_tree
import sys
import time


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    MST = nx.minimum_spanning_tree(G)
    best_score = average_pairwise_distance_fast(MST)
    # if a node of T has degree 1, let's delete it and calculate the pairwise distance
    BEST_MST = MST
    leaves = [key[0] for key in MST.degree() if key[1] == 1].sort(reverse=True)
    if leaves:
        for leaf in leaves:
            BEST_MST.remove_node(leaf)
            potential_score = average_pairwise_distance_fast(BEST_MST)
            if potential_score > best_score:
                BEST_MST = MST  # change it back
            else:
                MST = BEST_MST  # change our MST

    return BEST_MST


def run_solver(full=True):
    """
    Runs the solve() function on all graphs in the inputs folder and saves outputs
    """
    # start timer
    start_time = time.perf_counter()
    # find files
    sizes = ["small", "medium", "large"]
    num_graphs = [303, 303, 400]
    # we can just do small graphs if the parameter full is set to False
    if not full:
        num_graphs = num_graphs[0:1]

    # loop through all inputs and create outputs
    for i in range(len(num_graphs)):
        for j in range(1, num_graphs[i] + 1):
            filepath = sizes[i]+"-"+str(j)
            G = read_input_file("inputs/"+filepath+".in")
            T = solve(G)
            write_output_file(T, "outputs/"+filepath+".out")
    end_time = time.perf_counter()
    print(f"Process complete. Total time {end_time - start_time:0.4f} seconds")


if __name__ == '__main__':
    run_solver()

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     T = solve(G)
#     assert is_valid_network(G, T)
#     print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
#     write_output_file(T, 'out/test.out')