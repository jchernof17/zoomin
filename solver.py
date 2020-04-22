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
    best_T = MST.copy()
    best_score = average_pairwise_distance_fast(best_T)
    # if a node of T has degree 1, let's delete it and calculate the pairwise distance

    sources_and_sinks = [pair[0] for pair in MST.degree() if pair[1] == 1]  # get vertices with degree 1
    deletable_edges = [e for e in MST.edges if e[0] in sources_and_sinks or e[1] in sources_and_sinks]
    for edge in deletable_edges:
        new_T = best_T.copy()
        new_T.remove_edge(*edge)
        if not new_T[edge[0]] and len(new_T) > 1 and len(MST[edge[0]]) == 1:
            new_T.remove_node(edge[0])
        if not new_T[edge[1]] and len(new_T) > 1 and len(MST[edge[1]]) == 1:
            new_T.remove_node(edge[1])
        if not new_T.edges:
            new_score = 0
        else:
            new_score = average_pairwise_distance_fast(new_T)
        if new_score <= best_score:
            best_T = new_T
            best_score = new_score

    return best_T


def run_solver(full=True, file=""):
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
    if not file:
        for i in range(len(num_graphs)):
            for j in range(1, num_graphs[i] + 1):
                filepath = sizes[i]+"-"+str(j)
                G = read_input_file("inputs/"+filepath+".in")
                T = solve(G)
                write_output_file(T, "outputs/"+filepath+".out")
    else:
        filepath = file
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
