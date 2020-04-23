import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
from generator import generate_tree
import sys
import time

def sublists_from_list(lst):
    sublists = [[]]
    for i in range(len(lst)+1):
        for j in range(i+1, len(lst)+1):
            sub = lst[i:j]
            sublists.append(sub)

    return sublists


def solve(G, T):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

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

    # Now try getting a 0
    if any([pair[1] >= len(G)-1 for pair in G.degree()]):
        # find that node and return just that
        center_node = max(G.degree, key=lambda pair: pair[1])[0]
        if len(G)-1 == G.degree(center_node) - ((center_node, center_node) in G.edges):
            EMPTY_G = G.copy()
            for e in EMPTY_G.edges:
                EMPTY_G.remove_edge(*e)
            for node in range(len(G)):
                if node != center_node:
                    EMPTY_G.remove_node(node)
            if not EMPTY_G.edges:
                new_score = 0
            else:
                new_score = average_pairwise_distance_fast(EMPTY_G)
            if new_score <= best_score:
                best_T = EMPTY_G
                best_score = new_score
    """
    # load existing graph
    existing_best_score = 0
    if T and T.edges:
        existing_best_score = average_pairwise_distance_fast(T)
    best_T = T
    best_score = existing_best_score
    # Random guessing/brute force time
    if len(G):
        # create sublists
        sublists = sublists_from_list(list(G.nodes))
        for lst in sublists:
            # remove sublist nodes from original graph
            TEST_G = G.copy()
            TEST_G.remove_nodes_from(lst)
            # try the steiner tree method
            if len(TEST_G) and nx.is_connected(TEST_G):
                TEST_T = steiner_tree(TEST_G, TEST_G.nodes)
                if len(TEST_T) and nx.is_tree(TEST_T) and nx.is_dominating_set(G, TEST_T.nodes):
                    new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
                    if new_score < best_score:
                        best_T = TEST_T
                        best_score = new_score
    if best_score < existing_best_score:
        print("yes ----- "+str(-100 * (best_score - existing_best_score)/existing_best_score))

    else:
        print("no, " + str(existing_best_score))
    return best_T


def run_solver(full=False, file=""):
    """
    Runs the solve() function on all graphs in the inputs folder and saves outputs
    """
    # start timer
    start_time = time.perf_counter()
    # find files
    sizes = ["small", "medium", "large"]
    num_graphs = [303, 303, 400]
    # we can just do a subset of graphs if the parameter full is set to False
    sizes = ["medium"]
    GRAPH_RANGE = range(200, 304)
    # loop through all inputs and create outputs
    if not file:
        for size in sizes:
            for j in GRAPH_RANGE:
                filepath = size+"-"+str(j)
                G = read_input_file("inputs/"+filepath+".in")
                print("analyzing "+filepath)
                EXISTING_T = read_output_file("outputs/"+filepath+".out", G)
                T = solve(G, EXISTING_T)
                write_output_file(T, "outputs/"+filepath+".out")
    else:  # file-specific running
        filepath = file
        G = read_input_file("inputs/"+filepath+".in")
        EXISTING_T = read_output_file("outputs/"+filepath+".out", G)
        T = solve(G, EXISTING_T)
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
