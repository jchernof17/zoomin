import networkx as nx
from networkx.algorithms.approximation import steiner_tree, min_weighted_dominating_set, min_edge_dominating_set
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path
from networkx.algorithms.mis import maximal_independent_set
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
from generator import generate_tree
import sys
import time
from random import sample, randint

### CONTROL SWITCHES ###

# INPUT FILES
file = "large-66"
START = 1  # Set this to some number between 1 and 303
RUN_LIST_SMALL = False
RUN_LIST_MEDIUM = False
RUN_LIST_LARGE = True

# STRATEGIES
BRUTE_FORCE = False
MAX_SPANNING_TREE = False
DOMINATING_SET = False
MAXIMUM_SUBLISTS = 20000
BRUTE_EDGES = True

# DEBUGGING
TIME_EACH_OUTPUT = True


def subedgelists_from_graph(G):
    lst = list(G.edges)
    double_the_min_number_of_edges = len(min_edge_dominating_set(G))
    upper_bound = min([len(G.nodes) - 1,len(lst)])
    print("analyzing len " + str(double_the_min_number_of_edges) + ":" + str(upper_bound))
    sublists = []
    for _ in range(MAXIMUM_SUBLISTS):
        sublist = sample(lst, k=randint(double_the_min_number_of_edges, upper_bound))
        if sublist not in sublists:
            sublists.append(sublist)

    return sublists


def sublists_from_graph(G):
    lst = list(G.nodes)
    sublists = []
    if len(lst) > 50:
        for node in lst:
            for _ in range(MAXIMUM_SUBLISTS // len(lst)):
                try:
                    MIS_G = maximal_independent_set(G, [node])
                    while len(MIS_G) < min(3, len(lst) / 8):
                        MIS_G = maximal_independent_set(G, [node])
                    sub = list(MIS_G)
                    if sub not in sublists:
                        sublists.append(sub)
                except:
                    pass
    else:
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
    method_start_time = time.perf_counter()
    # load existing graph
    existing_best_score = 0
    if T and T.edges:
        existing_best_score = average_pairwise_distance_fast(T)
    else:  # if we already calculated a score of 0, skip this analysis
        return T
    best_T, best_score = T, existing_best_score

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
    # Edge-based brute force
    if len(G) and BRUTE_EDGES:
        subedgelists = subedgelists_from_graph(G)
        for sublist in subedgelists:
            TEST_T = G.edge_subgraph(sublist).copy()
            # print(sublist)
            if len(TEST_T) and nx.is_tree(TEST_T) and nx.is_dominating_set(G, TEST_T.nodes):
                new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
                if new_score < best_score:
                    best_T = TEST_T
                    best_score = new_score

    # Random guessing/brute force time
    if len(G) and BRUTE_FORCE:
        # create sublists
        sublists = sublists_from_graph(G)
        print("checking on " + str(len(sublists)) + " sublists")
        for lst in sublists:
            # remove sublist nodes from original graph
            TEST_G = G.copy()
            # nodes_to_remove = [node for node in G.nodes if node not in lst]
            # TEST_G.remove_nodes_from(nodes_to_remove)
            # try the steiner tree method
            TEST_T = steiner_tree(TEST_G, lst)
            if len(TEST_T) and nx.is_tree(TEST_T) and nx.is_dominating_set(G, TEST_T.nodes):
                new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
                if new_score < best_score:
                    best_T = TEST_T
                    best_score = new_score

    # max spanning tree
    if len(G) and MAX_SPANNING_TREE:
        MAXST = nx.maximum_spanning_tree(G)
        if len(MAXST) and nx.is_tree(MAXST) and nx.is_dominating_set(G, MAXST.nodes):
                new_score = 0 if not MAXST.edges else average_pairwise_distance_fast(MAXST)
                if new_score < best_score:
                    best_T = MAXST
                    best_score = new_score

    # dominating set time
    if len(G) and DOMINATING_SET:

        DS = list(min_weighted_dominating_set(G))
        TEST_T = steiner_tree(G, DS)
        if len(TEST_T) and nx.is_tree(TEST_T):
            new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
            if new_score < best_score:
                best_T = TEST_T
                best_score = new_score

    if best_score < existing_best_score:
        print("yes ----- "+str(round(-100 * (best_score - existing_best_score)/existing_best_score,2)) + "% ----- new score "+ str(round(best_score,4)))

    else:
        # print("no, " + str(existing_best_score))
        pass
    method_end_time = time.perf_counter()
    time_seconds = method_end_time - method_start_time
    if TIME_EACH_OUTPUT:
        print("time: " + str(round(time_seconds, 2)) + "s")
    return best_T


def run_solver():
    """
    Runs the solve() function on all graphs in the inputs folder and saves outputs
    """
    # start timer
    start_time = time.perf_counter()
    # find files
    num_graphs = []
    # we can just do a subset of graphs if the parameter full is set to False
    sizes = []
    if RUN_LIST_SMALL:
        sizes.append("small")
        num_graphs.append(304)
    if RUN_LIST_MEDIUM:
        sizes.append("medium")
        num_graphs.append(304)
    if RUN_LIST_LARGE:
        sizes.append("large")
        num_graphs.append(401)
    # loop through all inputs and create outputs
    if not file:
        for i in range(len(sizes)):
            size = sizes[i]
            GRAPH_RANGE = range(START, num_graphs[i])
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
