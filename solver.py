import networkx as nx
from networkx.algorithms.approximation import steiner_tree, min_edge_dominating_set, min_weighted_dominating_set
# from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path
from networkx.algorithms.mis import maximal_independent_set
from parse import read_input_file, write_output_file, read_output_file
from utils import average_pairwise_distance_fast
from joblib import Parallel, delayed
import multiprocessing

import time
from random import sample, randint

# CONTROL SWITCHES ###

# INPUT FILES
improvable_small = [1, 4, 7, 8, 10, 11, 15, 16, 17, 18, 20, 23, 24, 27, 28, 30, 31, 33, 34, 35, 37, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 51, 52, 55, 56, 58, 61, 63, 64, 65, 66, 67, 70, 71, 72, 75, 77, 78, 80, 81, 82, 83, 84, 87, 88, 89, 92, 95, 97, 99, 103, 104, 105, 108, 113, 116, 117, 118, 119, 121, 122, 124, 126, 128, 129, 130, 131, 133, 134, 135, 136, 137, 139, 141, 142, 143, 144, 146, 147, 149, 150, 151, 153, 155]
file = ""
START = 1  # Set this to some number between 1 and 303
RUN_LIST_SMALL = True
RUN_LIST_MEDIUM = False
RUN_LIST_LARGE = False

# STRATEGIES
BRUTE_FORCE = True
MAX_SPANNING_TREE = False
DOMINATING_SET = False
MAXIMUM_SUBLISTS = 1000
MAX_SECONDROUND_SUBLISTS = 2000
BRUTE_EDGES = False
EDGE_TINKERING = False
KRUSKAL_STARTER = False

# DEBUGGING
TIME_EACH_OUTPUT = True
SHOW_UPDATE_RESULT = True


def subedgelists_from_graph(G):
    lst = list(G.edges)
    # the maximum degree of any node in the graph G
    max_degree = max([out[1] for out in nx.degree(G)])
    # there is no way the number of edges is less than the power max_degree has to be raised to in order to reach the number of vertices!
    lower_bound = max([len(G) // max_degree - 1, 1])

    # there is no way the number of edges is >= the number of vertices!
    upper_bound = min([len(G) - 1, len(lst)])
    # print("analyzing len " + str(double_the_min_number_of_edges) + ":" + str(upper_bound))
    sublists = []
    for _ in range(MAXIMUM_SUBLISTS):
        sublist = sample(lst, k=randint(lower_bound, upper_bound))
        if sublist not in sublists:
            sublists.append(sublist)

    return sublists


def sublists_from_graph(G, max_iters=MAXIMUM_SUBLISTS):
    lst = list(G.nodes)
    sublists = []
    max_degree = max([out[1] for out in nx.degree(G)])
    if len(lst) > 50 and False:
        for node in lst:
            for _ in range(max_iters // len(lst)):
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
        for _ in range(max_iters):
            sublist = sample(G.nodes, randint(min([len(G), len(G) // max_degree]), len(G)))
            if sublist not in sublists:
                sublists.append(sublist)
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

    # Kruskal-like method (doesn't work yet)
    if len(G) > 10 and len(T) > 4 and KRUSKAL_STARTER:
        # Strategy: start with the best edges, connect the rest of the tree, and see what we've got
        edges = sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1))
        # print(edges)
        edge = edges.pop()
        edge = edges.pop()
        TEST_G = G.edge_subgraph([edge[:2]]).copy()
        while len(TEST_G) < len(T) // 2:
            TEST_G.add_edges_from([edges.pop()])
        TEST_T = steiner_tree(G, TEST_G.nodes)
        if len(TEST_T) and nx.is_tree(TEST_T) and nx.is_dominating_set(G, TEST_T.nodes):
            new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
            if new_score < best_score:
                best_T = TEST_T
                best_score = new_score
                # If we get a record, we continue trying to improve
                if SHOW_UPDATE_RESULT:
                    print("|___improvement of " + str(round(-100 * (best_score - existing_best_score)/existing_best_score, 2)) + "%" + " detected")

    # Edge Tinkering method
    if len(G) and EDGE_TINKERING:
        candidate_nodes = [node for node in G.nodes if node not in T.nodes]
        candidate_edges = sorted([e for e in G.edges if e not in list(T.edges) and ((e[0] in candidate_nodes) ^ (e[1] in candidate_nodes))])
        # print(list(T.edges))
        max_iters = 10000
        i = 0
        recalculate = False
        while i < max_iters:
            if recalculate:
                candidate_nodes = [node for node in G.nodes if node not in best_T.nodes]
                candidate_edges = sorted([e for e in G.edges if e not in list(best_T.edges) and ((e[0] in candidate_nodes) ^ (e[1] in candidate_nodes))])
                recalculate = False
            add_edge_sample = sample(candidate_edges, k=randint(min([1, len(candidate_edges)]), min([12, len(candidate_edges)])))
            add_edge_sample = []
            # print(edge_sample)
            edge_list = list(T.edges)
            for edge in add_edge_sample:
                edge_list.append(edge)
            # print(edge_list)
            TEST_T = G.edge_subgraph(edge_list).copy()
            nodes_of_degree_one = [node[0] for node in G.degree if node[1] == 1]
            deletable_edges = [e for e in TEST_T.edges if ((e[0] in nodes_of_degree_one) ^ e[1] in nodes_of_degree_one)]

            remove_edge_sample = sample(deletable_edges, k=randint(0, min([len(deletable_edges) // 2, 4])))
            TEST_T.remove_edges_from(remove_edge_sample)

            if len(TEST_T) and nx.is_tree(TEST_T) and nx.is_dominating_set(G, TEST_T.nodes):
                    new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
                    if new_score < best_score:
                        if SHOW_UPDATE_RESULT:
                            print("|")
                        best_T = TEST_T
                        best_score = new_score
                        max_iters += 3000
                        recalculate = True
            i += 1

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

    # Random guessing/brute force
    if len(G) and BRUTE_FORCE:
        # create sublists
        sublists = sublists_from_graph(G)
        # Print for debug only
        # print("checking on " + str(len(sublists)) + " sublists")
        i = 0
        while i < len(sublists):
            lst = sublists[i]
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
                    # If we get a record, we continue trying to improve
                    if SHOW_UPDATE_RESULT:
                        print("|___improvement of " + str(round(-100 * (best_score - existing_best_score)/existing_best_score, 2)) + "%" + " detected")
                    sublists.extend(sublists_from_graph(G, max_iters=MAX_SECONDROUND_SUBLISTS))
            i += 1

    # max spanning tree
    if len(G) and MAX_SPANNING_TREE:
        MAXST = nx.maximum_spanning_tree(G)
        if len(MAXST) and nx.is_tree(MAXST) and nx.is_dominating_set(G, MAXST.nodes):
                new_score = 0 if not MAXST.edges else average_pairwise_distance_fast(MAXST)
                if new_score < best_score:
                    best_T = MAXST
                    best_score = new_score

    # dominating set 
    if len(G) and DOMINATING_SET:

        DS = list(min_weighted_dominating_set(G))
        TEST_T = steiner_tree(G, DS)
        if len(TEST_T) and nx.is_tree(TEST_T):
            new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
            if new_score < best_score:
                best_T = TEST_T
                best_score = new_score

    if best_score < existing_best_score and SHOW_UPDATE_RESULT:
        print("|yes ----- " + str(round(-100 * (best_score - existing_best_score)/existing_best_score, 2)) + "% ----- new score "+ str(round(best_score,4)))

    elif SHOW_UPDATE_RESULT:
        # print("no, " + str(round(existing_best_score,2)))
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
    num_cores = multiprocessing.cpu_count()
    outputs = []

    def solver(size, index):
        filepath = size+"-"+str(index)
        G = read_input_file("inputs/"+filepath+".in")
        print("analyzing "+filepath)
        EXISTING_T = read_output_file("outputs/"+filepath+".out", G)
        # outputs.append((solve(G, EXISTING_T), filepath))
        write_output_file(solve(G, EXISTING_T), "outputs/"+filepath+".out")

    if not file:
        for i in range(len(sizes)):
            size = sizes[i]
            GRAPH_RANGE = range(START, num_graphs[i])
            Parallel(n_jobs=num_cores)(delayed(solver)(size, j) for j in improvable_small)
            '''
            for j in improvable_small:
                filepath = size+"-"+str(j)
                G = read_input_file("inputs/"+filepath+".in")
                print("analyzing "+filepath)
                EXISTING_T = read_output_file("outputs/"+filepath+".out", G)
                # outputs.append((solve(G, EXISTING_T), filepath))
                write_output_file(solve(G, EXISTING_T), "outputs/"+filepath+".out")
            '''
    else:  # file-specific running
        filepath = file
        G = read_input_file("inputs/"+filepath+".in")
        EXISTING_T = read_output_file("outputs/"+filepath+".out", G)
        T = solve(G, EXISTING_T)
        write_output_file(T, "outputs/"+filepath+".out")
    for output in outputs:
        # write_output_file(output[0], "outputs/"+output[1]+".out")
        pass
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
