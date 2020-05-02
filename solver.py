import networkx as nx
from networkx.algorithms.approximation import steiner_tree, min_edge_dominating_set, min_weighted_dominating_set
from networkx.algorithms.shortest_paths.weighted import dijkstra_path
from networkx.algorithms.mis import maximal_independent_set
from parse import read_input_file, write_output_file, read_output_file
from utils import average_pairwise_distance_fast, edge_lower_bound, edge_upper_bound
from joblib import Parallel, delayed
import multiprocessing
#inferior_outputs = ""
#import inferior_outputs
import time
from random import sample, randint, choices

# CONTROL SWITCHES ###

# INPUT FILES
improvable = ["small-1", "small-7", "small-15", "small-16", "small-17", 
"small-18", "small-27", "small-41", "small-43", "small-45", "small-55",
"small-66", "small-71", "small-72", "small-75", "small-78", 
"small-89", "small-95", "small-99", "small-117", "small-121",
"small-126", "small-129", "small-131", "small-133", "small-136", "small-144",
"small-161", "small-166", "small-177", "small-178", 
"small-194", "small-205", "small-206", "small-213", "small-217", 
"small-227", "small-228", "small-231", "small-234", 
"small-237", "small-239", "small-242", "small-253", "small-258", "small-260", 
"small-269", "small-274", "small-278","small-287", "small-291", "small-301", 
"medium-1", "medium-4",
"medium-6", "medium-7", "medium-11", "medium-15", "medium-16",
"medium-17", "medium-18", "medium-21", "medium-23",
"medium-26", "medium-27", "medium-28", "medium-29", "medium-30", "medium-31", "medium-34", 
"medium-35", "medium-37", "medium-38", "medium-39", "medium-40", "medium-41", "medium-42", 
"medium-43", "medium-44", "medium-45", "medium-48", "medium-49", "medium-51", "medium-52", 
"medium-55", "medium-56", 'medium-58', 'medium-61', 'medium-63', 'medium-64', 'medium-65', 
'medium-66', 'medium-67', 'medium-71', 'medium-72', 'medium-75', 'medium-77', 'medium-78', 
'medium-80', 'medium-81', 'medium-82', 'medium-83', 'medium-87', 'medium-89', 'medium-92', 
'medium-95', 'medium-99', 'medium-100', 'medium-101', 'medium-104', 'medium-106', 'medium-113', 
'medium-114', 'medium-115', 'medium-117', 'medium-118', 'medium-119', 'medium-120', 'medium-121', 
'medium-124', 'medium-126', 'medium-127', 'medium-128', 'medium-129', 'medium-130', 'medium-131', 
'medium-133', 'medium-134', 'medium-135', 'medium-136', 'medium-137', 'medium-139', 'medium-141', 
'medium-143', 'medium-146', 'medium-147', 'medium-150', 'medium-154', 'medium-155', 'medium-156', 
'medium-158', 'medium-160', 'medium-161', 'medium-162', 'medium-163', 'medium-166', 'medium-167', 
'medium-168', 'medium-169', 'medium-172', 'medium-173', 'medium-174', 'medium-176', 'medium-177',
'medium-178', 'medium-179', 'medium-181', 'medium-182', 'medium-183', 'medium-185', 'medium-186', 
'medium-187', 'medium-188', 'medium-192', 'medium-194', 'medium-195', 'medium-196', 'medium-199', 
'medium-200', 'medium-201', 'medium-202', 'medium-206', 'medium-207', 'medium-210', 'medium-211', 
'medium-212', 'medium-213', 'medium-214', 'medium-215', 'medium-217', 'medium-220', 'medium-222', 
'medium-225', 'medium-226', 'medium-230', 'medium-231', 'medium-232', 'medium-236', 'medium-237', 
'medium-238', 'medium-239', 'medium-241', 'medium-242', 'medium-246', 'medium-247', 'medium-250', 
'medium-251', 'medium-253', 'medium-254', 'medium-255', 'medium-256', 'medium-258', 'medium-261', 
'medium-262', 'medium-264', 'medium-266', 'medium-267', 'medium-269', 'medium-270', 'medium-271',
'medium-274', 'medium-275', 'medium-278', 'medium-279', 'medium-285', 'medium-286', 'medium-287', 
'medium-288', 'medium-291', 'medium-293', 'medium-294', 'medium-295', 'medium-297', 'medium-299', 
'medium-301', 'large-1', 'large-3', 'large-4', 'large-6', 'large-7', 'large-8', 'large-10', 
'large-11', 'large-12', 'large-14', 'large-15', 'large-16', 'large-17', 'large-18', 'large-20', 
'large-21', 'large-22', 'large-23', 'large-24', 'large-25', 'large-26', 'large-27', 'large-28', 
'large-30', 'large-31', 'large-34', 'large-35', 'large-37', 'large-38', 'large-39', 'large-40',
'large-41', 'large-42', 'large-43', 'large-44', 'large-45', 'large-47', 'large-48', 'large-49', 
'large-51', 'large-52', 'large-53', 'large-55', 'large-56', 'large-57', 'large-58', 'large-59', 
'large-61', 'large-63', 'large-64', 'large-66', 'large-67', 'large-70', 'large-72', 'large-73', 
'large-75', 'large-77', 'large-78', 'large-79', 'large-80', 'large-81', 'large-82', 'large-83', 
'large-84','large-86', 'large-87', 'large-89', 'large-91', 'large-92', 'large-93', 'large-95', 
'large-96', 'large-97', 'large-99', 'large-100', 'large-101', 'large-103', 'large-104', 'large-105',
'large-106', 'large-108', 'large-112', 'large-113', 'large-114', 'large-117', 'large-118', 
'large-119', 'large-120', 'large-121', 'large-122', 'large-124', 'large-126', 'large-127', 
'large-128', 'large-129', 'large-130', 'large-131', 'large-133', 'large-134', 'large-135', 
'large-136', 'large-137', 'large-139', 'large-141', 'large-143', 'large-145', 'large-146', 
'large-147', 'large-150', 'large-151', 'large-153', 'large-154', 'large-155', 'large-156', 
'large-158', 'large-160', 'large-161', 'large-162', 'large-163', 'large-166', 'large-167',
'large-168', 'large-169', 'large-171', 'large-172', 'large-173', 'large-174', 'large-176', 
'large-177', 'large-178', 'large-180', 'large-181', 'large-182', 'large-183', 'large-184', 
'large-185', 'large-186', 'large-187', 'large-188', 'large-191', 'large-192', 'large-193', 
'large-195', 'large-196', 'large-198', 'large-199', 'large-200', 'large-201', 'large-202', 
'large-205', 'large-206', 'large-207', 'large-209', 'large-211', 'large-212', 'large-213', 
'large-214', 'large-215', 'large-217', 'large-218', 'large-219', 'large-220', 'large-222', 
'large-225', 'large-226', 'large-228', 'large-230', 'large-231', 'large-232', 'large-233', 
'large-234', 'large-237', 'large-238', 'large-239', 'large-241', 'large-242', 'large-243', 
'large-244', 'large-246', 'large-247', 'large-250', 'large-251', 'large-252', 'large-253', 
'large-254', 'large-255', 'large-256', 'large-258', 'large-260', 'large-261', 'large-262', 
'large-264', 'large-265', 'large-266', 'large-267', 'large-268', 'large-269', 'large-270', 
'large-271', 'large-272', 'large-273', 'large-274', 'large-275', 'large-278', 'large-279', 
'large-282', 'large-283', 'large-284', 'large-285', 'large-286', 'large-287', 'large-288', 
'large-290', 'large-291', 'large-293', 'large-294', 'large-295', 'large-297', 'large-299', 
'large-300', 'large-301', 'large-302', 'large-304', 'large-305', 'large-306', 'large-307', 
'large-308', 'large-309', 'large-310', 'large-311', 'large-312', 'large-313', 'large-314', 'large-315', 'large-316', 'large-317', 'large-318', 'large-319', 'large-320', 'large-321', 'large-322', 'large-323', 'large-324', 'large-325', 'large-326', 'large-327', 'large-328', 'large-329', 'large-330', 'large-331', 'large-332', 'large-333', 'large-334', 'large-335', 'large-336', 'large-337', 'large-338', 'large-339', 'large-340', 'large-341', 'large-342', 'large-343', 'large-344', 'large-345', 'large-346', 'large-347', 'large-348', 'large-349', 'large-350', 'large-351', 'large-352', 'large-353', 'large-354', 'large-355', 'large-356', 'large-357', 'large-358', 'large-359', 'large-360', 'large-361', 'large-362', 'large-363', 'large-364', 'large-365', 'large-366', 'large-367', 'large-368', 'large-369', 'large-370', 'large-371', 'large-372', 'large-373', 'large-374', 'large-375', 'large-376', 'large-377', 'large-378', 'large-379', 'large-380', 'large-381', 'large-382', 'large-383', 'large-384', 'large-385', 'large-386', 'large-387', 'large-388', 'large-389', 'large-390', 'large-391', 'large-392', 'large-393', 'large-394', 'large-395', 'large-396', 'large-397', 'large-398', 'large-399', 'large-400']
bad_small = [file for file in improvable if "small" in file]
bad_medium = [file for file in improvable if "medium" in file]
bad_large = [file for file in improvable if "large" in file]

# Split up the large file list into four subsections for easier parallelizing
size = len(bad_large) // 4
bad_large_1 = bad_large[:size]
bad_large_2 = bad_large[size:2 * size]
bad_large_3 = bad_large[2 * size:3 * size]
bad_large_4 = bad_large[3 * size:]
file = ""
START = 1  # Set this to some number between 1 and 303
RUN_LIST_SMALL = True
RUN_LIST_MEDIUM = False
RUN_LIST_LARGE_1 = False
RUN_LIST_LARGE_2 = False
RUN_LIST_LARGE_3 = False
RUN_LIST_LARGE_4 = False
ONLY_RUN_IMPROVABLE = True  # don't you dare set this to false...

# STRATEGIES
BRUTE_FORCE = True
MAX_SPANNING_TREE = False
DOMINATING_SET = False
MAXIMUM_SUBLISTS = 65536
MAX_SECONDROUND_SUBLISTS = 1024
BRUTE_EDGES = True
EDGE_TINKERING = True
KRUSKAL_STARTER = True
TRY_SMALL_NUM_EDGES = False
LARGE_SHORTEST_PATH = False
DISPLAY_HUD = False

# DEBUGGING
TIME_EACH_OUTPUT = True
SHOW_UPDATE_RESULT = True


def subedgelists_from_graph(G, T=None):
    # lst = sorted(list(G.edges), key=lambda e: e[0])
    lst = sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1))
    weights = [1 / x[2].get('weight') for x in lst]
    #print(str(weights[0]) + "\t" + str(weights[len(lst) - 1]))
    density = nx.density(G)
    # the maximum degree of any node in the graph G
    max_degree = max([out[1] for out in nx.degree(G)])
    # there is no way the number of edges is less than the power max_degree has to be raised to in order to reach the number of vertices!
    lower_bound = max([len(G) // max_degree - 1, 1])
    # there is no way the number of edges is >= the number of vertices!
    upper_bound = min([len(G) - 1, len(lst)])
    if T:
        if len(T) < 15:  # dense graph
            lower_bound = max([lower_bound, int(len(list(T.edges)) - 6)])
            upper_bound = min([upper_bound, int(len(list(T.edges)) + 6)])
        else:
            lower_bound = max([lower_bound, int(len(list(T.edges)) * 0.7)])
            upper_bound = min([upper_bound, int(len(list(T.edges)) * 1.3)])
    #lower_bound = max([0, edge_lower_bound(G)])
    #upper_bound = min([edge_upper_bound(G), 99])
    # print("analyzing len " + str(double_the_min_number_of_edges) + ":" + str(upper_bound))
    sublists = []
    #print("edge: \t (" + str(lower_bound) + ") \t (" + str(upper_bound) + ") \t (" + str(len(T) - 1) + ")")
    # print("currently have " + str(len(list(T.edges))) + " in best tree, so searching in [" + str(lower_bound) + ":" + str(upper_bound) + "]")
    for _ in range(MAXIMUM_SUBLISTS):
        # sublist = sorted(sample(lst, k=randint(lower_bound, upper_bound)), key=lambda e: (e[0], e[1]))
        sublist = []
        vertex_set = {}
        max_tries = 0
        # max_list_size = randint(lower_bound, upper_bound)
        while max_tries < 2500 and not nx.is_dominating_set(G, vertex_set) and len(vertex_set) < upper_bound:
            edge = choices(lst, weights=weights)[0]
            max_tries += 1
            if not (edge[0] in vertex_set) or not (edge[1] in vertex_set):
                tup = (edge[0], edge[1])
                sublist.append(tup)
                vertex_set[edge[0]] = vertex_set[edge[1]] = 1
        if nx.is_dominating_set(G, vertex_set):
            sublist = sorted(sublist)
        if sublist not in sublists:
            sublists.append(sublist)

    return sublists


def sublists_from_graph(G, max_iters=MAXIMUM_SUBLISTS, T=""):
    lst = list(G.nodes)
    sublists = []
    max_degree = max([out[1] for out in nx.degree(G)])
    nodes_in_degree_order = sorted(list(G.nodes), key=lambda v: G.degree[v])
    weights = [G.degree[v] for v in nodes_in_degree_order]
    if len(lst) > 50 and False:
        for node in lst:
            for _ in range(max_iters // len(lst)):
                try:
                    MIS_G = maximal_independent_set(G, [node])
                    while len(MIS_G) < min(3, len(lst) / 8):
                        MIS_G = sorted(maximal_independent_set(G, [node]))
                    sub = list(MIS_G)
                    if sub not in sublists:
                        sublists.append(sub)
                except:
                    pass
    else:
        lower_bound = min([len(G), len(G) // max_degree])
        upper_bound = len(G)
        if T:
            lower_bound = max([lower_bound, (len(T) * 0.5)])
            upper_bound = min([upper_bound, (len(T) * 1.3)])
        density = nx.density(G)
        #lower_bound = max([0, edge_lower_bound(G)])
        #upper_bound = min([edge_upper_bound(G), 99])
        #print("nodes: \t (" + str(lower_bound) + ") \t (" + str(upper_bound) + ") \t (" + str(len(T) - 1) + ")")
        for _ in range(max_iters):
            num_nodes = randint(int(lower_bound), int(upper_bound))
            sublist = sorted(choices(nodes_in_degree_order, weights=weights, k=num_nodes))
            if sublist not in sublists:
                sublists.append(sublist)
    return sublists

def display_stats(G, T, filename=""):
    number_of_nodes = G.number_of_nodes()
    number_of_edges = len(list(G.edges))
    nodes_in_tree = T.number_of_nodes()
    edges_in_tree = len(list(T.edges))
    density = round(100 * nx.density(G), 2)
    node_ratio = round(100 * nodes_in_tree / number_of_nodes, 2)
    edge_ratio = round(100 * edges_in_tree / number_of_edges, 2)
    # print("(" + filename + ") \t (density) = (" + str(density) + ") \t (vr, er) = (" + str(node_ratio) + "," + str(edge_ratio) + ")")
    # print("("+filename+") \t (density) = (" + str(number_of_nodes)+","+str(number_of_edges)+") \t (vr, er) = (" + node_ratio + "," + edge_ratio + ")")
    return density, node_ratio, edge_ratio, number_of_nodes

def solve(G, T, filename=""):
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
    edges_of_G = list(G.edges)

    if len(G) and TRY_SMALL_NUM_EDGES:
        # print("(" + filename + ") - attempting small edges selection")
        small_lsts = []
        max_degree_vertex = max([out[0] for out in nx.degree(G)], key=lambda v: G.degree[v])
        edges_of_max_vertex = list(G.edges([max_degree_vertex]))
        neighbors_of_max_vertex = sorted(list(G.neighbors(max_degree_vertex)), key=lambda v: G.degree[v])
        for neighbor in neighbors_of_max_vertex:
            if neighbor != max_degree_vertex and nx.is_dominating_set(G, [neighbor, max_degree_vertex]):
                # create that edge
                edge = (max_degree_vertex, neighbor)
                TEST_T = G.edge_subgraph([edge]).copy()
                new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
                if new_score < best_score:
                    best_T = TEST_T
                    best_score = new_score
                    # If we get a record, we continue trying to improve
                    if SHOW_UPDATE_RESULT:
                        print("(" + filename + ") ___ improvement of " + str(round(-100 * (best_score - existing_best_score)/existing_best_score, 2)) + "%" + " detected (small num edges)")



    # Kruskal-like method (doesn't work yet)
    if len(G) > 10 and KRUSKAL_STARTER:
        # Strategy: add the smallest edge that adds 1 node to the tree
        edges = sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1))
        # print(edges)
        nodes = best_T.nodes
        one_in_tree = [(e[0], e[1]) for e in edges if (e[0] in nodes) ^ (e[1] in nodes)]
        i = 0
        while i < len(one_in_tree):
            edge = one_in_tree[i]
            T_edges = list(best_T.edges)
            T_edges.append(edge)
            TEST_T = G.edge_subgraph(T_edges).copy()
            if len(TEST_T) and nx.is_tree(TEST_T) and nx.is_dominating_set(G, TEST_T.nodes):
                new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
                if new_score < best_score:
                    best_T = TEST_T
                    best_score = new_score
                    nodes = best_T.nodes
                    one_in_tree = [(e[0], e[1]) for e in edges if (e[0] in nodes) ^ (e[1] in nodes)]
                    i = 0
                    # If we get a record, we continue trying to improve
                    if SHOW_UPDATE_RESULT:
                        print("(" + filename + ") ___ improvement of " + str(round(-100 * (best_score - existing_best_score)/existing_best_score, 2)) + "%" + " detected (Kruskal)")
            i = i + 1

    # Replace Large Edges
    if len(G) and LARGE_SHORTEST_PATH:
        # Replace the largest edges in the tree with a shorter path between the nodes
        edges = sorted(best_T.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True)
        iterations = len(edges)
        for i in range(iterations):
            largest_edge = edges[0]
            edges = edges[1:]
            largest_edge = (largest_edge[0], largest_edge[1])
            T_edges = list(best_T.edges)
            T_edges.remove(largest_edge)
            source, target = largest_edge[0], largest_edge[1]
            path = nx.dijkstra_path(G, source, target)
            new_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            T_edges.extend(new_edges)
            TEST_T = G.edge_subgraph(T_edges).copy()
            if len(TEST_T) and nx.is_tree(TEST_T) and nx.is_dominating_set(G, TEST_T.nodes):
                new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
                if new_score < best_score:
                    best_T = TEST_T
                    best_score = new_score
                    if SHOW_UPDATE_RESULT:
                        print("(" + filename + ") ___ improvement of " + str(round(-100 * (best_score - existing_best_score)/existing_best_score, 2)) + "%" + " detected (Replace Large)")

    # Edge Tinkering method
    if len(G) and EDGE_TINKERING:
        candidate_nodes = [node for node in G.nodes if node not in T.nodes]
        candidate_edges = sorted([e for e in G.edges if e not in list(T.edges) and ((e[0] in candidate_nodes) ^ (e[1] in candidate_nodes))])
        # print(list(T.edges))
        max_iters = MAXIMUM_SUBLISTS
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
                    max_iters += MAX_SECONDROUND_SUBLISTS
                    recalculate = True
                    if SHOW_UPDATE_RESULT:
                        print("(" + filename + ") ___ improvement of " + str(round(-100 * (best_score - existing_best_score) / existing_best_score, 2)) + "%" + " detected (Edge Tinkering)")
            i += 1

    # Edge-based brute force
    if len(G) and BRUTE_EDGES:
        subedgelists = subedgelists_from_graph(G, best_T)
        for sublist in subedgelists:
            TEST_T = G.edge_subgraph(sublist).copy()
            # print(sublist)
            if len(TEST_T) and nx.is_tree(TEST_T) and nx.is_dominating_set(G, TEST_T.nodes):
                new_score = 0 if not TEST_T.edges else average_pairwise_distance_fast(TEST_T)
                if new_score < best_score:
                    if SHOW_UPDATE_RESULT:
                        print("(" + filename + ") _EDGE_ improvement of " + str(round(-100 * (new_score - existing_best_score) / existing_best_score, 2)) + "%" + " detected (" + str(len(list(best_T.edges))) + " --> " + str(len(list(TEST_T.edges))) + ")")
                    best_T = TEST_T
                    best_score = new_score
    # Random guessing/brute force
    if len(G) and BRUTE_FORCE:
        # create sublists
        sublists = sublists_from_graph(G, T=best_T)
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
                    if SHOW_UPDATE_RESULT:
                        print("(" + filename + ") _NODE_ improvement of " + str(round(-100 * (new_score - existing_best_score) / existing_best_score, 2)) + "%" + " detected (" + str(len(best_T)) + " --> " + str(len(TEST_T)) + ")")
                    best_T = TEST_T
                    best_score = new_score
                    # If we get a record, we continue trying to improve
                    sublists.extend(sublists_from_graph(G, max_iters=MAX_SECONDROUND_SUBLISTS, T=best_T))
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

    if not nx.is_dominating_set(G, best_T) or not nx.is_tree(best_T):  # uh oh
        best_T = nx.minimum_spanning_tree(G)
    if best_score < existing_best_score and SHOW_UPDATE_RESULT:
        print("|yes ___ (" + filename + ") " + str(round(-100 * (best_score - existing_best_score) / existing_best_score, 2)) + "% ----- new score " + str(round(best_score, 4)))

    elif SHOW_UPDATE_RESULT:
        # print("no, " + str(round(existing_best_score,2)))
        pass
    method_end_time = time.perf_counter()
    time_seconds = method_end_time - method_start_time
    if TIME_EACH_OUTPUT:
        print("time: " + str(round(time_seconds, 2)) + "s")
    return best_T

def stats_summarizer():
    results = []
    if DISPLAY_HUD:
        for filename in bad_small:
            G = read_input_file("inputs/"+ filename +".in")
            T = read_output_file("outputs/"+filename+".out", G)
            res = display_stats(G, T, filename)
            results.append(res)
    results = sorted(results, key=lambda r: r[0])  # sort by densities
    for result in results:
        print(str(result[0]) + "\t" + str(result[1]))

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
    # if RUN_LIST_LARGE:
    #     sizes.append("large")
    #     num_graphs.append(401)
    # loop through all inputs and create outputs
    num_cores = multiprocessing.cpu_count()
    outputs = []

    def solver(size="", index="", filename=""):
        if not size and not index:
            filepath = filename
        else:
            filepath = size + "-" + str(index)
        G = read_input_file("inputs/"+ filepath +".in")
        print("analyzing "+filepath)
        EXISTING_T = read_output_file("outputs/"+filepath+".out", G)
        # outputs.append((solve(G, EXISTING_T), filepath))
        write_output_file(solve(G, EXISTING_T, filename=filepath), "outputs/"+filepath+".out")

    if not file and ONLY_RUN_IMPROVABLE:
        if RUN_LIST_SMALL:
            Parallel(n_jobs=num_cores)(delayed(solver)(filename=file) for file in bad_small)
        if RUN_LIST_MEDIUM:
            Parallel(n_jobs=num_cores)(delayed(solver)(filename=file) for file in bad_medium)
        if RUN_LIST_LARGE_1:
            Parallel(n_jobs=num_cores)(delayed(solver)(filename=file) for file in bad_large_1)
        if RUN_LIST_LARGE_2:
            Parallel(n_jobs=num_cores)(delayed(solver)(filename=file) for file in bad_large_2)
        if RUN_LIST_LARGE_3:
            Parallel(n_jobs=num_cores)(delayed(solver)(filename=file) for file in bad_large_3)
        if RUN_LIST_LARGE_4:
            Parallel(n_jobs=num_cores)(delayed(solver)(filename=file) for file in bad_large_4)

    elif not file and not ONLY_RUN_IMPROVABLE:
        for i in range(len(sizes)):
            size = sizes[i]
            GRAPH_RANGE = range(START, num_graphs[i])
            Parallel(n_jobs=num_cores)(delayed(solver)(size, j) for j in GRAPH_RANGE)
    elif file:  # file-specific running
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
    if DISPLAY_HUD:
        stats_summarizer()
    else:
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
