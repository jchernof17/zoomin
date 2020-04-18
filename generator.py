import networkx as nx
from parse import read_input_file, write_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import matplotlib as plt
import random
import sys

def generate_tree(n):
    """
    Args:
        n: int
    Returns:
        T: networkx.Graph [Tree of n vertices]
    """
    T = nx.random_tree(n,1)
    #plt.subplot(121)
    nx.draw(T)
    return T


def generate_graph(n):
    """
    Args:
        n: int

    REturns:
        G: networkx.Graph [Connected graph of n vertices]
    """
    G = nx.fast_gnp_random_graph(n, 0.5)
    while not nx.is_connected(G):
        G = nx.fast_gnp_random_graph(n, 0.5)
    for (u, v) in G.edges():
        weight = round(random.uniform(0.001, 99.999), 3)
        G.edges[u, v]['weight'] = weight
    return G


nums = [25, 50, 100]

for num in nums:
    G = generate_graph(num)
    print(G)
    write_input_file(G, str(num)+".in")
