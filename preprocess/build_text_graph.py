
import os
import sys
from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader
import networkx as nx
# This file is for constructing text graphs, which are intended to be added to the data file for training.

def buildGraph(nodes,edges):

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)
    for edge in edges:
        G.add_edge(nodes[int(edge[0])-1],nodes[int(edge[1])-1])
    return G


if __name__ == '__main__':

    states = torch.load("") # your states node file
    text_graphs = []
    for item in states:
        nodes,graphs = item
        graph = buildGraph(nodes,graphs)
        text_graphs.append(text_graphs)

    torch.save(text_graphs,"") # your text graphs filename
            
    print('\n done!')
        

    
