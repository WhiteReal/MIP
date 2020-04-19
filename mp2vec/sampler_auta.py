#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : White
# @Email   : baipz1993@gmail.com
# @File    : sampler_auta.py

import numpy as np
import random
import time
import tqdm
import dgl
import torch
import torch_geometric.utils as utils
import sys
import os


num_walks_per_node = 1000
walk_length = 100


def construct_graph(p_p_g, a_a_g, p_a_g):
    p_p_edges = p_p_g.edge_index
    p_p_edges = utils.sort_edge_index(p_p_edges)[0]
    p_p_edges = utils.to_undirected(p_p_edges)
    p_p_edges = utils.remove_self_loops(p_p_edges)[0]
    a_a_edges = a_a_g.edge_index
    a_a_edges = utils.sort_edge_index(a_a_edges)[0]
    a_a_edges = utils.to_undirected(a_a_edges)
    a_a_edges = utils.remove_self_loops(a_a_edges)[0]
    p_a_edges = p_a_g.edge_index
    p_a_edges = utils.sort_edge_index(p_a_edges)[0]
    p_a_edges = utils.remove_self_loops(p_a_edges)[0]
    paper_paper_graph = dgl.graph((p_p_edges[0], p_p_edges[1]), 'paper', 'pp')
    author_author_graph = dgl.graph((a_a_edges[0], a_a_edges[1]), 'author', 'aa')
    paper_author_graph = dgl.bipartite((p_a_edges[0], p_a_edges[1]), 'paper', 'pa', 'author',
                                       num_nodes=(paper_paper_graph.number_of_nodes(), author_author_graph.number_of_nodes()))
    author_paper_graph = dgl.bipartite((p_a_edges[1], p_a_edges[0]), 'author', 'ap', 'paper',
                                       num_nodes=(author_author_graph.number_of_nodes(), paper_paper_graph.number_of_nodes()))
    hg = dgl.hetero_from_relations([author_author_graph, author_paper_graph, paper_author_graph, paper_paper_graph])

    return hg


def generate_metapath(hg):
    # metapath: aappa
    # hg = construct_graph()
    output_path = open(os.path.join('/Users/white/PycharmProjects/deeplearning/Aminer_data/output/', "output_path.txt"), "w")
    type_dict = {1: 'a', 2: 'a', 3: 'p', 0: 'p'}
    count = 0
    for author_idx in tqdm.trange(hg.number_of_nodes('author')):
        traces, _ = dgl.sampling.random_walk(
            hg, [author_idx] * num_walks_per_node, metapath=['aa', 'ap', 'pp', 'pa'] * walk_length)
        for tr in traces:
            tr_list = tr.tolist()
            tr_path = list()
            for idx, node in enumerate(tr_list):
                if node == -1:
                    break
                type_node = type_dict[(idx + 1) % 4]
                tr_path.append('{}{}'.format(type_node, node))
            outline = ' '.join(tr_path)
            print(outline, file=output_path)
    output_path.close()


if __name__ == '__main__':
    ####
    p_p_g = torch.load('/Users/white/PycharmProjects/deeplearning/Aminer_data/output/graph/paper-paper-all.pt')
    a_a_g = torch.load('/Users/white/PycharmProjects/deeplearning/Aminer_data/output/graph/author-author.pt')
    p_a_g = torch.load('/Users/white/PycharmProjects/deeplearning/Aminer_data/output/graph/paper-author-all.pt')
    ####
    hg = construct_graph(p_p_g, a_a_g, p_a_g)
    generate_metapath(hg)
