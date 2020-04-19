#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : White
# @Email   : baipz1993@gmail.com
# @File    : metapath2vec_decoder.py

import pickle
import torch
import numpy as np
import time
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from src.decoder import multiClassInnerProductDecoder
from src.utils import *

torch.manual_seed(1111)
np.random.seed(1111)
train_set_ratio = 0.9
learning_rate = 0.01
EPOCH_NUM = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    ####
    author_author_graph = torch.load('../Aminer_data/output/graph/author-author.pt')
    node_embedding_path = '../../Aminer_data/output/output_embedding.txt'
    ####
    author2label = dict(author_author_graph.node_label)
    author_embed_dict = dict()
    with open(node_embedding_path, 'r') as rf:
        next(rf)
        for line in rf:
            embed = line.strip().split(' ')
            if embed[0].startswith('a'):
                author_embed_dict[int(embed[0].strip('a'))] = []
                for i in range(1, len(embed), 1):
                    author_embed_dict[int(embed[0].strip('a'))].append(float(embed[i]))

    author_embed_list = list()
    author_label_list = list()

    for author_idx in author_embed_dict.keys():
        author_embed_list.append(author_embed_dict[author_idx])
        author_label_list.append(author2label[author_idx])

    embeddings = torch.tensor(author_embed_list)
    labels = torch.tensor(author_label_list)

    # split
    index_array = [i for i in range(len(labels))]
    np.random.shuffle(index_array)
    training_embed = embeddings[index_array[:int(len(index_array) * train_set_ratio)]]
    training_label = labels[index_array[:int(len(index_array) * train_set_ratio)]]
    test_embed = embeddings[index_array[int(len(index_array) * train_set_ratio):]]
    test_label = labels[index_array[int(len(index_array) * train_set_ratio):]]

    mcip = multiClassInnerProductDecoder(embeddings.shape[1], len(labels.unique())).to(device)
    optimizer = torch.optim.Adam(mcip.parameters(), lr=learning_rate)

    # train and test
    for epoch in range(EPOCH_NUM):
        time_begin = time.time()
        mcip.train()
        optimizer.zero_grad()

        score = mcip(training_embed)
        pred = torch.argmax(score, dim=1)

        loss = -torch.log(score[range(score.shape[0]), training_label] + EPS).mean()
        loss.backward()
        optimizer.step()

        micro, macro = micro_macro(training_label, pred)

        # out.train_out[epoch] = np.array([micro, macro])

        print('{:3d}   loss:{:0.4f}   micro:{:0.4f}   macro:{:0.4f}'
              .format(epoch, loss.tolist(), micro, macro))

        mcip.eval()
        score = mcip(test_embed)
        pred = torch.argmax(score, dim=1)

        micro, macro = micro_macro(test_label, pred)
        print('{:3d}   loss:{:0.4f}   micro:{:0.4f}   macro:{:0.4f}    time:{:0.1f}\n'
              .format(epoch, loss.tolist(), micro, macro, (time.time() - time_begin)))

