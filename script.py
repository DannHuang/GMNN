import numpy
import torch
from semisupervised.codes import loader

# net_file = './semisupervised/data/cora' + '/net.txt'
# vocab_node = loader.Vocab(net_file, [0, 1])
# graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
# graph.to_symmetric(1.0)
# adj = graph.get_sparse_adjacency(False)

def test(a,b):
    return a+1,b+1

print(type(test(1,2)))