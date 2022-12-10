import numpy
import torch
from semisupervised.codes import loader

# net_file = './semisupervised/data/cora' + '/net.txt'
# vocab_node = loader.Vocab(net_file, [0, 1])
# graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
# graph.to_symmetric(1.0)
# adj = graph.get_sparse_adjacency(False)

a = []
b= []
a += [(1,2)]
b += a
print(type(b))
print(len(b))
print(b)
for d, v in b:
    print(d)
    print(v)