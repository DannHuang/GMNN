import sys
import os
import copy
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trainer import Trainer
from gnn import GNNq, GNNp, GNNp_concat, GNNq_feature, GNNp_singelGCN
import loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='/')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--MC_smp', type=int, default=10, help='Number of Monte-Carlo sampling')
parser.add_argument('--use_gold', type=int, default=1, help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--concat', type=int, default=1, help='Whether concat node features into inputs to GNN-p')
parser.add_argument('--compare', type=int, default=1, help='Comparing GNN-p w/ concat and w/o')
parser.add_argument('--IR', type=int, default=1, help='Importance Ratio adjust Expectation')
parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
parser.add_argument('--draw', type=str, default='max', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--seed', type=int, default=1, help = 'random seed')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

# # Random Seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)

# # Read-In Data
net_file = opt['dataset'] + '/net.txt'
label_file = opt['dataset'] + '/label.txt'
feature_file = opt['dataset'] + '/feature.txt'
train_file = opt['dataset'] + '/train.txt'
dev_file = opt['dataset'] + '/dev.txt'
test_file = opt['dataset'] + '/test.txt'

vocab_node = loader.Vocab(net_file, [0, 1])
vocab_label = loader.Vocab(label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

opt['num_node'] = len(vocab_node)
opt['num_feature'] = len(vocab_feature)
opt['num_class'] = len(vocab_label)

graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
graph.to_symmetric(opt['self_link_weight'])
feature.to_one_hot(binary=True)
adj = graph.get_sparse_adjacency(opt['cuda'])

with open(train_file, 'r') as fi:
    idx_train = [vocab_node.stoi[line.strip()] for line in fi]
with open(dev_file, 'r') as fi:
    idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
with open(test_file, 'r') as fi:
    idx_test = [vocab_node.stoi[line.strip()] for line in fi]
idx_all = list(range(opt['num_node']))

# # Global Tensor Init
inputs = torch.Tensor(feature.one_hot)
target = torch.LongTensor(label.itol)
# Index Seperate
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)
# Input/Output of 2 GNNs
inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
target_q = torch.zeros(opt['num_node'], opt['num_class'])
inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
target_p = torch.zeros(opt['num_node'], opt['num_class'])
# Importance Ratio Adjustment
if opt['IR'] == 1:
    probs_q = torch.zeros(opt['num_node'])
    probs_q_concat = torch.zeros(opt['num_node'])
# Feature concatenated to GNN-p input
if opt['concat'] == 1:
    # inputs_p_concat = torch.zeros(opt['num_node'], opt['num_class'] + opt['hidden_dim'])
    hidden = torch.zeros(opt['num_node'], opt['hidden_dim'])
if opt['compare'] == 1:
    # Maintain additional tensor for comparision
    target_q_concat = torch.zeros(opt['num_node'], opt['num_class'])
    target_p_concat = torch.zeros(opt['num_node'], opt['num_class'])
# Send to cuda if available
if opt['cuda']:
    inputs = inputs.cuda()
    target = target.cuda()

    idx_train = idx_train.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()

    inputs_q = inputs_q.cuda()
    target_q = target_q.cuda()
    inputs_p = inputs_p.cuda()
    target_p = target_p.cuda()
    # inputs_p_concat = inputs_p_concat.cuda() if opt['concat'] == 1 else None
    hidden = hidden.cuda() if opt['concat'] == 1 else None
    target_q_concat = target_q_concat.cuda() if opt['compare'] == 1 else None
    target_p_concat = target_p_concat.cuda() if opt['compare'] == 1 else None

# # Init GNNs
gnnq = GNNq(opt, adj)
trainer_q = Trainer(opt, gnnq)
# Maintain another GNN-q for comparison
if opt['compare'] == 1:
    gnnq_concat = GNNq_feature(opt, adj)
    trainer_q_concat = Trainer(opt, gnnq_concat)

gnnp = GNNp(opt, adj)
trainer_p = Trainer(opt, gnnp)

if opt['concat'] == 1:
    gnnp_concat = GNNp_singelGCN(opt, adj)
    trainer_p_concat = Trainer(opt, gnnp_concat)

def init_q_data():
    inputs_q.copy_(inputs)
    temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
    temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)    # idx_train = target class, o.w. = 0
    target_q[idx_train] = temp

def update_p_data():
    # update inputs and target (GNN-q's output) to GNN-p
    if opt['compare'] == 1:
        preds = trainer_q.predict(inputs_q, opt['tau'])
        preds_concat, hidden_q = trainer_q_concat.predict_o_hidden(inputs_q, opt['tau'])
        if opt['draw'] == 'exp':
            inputs_p.copy_(preds)
            target_p.copy_(preds)
            target_p_concat.copy_(preds_concat)
        elif opt['draw'] == 'max':
            idx_lb = torch.max(preds, dim=-1)[1]
            if opt['IR'] == 1: probs_q.copy_(torch.max(preds, dim=-1)[0])
            inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            idx_lb_concat = torch.max(preds_concat, dim=-1)[1]
            if opt['IR'] == 1: probs_q_concat.copy_(torch.max(preds_concat, dim=-1)[0])
            target_p_concat.zero_().scatter_(1, torch.unsqueeze(idx_lb_concat, 1), 1.0)
        elif opt['draw'] == 'smp':
            idx_lb = torch.multinomial(preds, 1).squeeze(1)
            if opt['IR'] == 1: probs_q.copy_(preds[(torch.arange(preds.shape[0]), idx_lb)])
            inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            idx_lb_concat = torch.multinomial(preds_concat, 1).squeeze(1)
            if opt['IR'] == 1: probs_q_concat.copy_(preds_concat[(torch.arange(preds_concat.shape[0]), idx_lb_concat)])
            target_p_concat.zero_().scatter_(1, torch.unsqueeze(idx_lb_concat, 1), 1.0)
        if opt['use_gold'] == 1:
            temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
            temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
            inputs_p[idx_train] = temp
            target_p[idx_train] = temp
            target_p_concat[idx_train] = temp
            # temp = torch.zeros(idx_train.size(0), inputs_p_concat.size(1)).type_as(target_q)
            # temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
            # inputs_p_concat[idx_train] = temp
        hidden.copy_(hidden_q)
    else:
        preds = trainer_q.predict(inputs_q, opt['tau'])
        if opt['draw'] == 'exp':
            inputs_p.copy_(preds)
            target_p.copy_(preds)
        elif opt['draw'] == 'max':
            idx_lb = torch.max(preds, dim=-1)[1]
            inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            if opt['IR'] == 1: probs_q.copy_(torch.max(preds, dim=-1)[0])
        elif opt['draw'] == 'smp':
            idx_lb = torch.multinomial(preds, 1).squeeze(1)
            if opt['IR'] == 1: probs_q.copy_(preds[(torch.arange(preds.shape[0]), idx_lb)])
            inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        if opt['concat'] == 1:
            # inputs_p_concat.copy_(torch.cat((inputs_p, inputs_q), 1))
            if opt['use_gold'] == 1:
                temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
                temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
                target_p[idx_train] = temp
                # temp = torch.zeros(idx_train.size(0), inputs_p_concat.size(1)).type_as(target_q)
                temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
                # inputs_p_concat[idx_train] = temp
        else:
            if opt['use_gold'] == 1:
                temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
                temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
                inputs_p[idx_train] = temp
                target_p[idx_train] = temp

def update_p_concat_data():
    if opt['compare'] == 1:
        inputs_p_concat = torch.cat((inputs_p, inputs_q), 1)
        if opt['use_gold'] == 1:
            temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
            temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
            target_p[idx_train] = temp
            temp = torch.zeros(idx_train.size(0), inputs_p_concat.size(1)).type_as(target_q)
            temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
            inputs_p_concat[idx_train] = temp
    else:
        preds = trainer_q.predict(inputs_q, opt['tau'])
        if opt['draw'] == 'exp':
            inputs_p.copy_(preds)
            target_p.copy_(preds)
        elif opt['draw'] == 'max':
            idx_lb = torch.max(preds, dim=-1)[1]
            inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        elif opt['draw'] == 'smp':
            idx_lb = torch.multinomial(preds, 1).squeeze(1)
            inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        inputs_p_concat = torch.cat((inputs_p, inputs_q), 1)
        if opt['use_gold'] == 1:
            temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
            temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
            target_p[idx_train] = temp
            temp = torch.zeros(idx_train.size(0), inputs_p_concat.size(1)).type_as(target_q)
            temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
            inputs_p_concat[idx_train] = temp

def update_q_data():
    # TODO: add Monte Carlo estimates, or MPE estimates of GNN-p
    if opt['compare'] == 1:
        preds_concat = trainer_p_concat.predict_w_hidden(inputs_p, hidden)
        preds = trainer_p.predict(inputs_p)
        target_q_concat.copy_(preds_concat)
        target_q.copy_(preds)
    else:
        if opt['concat'] == 1:
            preds = trainer_p_concat.predict_w_hidden(inputs_p, hidden)
        else:
            preds = trainer_p.predict(inputs_p)
        target_q.copy_(preds)
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        target_q[idx_train] = temp
        if opt['compare'] == 1: target_q_concat[idx_train] = temp

def update_q_data_by_gradient():
    if opt['compare'] == 1:
        preds_concat = trainer_p_concat.predict_by_gradient_w_hidden(inputs_p, hidden, target_p_concat, idx_all)
        preds = trainer_p.predict_by_gradient(inputs_p, target_p, idx_all)
        target_q_concat.copy_(preds_concat)
        target_q.copy_(preds)
    else:
        if opt['concat'] == 1:
            preds = trainer_p_concat.predict_by_gradient_w_hidden(inputs_p, hidden, target_p_concat, idx_all)
        else:
            preds = trainer_p.predict(inputs_p, target_p, idx_all)
        target_q.copy_(preds)
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        target_q[idx_train] = temp
        if opt['compare'] == 1: target_q_concat[idx_train] = temp

def pre_train(epoches):
    best = 0.0
    init_q_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_q.update_soft(inputs_q, target_q, idx_train)
        _, preds, accuracy_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
        _, preds, accuracy_test = trainer_q.evaluate(inputs_q, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
        if accuracy_dev > best:
            best = accuracy_dev
            state = dict([('model', copy.deepcopy(trainer_q.model.state_dict())), ('optim', copy.deepcopy(trainer_q.optimizer.state_dict()))])
    trainer_q.model.load_state_dict(state['model'])
    trainer_q.optimizer.load_state_dict(state['optim'])
    if opt['compare'] == 1:
        trainer_q_concat.model.load_state_dict(state['model'])
        trainer_q_concat.optimizer.load_state_dict(state['optim'])
    return results

def train_p(epoches):
    update_p_data()
    results = [[], []] if opt['compare'] == 1 else []
    for epoch in range(epoches):
        if opt['compare'] == 1:

            if opt['IR'] == 1: loss = trainer_p_concat.update_soft_IR_w_hidden(inputs_p, hidden, target_p_concat, idx_all, probs_q_concat)
            else: loss = trainer_p_concat.update_soft_w_hidden(inputs_p, hidden, target_p_concat, idx_all)
            _, preds, accuracy_dev = trainer_p_concat.evaluate_w_hidden(inputs_p, hidden, target, idx_dev)
            _, preds, accuracy_test = trainer_p_concat.evaluate_w_hidden(inputs_p, hidden, target, idx_test)
            results[1] += [(accuracy_dev, accuracy_test)]

            if opt['IR'] == 1: loss = trainer_p.update_soft_IR(inputs_p, target_p, idx_all, probs_q)
            else: loss = trainer_p.update_soft(inputs_p, target_p, idx_all)
            _, preds, accuracy_dev = trainer_p.evaluate(inputs_p, target, idx_dev)
            _, preds, accuracy_test = trainer_p.evaluate(inputs_p, target, idx_test)
            results[0] += [(accuracy_dev, accuracy_test)]

        else:
            if opt['concat'] == 1:
                if opt['IR'] == 1: loss = trainer_p_concat.update_soft_IR_w_hidden(inputs_p, hidden, target_p_concat, idx_all, probs_q)
                else: loss = trainer_p_concat.update_soft_w_hidden(inputs_p, hidden, target_p_concat, idx_all)
                _, preds, accuracy_dev = trainer_p_concat.evaluate_w_hidden(inputs_p, hidden, target, idx_dev)
                _, preds, accuracy_test = trainer_p_concat.evaluate_w_hidden(inputs_p, hidden, target, idx_test)
                results += [(accuracy_dev, accuracy_test)]
            else:
                if opt['IR'] == 1: loss = trainer_p.update_soft_IR(inputs_p, target_p, idx_all, probs_q)
                else: loss = trainer_p.update_soft(inputs_p, target_p, idx_all)
                _, preds, accuracy_dev = trainer_p.evaluate(inputs_p, target, idx_dev)
                _, preds, accuracy_test = trainer_p.evaluate(inputs_p, target, idx_test)
                results += [(accuracy_dev, accuracy_test)]
    return results

# def train_p_concat(epoches):
    update_p_concat_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_p_concat.update_soft(inputs_p_concat, target_p, idx_all)
        _, preds, accuracy_dev = trainer_p_concat.evaluate(inputs_p_concat, target, idx_dev)
        _, preds, accuracy_test = trainer_p_concat.evaluate(inputs_p_concat, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results

def train_q(epoches):
    update_q_data_by_gradient()
    results = [[], []] if opt['compare'] == 1 else []
    for epoch in range(epoches):
        if opt['compare']:
            # if opt['IR'] == 1: loss = trainer_q.update_soft_IR(inputs_q, target_q, idx_all, probs_q)
            loss = trainer_q.update_soft(inputs_q, target_q, idx_all)
            _, preds, accuracy_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
            _, preds, accuracy_test = trainer_q.evaluate(inputs_q, target, idx_test)
            results[0] += [(accuracy_dev, accuracy_test)]
            # if opt['IR'] == 1: loss = trainer_q_concat.update_soft_IR(inputs_q, target_q_concat, idx_all, probs_q_concat)
            loss = trainer_q_concat.update_soft_hidden(inputs_q, target_q_concat, idx_all)
            _, preds, accuracy_dev = trainer_q_concat.evaluate_o_hidden(inputs_q, target, idx_dev)
            _, preds, accuracy_test = trainer_q_concat.evaluate_o_hidden(inputs_q, target, idx_test)
            results[1] += [(accuracy_dev, accuracy_test)]
        else:
            # if opt['IR'] == 1: loss = trainer_q.update_soft_IR(inputs_q, target_q, idx_all, probs_q)
            loss = trainer_q.update_soft(inputs_q, target_q, idx_all)
            _, preds, accuracy_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
            _, preds, accuracy_test = trainer_q.evaluate(inputs_q, target, idx_test)
            results += [(accuracy_dev, accuracy_test)]
    return results

def get_accuracy(results):
    """
    Return test accuracy with best dev accuracy
    """
    best_dev, acc_test = 0.0, 0.0
    for d, t in results:
        if d > best_dev:
            best_dev, acc_test = d, t
    return acc_test

base_results, q_results, p_results, p_concat_results, q_concat_results = [], [], [], [], []
base_results += pre_train(opt['pre_epoch'])
print(f'Pre-train accuracy: q={get_accuracy(base_results)*100:.1f}%')
for k in range(opt['iter']):
    print(f'\tFixed-point Iteration {k+1}:')
    if opt['compare'] == 1:
        for it in range(opt['MC_smp']):
            results_p = train_p(opt['epoch'])
        p_results += results_p[0]
        p_concat_results += results_p[1]

        # TODO: add a greedy MPE-search to maintain a best prediction
        # sampler_q = Trainer(opt, gnnq)
        # sampler_q_concat = Trainer(opt, gnnq_concat)
        for it in range(1):
            # preds = sampler_q.predict(inputs_q, opt['tau'])
            # preds_concat = sampler_q_concat.predict(inputs_q, opt['tau'])
            # idx_lb = torch.multinomial(preds, 1).squeeze(1)
            # if opt['IR'] == 1: probs_q.copy_(preds[(torch.arange(preds.shape[0]), idx_lb)])
            # inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            # idx_lb_concat = torch.multinomial(preds_concat, 1).squeeze(1)
            # if opt['IR'] == 1: probs_q_concat.copy_(preds_concat[(torch.arange(preds_concat.shape[0]), idx_lb_concat)])
            # inputs_p_concat.copy_(torch.cat((inputs_p, inputs_q), 1))

            results_q = train_q(opt['epoch'])

        q_results += results_q[0]
        q_concat_results += results_q[1]
        print(f'\t\tw/ concatenation: p={get_accuracy(results_p[1])*100:.1f}%, q={get_accuracy(results_q[1])*100:.1f}%\n\t\t'
          f'w/o concatenation: p={get_accuracy(results_p[0])*100:.1f}%, q={get_accuracy(results_q[0])*100:.1f}%')
    else:
        for it in range(opt['MC_smp']):
            if opt['concat'] == 1:
                p_concat_results += train_p(opt['epoch'])
            else:
                p_results += train_p(opt['epoch'])
        q_results += train_q(opt['epoch'])

# acc_test = get_accuracy(p_concat_results)

if opt['compare'] == 1:
    print(f'Test accuracy w/ concatenation: p={get_accuracy(p_concat_results)*100:.1f}%, q={get_accuracy(q_concat_results)*100:.1f}%\n'
          f'Test accuracy w/o concatenation: p={get_accuracy(p_results)*100:.1f}%, q={get_accuracy(q_results)*100:.1f}%')
else:
    if opt['concat'] == 1:
        print(f'Test accuracy w/ concatenation: p={get_accuracy(p_concat_results) * 100:.1f}%, q={get_accuracy(q_results) * 100:.1f}%')
    else:
        print(f'Test accuracy w/o concatenation: p={get_accuracy(p_results) * 100:.1f}%, q={get_accuracy(q_results) * 100:.1f}%')
if opt['save'] != '/':
    trainer_q.save(opt['save'] + '/gnnq.pt')
    trainer_p.save(opt['save'] + '/gnnp.pt')

