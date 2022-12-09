import math
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class Trainer(object):
    # CSE Loss
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        self.model.reset()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def update(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft_IR_w_hidden(self, inputs, hidden, target, idx, probs):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            hidden = hidden.cuda()
            target = target.cuda()
            idx = idx.cuda()
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs, hidden)
        logits = torch.log_softmax(logits, dim=-1)
        # loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
        # adjust by importance ratio
        inds = torch.where(self.model.adj.to_dense()>0, 1, 0)
        IR = torch.prod(inds * probs, dim=-1)

        loss = -torch.sum(torch.sum(target[idx] * logits[idx], dim=-1) * IR)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft_IR(self, inputs, target, idx, probs):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        # loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
        # adjust by importance ratio
        inds = torch.where(self.model.adj.to_dense()>0, 1, 0)
        IR = torch.prod(inds * probs, dim=-1)

        loss = -torch.sum(torch.sum(target[idx] * logits[idx], dim=-1) * IR)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft_w_hidden(self, inputs, hidden, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            hidden = hidden.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs, hidden)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft_hidden(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits, _ = self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate_w_hidden(self, inputs, hidden, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            hidden = hidden.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()

        logits = self.model(inputs, hidden)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return loss.item(), preds, accuracy.item()
    
    def evaluate(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return loss.item(), preds, accuracy.item()

    def evaluate_o_hidden(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()

        logits, _ = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return loss.item(), preds, accuracy.item()

    def predict(self, inputs, tau=1):
        # output distribution
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits
    
    def predict_by_gradient_w_hidden(self, inputs, hidden, target, idx):
        psd_label = torch.zeros_like(inputs)
        psd_label.copy_(inputs)
        if self.opt['cuda']:
            psd_label = psd_label.cuda()
            hidden = hidden.cuda()
            target = target.cuda()
            idx = idx.cuda()
        for p in self.parameters: p.requires_grad = False
        psd_label.requires_grad = True
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(psd_label, hidden)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))

        loss.backward()
        self.optimizer.step()

        # labels = torch.softmax(psd_label, dim=-1).detach()
        labels = psd_label.detach()

        for p in self.parameters: p.requires_grad = True

        return labels

    def predict_by_gradient(self, inputs, target, idx):
        psd_label = torch.zeros_like(inputs)
        psd_label.copy_(inputs)
        if self.opt['cuda']:
            psd_label = psd_label.cuda()
            target = target.cuda()
            idx = idx.cuda()
        for p in self.parameters: p.requires_grad = False
        psd_label.requires_grad = True
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(psd_label)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))

        loss.backward()
        self.optimizer.step()

        # labels = torch.softmax(psd_label, dim=-1).detach()
        labels = psd_label.detach()
        
        for p in self.parameters: p.requires_grad = True

        return labels         

    def predict_w_hidden(self, inputs, hidden, tau=1):
        # output distribution
        if self.opt['cuda']:
            inputs = inputs.cuda()
            hidden = hidden.cuda()

        self.model.eval()

        logits = self.model(inputs, hidden) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    def predict_o_hidden(self, inputs, tau=1):
        # output distribution
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()
        logits, hidden = self.model(inputs)
        logits /= tau
        logits = torch.softmax(logits, dim=-1).detach()

        return logits, hidden

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
