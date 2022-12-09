import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/citeseer'
opt['hidden_dim'] = 16
opt['input_dropout'] = 0.5
opt['dropout'] = 0
opt['optimizer'] = 'rmsprop'
opt['lr'] = 0.05
opt['decay'] = 5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 100
opt['epoch'] = 10
<<<<<<< HEAD
opt['iter'] = 5
=======
opt['iter'] = 15
>>>>>>> b1d8e26 (run experiment)
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['tau'] = 0.1
opt['concat'] = 1
opt['compare'] = 1 * opt['concat']
<<<<<<< HEAD
opt['IR'] = 1
=======
opt['IR'] = 0
>>>>>>> b1d8e26 (run experiment)
opt['MC_smp'] = 50

def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

<<<<<<< HEAD
for k in range(10):
=======
for k in range(1):
>>>>>>> b1d8e26 (run experiment)
    seed = k + 1
    opt['seed'] = seed
    print(f'Experiment{k}:'+'-'*17)
    run(opt)
