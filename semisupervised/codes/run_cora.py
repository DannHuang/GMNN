import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/cora'
opt['hidden_dim'] = 16
opt['input_dropout'] = 0.5
opt['dropout'] = 0
opt['optimizer'] = 'rmsprop'
opt['lr'] = 0.05
opt['decay'] = 5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 100
opt['epoch'] = 10   # Gradient descent iter
<<<<<<< HEAD
<<<<<<< HEAD
opt['iter'] = 5    # fixed-point iter
=======
opt['iter'] = 15    # fixed-point iter
>>>>>>> b1d8e26 (run experiment)
=======
opt['iter'] = 5    # fixed-point iter
>>>>>>> 29cffe9 (prepare for positional-embd test)
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['tau'] = 0.1
opt['concat'] = 1
opt['compare'] = 1 * opt['concat']
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
opt['IR'] = 0
opt['MC_smp'] = 50
=======
=======
>>>>>>> 1c20b2d (logger update)
opt['IR'] = 1
opt['MC_smp'] = 10
>>>>>>> da42629 (IR test)
=======
opt['IR'] = 0
opt['MC_smp'] = 50
>>>>>>> b1d8e26 (run experiment)

def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

for k in range(10):
    seed = k + 1
    opt['seed'] = seed
<<<<<<< HEAD
<<<<<<< HEAD
    print(f'Experiment{k+1}:'+'-'*17)
=======
    print(f'Experiment{k}:'+'-'*17)
>>>>>>> 263f96e (logger-update)
=======
    print(f'Experiment{k+1}:'+'-'*17)
>>>>>>> 1c20b2d (logger update)
    run(opt)
