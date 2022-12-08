#!/usr/bin/env python
# coding: utf-8

import sys, io, os, math, torch, csv, time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad

from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import U1,U2,U3,RandomUnitary

import base.encoder as enc
from base.encoder import QulacsEncoderFactory
from base.model_regression import qcNN, cNN, qNN, count_obs, xavier_init
from base.data_gen_regression import DataGen
from base.func import get_rmse, torch_fix_seed

start = time.time()

## set params ##
seeds   = 0
X_dim   = 1000
M_train = 1000
it_end  = 1000

### set random seed ###
torch_fix_seed(seed=seeds)
path = 'out_{}'.format(seeds)
if not os.path.isdir(path):
        os.mkdir(path)

### main task ###
for nqubit in [2,3,4,5]:
    print("----- n = {} -----".format(nqubit))

    ## Data preparation ## 
    dgen = DataGen(nqubit=nqubit, M=M_train, encoder="manual_haar", seed=seeds) 
    x_train, y_train, x_test, y_test, gen_enc = dgen.gen_data() 

    ### Learning ###### 
    # qcNN # 
    model_qc = qcNN(nqubit=nqubit, M=M_train, X_dim=X_dim, seed=seeds, gen_enc=gen_enc) 
    loss_learn_qc, y_pred_learn_qc = model_qc.learn(x_train, y_train, it_end=it_end) 
    rmse_learn_qc = get_rmse(loss_learn_qc, "learn", "qcNN") 

    # cNN # 
    model_c = cNN(nqubit=nqubit, X_dim=X_dim, seed=seeds) 
    loss_learn_c, y_pred_learn_c = model_c.learn(x_train, y_train, it_end=it_end) 
    rmse_learn_c = get_rmse(loss_learn_c, "learn", "cNN") 

    # qNN # 
    model_q = qNN(nqubit=nqubit, X_dim=X_dim, Lq=10, seed=seeds) 
    loss_learn_q, y_pred_learn_q = model_q.learn(x_train, y_train, it_end=it_end) 
    rmse_learn_q = get_rmse(loss_learn_q, "learn", "qNN") 

    ### Testing ###### 
    # qcNN # 
    loss_test_qc, y_pred_test_qc = model_qc.test(x_test, y_test) 
    rmse_test_qc = get_rmse(loss_test_qc, "test", "qcNN") 

    # cNN # 
    loss_test_c, y_pred_test_c = model_c.test(x_test, y_test) 
    rmse_test_c = get_rmse(loss_test_c, "test", "cNN") 

    # qNN # 
    loss_test_q, y_pred_test_q = model_q.test(x_test, y_test) 
    rmse_test_q = get_rmse(loss_test_q, "test", "qNN") 


    with open('out_{}/rmse_n{}.csv'.format(seeds, nqubit), 'w') as f: 
        writer = csv.writer(f) 
        writer.writerow(['rmse_learn_qc', 'rmse_learn_c', 'rmse_learn_q', 'rmse_test_qc', 'rmse_test_c', 'rmse_test_q']) 
        writer.writerow([rmse_learn_qc, rmse_learn_c, rmse_learn_q, rmse_test_qc, rmse_test_c, rmse_test_q]) 

print("Finish !") 
print(f'elapsed time: {time.time() - start:.1f}s')

