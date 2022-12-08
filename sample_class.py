#!/usr/bin/env python
# coding: utf-8

import sys, io, os, math, torch, csv, time
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

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
from base.model_class import qcNN, cNN, qNN, count_obs, xavier_init
from base.data_gen_class import DataGen
from base.func import get_score, torch_fix_seed

start = time.time()

## set params ##
seed    = 0 
n0      = 1000
M_train = 1000
it_end  = 1000

### set random seed ###
torch_fix_seed(seed=seed)
path = 'out_{}'.format(seed)
if not os.path.isdir(path):
    os.mkdir(path)

### main task ### 
for nqubit in [2,3,4,5]:
    print("----- n = {} -----".format(nqubit))

    ## Data preparation ##
    dgen = DataGen(nqubit=nqubit, M=M_train, encoder="manual_haar", seed=seed)
    x_train, y_train, _, x_test, y_test, __, gen_enc = dgen.gen_data() 


    ### Learning ######
    # qcNN #
    model_qc = qcNN(nqubit=nqubit, M=M_train, n0=n0, seed=seed, gen_enc=gen_enc)
    loss_learn_qc, y_pred_learn_qc = model_qc.learn(x_train, y_train, it_end=it_end)
    score_learn_qc = get_score(y_pred_learn_qc, y_train, "learn", "qcNN")

    # cNN #
    model_c = cNN(nqubit=nqubit, n0=n0, seed=seed)
    loss_learn_c, y_pred_learn_c = model_c.learn(x_train, y_train, it_end=it_end)
    score_learn_c = get_score(y_pred_learn_c, y_train, "learn", "cNN")

    # qNN #
    model_q = qNN(nqubit=nqubit, Lq=10, seed=seed)
    loss_learn_q, y_pred_learn_q = model_q.learn(x_train, y_train, it_end=it_end)
    score_learn_q = get_score(y_pred_learn_q, y_train, "learn", "qNN")


    ### Testing ######
    # qcNN #
    loss_test_qc, y_pred_test_qc = model_qc.test(x_test, y_test)
    score_test_qc = get_score(y_pred_test_qc, y_test, "test", "qcNN")

    # cNN #
    loss_test_c, y_pred_test_c = model_c.test(x_test, y_test)
    score_test_c = get_score(y_pred_test_c, y_test, "test", "cNN")

    # qNN #
    loss_test_q, y_pred_test_q = model_q.test(x_test, y_test)
    score_test_q = get_score(y_pred_test_q, y_test, "test", "qNN")


    with open('out_{}/scores_n{}.csv'.format(seed, nqubit), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['score_learn_qc', 'score_learn_c', 'score_learn_q', 'score_test_qc', 'score_test_c', 'score_test_q'])
        writer.writerow([score_learn_qc, score_learn_c, score_learn_q, score_test_qc, score_test_c, score_test_q])

print("Finish !")
print(f'elapsed time: {time.time() - start:.1f}s')


