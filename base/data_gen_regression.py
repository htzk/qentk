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
from base.model_regression import count_obs


### Core function ###
class DataGen:
    def __init__(self, nqubit, M, seed=0, encoder="manual_haar", gen_enc=None):
        self.nqubit = nqubit
        self.M  = M   # num of training data
        self.Mt = 100 # num of test data
        self.encoder = encoder
        self.seed = seed
        np.random.seed(seed=self.seed)

        ## inport gen_enc
        if gen_enc != None:
            self.encoder = gen_enc
        else:
            factory = QulacsEncoderFactory()
            self.encoder = factory.create(self.encoder, self.nqubit)

        obs_dg = [Observable(self.nqubit) for _ in range(self.nqubit)]
        for i in range(len(obs_dg)):
            obs_dg[i].add_operator(1., f'Z {i}')  # measuring all qubit
        self.obs = obs_dg

        ## random input data to Quantum Circuit
        self.x_in = np.random.uniform([0]*self.nqubit, [2*np.pi]*self.nqubit, size=(self.M +self.Mt, self.nqubit))

        st_list = []
        for x in self.x_in:
            st = QuantumState(self.nqubit)
            input_gate = self.encoder.encode(x, self.seed)
            input_gate.update_quantum_state(st)
            st_list.append(st.copy())
        self.input_state_list = st_list  # state after U_enc

    def gen_data(self, noise_level=1e-4):
        y_list = []
        y_list_l = []
        y_list_t = []
        noise_l = np.random.normal(0, noise_level, self.M)
        noise_t = np.random.normal(0, noise_level, self.Mt)
        Xj_lists =[]
        for j in range(self.M +self.Mt):
            sv_xin = self.input_state_list[j]

            ## Grobal obs
            sv = sv_xin.get_vector()
            rho = np.outer(sv, np.conjugate(sv))
            f_xin = 0
            for i in range(2**self.nqubit):
                f_xin += rho[i][i] * count_obs(i)

            if j < self.M:
                y_list_l.append(f_xin.real)
                y_list.append(f_xin.real)
            else:
                y_list_t.append(f_xin.real)
                y_list.append(f_xin.real)

        y_list_l = y_list_l / np.sqrt(np.var(y_list)) 
        y_list_t = y_list_t / np.sqrt(np.var(y_list)) 
        y_list_l = y_list_l + noise_l 
        y_list_t = y_list_t + noise_t

        return torch.tensor(self.x_in[0:self.M], dtype=torch.float32), y_list_l, \
               torch.tensor(self.x_in[self.M:self.M+self.Mt], dtype=torch.float32), y_list_t, self.encoder

