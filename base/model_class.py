#!/usr/bin/env python
# coding: utf-8

import sys, io, os, math, torch
import matplotlib
import matplotlib.cm as cm
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
from base.encoder import QulacsEncoderFactory
import base.encoder as enc


# sub-function
def count_obs(num): 
    bin_num = bin(num)[2:] 
    count = 0 
    for i in bin_num: 
        if int(i) == 0: 
            count -= 0 
        elif int(i) == 1: 
            count += 1 
    return count

# NN definition
def net_cNN(X, WxhcNN, bxhcNN, WhycNN, bhycNN): 
    h= torch.relu(X @ WxhcNN + bxhcNN.repeat(X.size(0), 1))    # for classification 
    y= torch.sigmoid(h @ WhycNN + bhycNN.repeat(h.size(0), 1)) # for classification 
    return y

def net_qcNN(X, WxyqcNN, bxyqcNN): 
    y= torch.sigmoid(X @ WxyqcNN + bxyqcNN.repeat(X.size(0), 1)) # for classification 
    return y

def xavier_init(size): 
    in_dim= size[0] 
    xavier_stddev= 1./np.sqrt(in_dim/2.) 
    return Variable(torch.randn(*size)*xavier_stddev, requires_grad=True)

# Model definition
class qcNN:
    def __init__(self, nqubit, M, X_dim, encoder="manual_haar", seed=0, gen_enc=None):
        self.nqubit = nqubit
        self.M = M  # num of data
        self.X_dim = X_dim
        self.encoder = encoder
        self.seed = seed
        np.random.seed(seed=self.seed)

        # Define params
        self.WxyqcNN= xavier_init(size=[X_dim, 1]) 
        self.bxyqcNN= Variable(torch.zeros(1), requires_grad=True)
        qcNN_params= [self.WxyqcNN, self.bxyqcNN]

        # inport gen_enc
        if gen_enc != None:
            self.encoder = gen_enc
        else:
            factory = QulacsEncoderFactory()
            self.encoder = factory.create(self.encoder, self.nqubit)

        # Preparing measurement
        obs_dg = [Observable(self.nqubit) for _ in range(self.nqubit)]
        for i in range(len(obs_dg)):
            obs_dg[i].add_operator(1., f'Z {i}')  # measuremnt all qubit
        self.obs = obs_dg

        ## Parameter for optimizer
        beta1  = 0.5  
        beta2  = 0.99 
        cnn_lr = 0.01 
        self.optimizer= torch.optim.Adam(qcNN_params, lr=cnn_lr, betas=(beta1, beta2), amsgrad=True)

    def learn(self, x_train, y_train, it_end):
        label = Variable(torch.tensor([y_train], dtype=torch.float32).T).clone()

        st_list = []
        for x in x_train:
            st = QuantumState(self.nqubit)
            input_gate = self.encoder.encode(x, self.seed)
            input_gate.update_quantum_state(st)
            st_list.append(st.copy())
        input_state_list = st_list  # state after U_enc

        ## random unitary gate set
        self.outgate_list = []
        for j in range(self.X_dim):
            u_out = QuantumCircuit(self.nqubit)
            ## m = 1
            for i in range(self.nqubit):
                u_out.add_gate(RandomUnitary([i]))
            self.outgate_list.append(u_out)  # rondom u_gate for j

            ## m = 2
        #   for i in range(int(np.floor(self.nqubit/2.))):
        #       u_out.add_gate(RandomUnitary([2*i, 2*i+1]))
        #   self.outgate_list.append(u_out) # rondom u_gate for j

            ## m = 4
        #   for i in range(int(np.floor(self.nqubit/4.))):
        #       u_out.add_gate(RandomUnitary([2*i, 2*i+1, 2*i+2, 2*i+3]))
        #   outgate_list.append(u_out) # rondom u_gate for j

        ## calculate output state from QC
        Xj_lists = []
        for j in range(self.M):
            Xj_list = []
            for k in range(len(self.outgate_list)):
                sv = input_state_list[j]
                self.outgate_list[k].update_quantum_state(sv)
                ## Calculating observable
                sv_ = sv.get_vector()
                rho = np.outer(sv_, np.conjugate(sv_))
                f_xin = 0.
                for i in range(2**self.nqubit):
                    f_xin += rho[i][i] * count_obs(i)
                Xj_list.append(f_xin.real)

            Xj_lists.append(Xj_list)
        X_data = DataLoader(torch.tensor(Xj_lists, dtype=torch.float32), batch_size=len(x_train), shuffle=False)
        losses = []
        for epoch in range(it_end):
            self.optimizer.zero_grad()
            ### Start training
            for data in X_data:
                y_pred = net_qcNN(data, self.WxyqcNN, self.bxyqcNN)
                loss = F.binary_cross_entropy(y_pred, label)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

        return loss, y_pred

    def test(self, x_test, y_test):
        label = Variable(torch.tensor([y_test], dtype=torch.float32).T).clone()
        st_list = []
        for x in x_test:
            st = QuantumState(self.nqubit)
            input_gate = self.encoder.encode(x, self.seed)
            input_gate.update_quantum_state(st)
            st_list.append(st.copy())

        input_state_list = st_list  # state after U_enc
        Xj_lists = []
        for j in range(len(x_test)):
            Xj_list = []
            for k in range(len(self.outgate_list)):
                sv = input_state_list[j]
                self.outgate_list[k].update_quantum_state(sv)
                ## Grobal obs
                sv_ = sv.get_vector()
                rho = np.outer(sv_, np.conjugate(sv_))
                f_xin = 0.
                for i in range(2**self.nqubit):
                    f_xin += rho[i][i] * count_obs(i)

                Xj_list.append(f_xin.real)
            Xj_lists.append(Xj_list)
        X_data = DataLoader(torch.tensor(Xj_lists, dtype=torch.float32), batch_size=len(x_test), shuffle=False)

        ### Start testing
        y_pred = net_qcNN(torch.tensor(Xj_lists, dtype=torch.float32), self.WxyqcNN, self.bxyqcNN)
        loss = F.binary_cross_entropy(y_pred, label)

        return loss, y_pred

class cNN:
    def __init__(self, nqubit, X_dim, seed):
        self.nqubit = nqubit
        self.seed = seed
        
        # Define params
        self.WxhcNN= xavier_init(size=[self.nqubit, X_dim])
        self.bxhcNN= Variable(torch.zeros(X_dim), requires_grad=True)
        self.WhycNN= xavier_init(size=[X_dim, 1])
        self.bhycNN= Variable(torch.zeros(1), requires_grad=True)
        cNN_params= [self.WxhcNN, self.bxhcNN, self.WhycNN, self.bhycNN]

        # Setting for optimizer
        beta1  = 0.5
        beta2  = 0.99
        dnn_lr = 0.01 
        self.optimizer= torch.optim.Adam(cNN_params, lr=dnn_lr, betas=(beta1, beta2), amsgrad=True)

    def learn(self, x_train, y_train, it_end):
        X_data = DataLoader(x_train, batch_size=len(x_train), shuffle=False)
        label  = Variable(torch.tensor([y_train], dtype=torch.float32).T).clone()
        losses = []
        for epoch in range(it_end):
            self.optimizer.zero_grad()

            ### Start training
            for data in X_data:
                y_pred = net_cNN(data, self.WxhcNN, self.bxhcNN, self.WhycNN, self.bhycNN)
                loss = F.binary_cross_entropy(y_pred, label)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
        return loss, y_pred.detach().numpy()

    def test(self, x_test, y_test):
        label = Variable(torch.tensor([y_test], dtype=torch.float32).T).clone()

        ### Start testing
        y_pred = net_cNN(x_test, self.WxhcNN, self.bxhcNN, self.WhycNN, self.bhycNN)
        loss = F.binary_cross_entropy(y_pred, label)
        return loss, y_pred

class qNN:
    def __init__(self, nqubit, X_dim, Lq, encoder="manual_haar", seed=0):
        self.nqubit = nqubit
     #  self.M = M  # num of dataset
        self.X_dim = X_dim
        self.encoder = encoder
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.Lq = Lq

        factory = QulacsEncoderFactory()
        self.encoder = factory.create(self.encoder, self.nqubit)

        # Setting Observable
        obs_dg = [Observable(self.nqubit) for _ in range(self.nqubit)]
        for i in range(len(obs_dg)):
            obs_dg[i].add_operator(1., f'Z {i}')  
        self.obs = obs_dg

        ## random parameter for PQC
        self.theta = Variable(2*np.pi*torch.rand(self.nqubit*self.Lq*3), requires_grad=True)

        # Setting of optimizer
        beta1  = 0.5
        beta2  = 0.99
        cnn_lr = 0.01 
        self.optimizer= torch.optim.Adam([self.theta], lr=cnn_lr, betas=(beta1, beta2), amsgrad=True)

    def get_loss(self, y_target, y_pred):
        loss = F.binary_cross_entropy(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32))
        return loss

    def set_gradient(self, theta, x_train, y_train):
        param_grad=[]
        y_pred=[]
        for x in x_train:
            y_pred.append(self._sample(x, theta))

        for i,p in enumerate(theta):
            p_plus = theta.detach().clone()  
            p_minus = theta.detach().clone()  
            p_plus[i] = theta[i] + np.pi / 2.
            p_minus[i] = theta[i] - np.pi / 2.
            yp_list=[]
            ym_list=[]
            for x in x_train:
                plus_y = self._sample(x, p_plus) 
                minus_y = self._sample(x, p_minus) 
                yp_list.append(plus_y)
                ym_list.append(minus_y)

            loss_p = self.get_loss(y_train, yp_list)
            loss_m = self.get_loss(y_train, ym_list)
            grad = 0.5*(loss_p - loss_m).clone().detach()
            param_grad.append(grad)

        theta.grad = Variable(torch.tensor(param_grad, dtype=torch.float32))
        return theta, y_pred

    def _sample(self, x, theta):
        encode_gate = self.encoder.encode(x, self.seed)
        pqc = self.create_pqc(self.Lq, theta)

        st = QuantumState(self.nqubit)
        encode_gate.update_quantum_state(st)
        pqc.update_quantum_state(st)

        ## Partial trace at 1st qubit(manual) 
        sv = st.get_vector() 
        rho = np.outer(sv, np.conjugate(sv)) 
        f_xin = 0. 
        for i in range(0, 2**self.nqubit, 2): 
            f_xin += rho[i][i]

        return f_xin.real

    def create_pqc(self, Lq, theta):
        qc = QuantumCircuit(self.nqubit)
        for k in range(Lq):
            for j in range(self.nqubit):
                qc.add_U3_gate(j, theta[0+ 3*(j+(k*self.nqubit))], theta[1+ 3*(j+(k*self.nqubit))], theta[2+ 3*(j+(k*self.nqubit))])
                if j == (self.nqubit-1):
                    for i in range(self.nqubit):
                        qc.add_CNOT_gate(i%self.nqubit, (i+1)%self.nqubit)
        return qc

    def learn(self, x_train, y_train, it_end):
        losses = []
        for epoch in range(it_end):
            self.optimizer.zero_grad()
            self.theta, y_pred = self.set_gradient(self.theta, x_train, y_train)
            loss = F.binary_cross_entropy(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            losses.append(loss)
            self.optimizer.step()
            print(loss)

        return loss, y_pred

    def test(self, x_test, y_test):
        ### Start test 
        y_pred=[] 
        for x in x_test: 
            y_pred.append(self._sample(x, self.theta)) 
        loss = F.binary_cross_entropy(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        return loss, y_pred


