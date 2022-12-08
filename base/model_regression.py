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
    h= torch.sigmoid(X @ WxhcNN + bxhcNN.repeat(X.size(0), 1))
    y = h @ WhycNN + bhycNN.repeat(h.size(0), 1)
    return y

def net_qcNN(X, WxyqcNN, bxyqcNN): 
 #  h= torch.sigmoid(X @ WxhqcNN + bxhqcNN.repeat(X.size(0), 1)) # for L=2
 #  y = h @ WhyqcNN + bhyqcNN.repeat(X.size(0), 1) # for L=2

    y = X @ WxyqcNN + bxyqcNN.repeat(X.size(0), 1) # for L1
    return y

def net_qNN(X, WxyqNN): 
    y = X * WxyqNN 
    return y

def xavier_init(size): 
    in_dim= size[0] 
    xavier_stddev= 1./np.sqrt(in_dim/2.) 
    return Variable(torch.randn(*size)*xavier_stddev, requires_grad=True)

# Model definition
class qcNN:
    def __init__(self, nqubit, M, n0, encoder="manual_haar", seed=0, gen_enc=None):
        self.nqubit = nqubit
        self.M = M  # num of data
        self.X_dim = n0
        self.encoder = encoder
        self.seed = seed
        np.random.seed(seed=self.seed)

        # Define params
        self.WxyqcNN= xavier_init(size=[self.X_dim, 1]) 
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
        cnn_lr = 0.1 
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
                loss_func = torch.nn.MSELoss() 
                loss = loss_func(y_pred, label) 
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
        loss_func = torch.nn.MSELoss() 
        loss = loss_func(y_pred, label)

        return loss, y_pred

class cNN:
    def __init__(self, nqubit, n0, seed):
        self.nqubit = nqubit
        self.seed = seed
        
        # Define params
        self.WxhcNN= xavier_init(size=[self.nqubit, n0])
        self.bxhcNN= Variable(torch.zeros(n0), requires_grad=True)
        self.WhycNN= xavier_init(size=[n0, 1])
        self.bhycNN= Variable(torch.zeros(1), requires_grad=True)
        cNN_params= [self.WxhcNN, self.bxhcNN, self.WhycNN, self.bhycNN]

        # Setting for optimizer
        beta1  = 0.5
        beta2  = 0.99
        dnn_lr = 0.1 
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
                loss_func = torch.nn.MSELoss() 
                loss = loss_func(y_pred, label) 
                losses.append(loss.item()) 
                loss.backward() 
                self.optimizer.step()

        return loss, y_pred.detach().numpy()

    def test(self, x_test, y_test):
        label = Variable(torch.tensor([y_test], dtype=torch.float32).T).clone()

        ### Start testing
        y_pred = net_cNN(x_test, self.WxhcNN, self.bxhcNN, self.WhycNN, self.bhycNN)
        loss_func = torch.nn.MSELoss() 
        loss = loss_func(y_pred, label)
        return loss, y_pred

class qNN:
    def __init__(self, nqubit, Lq, encoder="manual_haar", seed=0):
        self.nqubit = nqubit
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

        self.WxyqNN= Variable(torch.ones(1), requires_grad=True)
        self.theta = Variable(2*np.pi*torch.rand(self.nqubit*self.Lq*3), requires_grad=True)
        qNN_params = [self.WxyqNN, self.theta]

        # Setting of optimizer
        beta1  = 0.5
        beta2  = 0.99
        cnn_lr = 0.05 
        self.optimizer= torch.optim.Adam(qNN_params, lr=cnn_lr, betas=(beta1, beta2), amsgrad=True)

    def set_gradient(self, theta, x_train, y_train, y_pred):
        param_grad=[]

        for i,p in enumerate(theta):
            p_plus = theta.detach().clone()  
            p_minus = theta.detach().clone()  
            p_plus[i] = theta[i] + np.pi / 2.
            p_minus[i] = theta[i] - np.pi / 2.
            yp_list=[]
            ym_list=[]
            for x in x_train: 
                plus_y = net_qNN(self._sample(x, p_plus), self.WxyqNN) 
                minus_y = net_qNN(self._sample(x, p_minus), self.WxyqNN) 
                yp_list.append(plus_y) 
                ym_list.append(minus_y) 

            diff_pred = y_pred - y_train 
            diff_pm = torch.tensor(yp_list) - torch.tensor(ym_list) 
            grad =  (diff_pm * diff_pred).sum()/len(y_train) 
            param_grad.append(grad)

        theta.grad = Variable(torch.tensor(param_grad, dtype=torch.float32))
        return theta

    def _sample(self, input_state, theta): 
        st = input_state.copy() 
        pqc = self.create_pqc(self.Lq, theta) 
        pqc.update_quantum_state(st) 

        ## Grobal obs 
        sv = st.get_vector() 
        rho = np.outer(sv, np.conjugate(sv)) 
        f_xin = 0. 
        for i in range(2**self.nqubit): 
            f_xin += rho[i][i] * count_obs(i) 

        return torch.tensor(f_xin.real)

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
        label = Variable(torch.tensor([y_train], dtype=torch.float32)).clone()  
        st_list = [] 
        for x in x_train: 
            st = QuantumState(self.nqubit) 
            input_gate = self.encoder.encode(x, self.seed) 
            input_gate.update_quantum_state(st) 
            st_list.append(st.copy()) 

        losses = [] 
        for epoch in range(it_end): 
            self.optimizer.zero_grad() 
            qnn_out = [] 
            for j in range(len(y_train)): 
                qnn_out.append(self._sample(st_list[j], self.theta)) 
            QO_list = DataLoader(torch.tensor(qnn_out, dtype=torch.float32), batch_size=len(x_train), shuffle=False) 

            for data in QO_list: 
                y_pred = torch.reshape(net_qNN(data, self.WxyqNN), (1,len(x_train))) 
                self.theta = self.set_gradient(self.theta, st_list, label, y_pred) 

                loss_func = torch.nn.MSELoss() 
                loss = loss_func(y_pred, label) 
                losses.append(loss) 
                loss.backward() 
                self.optimizer.step() 
        return loss, y_pred

    def test(self, x_test, y_test):
        ### Start test 
        y_pred=[] 
        for x in x_test: 
            st = QuantumState(self.nqubit) 
            input_gate = self.encoder.encode(x, self.seed) 
            input_gate.update_quantum_state(st) 
            y_pred.append(net_qNN(self._sample(st, self.theta), self.WxyqNN))

        loss_func = torch.nn.MSELoss() 
        loss = loss_func(torch.tensor(y_pred), torch.tensor(y_test)) 

        return loss, y_pred


