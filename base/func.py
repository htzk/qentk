#!/usr/bin/env python
# coding: utf-8

import sys, io, os, math, torch, csv, time
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

### set random seed ###
def torch_fix_seed(seed=0): 
    # Numpy 
    np.random.seed(seed) 
    # Pytorch 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.use_deterministic_algorithms = True

    ### Calculate RMSE ###
def get_rmse(mse, task="task", net="net"):
    print("RMSE:" +task +"(" +net +")")
    print(math.sqrt(mse))
    return math.sqrt(mse)

    ### Calculate score ###
def get_score(y_pred, y_label, task="task", net="net"):
    print("Score:" +task +"(" +net +")")

    yb_pred=[]
    score=0.
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5:
            yb_pred.append(0)
    
        else:
            yb_pred.append(1)
        if yb_pred[i] == y_label[i]:
            score += 1.
        else:
            score += 0.
    print(score/len(y_pred))
    return score/len(y_pred)



