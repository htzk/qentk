# Quantum-enhanced neural networks in the neural tangent kernel framework

We propose a quantum-classical nerural network (qcNN), shown in Fig.1 or Fig.2 (b), and analyszed the machine learning performance compared to other conventional models shown in Fig.2 (c) and (d).
The python source codes shared in this page is used for "Section IV-C. Advantage of qcNN over full-classical and full-quantum models".

<p align="center">
<img src="https://github.com/htzk/qentk/blob/main/FIG1.JPG" width="700">
Fig.1: Overview of the proposed qcNN
<img src="https://github.com/htzk/qentk/blob/main/FIG2.JPG" width="700">
Fig.2: Models
</p>

## Requirements

|Software|Version|
|:---:|:---:|
|Python|3.8.8|
|Qulacs|0.3.0|
|Torch|1.9.1|
|numpy|1.23.3|

Qulacs is a Python/C++ library for fast simulation of large, noisy, or parametric quantum circuits, developed by QunaSys, Osaka University and NTT.
https://github.com/qulacs/qulacs

## Usage

see [sample_regression.py](https://github.com/htzk/qentk/blob/main/sample_regression.py) and [sample_class.py](https://github.com/htzk/qentk/blob/main/sample_class.py)

```python
## set params ##
seed    = 0
n0      = 1000 # the number of nodes of classical neural network
M_train = 1000 # the number of training data
it_end  = 1000 # the number of training iteration
```

About simulation time:  
The calculation time is heavily depending on the model and parameter settings.
Typical time is as follows with our environment.  
  qcNN and qNN are fast; 5 ~ 15 sec. for (n0, M_train, it_end, #qubit)=(1000, 1000, 1000, 2)  
  qNN takes much longer time; 5 min. for (n0, M_train, it_end, #qubit)=(100, 100, 100, 2)  
It increases linearly with n0, M_train and it_end, and exponentially with #qubit.
