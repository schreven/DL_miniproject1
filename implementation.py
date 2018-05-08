# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:02:13 2018

@author: Bob
"""
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import FloatTensor as Tensor

import matplotlib.pyplot as plt
import numpy as np

import dlc_bci as bci
train_input , train_target = bci.load(root = './ data_bci')
print ('train_input:', str ( type ( train_input ) ) , train_input.size () )
print ('train_target:', str ( type ( train_target ) ) , train_target.size () )
test_input , test_target = bci.load ( root = './data˙bci' , train = False )
print ('test_input:', str ( type ( test_input ) ) , test_input.size () )
print ('test target:', str ( type ( test_target ) ) , test_target.size () )





class Net1(nn.Module):
    def __init__(self, size, nb_hidden):
        self.size = size
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(self.size[1]*self.size[2], nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1,self.size[1]*self.size[2])))
        x = self.fc2(x)
        return x
    
class Net2(nn.Module):
    def __init__(self, size, nb_hidden, chn, ker_conv, ker_pool):
        self.size = size
        super(Net2, self).__init__()
        self.ker_pool = ker_pool
        self.ker_conv = ker_conv
        self.chn = chn
        self.conv1 = nn.Conv1d(size[1], chn[0], kernel_size=ker_conv[0])
        self.conv2 = nn.Conv1d(chn[0], chn[1], kernel_size=ker_conv[1])
        self.fc1 = nn.Linear(chn[1]*(((size[2]-ker_conv[0]+1)/ker_pool[0])-ker_conv[1]+1)/ker_pool[1],nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.tanh(F.max_pool1d(self.conv1(x), kernel_size=self.ker_pool[0]))
        x = F.tanh(F.max_pool1d(self.conv2(x), kernel_size=self.ker_pool[1]))
        x = F.tanh(self.fc1(x.view(-1,self.chn[1]*(((size[2]-self.ker_conv[0]+1)/self.ker_pool[0])-self.ker_conv[1]+1)/self.ker_pool[1])))
        x = self.fc2(x)
        return x
    

def train_model(model, train_input, train_target, mini_batch_size, use_tol):
    
    criterion = nn.CrossEntropyLoss()
    eta = 1e-3

    for e in range(0, 500):
        sum_loss = 0
        optimizer = torch.optim.Adam(model.parameters(), lr = eta)
        for b in range(0, train_input.size(0), mini_batch_size):
            if b+mini_batch_size > train_input.size(0):
                mini_batch_size_var = train_input.size(0)-b
            else: mini_batch_size_var = mini_batch_size
            model.train()
            output = model(train_input.narrow(0, b, mini_batch_size_var))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size_var))
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.data[0]
            optimizer.step()
        if e%100 ==0:
            print(e, sum_loss)
        #if e%500 ==0:
         #   eta = 1e-4
        if sum_loss <= 10**(-10) and use_tol ==True:
            print('tolerance attained at iteration {}'.format(e))
            return

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        if b+mini_batch_size > data_input.size(0):
            mini_batch_size_var = data_input.size(0)-b
        else: mini_batch_size_var = mini_batch_size
        model.eval()
        output = model(data_input.narrow(0, b, mini_batch_size_var))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(0, mini_batch_size_var):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors





size = train_input.size()
use_cuda = True    


nb_hiddens = [1000]
mini_batch_sizes = [316]
train_error = np.zeros([len(nb_hiddens),len(mini_batch_sizes)])
test_error = np.zeros([len(nb_hiddens),len(mini_batch_sizes)])

mu, std = train_input.data.mean(), train_input.data.std()
train_input.data.sub_(mu).div_(std)


for h_, nb_hidden in enumerate(nb_hiddens):
    train_input, train_target = Variable(train_input), Variable(train_target)
    test_input, test_target = Variable(test_input), Variable(test_target)
    model = Net2(size, nb_hidden, [50,200], [5,4], [2,2])
    
    if use_cuda:
        train_input, train_target = train_input.cuda(), train_target.cuda()
        test_input, test_target = test_input.cuda(), test_target.cuda()
        model = model.cuda()
    for s_, mini_batch_size in enumerate(mini_batch_sizes):
        train_model(model, train_input, train_target, mini_batch_size, False)
        train_error[h_,s_] = compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100
        test_error[h_,s_] = compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
        print('train_error {:.02f}; test_error {:.02f}'.format(train_error[h_,s_],test_error[h_,s_]))
    
for s_ in range(len(mini_batch_sizes)):
    plt.plot(nb_hiddens,train_error[:,s_])
    plt.plot(nb_hiddens,test_error[:,s_])
legends = np.append(['tr, size:{0:0d}'.format(si) for si in mini_batch_sizes],['te, size:{0:0d}'.format(si) for si in mini_batch_sizes])
plt.legend(legends)
    
    
    