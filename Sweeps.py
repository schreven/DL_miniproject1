# -*- coding: utf-8 -*-
"""
Created on Tue May  8 19:24:37 2018

@author: Bob
"""
import numpy as np


def Sweep_nb_hidden_batch_size(nb_hiddens, mini_batch_sizes):
    train_error = np.zeros([len(nb_hiddens),len(mini_batch_sizes)])
    test_error = np.zeros([len(nb_hiddens),len(mini_batch_sizes)])
    
    for h_, nb_hidden in enumerate(nb_hiddens):
        train_input, train_target = Variable(train_input), Variable(train_target)
        test_input, test_target = Variable(test_input), Variable(test_target)
        model = Net2(size, nb_hidden, chn_conv, ker_conv, ker_pool, len_IN_lin)
        
        if use_cuda:
            train_input, train_target = train_input.cuda(), train_target.cuda()
            test_input, test_target = test_input.cuda(), test_target.cuda()
            model = model.cuda()
        for s_, mini_batch_size in enumerate(mini_batch_sizes):
            train_model(model, criterion, train_input, train_target, mini_batch_size, False)
            train_error[h_,s_] = compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100
            test_error[h_,s_] = compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
            print('train_error {:.02f}; test_error {:.02f}'.format(train_error[h_,s_],test_error[h_,s_]))
        
    for s_ in range(len(mini_batch_sizes)):
        plt.plot(nb_hiddens,train_error[:,s_])
        plt.plot(nb_hiddens,test_error[:,s_])
    legends = np.append(['tr, size:{0:0d}'.format(si) for si in mini_batch_sizes],['te, size:{0:0d}'.format(si) for si in mini_batch_sizes])
    plt.legend(legends)