#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implement Label-Spreading
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA

import cifar10
from ls import Label_spreading

def get_pca_data(n_data, data_dim, dataset):
    def mnist():
        dataset = fetch_mldata('MNIST original')
        n_train = 60000
        data = dataset['data'].astype(np.float32)[:n_train] / 255.
        label = dataset['target'].astype(np.int32)[:n_train]
        return n_train, data, label
    
    def cifar():
        dataset = cifar10.load()
        n_train = 50000
        data = dataset['train']['data'].reshape((n_train, 32*32*3)) / 255.
        label = dataset['train']['target']
        return n_train, data, label
    
    get_data = (mnist)if(dataset == 'mnist')else(cifar)
    n_train, data, label = get_data() 
    pca = PCA(n_components = data_dim)
    app_pca = pca.fit(data).transform(data)
    
    perm = np.random.permutation(n_train)
    train_data = app_pca[perm[:n_data]].reshape((n_data, data_dim))
    test_data = app_pca[perm[n_data:]].reshape((n_train-n_data, data_dim))
    print('train_data_dim:{}, val_data_dim{}'.format(train_data.shape, test_data.shape))
    train_label = label[perm[:n_data]]
    test_label = label[perm[n_data:]]
    return train_data, train_label, test_data, test_label

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_data', default=10000, type=int)
    parser.add_argument('--data_dim', default=2, type=int)
    parser.add_argument('--dataset', default='mnist', type=str)
    args = parser.parse_args()
    
    print('pca+label-spreading')
    train_data, train_target, test_data, test_target = get_pca_data(args.n_data, args.data_dim, args.dataset)
    
    n_testlabel = len(test_target)
    array = np.concatenate((test_data, train_data), axis=0)
    target = np.concatenate((test_target, train_target), axis=0)
    
    labels = -np.ones(len(test_target) + len(train_target))
    for i in range(len(train_target)):
        labels[len(test_target)+i] = train_target[i]
    
    ####################################
    print("Start label spreading...")
    ls = Label_spreading(array, labels, target, n_testlabel)
    ls.fit()
    ls.get_predict_labels()
    ls.evaluate()
    print('accuracy=', ls.accuracy)   
