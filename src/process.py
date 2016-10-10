#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

itr = 1
output_dim = 30 
dataset = 'mnist'

for n_train in [100, 500, 1000, 2000, 3000]:
    for i in range(itr):
        print("pca+label-spreading_trainlabeled:", n_train, "_outputdim:", output_dim)
        os.system( "python apply_pca_ls.py --n_data {} --data_dim {} --dataset {}".format(n_train, output_dim, dataset) )
