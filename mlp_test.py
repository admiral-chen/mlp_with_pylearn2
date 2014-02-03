__authors__ = "Yifei Chen"

import sys
import os
import random

import numpy as np
import theano

import pylearn2
import pylearn2.datasets as p2dts
import pylearn2.models as p2mdl
import pylearn2.costs as p2cst
import pylearn2.training_algorithms as p2alg
import pylearn2.termination_criteria as p2tercri



def main():
    n_input = 978
    n_hidden1 = 200
    n_hidden2 = 200
    n_output = 1
    w_decay_coef = [1e-4, 1e-4, 1e-4]


    '''
    Data
    '''
    # load raw complete dataset
    X, y = p2dts.geo.build_X_y(target_gene_idx)
    # normalize
    X_ava = X.mean(axis=0); X_std = X.std(axis=0)
    X = (X - X_ava[numpy.newaxis,:]) / X_std[numpy.newaxis,:]
    y = (y - y.mean()) / y.std()
    # partition into train, valid, test
    random.seed(0)
    sample_idx_train = random.sample(range(12031), 9600)
    sample_idx_test = [x for x in range(12031) if x not in sample_idx_train]
    X_tr = X[sample_idx_train[0:8000], :];  y_tr = y[sample_idx_train[0:8000]];
    X_va = X[sample_idx_train[8001:9600], :];  y_tr = y[sample_idx_train[8001:9600]];
    X_te = X[sample_idx_test, :];   y_te = y[sample_idx_test];
    # wrap up into pylearn2 format
    dataset_train = p2dts.dense_design_matrix.DenseDesignMatrix(X=X_tr, y=y_tr)
    dataset_valid = p2dts.dense_design_matrix.DenseDesignMatrix(X=X_va, y=y_va)
    dataset_test = p2dts.dense_design_matrix.DenseDesignMatrix(X=X_te, y=y_te)

    
    '''
    Model
    '''
    model = p2mdl.mlp.MLP(\
        nvis = n_input,
        layers = [
            p2mdl.mlp.Tanh(layer_name='h1', dim=n_hidden1, istdev=0.01),
            p2mdl.mlp.Tanh(layer_name='h2', dim=n_hidden2, istdev=0.01),
            p2mdl.mlp.Linear(layer_name='y', dim=n_output, istdev=0.01)
            ]
        )


    '''
    Algorithm
    '''
    algorithm = p2alg.sgd.SGD(
        batch_size=100, learning_rate=0.1, initial_momentum=0.5,
        monitoring_dataset = {'train': dataset_train, 'valid': dataset_valid},
        termination_criterion = p2tercri.Or(
            criteria = [p2tercri.EpochCounter(max_epochs=2000),
                        p2tercri.MonitorBased(
                            channel_name="valid_objective",
                            prop_decrease=1e-5, N=4)]),
        cost = p2cst.cost.SumOfCosts(costs=[p2cst.costs.mlp.Default(),
                                            p2cst.costs.mlp.WeightDecay(coeffs=w_decay_coef)]),
        update_callbacks = p2alg.sgd.ExponentialDecay(decay_factor=1.0000003, min_lr=.000001)
        )


    '''
    Extensions
    '''
    extensions=[p2alg.sgd.MomentumAdjustor(start=0, saturate=200, final_momentum=.99)]


    '''
    Train
    '''
    train = pylearn2.train.Train(dataset=dataset_train, model=model,
                                 algorithm=algorithm, extensions=extensions,
                                 save_path=save_path, save_freq=100)
    train.main_loop()


    '''
    Test
    '''
    # TODO
    # scale back




if __name__=='__main__':
    main()
