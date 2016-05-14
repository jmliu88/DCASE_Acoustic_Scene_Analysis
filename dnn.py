#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# dnn classifier
import lasagne
import theano
import theano.tensor as T
import numpy as np
import time

from src.ui import *
from src.general import *
from src.files import *

from src.features import *
from src.dataset import *
from src.evaluation import *
import batch
import pdb

def onehot(y):
    y_hat = np.zeros(shape = (y.shape[0], 15),dtype='int32')
    for i in range(y.shape[0]):
        y_hat[i,int(y[i])] = 1
    return y_hat
def calc_error(data_test, predict):
    ''' return error, cost on that set'''

    b = batch.Batch(data_test, max_batchsize=5000, seg_window=15, seg_hop=5)
    err = 0
    cost_val=0
    eps = 1e-50
    for (x,y_lab,_) in b:
        decision=predict(x.reshape((x.shape[0],-1)).astype('float32')) + eps
        pred_label= np.argmax(decision,axis=-1)
        y = onehot(y_lab)

        cost_val += -np.sum(y*np.log(decision))
        #pdb.set_trace()
        err += np.sum( (pred_label!= y_lab ))
    err = err/len(b.index_bkup)
    cost_val = cost_val /len(b.index_bkup)
    return err , cost_val

# create neural network
def build(input_var, depth=3, width = 1024, num_class=15, drop_input=.2, drop_hidden=.5, feat_dim=60):
    # feature_data: numpy.ndarray [shape=(t, feature vector length)]
    # depth: number of hidden layers
    # width: number of units in each hidden layer
    #feature_length = feature_data.shape[1]    # feature_data shape???
    network = lasagne.layers.InputLayer(shape=(None,feat_dim),
                                        input_var = input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    # create hidden layers and dropout
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(network,
                                            width,
                                            nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)

    # output layer
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, num_class, nonlinearity=softmax)
    return network

# train dnn
def do_train(data, data_val, data_test,  **classifier_parameters):
    '''
    return ??
    '''

    batch_maker = batch.Batch(data, isShuffle = True, seg_window=15, seg_hop=5)
    num_epochs = 10000
    #num_epochs = 3
    # prepare theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.imatrix('targets')  # ??

    network = build(input_var,**classifier_parameters)

    # create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction,target_var)
    loss = loss.mean()

    # create update expressions for training: SGD with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    train = theano.function([input_var, target_var],loss , updates=updates)
    # create a loss expression for validation/testing
    # 'deterministic' disable dropout layers
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    predict = theano.function( [input_var], lasagne.layers.get_output(network, deterministic=True))

    # training loop
    print("Starting training...")
    epoch = 0
    #no_best = 1
    no_best = 70
    best_cost = np.inf
    best_epoch = epoch
    model_params = []
    # TO REMOVE
    #model_params.append(lasagne.layers.get_all_param_values(nnet))
    while epoch < num_epochs:

        start_time = time.time()
        cost_train = 0
        for _, (x ,y ,_) in enumerate(batch_maker):
            x =x.reshape((x.shape[0], -1))
            y=onehot(y)

            assert(not np.any(np.isnan(x)))
            cost_train+= train(x, y) *x .shape[0]#*x .shape[1]
            assert(not np.isnan(cost_train))
        cost_train = cost_train/ len(batch_maker.index_bkup)
        err_val, cost_val = calc_error(data_val,predict)

        err_test, cost_test = calc_error(data_test,predict)
            #cost_val, err_val = 0, 0
        #pdb.set_trace()
        end_time = time.time()

        print "epoch: {} ({}s), training cost: {}, val cost: {}, val err: {}, test cost {}, test err: {}".format(epoch, end_time-start_time, cost_train, cost_val, err_val, cost_test, err_test)
        model_params.append(lasagne.layers.get_all_param_values(network))
        check_path('dnn')
        save_data('dnn/epoch_{}.autosave'.format(epoch), (classifier_parameters, model_params[best_epoch]))
        #savename = os.path.join(modelDir,'epoch_{}.npz'.format(epoch))
        #files.save_model(savename,structureDic,lasagne.layers.get_all_param_values(nnet))
        is_better = False
        if cost_val < best_cost:
            best_cost =cost_val
            best_epoch = epoch
            is_better = True
        if epoch - best_epoch >= no_best:
            ## Early stoping
            break
        epoch += 1
    return (classifier_parameters, model_params[best_epoch])



def build_model(model_params):
    input_var = T.matrix('inputs')

    network = build(input_var, **model_params[0])
    lasagne.layers.set_all_param_values(network,model_params[1])

    prediction = lasagne.layers.get_output(network, deterministic=True)

    predict = theano.function([input_var], prediction)

    return predict

# do_classification_dnn: classification for given feature data
def do_classification(feature_data, predict):
    '''
    input feature_data
    return classification results
    '''
    # ???
    x, _ = batch.make_batch(feature_data,15,5)
    decision = predict(x.reshape((x.shape[0],-1)))
    pred_label = np.argmax(np.sum(decision,axis=0), axis = -1)
    return batch.labels[pred_label]
