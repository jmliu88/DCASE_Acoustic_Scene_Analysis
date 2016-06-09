import lasagne
import theano
import theano.tensor as T
import numpy as np
import time
import batch
from layers import *
from utils import *

import pdb
from src.ui import *
from src.general import *
from src.files import *

from src.features import *
from src.dataset import *
from src.evaluation import *


def calc_error(b, predict):
    ''' return error, cost on that set'''

    eps = 1e-10
    err = 0
    cost_val=0
    for (x,y_lab,m) in b:
        x=batch.make_context(x,15)
        y = onehot(y_lab)
        decision=predict(x.astype('float32')) + eps
        pred_label= np.argmax(decision,axis=-1)

        cost_val += -np.sum(y*np.log(decision))
        #pdb.set_trace()
        err += np.sum( np.expand_dims(pred_label,axis=1) != y_lab )
    err = err/float(len(b.index_bkup))
    cost_val = cost_val /len(b.index_bkup)
    return err , cost_val

def build(input_var, dropout_rate_dense = 0.2, dropout_rate_pre = 0.2, n_layers_pre = 3, n_hidden_pre = 125, n_dense = 3, n_hidden_dense= 256, n_attention=10, n_class = 10, max_length = 1000, feat_dim = 60, return_layers = False):

    l_in = lasagne.layers.InputLayer(
        shape=(None, max_length, feat_dim), name='Input', input_var = input_var)
    #l_mask = lasagne.layers.InputLayer(
        #shape= (None, max_length), name='Mask input', input_var= mask)
    #input_shape = (None, max_length, feat_dim)
# Construct network
    #n_batch, n_seq, n_features = l_in .input_var.shape
# Store a dictionary which conveniently maps names to layers we will
# need to access later
    layer = l_in
    layers = {'in':layer}
    layer = lasagne.layers.GaussianNoiseLayer(layer)
    layer = lasagne.layers.ReshapeLayer(layer , (-1 ,feat_dim))
    for iLayer in range(n_layers_pre):
        layer = lasagne.layers.DropoutLayer(layer,p=dropout_rate_pre)
        layer = lasagne.layers.DenseLayer(layer, num_units=n_hidden_pre, nonlinearity = lasagne.nonlinearities.leaky_rectify,W = lasagne.init.Orthogonal(np.sqrt(2/(1+0.01**2))),b = lasagne.init.Constant(1)) ## W_{yh_back}+b
        layers['dense_pre_%d'%iLayer] = layer

    layer = lasagne.layers.ReshapeLayer(layer , (-1 ,max_length , n_hidden_pre))
    attention_layers = []
    att_class_logit = []
    for i_class in range(n_class):
        att_representation = AttentionLayer(layer)
        attention_layers.append(att_representation)
        for iLayer in range(n_dense):
            att_representation= lasagne.layers.DropoutLayer(att_representation,p = dropout_rate_dense)
            att_representation= lasagne.layers.DenseLayer(att_representation, num_units= n_hidden_dense, nonlinearity = lasagne.nonlinearities.leaky_rectify,W = lasagne.init.Orthogonal(np.sqrt(2/(1+0.01**2))),b = lasagne.init.Constant(1)) ## W_{yh_back}+b
            #layer = lasagne.layers.DenseLayer(layer, num_units= n_hidden_dense, nonlinearity = lasagne.nonlinearities.tanh,W = lasagne.init.Orthogonal(np.sqrt(2/(1+0.01**2))),b = lasagne.init.Constant(1)) ## W_{yh_back}+b
            layers['dense_%d_%d'%(iLayer,i_class)] =att_representation
        att_representation = lasagne.layers.DenseLayer(att_representation,num_units=1,nonlinearity=None)
        att_class_logit.append(att_representation)

    layer = lasagne.layers.ConcatLayer(att_class_logit)
    layer = lasagne.layers.NonlinearityLayer(layer,lasagne.nonlinearities.softmax)
    layers['output'] = layer
    #pdb.set_trace()
    #l_out = lasagne.layers.ReshapeLayer(layer, (-1 ,max_length, n_class))
    #layers['out'] =l_out
    layers['out'] = layer
    return layers['out'], layers
def cost_prev(output,target_output,mask):
    return -T.sum(mask.dimshuffle(0, 1, 'x') *
                target_output*T.log(output))/(mask.shape[0])
def cost(output,target_output):
    return -T.sum(target_output*T.log(output))/(output.shape[0])
def do_train(data, data_val, data_test, **classifier_parameters):
    ''' input
        -------
        data: {label:np.array(features)}
        classifier_param: {n_layers:3,...}

        output
        ------
        model_param = {structure:
                        {n_layers: int,
                         n_dense: int...},
                       params:lasagne.layers.get_all_params(l_out)
                      }
    '''
    import time
    batch_maker = batch.Batch(data, isShuffle = True, seg_window = classifier_parameters['max_length'], seg_hop = classifier_parameters['max_length']/2, max_batchsize=400)
    b_v = batch.Batch(data_val, isShuffle = True, seg_window = classifier_parameters['max_length'], seg_hop = classifier_parameters['max_length']/2, max_batchsize=400)
    b_t = batch.Batch(data_test, isShuffle = True, seg_window = classifier_parameters['max_length'], seg_hop = classifier_parameters['max_length']/2, max_batchsize=400)

    input_var = T.tensor3('input')
    mask = T.matrix('mask')
    target_output = T.matrix('target_output')
    network,layers = build(input_var, **classifier_parameters)

    eps = 1e-10
    loss_train = cost(lasagne.layers.get_output(
        network,  deterministic=False)+eps,target_output)
    loss_eval  = cost(lasagne.layers.get_output(
        network,  deterministic=True)+eps,target_output)
    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.adadelta(loss_train,all_params,learning_rate=1.0)
    #updates = lasagne.updates.momentum(loss_train , all_params,
                                    #learning_rate, momentum)
    pred_fun = lasagne.layers.get_output(
            network, deterministic=True)
    train = theano.function([input_var, target_output],loss_train , updates=updates)
    #compute_cost = theano.function([input_var, target_output, mask],loss_eval)
    predict = theano.function( [input_var], pred_fun)


#theano.config.warn_float64='pdb'
    print "start training"

    #err, cost_test = calc_error(data_val,predict)
    epoch = 0
    no_best = 70
    best_cost = np.inf
    best_epoch = epoch
    model_params = []
    # TO REMOVE
    #model_params.append(lasagne.layers.get_all_param_values(network))
    while epoch < 500:

        start_time = time.time()
        cost_train = 0
        for _, (x ,y ,m) in enumerate(batch_maker):
            x=batch.make_context(x,15)
            x =x .astype('float32')
            m=m.astype('float32')
            y = onehot(y)
            y=y.astype('float32')

            assert(not np.any(np.isnan(x)))
            cost_train+= train(x, y) *x .shape[0]#*x .shape[1]
            assert(not np.isnan(cost_train))
        cost_train = cost_train/ len(batch_maker.index_bkup)
        err_val, cost_val = calc_error(b_v,predict)

        err_test, cost_test = calc_error(b_t,predict)
            #cost_val, err_val = 0, 0
        #pdb.set_trace()
        end_time = time.time()

        is_better = False
        if cost_val < best_cost:
            best_cost =cost_val
            best_epoch = epoch
            is_better = True

        if is_better:
            print "epoch: {} ({}s), training cost: {}, val cost: {}, val err: {}, test cost {}, test err: {}, New best.".format(epoch, end_time-start_time, cost_train, cost_val, err_val, cost_test, err_test)
        else:
            print "epoch: {} ({}s), training cost: {}, val cost: {}, val err: {}, test cost {}, test err: {}".format(epoch, end_time-start_time, cost_train, cost_val, err_val, cost_test, err_test)

        sys.stdout.flush()
        model_params.append(lasagne.layers.get_all_param_values(network))
        #check_path('dnn')
        #save_data('dnn/epoch_{}.autosave'.format(epoch), (classifier_parameters, model_params[best_epoch]))
        #savename = os.path.join(modelDir,'epoch_{}.npz'.format(epoch))
        #files.save_model(savename,structureDic,lasagne.layers.get_all_param_values(network))
        if epoch - best_epoch >= no_best:
            ## Early stoping
            print "Training stops, best epoch is {}".format(best_epoch)
            break
        epoch += 1
    return (classifier_parameters, model_params[best_epoch])
def validate(data,data_val, predict):
    err_val, cost_val = calc_error(data_val,predict)
    err_train, cost_train = calc_error(data_val,predict)
    print err_val, cost_val
    print err_train, cost_train


def build_model(params, return_layers=False):
    input_var = T.tensor3('input')
    mask = T.matrix('mask')
    target_output = T.tensor3('target_output')

    network,layers = build(input_var, **params[0])
    lasagne.layers.set_all_param_values(network,params[1])

    pred_fun = lasagne.layers.get_output( network, deterministic=True)
    predict = theano.function( [input_var], pred_fun)

    if return_layers:
        return predict, layers
    return predict


def do_classification(feature_data, predict, params):
    length = params[0]['max_length']
    x, m = batch.make_batch(feature_data,length,length)
    x=batch.make_context(x,15)
    #decision = predict(np.expand_dims(feature_data,axis=0).astype('float32'), np.ones(shape=(1,feature_data.shape[0])))
    decision = predict(x)
    pred_label = np.argmax(np.sum(decision,axis=0), axis = -1)
    return batch.labels[pred_label]
