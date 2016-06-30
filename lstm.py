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

def framewise_onehot(y, length=1000):
    y_hat = np.zeros(shape = (y.shape[0], length, 15))
    for i in range(y.shape[0]):
        y_hat[i,:,int(y[i])] = 1
    return y_hat

def calc_error(b, predict):
    ''' return error, cost on that set'''

    eps = 1e-10
    err = 0
    cost_val=0
    for (x,y,m) in b:
        x = batch.make_context(x,15)
        y = framewise_onehot(y,x.shape[1])
        decision=predict(x.astype('float32'),m.astype('float32')) + eps
        pred_label= np.argmax(decision,axis=2)
        y_lab = np.argmax(y,axis=2)

        cost_val += -np.sum(y*np.log(decision))
        #pdb.set_trace()
        err += np.sum( (pred_label!= y_lab ))
    err = err/len(b.index_bkup)
    cost_val = cost_val /len(b.index_bkup)
    return err , cost_val

def build(input_var,mask, dropout_rate_blstm = 0.2, dropout_rate_dense = 0.2, n_layers = 3, n_dense = 3, n_hidden_blstm = 125, n_hidden_dense= 256, n_class = 10, max_length = 1000, feat_dim = 60):

    l_in = lasagne.layers.InputLayer(
        shape=(None, max_length, feat_dim), name='Input', input_var = input_var)
    l_mask = lasagne.layers.InputLayer(
        shape= (None, max_length), name='Mask input', input_var= mask)
    #input_shape = (None, max_length, feat_dim)
# Construct network
    #n_batch, n_seq, n_features = l_in .input_var.shape
# Store a dictionary which conveniently maps names to layers we will
# need to access later
    layer = l_in
    layers = {'in':layer}
    layer = lasagne.layers.GaussianNoiseLayer(layer)
    #layer = lasagne.layers.ReshapeLayer(layer , (-1 ,feat_dim))
    for iLayer in range(n_layers):
        #layer = lasagne.layers.DropoutLayer(layer,p=dropout_rate_blstm)
        #layer = lasagne.layers.ReshapeLayer(layer , (-1 ,max_length,n_hidden_blstm))

        l_forward_1 = lasagne.layers.LSTMLayer(
            layer, num_units=n_hidden_blstm/2, grad_clipping=5, name='Forward LSTM %d'%iLayer, mask_input=l_mask)
        l_backward_1 = lasagne.layers.LSTMLayer(
            layer, num_units=n_hidden_blstm/2, grad_clipping=5, backwards=True, name='Backwards LSTM %d'%iLayer,
            mask_input=l_mask)
        layer = lasagne.layers.ConcatLayer(
            [l_forward_1, l_backward_1], axis=-1, name='Sum 1')
        layers['forward_%d'%iLayer] = l_forward_1
        layers['backward_%d'%iLayer] = l_backward_1
        layers['blstm_%d'%iLayer] = layer
        l_forward_1.forgetgate=lasagne.layers.Gate(W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1), b=lasagne.init.Constant(1.))#w_in_to_ingate.set_value(val['w_in_to_ingate'].astype('float32'))
        l_backward_1.forgetgate=lasagne.layers.Gate(W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1), b=lasagne.init.Constant(1.))#w_in_to_ingate.set_value(val['w_in_to_ingate'].astype('float32'))

    layer = lasagne.layers.ReshapeLayer(layer , (-1,n_hidden_blstm))

    for iLayer in range(n_dense):
        layer = lasagne.layers.DropoutLayer(layer,p = dropout_rate_dense)
        #layer = lasagne.layers.DenseLayer(layer, num_units= n_hidden_dense, nonlinearity = lasagne.nonlinearities.leaky_rectify,W = lasagne.init.Orthogonal(np.sqrt(2/(1+0.01**2))),b = lasagne.init.Constant(1)) ## W_{yh_back}+b
        layer = lasagne.layers.DenseLayer(layer, num_units= n_hidden_dense, nonlinearity = lasagne.nonlinearities.tanh,W = lasagne.init.Orthogonal(np.sqrt(2/(1+0.01**2))),b = lasagne.init.Constant(1)) ## W_{yh_back}+b
        layers['dense_%d'%iLayer] = layer
    layer= lasagne.layers.DenseLayer(
        layer, num_units=n_class, nonlinearity=lasagne.nonlinearities.softmax, W = lasagne.init.Normal(0.1,0),b = lasagne.init.Constant(1),
        name='Dense Softmax')
    layers['output'] = layer
    #pdb.set_trace()
    l_out = lasagne.layers.ReshapeLayer(layer, (-1 ,max_length, n_class))
    layers['out'] =l_out
    return l_out, layers
def cost_prev(output,target_output,mask):
    return -T.sum(mask.dimshuffle(0, 1, 'x') *
                target_output*T.log(output))/(mask.shape[0])
def cost(output,target_output,mask):
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
    import pdb
    batch_maker = batch.Batch(data, isShuffle = True, seg_window = classifier_parameters['max_length'], seg_hop = classifier_parameters['max_length']/2)
    b_v = batch.Batch(data_val, max_batchsize=500, seg_window = classifier_parameters['max_length'], seg_hop = classifier_parameters['max_length']/2)
    b_t = batch.Batch(data_test, max_batchsize=500, seg_window = classifier_parameters['max_length'], seg_hop = classifier_parameters['max_length']/2)

    input_var = T.tensor3('input')
    mask = T.matrix('mask')
    target_output = T.tensor3('target_output')
    nnet,layers = build(input_var, mask, **classifier_parameters)

    eps = 1e-10
    loss_train = cost(lasagne.layers.get_output(
        nnet,  deterministic=False)+eps,target_output,mask)
    loss_eval  = cost(lasagne.layers.get_output(
        nnet,  deterministic=True)+eps,target_output, mask)
    all_params = lasagne.layers.get_all_params(nnet, trainable=True)
    updates = lasagne.updates.adadelta(loss_train,all_params,learning_rate=1.0)
    #updates = lasagne.updates.momentum(loss_train , all_params,
                                    #learning_rate, momentum)
    pred_fun = lasagne.layers.get_output(
            nnet, deterministic=True)
    train = theano.function([input_var, target_output, mask],loss_train , updates=updates)
    #compute_cost = theano.function([input_var, target_output, mask],loss_eval)
    predict = theano.function( [input_var, mask], pred_fun)


#theano.config.warn_float64='pdb'
    print "start training"

    #err, cost_test = calc_error(data_val,predict)
    epoch = 0
    no_best = 70
    best_cost = np.inf
    best_epoch = epoch
    model_params = []
    # TO REMOVE
    #model_params.append(lasagne.layers.get_all_param_values(nnet))
    while epoch < 10000:

        start_time = time.time()
        cost_train = 0
        for _, (x ,y ,m) in enumerate(batch_maker):
            x =x .astype('float32')
            x = batch.make_context(x,15)
            m = np.ones_like(m)
            m=m.astype('float32')
            y = framewise_onehot(y, x.shape[1])
            y=y.astype('float32')

            assert(not np.any(np.isnan(x)))
            cost_train+= train(x, y, m) *x .shape[0]#*x .shape[1]
            assert(not np.isnan(cost_train))
        cost_train = cost_train/ len(batch_maker.index_bkup)
        err_val, cost_val = calc_error(b_v,predict)

        err_test, cost_test = calc_error(b_t,predict)
            #cost_val, err_val = 0, 0
        #pdb.set_trace()
        end_time = time.time()

        print "epoch: {} ({}s), training cost: {}, val cost: {}, val err: {}, test cost {}, test err: {}".format(epoch, end_time-start_time, cost_train, cost_val, err_val, cost_test, err_test)
        model_params.append(lasagne.layers.get_all_param_values(nnet))
        check_path('lstm')
        save_data('lstm/epoch_{}.autosave'.format(epoch), (classifier_parameters, model_params[best_epoch]))
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
def validate(data,data_val, predict):
    err_val, cost_val = calc_error(data_val,predict)
    err_train, cost_train = calc_error(data_val,predict)
    print err_val, cost_val
    print err_train, cost_train


def build_model(params):
    input_var = T.tensor3('input')
    mask = T.matrix('mask')
    target_output = T.tensor3('target_output')

    nnet,layers = build(input_var, mask, **params[0])
    lasagne.layers.set_all_param_values(nnet,params[1])

    pred_fun = lasagne.layers.get_output( nnet, deterministic=True)
    predict = theano.function( [input_var, mask], pred_fun)

    return predict


def do_classification(feature_data, predict, params):
    length = params[0]['max_length']
    x, m = batch.make_batch(feature_data,length,length)
    x = batch.make_context(x,15)
    #decision = predict(np.expand_dims(feature_data,axis=0).astype('float32'), np.ones(shape=(1,feature_data.shape[0])))
    decision = predict(x, m)
    pred_label = np.argmax(np.sum(decision,axis=(0,1)), axis = -1)
    return batch.labels[pred_label]
