import lasagne
import theano
#from lasagne_cough import run_lstm

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
    layer = lasagne.layers.ReshapeLayer(layer , (-1 ,feat_dim))
    for iLayer in range(n_layers):
        layer = lasagne.layers.DropoutLayer(layer,p=dropout_rate_blstm)
        layer = lasagne.layers.ReshapeLayer(layer , (-1 ,max_length,n_hidden_dense))

        l_forward_1 = lasagne.layers.LSTMLayer(
            layer, num_units=n_hidden_blstm/2, name='Forward LSTM %d'%iLayer, mask_input=l_mask)
        l_backward_1 = lasagne.layers.LSTMLayer(
            layer, num_units=n_hidden_blstm/2, backwards=True, name='Backwards LSTM %d'%iLayer,
            mask_input=l_mask)
        layer = lasagne.layers.ConcatLayer(
            [l_forward_1, l_backward_1], axis=-1, name='Sum 1')
        layers['forward_%d'%iLayer] = l_forward_1
        layers['backward_%d'%iLayer] = l_backward_1
        layers['blstm_%d'%iLayer] = layer

        layer = lasagne.layers.ReshapeLayer(layer , (-1,n_hidden_blstm))
        #layer = lasagne.layers.DropoutLayer(layer, p=dropout_rate)
        #layer = lasagne.layers.DenseLayer(layer , num_units=n_hidden_dense, nonlinearity=lasagne.nonlinearities.leaky_rectify, W = lasagne.init.Normal(0.1,0),b = lasagne.init.Constant(1)) ## W_{yh_back}+b
        #layer = lasagne.layers.DenseLayer(layer , num_units=n_hidden_dense, nonlinearity=lasagne.nonlinearities.linear, W = lasagne.init.Normal(0.1,0),b = lasagne.init.Constant(1)) ## W_{yh_back}+b

    #layer= lasagne.layers.ReshapeLayer(layer, (-1,n_hidden_dense))
    #layer= lasagne.layers.DenseLayer(
    #    layer, num_units=100, nonlinearity=lasagne.nonlinearities.tanh, W = lasagne.init.Normal(0.1,0),b = lasagne.init.Constant(1),
    #    name='preDense Softmax')
    #layers['dense_final'] = layer
    for iLayer in range(n_dense):
        layer = lasagne.layers.DropoutLayer(layer,p = dropout_rate_dense)
        layer = lasagne.layers.DenseLayer(layer, num_units= n_hidden_dense, nonlinearity = lasagne.nonlinearities.leaky_rectify,W = lasagne.init.Normal(0.1,0),b = lasagne.init.Constant(1)) ## W_{yh_back}+b
    layer= lasagne.layers.DenseLayer(
        layer, num_units=n_class, nonlinearity=lasagne.nonlinearities.sigmoid, W = lasagne.init.Normal(0.1,0),b = lasagne.init.Constant(1),
        name='Dense Softmax')
    layers['output'] = layer
    #pdb.set_trace()
    l_out = lasagne.layers.ReshapeLayer(layer, (-1 ,max_length))
    layers['out'] =l_out
def do_train_lstm(data, n_layers = 3, n_dense = 3, n_hidden_blstm = 125, n_hidden_dense= 256, n_class = 10):
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

    pass

def do_classification_lstm(feature_data, model_container):
    current_result = 1
    return current_result
