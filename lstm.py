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
        l_forward_1.forgetgate=lasagne.layers.gate(w_in=lasagne.init.normal(0.1), w_hid=lasagne.init.normal(0.1), w_cell=lasagne.init.normal(0.1), b=lasagne.init.constant(1.))#w_in_to_ingate.set_value(val['w_in_to_ingate'].astype('float32'))
        l_backward_1.forgetgate=lasagne.layers.gate(w_in=lasagne.init.normal(0.1), w_hid=lasagne.init.normal(0.1), w_cell=lasagne.init.normal(0.1), b=lasagne.init.constant(1.))#w_in_to_ingate.set_value(val['w_in_to_ingate'].astype('float32'))

        layer = lasagne.layers.ReshapeLayer(layer , (-1,n_hidden_blstm))

    for iLayer in range(n_dense):
        layer = lasagne.layers.DropoutLayer(layer,p = dropout_rate_dense)
        layer = lasagne.layers.DenseLayer(layer, num_units= n_hidden_dense, nonlinearity = lasagne.nonlinearities.leaky_rectify,W = lasagne.init.Normal(0.1,0),b = lasagne.init.Constant(1)) ## W_{yh_back}+b
        layers['dense_%d'%iLayer] = layer
    layer= lasagne.layers.DenseLayer(
        layer, num_units=n_class, nonlinearity=lasagne.nonlinearities.sigmoid, W = lasagne.init.Normal(0.1,0),b = lasagne.init.Constant(1),
        name='Dense Softmax')
    layers['output'] = layer
    #pdb.set_trace()
    l_out = lasagne.layers.ReshapeLayer(layer, (-1 ,max_length))
    layers['out'] =l_out
    return l_out, layers
def do_train_lstm(data, dropout_rate_blstm = 0.2, dropout_rate_dense = 0.2, n_layers = 3, n_dense = 3, n_hidden_blstm = 125, n_hidden_dense= 256, n_class =n_class, max_length = 1000, feat_dim = 60):
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
    input_var = T.tensor3('input')
    mask = T.matrix('mask')
    target_output = T.matrix('target_output')
    nnet,layers = build(input_var, mask, dropout_rate_blstm =dropout_rate_blstm, dropout_rate_dense = dropout_rate_dense, n_layers =n_layers, n_dense = n_dense , n_hidden_blstm = n_hidden_blstm , n_hidden_dense=  n_hidden_dense, n_class =n_class, max_length = 1000, feat_dim =feat_dim)

    pass

def do_classification_lstm(feature_data, model_container):
    current_result = 1
    return current_result
