#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# dnn classifier

'''
import numpy
import theano
import theano.tensor as T
import lasagne

feature_data:  array-(1501,60)
data['beach']: array-(*   ,60)

'''
# create neural network
def build_dnn(feature_data, depth=3, width = 1024, drop_input=.2, drop_hidden=.5):
	# feature_data: numpy.ndarray [shape=(t, feature vector length)]
	# depth: number of hidden layers
	# width: number of units in each hidden layer
	feature_length = feature_data.shape[1]   # feature_data shape???
	network = lasagne.layers.InputLayer(shape=(None,1,feature_length),
										input_var = data)
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
	network = lasagne.layers.DenseLayer(network, 15, nonlinearity=softmax)
	return network
    

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) = len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs)-batchsize+1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


# train dnn 
def do_train(data, num_epochs=10):
	'''
	return 
	'''
    # prepare theano variables for inputs and targets
	input_var = T.tensor3('inputs')
	target_var = T.ivector('targets')
	
	network = build_dnn(input_var,
						int(depth),	int(width),
						float(drop_in),	float(drop_hid))

	# create a loss expression for training
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction,target_var)
	loss = loss.mean()
	# could add some weight decay here

	# create update expressions for training: SGD with momentum
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
		loss, params, learning_rate=0.01, momentum=0.9)

	# create a loss expression for validation/testing
	# 'deterministic' disable dropout layers
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_val)
	test_loss = test_loss.mean()

	# return test_prediction and target, delete code for err & acc
	
	
	# create an expression for classification accuracy
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatx)

	# compile a function performing a training step on a mini-batch
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	# compile a function computing the validation loss and accuracy
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	# training loop
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1
		
		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(
			val_acc / val_batches * 100))

	'''
	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
		inputs, targets = batch
		err, acc = val_fn(inputs, targets)
		test_err += err
		test_acc += acc
		test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
			test_acc / test_batches * 100))
	'''

    model_params = lasagne.layers.get_all_param_values(network)
    return model_params
	

'''
def load_feature_data():
	print("Loading features...")
	pass
	return data



if __name__ == '__main__':
	data = load_feature_data()
	
'''

def build_model(model_params):
	input_var = T.tensor3('inputs')
	target_var = T.ivector('targets')
	
	network = build_dnn(input_var,
						int(depth),	int(width),
						float(drop_in),	float(drop_hid))
	lasagne.layers.set_all_params(network,model_params)

	prediction = lasagne.layers.get_output(network, deterministic=True)

	predict = theano.function([input_var], prediction)

	return predict

# do_classification_dnn: classification for given feature data
def do_classification_dnn(feature_data, predict):
	'''
	input feature_data
	return classification results
	'''
	decision = predict(feature_data)






















