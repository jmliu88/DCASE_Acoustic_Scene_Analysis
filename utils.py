import numpy as np
def onehot(y):
    y_hat = np.zeros(shape = (y.shape[0], 15),dtype='int32')
    for i in range(y.shape[0]):
        y_hat[i,int(y[i])] = 1
    return y_hat
def framewise_onehot(y, length=1000):
    y_hat = np.zeros(shape = (y.shape[0], length, 15))
    for i in range(y.shape[0]):
        y_hat[i,:,int(y[i])] = 1
    return y_hat
