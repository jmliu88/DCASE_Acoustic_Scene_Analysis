import numpy as np
import pdb
import random
import sys
def label2index(lab):
# TODO
    pass
class Batch():
    def __init__(self, data, seg_size = 1000,  MAX_BATCH_SIZE = 100,isShuffle = False):
        self.data = data
        self.seg_size = seg_size
        self.get_seg()
        self.index = range(len(self.seg)) # index of samples. access data by this index later
        if isShuffle:
            random.shuffle(self.index)
        self.batchsize= MAX_BATCH_SIZE
        self.n_batch = len(self.index)/self.batchsize
    def get_seg(self):
        data = self.data
        seg_size = self.seg_size
        seg = []
        for k in data:
            num_frames = data[k].shape[0]
            for i_start in range(0,(num_frames-seg_size)/seg_size):
                seg.append((k,i_start,i_start+seg_size))
        self.seg = seg
    def get_batch(self):
        pdb.set_trace()
        batchsize = min(self.batchsize, len(self.index))
        x = np.zeros(shape=(batchsize,self.seg_size,self.data.values()[0].shape[1]))
        y = np.zeros(shape=(batchsize,1))
        m = np.ones(shape=(batchsize,self.seg_size,self.data.values()[0].shape[1]))
        for i in range(batchsize):
            k,st,en = self.seg[i]
            x[i] = self.data[k][st:en]
            y[i] = label2index(k)
        self.index = np.delete(self.index,range(batchsize))

        return x,y,m
## auxiliary functions below. makes Batch iteratable
    def reset(self):
        self.index = range(len(self.seg)) # index of samples. access data by this index later
        if self.isShuffle:
            random.shuffle(self.index)
    def __iter__(self):
        return self
    def next(self):
        if len(self.index) == 0:
            self.reset()
            raise StopIteration
        else:
            return self.get_batch()
if __name__ == '__main__':
    pass
    ## untested
    #data = sys.argv[1]
    #batch = batch_training(data)
    #len(batch .seg)
