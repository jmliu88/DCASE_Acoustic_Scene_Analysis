import numpy as np
import pdb
import random
import sys

from src.files import *
labels = load_labels('data/TUT-acoustic-scenes-2016-development/labels.txt')
#['residential_area',
#            'bus',
#            'cafe',
#            'car',
#            'city_center',
#            'forest_path',
#            'grocery_store',
#            'home',
#            'library',
#            'metro_station',
#            'office',
#            'train',
#            'tram',
#            'urban_park',
#            'beach',
#            'cafe/restaurant',
#            'park',
#            'lakeside_beach']
def label2index(lab):
    ''' Find the corresponding index of label from the global label variable.
        input:
        -----
        label in string
        output:
        -----
        index of the label
    '''
    dic = dict(zip(labels,range(len(labels))))
    try:
        ind = dic[lab]
    except KeyError as e:
        print "lab:%s is not in dict."%lab
        raise(e)
    return ind

class Batch():
    ''' Generate batch from data.
        e.g.:
        Batch_maker = Batch(data)
    '''
    def __init__(self, data, seg_window = 1000, seg_hop = 100,  max_batchsize = 100,isShuffle = False):
        self.data = data
        self.seg_window = seg_window
        self.seg_hop = seg_hop
        self.get_seg()
        self.index = range(len(self.seg)) # index of samples. access data by this index later
        self.index_bkup = self.index
        self.isShuffle = isShuffle
        if isShuffle:
            random.shuffle(self.index)
        self.batchsize= max_batchsize
        self.n_batch = len(self.index)/self.batchsize
    def get_seg(self):
        data = self.data
        seg_window = self.seg_window
        seg_hop = self.seg_hop
        seg = []
        for k in data:
            num_frames = data[k].shape[0]
            for i_start in range(0,(num_frames-seg_window)/seg_hop):
                seg.append((k,i_start*seg_hop,i_start*seg_hop+seg_window))
        self.seg = seg
    def get_batch(self):
        #pdb.set_trace()
        index = self.index
        batchsize = min(self.batchsize, len(self.index))
        x = np.zeros(shape=(batchsize,self.seg_window,self.data.values()[0].shape[1]),dtype='float32')
        y = np.zeros(shape=(batchsize,1),dtype='float32')
        m = np.ones(shape=(batchsize,self.seg_window),dtype='float32')
        for i in range(batchsize):
            k,st,en = self.seg[index[i]]
            x[i] = self.data[k][st:en]
            assert(not np.any(np.isnan(x[i])))
            y[i] = label2index(k)
        self.index = np.delete(self.index,range(batchsize))

        return x,y,m
## auxiliary functions below. makes Batch iteratable
    def reset(self):
        self.index = self.index_bkup # index of samples. access data by this index later
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
def make_batch(feature_data, size=1000, hop=100):
    length = feature_data.shape[0]
    if length>size:
        n_seg = (length-size)/hop
        x = np.zeros(shape=(n_seg, size,feature_data.shape[1]),dtype='float32')
        m = np.ones(shape=(n_seg, size),dtype='float32')
        for i in range(0,n_seg):
            x[i] = feature_data[i*hop:i*hop+size]
    else:
        x = np.lib.pad(feature_data,((0,size-length),(0,0)), 'constant', constant_values = 0)
        x = np.expand_dims(x,axis=0)
        m = np.zeros(shape=(1,size))
        m[:,:length] = 1
    return x,m
if __name__ == '__main__':
    label2index('a')
    pass
    ## untested
    #data = sys.argv[1]
    #batch = batch_training(data)
    #len(batch .seg)
