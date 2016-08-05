import os
import sys
import random
import numpy as np

def save(filename,dic):
    with open(filename,'w') as fid:
        for k in dic:
            for i in dic[k]:
                fid.write('%s\t%s\n'%(i,k))

def get_recordings_with_same_id(files, rand_int):
    record_id_val = os.path.basename(files[rand_int]).split('_')[0]
    val_ind = []
    ind = rand_int
    while ind<len(files):
        record_id = os.path.basename(files[ind]).split('_')[0]
        if record_id == record_id_val:
            val_ind.append(ind)
            ind = ind+1
        else:
            break
    ind = rand_int
    while ind>0:
        ind = ind-1
        record_id = os.path.basename(files[ind]).split('_')[0]
        if record_id == record_id_val:
            val_ind.append(ind)
        else:
            break
    return np.sort(val_ind)
if __name__ == '__main__':
    '''
        Usage: python train2train_val.py data/path/to/fold1_train.txt
    '''
    train_path = sys.argv[1]
    val_portion = 0.1
    train_portion = 1- val_portion
    val_path = train_path.replace('train','val')
    os.system('cp %s.bkup %s'%(train_path,train_path))
    #os.system('cp %s %s.bkup'%(train_path,train_path))

    with open(train_path,'r') as fid:
        all_line = fid.readlines()

    dic = {}
    for i in all_line:
        name, lab = i.strip().split()
        try:
            dic[lab].append(name)
        except:
            dic.update({lab:[name]})

    train_dic = {}
    val_dic = {}

    for k in dic:
        train_ind = []
        files = dic[k]
        indices= range(len(files))
        rand_int = random.randint(0, len(files)-1)
        rand_int2 = random.randint(0, len(files)-1)
        while abs(rand_int - rand_int2) < 6:
            rand_int2 = random.randint(0, len(files))
        val_ind = get_recordings_with_same_id(files,rand_int)
        val_ind = np.concatenate((val_ind ,get_recordings_with_same_id(files,rand_int2)),axis=0)
        train_ind = np.delete(indices,val_ind)

        train_files = [files[x] for x in train_ind]
        val_files = [files[x] for x in val_ind]
        train_dic.update({k:train_files})
        val_dic.update({k:val_files})
    save(train_path,train_dic)
    save(val_path,val_dic)

