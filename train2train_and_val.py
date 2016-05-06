import os
import sys
import random

def save(filename,dic):
    with open(filename,'w') as fid:
        for k in dic:
            for i in dic[k]:
                fid.write('%s\t%s\n'%(i,k))
if __name__ == '__main__':
    '''
        Usage: python train2train_val.py data/path/to/fold1_train.txt
    '''
    train_path = sys.argv[1]
    val_portion = 0.1
    train_portion = 1- val_portion
    val_path = train_path.replace('train','val')
    os.system('cp %s %s.bkup'%(train_path,train_path))

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
        files = dic[k]
        ind = range(len(files))
        random.shuffle(ind)
        pivot = int(len(files)*train_portion)
        train_ind = [files[x] for x in ind[:pivot]]
        val_ind = [files[x] for x in ind[pivot:]]
        train_dic.update({k:train_ind})
        val_dic.update({k:val_ind})
    save(train_path,train_dic)
    save(val_path,val_dic)

