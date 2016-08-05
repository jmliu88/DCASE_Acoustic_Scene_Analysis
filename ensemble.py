#!/usr/bin/env python
# -*- coding: utf-8 -*-
from src.dataset import *
from task1_scene_classification import *
import pdb
import sys
from pprint import pprint
import numpy as np

def to_string(entry):
    return " ".join(entry)
def calc_weight(dataset,result_path_list, mode):
    dic = {}
    [dic.setdefault(k,0) for k in result_path_list]
    for k in result_path_list:
        try:
            k = os.path.dirname(k)

            result = do_system_evaluation(dataset,k,mode)
            dic [k] = result['overall_accuracy']
        except:
            dic[k] = 0
    return dic

def prun_correleted_result(result_path_list, thres):

    result_dic = get_result_list(result_path_list)
    pick = {}
    max_uncor = 1
    correlation_matrix = np.zeros(shape=(len(result_dic),len(result_dic)))
    for i in range(len(result_dic.values())):
        for j in range(len(result_dic.values())):
            i_entry = [to_string(k) for k in result_dic.values()[i]]
            j_entry = [to_string(k) for k in result_dic.values()[j]]
            correlation_matrix[i,j] = sum(np.array(i_entry) != np.array(j_entry)) /float(len(i_entry))
    uncorrelated_sum = np.mean(correlation_matrix,axis = 1)
    max_uncor = np.max(uncorrelated_sum)
    max_uncor_ind = np.argmax(uncorrelated_sum)
    key = result_dic.keys()[max_uncor_ind]
    pick.update({key:result_dic.pop(key)})
    while max_uncor > thres:
        correlation_matrix = np.zeros(shape=(len(result_dic),len(pick)))
        for i in range(len(result_dic.values())):
            for j in range(len(pick.values())):

                i_entry = [to_string(k) for k in result_dic.values()[i]]
                j_entry = [to_string(k) for k in pick.values()[j]]
                correlation_matrix[i,j] = sum(np.array(i_entry) != np.array(j_entry)) /float(len(i_entry))
        uncorrelated_sum = np.mean(correlation_matrix,axis = 1)
        max_uncor = np.max(uncorrelated_sum)
        max_uncor_ind = np.argmax(uncorrelated_sum)
        key = result_dic.keys()[max_uncor_ind]
        pick.update({key:result_dic.pop(key)})

        pdb.set_trace()
    return pick


#def uncorrelated_vote(result_path_list, thres = 0.1):
#    result_dic = get_result_list(result_path_list)
#    result_dic = prun_correleted_result(result_dic, thres)
#    entry_list = [x[0] for x in result_dic.values()[0]]
#
#    entry_dic = {k:{} for k in entry_list}
#    for i_path in result_dic:
#        for i_line in result_dic[i_path]:
#            dirname = os.path.dirname(i_path)
#            entry_key, lab = i_line
#            try:
#                entry_dic[entry_key][lab] += weights[dirname]
#            except:
#                entry_dic[entry_key].update( {lab: weights[dirname]})
#
#    results = []
#    for i_entry in entry_dic:
#        v = entry_dic[i_entry]
#        lab = v.keys()[np.argmax(v.values())]
#        results.append('%s\t%s'%(i_entry,lab))
#
#    return results

def weighted_vote(result_path_list, weights):
    result_dic = get_result_list(result_path_list)
    entry_list = [x[0] for x in result_dic.values()[0]]

    entry_dic = {k:{} for k in entry_list}
    for i_path in result_dic:
        for i_line in result_dic[i_path]:
            dirname = os.path.dirname(i_path)
            entry_key, lab = i_line
            try:
                entry_dic[entry_key][lab] += weights[dirname]
            except:
                entry_dic[entry_key].update( {lab: weights[dirname]})

    results = []
    for i_entry in entry_dic:
        v = entry_dic[i_entry]
        lab = v.keys()[np.argmax(v.values())]
        results.append('%s\t%s'%(i_entry,lab))

    return results

        #dic[entry_list] = {result_dic[i_path]:]}
def get_prediction(result_filename):

    results=[]
    with open(result_filename, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            results.append(row)
    return results
def get_result_list(results_list):
    result = []
    dic = {}
    for i in results_list:
        try:
            i_result = get_prediction(i)
            dic[i] = i_result
            result.append(i_result)
        except:
            print i
            continue

    return dic
def vote(result_list):
    result_list = result_list.values()
    pred = []
    for i in range(len(result_list[1])):
        i_sample = [result_list[x][i] for x in range(len(result_list))]
        i_sample = ['\t'.join(x) for x in i_sample]
        dic={}
        l = i_sample[0]
        for i_pred in i_sample:
            dic[i_pred]= i_sample.count(i_pred)

            for k in dic:
                if dic[k]>dic[l]:
                    l = k
        pred.append(l)
    return pred
def write(savename,pred):
    with open(savename,'w') as fid:
	for i in pred:
	    fid.write('%s\n'%(i))


if __name__ == "__main__":

    dataset = TUTAcousticScenes_2016_DevelopmentSet('data/')

    save_path = sys.argv[2]
    check_path(save_path)

    ensemble_yaml = sys.argv[1]
    ensemble_params = load_parameters(ensemble_yaml)
    yamls = ensemble_params['models'].split()
    y_pred = []
    results = []

    if os.path.splitext(yamls[0])[-1] == '.yaml':
        for i_file in range(1,5):
            result_path_list = []
            for i_model in yamls:
                i_params = load_parameters(i_model)
                i_params = process_parameters(i_params)

                result_path = i_params['path']['results']
                result_path_list.append( os.path.join(result_path,'results_fold%d.txt'%i_file))
            if ensemble_params['type'] == "majority_vote":
                result_list = get_result_list(result_path_list)
                results = vote(result_list)
            if ensemble_params['type'] == "weighted_majority_vote":
                weight = calc_weight(dataset,result_path_list,'folds')
                results = weighted_vote(result_path_list, weight)
                #results.append(get_prediction(os.path.join(result_path,'results_fold%d.txt'%i_file)))
            if ensemble_params['type'] == "uncorrelated_vote":
                result_list = prun_correleted_result(result_path_list, 0.08 )
                print len(result_list)
                results = vote(result_list)
                pass

            save_file=os.path.join(save_path,'results_fold%d.txt'%i_file)
            write(save_file,results)

        do_system_evaluation(dataset,save_path,'folds')
    else:
        ## ensemble sumbision results
        result_path_list = []
        for i_model in yamls:
            result_path_list.append(i_model)
        if ensemble_params['type'] == "majority_vote":
            result_list = get_result_list(result_path_list)
            results = vote(result_list)
            #results.append(get_prediction(os.path.join(result_path,'results_fold%d.txt'%i_file)))
        save_file=os.path.join(save_path,'results_fold0.txt')
        write(save_file,results)
