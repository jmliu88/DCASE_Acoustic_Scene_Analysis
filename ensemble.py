#!/usr/bin/env python
# -*- coding: utf-8 -*-
from src.dataset import *
from task1_scene_classification import *
import sys
from pprint import pprint
import numpy as np
def get_prediction(result_filename):

    results=[]
    with open(result_filename, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            results.append(row)
    return results
def get_result_list(results_list):
    result = []
    for i in results_list:
        result.append( get_prediction(i))
    return result
def vote(result_list):
    pred = []
    for i in range(len(result_list[1])):
	i_sample = [result_list[x][i] for x in range(len(result_list))]
	dic={}
	for i_pred in i_sample:
	    i_pred = '\t'.join(i_pred)
	    dic[i_pred]= i_sample.count(i_pred)

	l = dic.keys()[0]
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

    result_path_list = []
    for i_file in range(1,5):
        for i_model in yamls:
            i_params = load_parameters(i_model)
            i_params = process_parameters(i_params)

            result_path = i_params['path']['results']
            result_path_list.append( os.path.join(result_path,'results_fold%d.txt'%i_file))
        result_list = get_result_list(result_path_list)
        if ensemble_params['type'] == "majority_vote":
            results = vote(result_list)
            #results.append(get_prediction(os.path.join(result_path,'results_fold%d.txt'%i_file)))
        save_file=os.path.join(save_path,'results_fold%d.txt'%i_file)
        write(save_file,results)



    do_system_evaluation(dataset,save_path,'folds')
