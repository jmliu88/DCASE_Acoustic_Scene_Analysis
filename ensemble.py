#!/usr/bin/env python
# -*- coding: utf-8 -*-
from src.dataset import *
from task1_scene_classification import *
import sys
from pprint import pprint
import numpy as np
if __name__ == "__main__":

    dataset = TUTAcousticScenes_2016_DevelopmentSet('data/')

    ensemble_yaml = sys.argv[1]
    ensemble_params = load_parameters(ensemble_yaml)
    yamls = ensemble_params['models'].split()
    y_pred = []
    for i_model in yamls:
        i_params = load_parameters(i_model)
        i_params = process_parameters(i_params)

        result_path = i_params['path']['results']
        y_true, i_y_pred, result = do_gether_results(dataset, result_path, 'folds')
        y_pred.append(i_y_pred)

    preds = np.array(y_pred)
    if ensemble_params['type'] == "majority_vote":
        for i in range(preds.shape[1]):
            i_sample = preds[:,i]
            i_sample = i_sample.tolist()
            dic={}
            for i_pred in i_sample:
                dic[i_pred]= i_sample.count(i_pred)

            l = dic.keys()[0]
            for k in dic:
	        if dic[k]>dic[l]:
	            l = k
            pred.append(l)


    sys.exit(1)
    do_system_evaluation(dataset,result,'folds')
