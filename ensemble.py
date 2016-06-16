#!/usr/bin/env python
# -*- coding: utf-8 -*-
from src.dataset import *
from task1_scene_classification import *
import sys
from pprint import pprint
import numpy as np
def get_prediction(result_filename):

    with open(result_filename, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            results.append(row)
    return results
if __name__ == "__main__":

    dataset = TUTAcousticScenes_2016_DevelopmentSet('data/')

    save_path = sys.argv[2]
    check_path(save_path)

    ensemble_yaml = sys.argv[1]
    ensemble_params = load_parameters(ensemble_yaml)
    yamls = ensemble_params['models'].split()
    y_pred = []
    for i_file in range(1,5):
        for i_model in yamls:
            i_params = load_parameters(i_model)
            i_params = process_parameters(i_params)

            result_path = i_params['path']['results']
            results.append(get_prediction(os.path.join(result_path,'results_fold%d.txt'%i_file)))

        preds = np.array(results)
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
    import pdb; pdb.set_trace()


    sys.exit(1)
    do_system_evaluation(dataset,result,'folds')
