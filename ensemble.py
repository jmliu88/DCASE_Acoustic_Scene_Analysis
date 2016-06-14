#!/usr/bin/env python
# -*- coding: utf-8 -*-
from src.dataset import *
from task1_scene_classification import *
import sys
from pprint import pprint
if __name__ == "__main__":

    dataset = TUTAcousticScenes_2016_DevelopmentSet('data/')

    ensemble_yaml = sys.argv[1]
    ensemble_params = load_parameters(ensemble_yaml)
    yamls = ensemble_params['models'].split()
    for i_model in yamls:
        i_params = load_parameters(i_model)
        i_params = process_parameters(i_params)

        result_path = i_params['path']['results']
        result = do_gether_results(dataset, result_path, 'folds')
    results = dataset
    result = []

    do_system_evaluation(dataset,result,'folds')
