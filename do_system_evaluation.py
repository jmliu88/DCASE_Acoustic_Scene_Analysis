#!/usr/bin/env python
# -*- coding: utf-8 -*-
from src.dataset import *
import task1_scene_classification
import sys
if __name__ == "__main__":
    result_file = sys.argv[1]
    if os.path.splitext(result_file)[-1] == '.yaml':
	params = task1_scene_classification.load_parameters(result_file)
	params = task1_scene_classification.process_parameters(params)
	result = params['path']['results']
    else:
        result = result_file
    print result
    dataset = TUTAcousticScenes_2016_DevelopmentSet('data/')
    task1_scene_classification.do_system_evaluation(dataset,result,'folds')
