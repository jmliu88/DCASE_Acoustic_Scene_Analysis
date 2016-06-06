#!/usr/bin/env python
# -*- coding: utf-8 -*-
from src.dataset import *
import task1_scene_classification
import sys
if __name__ == "__main__":
    result = sys.argv[1]
    dataset = TUTAcousticScenes_2016_DevelopmentSet('data/')
    task1_scene_classification.do_system_evaluation(dataset,result,'folds')
