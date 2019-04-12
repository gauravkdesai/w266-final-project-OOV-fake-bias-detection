# This is classification module 

### In this module where we have all the python code to run classifier end to end given model files. 

This module has following 3 files

1. wiki_parser.py -> Runs parser for given corpus of url, it reaches to web and extracts wiki features into npy files for each feature
2. classification.py -> This is copy of classification.py from Baly's Git repo. We had to create a copy so that we keep our classification consistent with base paper
3. run_classifier.py -> This script is called from results_comparisons.ipynb notebook to run classification on all 48 sets of model*features*labels
