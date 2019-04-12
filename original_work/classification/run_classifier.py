#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import classification as nmrc
importlib.reload(nmrc)
import pandas as pd
import concurrent.futures
import numpy as np
import pdb

news_media_data_folder_location='''../../News-Media_Reliability/data/'''
news_media_corpus_file = news_media_data_folder_location+'''corpus.csv'''
news_media_feature_location = news_media_data_folder_location+'''features/'''

model1_feature_location = '''../features/mimic_model_original.h5/'''
model2_feature_location = '''../features/mimic_model_complex.h5/'''
model3_feature_location = '''../features/mimic_model_complex_attention.h5/'''
model4_feature_location = '''../features/mimic_model_complex_attention_cosine_loss.h5/'''


def run_classification(name, feature_location, corpus = news_media_corpus_file
                       , task = 'bias', features = ['has_wiki'] ):
	'''Runs classification on given set of features in .nmpy format 
	and returns MAcro F1, Accuracy, MAE and Macri MAE

	params:
		name: Name identifying the model. Can be used for reporting purpose
		feature_location: folder name where .npy files for each of the features is located
		corpus : corpus file location where all labelled url data is stored
		task : bias or fact, which label needs to be classified
		features : list of features which should be used to extract .npy files 
			and used as input for classifier

	Returns:
		a dict containing
			name : same as input, indicates the model that was used
			feature_location : same as input, location of features
			task : same as input, bias or fact
			features : comma separated string of features
			F1 : Macro F1 score of classification
			Accuracy : Accuracy of classsification
			MAE : L1 score of classification
			MAE_U: Macro MAE returned by classification

	'''
    
    result = nmrc.Classification(corpus, features, task,feature_location)
    result_dict = {
                     'name'             : name
                    ,'feature_location' : feature_location
                    ,'task'             : task
                    ,'features'         : ",".join(features)
                    ,'F1'               : result[0]
                    ,'Accuracy'         : result[1]
                    ,'MAE'              : result[2]
                    ,'MAE_U'            : result[3]}
    
    return result_dict  


def run_all_classifiers():
	'''Runs classification on all 5 models for each of the two tasks and 6 
	features combinations. So total this function runs 60 classifications serially.
	Adds results of each classification in a list and returns that list of dict


	'''
	tasks = ['bias','fact']
	features = [ ['has_wiki'], ['wikicontent'], ['wikisummary'], ['wikitoc'], ['wikicategories']
	           ,['has_wiki','wikicontent','wikisummary','wikitoc','wikicategories']]
	feature_locations = [news_media_feature_location,model1_feature_location,model2_feature_location, model3_feature_location, model4_feature_location]
	model_names = ['original','original_replica','complex', 'complex_with_attention', 'complex_with_attention_cosine_loss']
	corpus = news_media_corpus_file

    result_list=[]
    for i,model_name in enumerate(model_names):
        feature_location =feature_locations[i]
        for task in tasks:
            for feature_list in features:
                print('corpus',corpus)
                print('feature_list',feature_list)
                print('task',task)
                print('feature_location',feature_location)

                result_list.append(run_classification(model_name, feature_location, corpus, task, feature_list))
                print('Completed {} classifications'.format(len(result_list)))
    return result_list

        
