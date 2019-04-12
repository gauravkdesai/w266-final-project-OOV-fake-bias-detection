#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json as js
import re

import tensorflow as tf
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras_self_attention import SeqSelfAttention

import pandas as pd
import numpy as np
import os

import gensim

from Embedding_Model_Creation import word2vec_preprocess
import pdb
import concurrent.futures


def get_wiki_api_url():
	'''Returns wikipedia api URL which can be used to pragmatically 
	extract different sections of wikipedia page'''
	return 'https://en.wikipedia.org/w/api.php'


def has_wiki_for_url(website_url):
	'''Returns True if wikipedia page exists for given website 
	else returns False'''
    wiki_info = requests.get(
        get_wiki_api_url(),
        params={
            'action': 'query',
            'format': 'json',
            'titles': website_url,
            'redirects' : ''

         }
    ).json()
    #print(js.dumps(wiki_info,indent=2) )
    page_0 = list(wiki_info['query']['pages'].values())[0]
    missing = 'missing' in page_0.keys()
    
    return not missing

def get_wiki_title_for_url(website_url):
	'''Gets wikipedia page title for given website if exists. 
	Else returns None'''
    wiki_info = requests.get(
        get_wiki_api_url(),
        params={
            'action': 'query',
            'format': 'json',
            'titles': website_url,
            'redirects' : ''

         }
    ).json()
    #print(js.dumps(wiki_info,indent=2) )
    page_0 = list(wiki_info['query']['pages'].values())[0]
    missing = 'missing' in page_0.keys()
    
    return None if missing else page_0['title']

def get_wiki_summary(title):
	'''Extracts summary from wikipedia page for given wikipedia title'''
    summary_extract = requests.get(
        get_wiki_api_url(),
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'export':None,
            'exintro': True,
            'explaintext': True,
         }
    ).json()
    summary = list(summary_extract['query']['pages'].values())[0]['extract']
    return summary

def get_wiki_content(title):
	'''Extracts content of wikipedia page for given wikipedia title'''

    content_extract = requests.get(
        get_wiki_api_url(),
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'revisions',
            'rvprop':'content',
            'formatversion':'2',
            'rvslots':'*'
         }
    ).json()
    raw_content = content_extract['query']['pages'][0]['revisions'][0]['slots']['main']['content']
    return raw_content

def get_wiki_toc(title):
	'''Extracts table of content of wikipedia page for given wikipedia title'''

    sections = requests.get(
        get_wiki_api_url(),
        params={
            'action': 'parse',
            'format': 'json',
            'page': title,
            'prop': 'sections',
            'rvprop':'content',
            'formatversion':'2',
            'rvslots':'*'
         }
    ).json()
    raw_content = " ".join([section['line'] for section in sections['parse']['sections']])
    return raw_content

def get_wiki_categories(title):
	'''Extracts categories under which wikipedia page is categorized for 
	given wikipedia title'''

    categories = requests.get(
        get_wiki_api_url(),
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'categories',
            'rvprop':'content',
            'formatversion':'2',
            'rvslots':'*'
         }
    ).json()
    raw_content = " ".join([category['title'] for category in categories['query']['pages'][0]['categories']])
    return raw_content

def remove_between_tags(text,start_tag,end_tag):
	'''Removes sub text falling inside start_tag and end_tag from give text

		params:
			text: String where you want to search for tags
			start_tag: start of tag or marker from where you want sub text 
				to be removed from text. e.g. <pre>
			end_tag: end of tag or marker till you want to remove sub text 
				from text. e.g. </pre>

		Returns:
			String which is subset of text with all the content falling 
			within start_tag and end_tag removed (including start_tag and end_tag)

	'''
    
    tag_pattern = start_tag+'[\s\S]*?'+end_tag
    return_text = re.sub(tag_pattern ,'', text, count=0)
    return return_text

def remove_between_tags_except_starts_with(text, start_tag, end_tag, starts_with):
	'''Removes sub text falling inside start_tag and end_tag from give text 
		except when subtext starts with 
		specific pattern in which case we keep everything as it is

		params:
			text: String where you want to search for tags
			start_tag: start of tag or marker from where you want sub text 
				to be removed from text. e.g. <pre>
			end_tag: end of tag or marker till you want to remove sub text 
				from text. e.g. </pre>
			starts_with: starting pattern of sub text that you want to keep

		Returns:
			String which is subset of text with all the content falling 
			within start_tag and end_tag removed except when the sub text 
			starts with 'starts_with' pattern in which case we keep the text 
			unchanged

	'''
    new_start_tag = start_tag+'(?!'+starts_with+')'
    return_text = remove_between_tags(text, new_start_tag, end_tag)
    
    return return_text

def remove_special_text(text, special_text):
	'''Remove all occurrences of a particular pattern from text non greedily'''

    return_text = re.sub(special_text ,' ', text, count=0)
    return return_text

def preprocess_content(text):
	'''Removes unwanted tags and patterns from text and returns the remaining 
	txt back. In particular it removes the tags like <ref> and any content 
	within the tag, any http links,	any braces or colons etc.

	'''
	tags_to_be_omitted = [('<ref.*>','</ref>')]
	tags_to_be_omitted_except_starts_with = [('{{','}}','hatnote')]
	patterns_to_be_deleted = ['{{hatnote\|', '}}', '\(', '\)', ']]', '\n', ':'
							  , '''\[\[''','''\[http.*\]''','='
	                          ,'\*', '<ref .*/>', '\|']


    #text=text.lower()
    for tag_set in tags_to_be_omitted:
        text = remove_between_tags(text, tag_set[0], tag_set[1])
    
    for tag_set in tags_to_be_omitted_except_starts_with:
        text = remove_between_tags_except_starts_with(text, tag_set[0], tag_set[1], tag_set[2])
        
    for pattern in patterns_to_be_deleted:
        text = remove_special_text(text, pattern)
        
    return text

def get_manual_wiki_override():
	'''For websites where wiki api can not find wikipedia page we have manually 
	found the wikipedia page and stored in 'wikipedia_data/wiki_manual_searched_pages.csv'. 
	This function will load the manual override file and return it as pandas 
	dataframe with two columns
	Input.url : website url for which we want to find out wikipedia page
	Answer.website : wikipedia page url 

	'''
	manual_wiki_data_file_name = '../wikipedia_data/wiki_manual_searched_pages.csv'
	manual_wiki_data = pd.read_csv(manual_wiki_data_file_name)
	return manual_wiki_data

def get_wiki_title_from_manual_file(url, manual_wiki_data):
	'''Gets wikipedia page title from manual override file for given website url'''

    if manual_wiki_data is not None:
        wiki_website= manual_wiki_data[manual_wiki_data['Input.url']==url]['Answer.website']
        #print('wiki_website',wiki_website)
        #print(wiki_website.size)
        if wiki_website.size > 0 and wiki_website.iloc[0] is not np.NaN and wiki_website.iloc[0] != 'N/A':
            wiki_word_split = wiki_website.iloc[0].split('/wiki/')
            if len(wiki_word_split)>1:
                return wiki_word_split[1]
    return None                          

def get_wiki_data_for_url(website_url, manual_wiki_data):
	'''Extracts wikipedia features for given website url. 
	If wikipedia api can not find wikipedia website then we also try manual 
	override file which we have built manually for cases where api fails to 
	find wikipedia page for given url

	'''
    data = {}
    data['url'] = website_url
    try:
        #print('website_url',website_url)
        # Check if the url is available in wiki as redirects
        wiki_title = get_wiki_title_for_url(website_url)
        #print('url title extracted',website_url)
        if wiki_title is None:
            wiki_title = get_wiki_title_from_manual_file(website_url, manual_wiki_data)
        if wiki_title is None:
            data['has_wiki'] = 0
        else:
            data['has_wiki'] = 1

            wiki_summary = preprocess_content(get_wiki_summary(wiki_title))
            data['wiki_summary'] = wiki_summary

            wiki_content = preprocess_content(get_wiki_content(wiki_title))
            data['wiki_content'] = wiki_content

            wiki_toc = preprocess_content(get_wiki_toc(wiki_title))
            data['wiki_toc'] = wiki_toc

            wiki_categories = preprocess_content(get_wiki_categories(wiki_title))
            data['wiki_categories'] = wiki_categories
    except Exception as e:
        print('Exception {} while getting wiki data for url {}'.format(e,website_url))
        data['has_wiki'] = 0
    return data



def get_wiki_data(website_urls):
	'''Returns wikipedia information for given list of urls
		params:
			website_urls: list of urls for which wikipedia information needs to be extracted

		Returns:
			a list of dict with each item in list corresponding to one url in website_urls
	

	'''
	#List containing dictionary of each url
    data_list = []

    # future object list for multi threaded process
    future_list = []

    #Get manual override file for cases where wiki api fails to find wiki page automatically
    manual_wiki_data = get_manual_wiki_override()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for website_url in website_urls:
            future_list.append(executor.submit(get_wiki_data_for_url,website_url,manual_wiki_data))

        # make sure we extract the results in same order as input so that labels can be matched in classification
        for future in future_list:
            data_list.append(future.result())
            if len(data_list)%100 == 0:
                print('Retrieved features for {} urls from wiki'.format(len(data_list)))

    print('Retrieved features for {} urls from wiki'.format(len(data_list)))
    return data_list

def encode_fact(fact_label):
	'''Encode factuality label from text to number for classification'''

	labels = {}
	labels['fact'] = {'low': 0, 'mixed': 1, 'high': 2}

    fact_label = fact_label.lower()
    return labels['fact'][fact_label]

def encode_bias(fact_label):
	'''Encode bias label from text to number for classification'''

	labels = {}
	labels['bias'] = {'extreme-right': 0, 'right': 1, 'right-center': 2
				, 'center': 3, 'left-center': 4, 'left': 5, 'extreme-left': 6}
    fact_label = fact_label.lower()
    return labels['bias'][fact_label]



def process_wiki_data(reload_wiki_urls=True):
	'''Reads website urls from news media corpus file, extracts wiki data for
	each of the url, and adds fact and bias labels from news media corpus at the
	end of each record. Wraps up all this information in pandas dataframe and 
	saves in a csv file. Also returns the dataframe for further use

	params:
		reload_wiki_urls: if we want to extract wiki data again then pass this True.
			Else the function will reuse already saved wiki extract from previous 
			run which will boost the performance by reducing web hits.

	Returns:
		Pandas dataframe containing url, its wiki features and fact and bias labels 	

	'''

	news_media_corpus_file = '''../../News-Media_Reliability/data/corpus.csv'''
	wiki_corpus_file = '''../wikipedia_data/wiki_corpus.csv'''

    if reload_wiki_urls:
        print('Retrieving url from news media corpus file')
        corpus_data = pd.read_csv(news_media_corpus_file)
        #print(corpus_data)
        source_urls = list(corpus_data['source_url_processed'])
        print('{} urls retrieved from news media corpus file'.format(len(source_urls)))
        wiki_data_for_urls = get_wiki_data(source_urls)
        print('Got {} url features from wiki'.format(len(wiki_data_for_urls)))
        wiki_df = pd.DataFrame(wiki_data_for_urls)
        
        corpus_data['fact'] = corpus_data['fact'].apply(lambda row:encode_fact(row))
        wiki_df = pd.concat([wiki_df,corpus_data['fact'] ], axis=1)
        
        corpus_data['bias'] = corpus_data['bias'].apply(lambda row:encode_bias(row))
        wiki_df = pd.concat([wiki_df, corpus_data['bias']], axis=1)
        #print(wiki_df)
        print('Writing wiki data to wiki_corpus.csv')
        wiki_df.to_csv(wiki_corpus_file) 
        print('Wiki data written to wiki_corpus.csv successfully')
    else:
        print('reload_wiki_urls is set to False, so not scrapping information from web again')
        wiki_df = pd.read_csv(wiki_corpus_file)

    return wiki_df

def get_char_dict_from_word2vec():
	'''Gets character dictionary from word2vec using gensim'''

	print('Loading word2vec from binary file')
	wordVec = gensim.models.KeyedVectors.load_word2vec_format(
	    '../word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)  
	print('Loading word2vec from binary file is complete')
	data_dict, char_dict=word2vec_preprocess.run()
	return char_dict

def get_padding(sequence, max_padding):
	'''Since our models expect word of max_padding length we clip the word to 
	max_padding length or left pad 0 to make length max_padding'''
    sequence = sequence[:max_padding] if len(sequence)>max_padding else sequence
    result = np.zeros(max_padding)
    result[max_padding-len(sequence):]=sequence
    return result

def get_padded_sequence(char_dict, token, max_padding=25):
	'''For given word extracts character encoding from chara_dict and clip 
	at max_padding'''
    sequence = [char_dict[c] if c in char_dict else 0 for c in token]
    return np.reshape(get_padding(sequencem, max_padding), (1,-1))

def load_lstm_model(path, model_h5_name):
    # Grabs our already created LSTM model which we trained and is
    # super awesome from some pathway
    if model_h5_name in model_load_custom_object_attention:
        model = load_model(path
                  , custom_objects=SeqSelfAttention.get_custom_objects())
    else:    
        model = load_model(path)
    return model

def return_avg_embedding(tokens, model, char_dict):
    '''Takes a list of tokens and finds average embedding.  If embedding does not exist
    in Word2Vec, we predict a proxy using "model"
    
    param 
    	tokens: list of english tokens
    	model: ML model which can predict 300d embedding from string input
    	char_dict: character dictionary from word to vector
    
    Return: 
    	returns a 300d embedding which is average of all tokens passed into function
    '''
    
    global wordVec
    
    # Initialize an array of zeros and a counter for averaging
    final_embedding = np.zeros(300)
    count = 0
    
    # For each token check if pre-trained word2vec exists, otherwise predict
    # using pre-trained neural model 
    if tokens is None or tokens is np.nan:
        return final_embedding

    for token in tokens.split():
        if not wordVec.vocab.get(token):
            # Predict embedding from ML model
            sequence = get_padded_sequence(char_dict, token)
            embedding = model.predict(sequence) 
            embedding=np.reshape(embedding, 300)
            
        else:
            embedding = wordVec[token]
      
        final_embedding += embedding
        count += 1
    
    # Find average value of embedding
    avg_embedding = final_embedding/count
    avg_embedding = np.reshape(avg_embedding,(300,1))

    return avg_embedding

## Model Paths



features_not_require_embeddings = ['has_wiki']

def calculate_and_save_feature_file(feature_file_dir, feature_key, wiki_df, current_model, char_dict):
	'''Calculate average embedding for each feature using a given model 
	and saves the embedding in numpy format

	params:
		feature_file_dir: Directory where numpy file containing embeddings for 
			given feature to be saved
		feature_key: feature name for which model needs to be run and numpy 
			file to be created
		wiki_df : dataframe containing wiki information for given feature 
			along with fact and bias labels
		current_model: model trained for providing embeddings for word not 
			seen in word2vec
		char_dict: Word 2 Vec character dictionary

	'''
	feature_files = {'has_wiki'     : '''has_wiki.npy''', 
                  'wiki_summary'    : '''wikisummary.npy''', 
                  'wiki_content'    : '''wikicontent.npy''',
                  'wiki_toc'        : '''wikitoc.npy''',
                  'wiki_categories' : '''wikicategories.npy'''
                }

    #pdb.set_trace()
    print(' ')
    feature_file_name = feature_file_dir+feature_files[feature_key]
    print('feature_file_name',feature_file_name)
    feature_file_df = pd.DataFrame(wiki_df['url'])


    print('Extracting feature', feature_key)

    if feature_key in features_not_require_embeddings:
        print('This feature does not need embedding, Saving {} records as it is'.format(wiki_df[feature_key].shape))
        feature_file_df = pd.concat([feature_file_df, wiki_df[feature_key]], axis=1)
        #print(feature_file_df)

    else:
        average_embedding_for_feature = wiki_df.apply(lambda row: return_avg_embedding(row[feature_key]
                                                                                  , current_model, char_dict)
                                                      , axis=1)
        print('Computed average embedding for feature {} for {} records'.format(feature_key
                                                                            , len(average_embedding_for_feature)))
        print('Average embedding shape',average_embedding_for_feature.shape)
        #print(average_embedding_for_feature)
        #print(type(average_embedding_for_feature))
        embedding_df = pd.DataFrame()
        for i in range(300):
            embedding_df[str(i)] = [em[i] for em in average_embedding_for_feature]
        #print('embedding_df',embedding_df)                               
        feature_file_df = pd.concat([feature_file_df, embedding_df], axis=1)
        #np.save(feature_file_name, average_embedding_for_feature)
        print('Average embeddings for feature {} saved to file {}'.format(feature_key,feature_file_name ))

    feature_file_df = pd.concat([feature_file_df, wiki_df['fact']], axis=1)
    feature_file_df = pd.concat([feature_file_df, wiki_df['bias']], axis=1)

    number_of_columns = len(feature_file_df.columns)
    column_names = range(number_of_columns)
    feature_file_df.columns = column_names
    print('Feature file shape',feature_file_df.shape)
    #print('feature_file_df',feature_file_df)
    np.save(feature_file_name, feature_file_df)

def run_wiki_parser():
    '''Extracts wiki features from web for given corpus file
    Pre-processes the wiki information with predefined rules
	Convert the features into 300d embeddings by using word2vec and custom models
	saves the embeddings in .npy files for each feature and each model used

    '''
    #Load wiki data either from saved file or scrapping the web
    wiki_df = process_wiki_data() 
    print('Wiki features loaded from file, records=',wiki_df.shape)
    #print(wiki_df)

    #Directory where models are saved in h5 file
    model_dir = '../Saved_Models'

    # model file names
	model_files =['''mimic_model_original.h5''',
	               '''mimic_model_complex.h5''',
	               '''mimic_model_complex_attention.h5''' ,
	               '''mimic_model_complex_attention_cosine_loss.h5'''
	]

	# models which require custom object while reading from file
	model_load_custom_object_attention = ['''mimic_model_complex_attention.h5'''
	                                      ,'''mimic_model_complex_attention_cosine_loss.h5''']

	# feature file location	                                      
	feature_dir='features'

	# char dict form word to vec
	char_dict = get_char_dict_from_word2vec()

    for model_file in model_files:
        model_path = model_dir+'''/'''+model_file
        print('-'*50)
        print('Using model from ',model_path)
        feature_file_dir = feature_dir+'/'+model_file+'/'
        if not os.path.exists(feature_file_dir):
            os.makedirs(feature_file_dir)

        current_model = load_lstm_model(model_path,model_file)

        for feature_key in feature_files.keys():
            calculate_and_save_feature_file(feature_file_dir,feature_key,wiki_df
            	, current_model, char_dict)

        print()
        print('Completed processing all {} features for model {}'
        	.format(len(feature_files),model_path))
        print('-'*50)
    
    print('-'*50)
    print('Finished with all {} models'.format(len(model_files)))
    
                                

if __name__ == '__main__':
    run_wiki_parser()






