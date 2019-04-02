# File to train LSTM model
import gensim
from string import ascii_lowercase, ascii_uppercase

import numpy as np


def load_word2vec(path):
    """
    Load pretrained word2vec model from file
    :param path: pathway to find pretrained word2vec model
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True) 
    return model


def create_char_dicts(non_letter_chars, lower_case=True, upper_case=True)
    """
    Create dictionary mapping characters to indices
    :param non_letter_chars: list of characters which should be supported other than letters
    :param lower_case: Should set of english lowercase letters be included; default True
    :param upper_case: Should set of english uppercase letters be included; default True
    """
    lower_case_letter_dict={}
    upper_case_letter_dict={}
    index_count = 1
    # Create a dictionary with upper and lower case letters and associated index
    # Note: We include underscores, hyphens, and apostrophes but ignore other characters
    # found in word2vec model, including chinese symbols, emojis, etc
    if lower_case:
        lower_case_letter_dict = {letter: int(index)+index_count for index, letter in enumerate(ascii_lowercase, start=1)}
        index_count += 26
    if upper_case:
        upper_case_letter_dict = {letter: int(index)+index_count for index, letter in enumerate(ascii_uppercase, start=1)} 
        index_count += 26
        
    chardict = {**lower_case_letter_dict, **upper_case_letter_dict}
    
    for char in non_letter_chars:
        chardict[char] = index_count
        index_count += 1

    # Creation of reverse character lookup for debugging and word creation
    reverse_chardict = {}
    for k,v in chardict.items():
        reverse_chardict[v] = k
    
    return chardict, reverse_chardict


def create_data_dict(model, chardict):
    """
    Using pretrained word2vec model and supported character dictionary generate
    a dictionary with valid words as keys and associated word2vec embeddings as values
    
    :param model: loaded word2vec model
    :param chardict: dictionary of charaters (keys) and associated indices (values)
    """
    
    # Less than 25 isn't part of original paper, but word2vec has some outrageous entries
    def include_word(word, chardict):
        """
        Function to determine if word can be included and perform any parsing
        """
        if (all(char in chardict.keys() for char in word)) & (len(word)<=25):
            return True

        return False


    # Create list of words which will be used in training/testing our model
    all_words = dict({})

    # For every word in word2vec model establish if it is "allowed"; if it is
    # add the word to our all_words dict, with the embedding as the value
    for idx, key in enumerate(model.wv.vocab):
        if include_word(key, chardict):
            all_words[key] = model.wv[key]
        else:
            pass
        
    return all_words


def train_model():
    return

def run():
    model = load_word2vec('./word2vec_model/GoogleNews-vectors-negative300.bin')
    supported_non_letter_characters = ['.','-','\'']
    chardict, reverse_chardict = create_char_dicts(supported_non_letter_characters)
    data_dict = create_data_dict(model, chardict)
    train_model()
    
    
    
    
    