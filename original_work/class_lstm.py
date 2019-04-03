# File to define our LSTM class

# Useful library for embeddings
import gensim
from string import ascii_lowercase, ascii_uppercase

import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence 
from keras import backend as K

import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, initializers
from tensorflow.keras.layers import Dropout, Embedding, Dense, LSTM, Bidirectional, Flatten
from tensorflow.keras.models import load_model
import sys
from sklearn.metrics.pairwise import cosine_similarity


class MimicLSTM():
    
    def __init__(self, layers, H, chardict, character_dim, data_dictionary, epochs=5, batch_size=100,
                 optimizer='adam', loss_function="mean_squared_error", train=True, full_set=False, load_path=''):
        
        """Init function.
        Args:
          layers: number of bidirectional LSTM layers
          H: hidden state dimension
          chardict: map of characters to indices
          character_dim: Embedding length for each character to be used in model
          output_dim: output dimension we are predicting
          data_dictionary: dictionary of words as keys and associated embeddings as values
          max_len: longest sequence in training data; needed to pad data
          optimizer: defaults to adam but can accept another input
        """
                    
        self.layers = layers
        self.H = H
        self.chardict = chardict
        self.character_dim = character_dim
        self.output_dim = int(list(data_dictionary.values())[0].shape[0])
        self.data_dictionary = data_dictionary
        self.optimizer=optimizer
        self.loss = loss_function
        
        if train:
            self.train_w, self.train_e, self.test_w, self.test_e, \
            self.words_index, self.embeddings = self.preprocess_data()

            self.lstm = self.build_model()
            self.train_model(epochs, batch_size, full_set)
            
        elif len(load_path)>0:
            self.load_model(load_path)
           
         
    def build_model(self):
        """
        Create Keras neural model based on input specifications
        """
        with tf.name_scope("model_creation"):
            
            # Initialize a Keras sequential model
            lstm = Sequential()
            
            # Add an embedding layer which houses are character level embeddings.  We use a 
            # mask to coincide with our earlier padding.  Embedding rows is equal to our character 
            # count, +1 for our 0 mask, and +1 because function is up to but not including that number
            lstm.add(Embedding(len(self.chardict)+2, self.character_dim, input_length=self.max_len, 
                               embeddings_initializer=initializers.RandomUniform(-.1,.1), mask_zero=True))
            
            # ADd a bidirectional LSTM layer for each layer in self.layers.  Last layer only outputs final
            # result, but intermediate layers return full sequence.  We default to concating forward
            # and backward outputs across all layers.
            for x in range(0,self.layers):
                if x == self.layers-1:
                    lstm.add(Bidirectional(LSTM(self.H, return_sequences=False),
                                                merge_mode='concat'))
                else:
                    lstm.add(Bidirectional(LSTM(self.H, return_sequences=True), merge_mode='concat'))
            
            # Add a fully connected layer which results in predictions equal to our stated output_dim
            lstm.add(Dense(self.output_dim, activation='tanh'))
            
#             def custom_loss(y_true, y_pred):
#                 y_true = tf.math.l2_normalize(y_true)
#                 y_pred = tf.math.l2_normalize(y_pred)
#                 loss = tf.losses.cosine_distance(y_true, y_pred)
#                 return loss
            
            def custom_cosine(y_true, y_pred):
                """
                Keras cosine_proximity doesn't subtract 1, 
                so can't be used as loss function
                """
                y_true = K.l2_normalize(y_true, axis=-1)
                y_pred = K.l2_normalize(y_pred, axis=-1)
                return 1-K.sum(y_true * y_pred, axis=-1)
           
            # Compile model and view high level summary for analysis
            lstm.compile(loss=self.loss, optimizer=self.optimizer)
            lstm.summary()
            
            
        return lstm

    
    def calc_max_len(self, train_words):
        """
        helper function to calculate longest sequence in data
        """
        max_len=0
        for word in train_words:
            if len(word) > max_len:
                max_len = len(word)
        return max_len
    
    
    def pad_data(self, sequences):
        """
        helper function which left pads sequences based on self.max_len
        """
        padded_data = sequence.pad_sequences(sequences, maxlen=self.max_len)
        return padded_data
    
    
    def preprocess_data(self):
        """
        Preprocess self.data_dictionary into padded sequences which can be 
        used for training and testing
        """
        words = []
        embeddings = []

        for k,v in self.data_dictionary.items():
            words.append(k)
            embeddings.append(v)

        # Convert characters to index references and all lists to numpy arrays
        words_index = [[self.chardict[char] for char in word] for word in words]
        words_index = np.array(words_index)
        embeddings = np.array(embeddings)

        # Establish train/test splits 
        train_w, test_w, train_e, test_e = train_test_split(words_index, 
                                                            embeddings, test_size=0.1, random_state=1)
        self.max_len = self.calc_max_len(train_w)
        
        # Pad our training and test data so we can handle variable length sequences
        train_w = self.pad_data(train_w)
        test_w = self.pad_data(test_w)
        
        return train_w, train_e, test_w, test_e, words_index, embeddings
    
    
    def train_model(self, epochs, batch_size, full_set):
        """
        Train precompiled LSTM model with specified parameters.  If full_set=True we are
        training the final model on all data, but if False we are training on subset
        and validating resultings.
        
        :param epochs: Number of epochs we should run in training
        :batch_size: number of examples to use before making gradient update
        :full_set: boolean of whether we are training on all data, or using train/test split
        """
        
        if full_set:
            full_data = self.pad_data(self.words_index)
            self.lstm.fit(full_data, self.embeddings, epochs=epochs, batch_size=batch_size)
        else:
            self.lstm.fit(self.train_w, self.train_e, validation_data = (self.test_w, self.test_e),
                          epochs = epochs, batch_size = batch_size)
        
        return 
    
                          
    def save_model(self, model, path):
        """
        helper function to save trained keras model
        """
        model.save(path) 
                          
        return "Save Succesful"
             
                          
    def load_model(self, path):
        """
        Helper function to load pretrained keras model
        """
        try:              
            self.lstm = load_model(path)
        except:
            print("Could not load file, not valid path")
                          
        return
                          
        


