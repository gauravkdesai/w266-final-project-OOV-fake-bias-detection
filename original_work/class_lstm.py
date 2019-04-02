# File to define our LSTM class

# Useful library for embeddings
import gensim
from string import ascii_lowercase, ascii_uppercase

import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence 

import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, initializers
from tensorflow.keras.layers import Dropout, Embedding, Dense, LSTM, Bidirectional, Flatten


class MimicLSTM():
    
    def __init__(self, layers, V, H, character_dim, output_dim, data_dictionary, optimizer='adam'):
        """Init function.
        
        Args:
          V: vocabulary size
          H: hidden state dimension
          layers: number of bidirectional LSTM layers
          character_dim: Embedding length for each character to be used in model
          output_dim: output dimension we are predicting
          data_dictionary: dictionary of words as keys and associated embeddings as values
          max_len: longest sequence in training data; needed to pad data
          optimizer: defaults to adam but can accept another input
        """
        
        self.layers = layers
        self.V = V
        self.H = H
        self.character_dim = character_dim
        self.output_dim = output_dim
        self.data_dictionary = data_dictionary
        self.max_len = self.calc_max_len()
        self.data = self.preprocess_data()
        self.optimizer=optimizer
        
        
    def build_model(self):
        """
        Create Keras neural model based on input specifications
        """
        with tf.name_scope("model_creation"):
            
            # Initialize a Keras sequential model
            self.lstm = Sequential()
            
            # Add an embedding layer which houses are character level embeddings.  We use a 
            # mask to coincide with our earlier padding
            self.lstm.add(Embedding(58, self.character_dim, input_length=max_len, 
                               embeddings_initializer=initializers.RandomUniform(-.1,.1), mask_zero=True))
            
            # ADd a bidirectional LSTM layer for each layer in self.layers.  Last layer only outputs final
            # result, but intermediate layers return full sequence.  We default to concating forward
            # and backward outputs across all layers.
            for x in range(0,self.layers):
                if x == self.layers-1:
                    self.lstm.add(Bidirectional(LSTM(self.H, return_sequences=False),
                                                merge_mode='concat'))
                else:
                    self.lstm.add(Bidirectional(LSTM(self.H, return_sequences=True), merge_mode='concat'))
            
            # Add a fully connected layer which results in predictions equal to our stated output_dim
            self.lstm.add(Dense(self.output_dim, activation='tanh'))
            
            # Compile model and view high level summary for analysis
            self.lstm.compile(loss="mean_squared_error", optimizer=self.optimizer)
            self.lstm.summary()
            
        return

    
    def calc_max_len(self):
        """
        helper function to calculate longest sequence in data
        """
        max_len=0
        for word in train_words:
            if len(word) > max_len:
                max_len = len(word)
        return max_len
    
    
    def pad_sequences(self, sequences):
        """
        Left pads sequences based on self.max_len
        """
        padded_data = sequence.pad_sequences(train_w, maxlen=self.max_len)
        return padded_data
    
    
    def preprocess_data(self):
        """
        Preprocess self.data_dictionary into padded sequences which can be 
        used for training and testing
        """
        words = []
        embeddings = []

        for k,v in all_words.items():
            words.append(k)
            embeddings.append(v)

        # Convert characters to index references and all lists to numpy arrays
        words_index = [[chardict[char] for char in word] for word in words]
        words_index = np.array(words_index)
        embeddings = np.array(embeddings)

        # Establish train/test splits 
        self.train_w, self.test_w, self.train_e, self.test_e = train_test_split(words_index, embeddings, 
                                                                                      test_size=0.1, random_state=1)
        self.max_len = self.calc_max_len()
        
        # Pad our training and test data so we can handle variable length sequences
        self.train_w = pad_sequences(self.train_w)
        self.test_w = pad_sequences(self.test_w)
        
        return
    
    
    def train_model(self):
        return
    
    def save_model(self):
        return
    
    

        
        