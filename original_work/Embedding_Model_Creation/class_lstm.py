# File to define our LSTM class


# Authors: Clay & Gaurav
# Objective: Create Mimic LSTM models as outlined by Pinter et. al in "Mimicking Word Embeddings using Subword RNNs"
#-------------------------------------------------------------------------------------------------------------------


# Imports used in class MimicLSTM
from string import ascii_lowercase, ascii_uppercase
import sys

import gensim

from keras import backend as K
from keras import Sequential, optimizers, initializers, regularizers
from keras.layers import Dropout, Embedding, Dense, LSTM, Bidirectional
from keras.models import load_model
from keras.preprocessing import sequence 

from keras_self_attention import SeqSelfAttention
import numpy as np
from sklearn.model_selection import train_test_split


class MimicLSTM():
    
    def __init__(self, layers, H, chardict, character_dim, data_dictionary, epochs=1, batch_size=1000,
                 recurrent_dropout=0.0, dense_dropout=0.0, use_attention=False, optimizer='adam',                                                loss_function="mean_squared_error", train=True, full_set=False, load_path=''):
        
        """
        Init function.
        Args:
          layers: number of bidirectional LSTM layers
          H: hidden state dimension
          chardict: map of characters to indices
          character_dim: Embedding length for each character to be used in model
          data_dictionary: dictionary of words as keys and associated embeddings as values
          epochs: Number or epochs to run during training
          batch_size: Number of samples to use in calculating each gradient update
          recurrent_dropout: Dropout rate in recurrent cells
          dense_dropout: Dropout rate in final fully connected layer
          use_attention: Add attention in between biLSTM layers if layers >= 2
          optimizer: defaults to adam but can accept another input
          loss_function: Loss function train model
          custom_loss: Whether to use cosine distance (Define below) instead of loss_function
          train: Whether to train the model
          full_set: If training the model, are we training on 100% of data, or withholding a validation set
          load_path: Path to load pre-trained model
        """
                    
        self.layers = layers
        self.H = H
        self.chardict = chardict
        self.character_dim = character_dim
        self.output_dim = int(list(data_dictionary.values())[0].shape[0])
        self.data_dictionary = data_dictionary
        self.optimizer=optimizer
        self.loss = loss_function
        self.use_attention = use_attention
        self.recurrent_dropout = recurrent_dropout
        self.dense_dropout = dense_dropout
    
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

        # Initialize a Keras sequential model
        lstm = Sequential()

        # Add an embedding layer which houses are character level embeddings.  We use a 
        # mask to coincide with our earlier padding.  Embedding rows is equal to our character 
        # count, +1 for our 0 mask, and +1 because function is up to but not including that number

        lstm.add(Embedding(len(self.chardict)+2, self.character_dim, input_length=self.max_len, 
                           embeddings_initializer=initializers.RandomNormal(), mask_zero=True))

        # ADd a bidirectional LSTM layer for each layer in self.layers.  Last layer only outputs final
        # result, but intermediate layers return full sequence.  We default to concating forward
        # and backward outputs across all layers.
        for x in range(0,self.layers):
            if x == self.layers-1:
                lstm.add(Bidirectional(LSTM(self.H, return_sequences=False, 
                                            recurrent_dropout=self.recurrent_dropout), merge_mode='concat'))
            else:
                lstm.add(Bidirectional(LSTM(self.H, return_sequences=True, 
                                            recurrent_dropout=self.recurrent_dropout), merge_mode='concat'))

                # If use_attention is set to true create an attention layer between each biLSTM layer
                if self.use_attention:
                    lstm.add(SeqSelfAttention(attention_activation='sigmoid',
                                             kernel_regularizer=regularizers.l2(1e-5)))


        # Apply dropout prior to fully connected layer; default value is 0 dropout
        lstm.add(Dropout(rate=self.dense_dropout))

        # Add a fully connected layer which results in predictions equal to our stated output_dim
        lstm.add(Dense(self.output_dim, activation='tanh'))                  

        # Compile model and view high level summary for analysis
        lstm.compile(loss=self.loss, optimizer=self.optimizer)
        lstm.summary()
                
        return lstm

    
    def calc_max_len(self, train_words):
        """
        helper function to calculate longest sequence in data
        :param train_words: list of words which are used in training data
        """
        max_len=0
        for word in train_words:
            if len(word) > max_len:
                max_len = len(word)
                
        return max_len
    
    
    def pad_data(self, sequences):
        """
        helper function which left pads sequences based on self.max_len
        :param sequences: list of sequences which we will pad 
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
        
        return self
    
                          
    def save_model(self, path):
        """
        helper function to save trained keras model
        :param path: Location of where to save file
        """
        try:
            self.lstm.save(path) 
            print("Save Succesful")
        except:
            print("Error saving file")
                          
        return self
             
                          
    def load_model(self, path):
        """
        Helper function to load pretrained keras model
        :param path: location of model to be loaded
        """
        try:              
            self.lstm = load_model(path)
            print("Load Succesful")
        except:
            print("Could not load file, not valid file or path")
                          
        return self
                          
        


