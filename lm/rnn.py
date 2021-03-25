import tensorflow as tf
import numpy as np
class rnn_model(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim,rnn_units):
        super(rnn_model,self).__init__()
        self.rnn_units=rnn_units
        self.embedding=tf.keras.layers.Embedding(vocab_size,embedding_dim)
        self.gru=tf.keras.layers.GRU(self.rnn_units,
                                     return_sequences=True)
        self.gru1=tf.keras.layers.GRU(self.rnn_units,
                                     return_sequences=True)
        self.fc=tf.keras.layers.Dense(vocab_size)
    def call(self,x):
        x=self.embedding(x)
        x=self.gru(x)
        x=self.gru1(x)
        x=self.fc(x)
        return x