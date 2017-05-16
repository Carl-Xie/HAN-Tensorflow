# encoding=utf-8

from tensorflow.contrib import rnn

import tensorflow as tf


class Model(object):

    def __init__(self):
        """init the model with hyper-parameters"""
        pass

    def predict(self, x):
        """forward calculation from x to y"""
        pass

    def loss(self, batch_x, batch_y):
        """calculate model loss"""
        pass

    def optimize(self, batch_x, batch_y):
        """optimize the model loss"""
        pass


class HierarchicalAttentionNetwork(Model):

    def __init__(self,
                 sess,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 word_ctx_size,
                 sentence_ctx_size,
                 num_classes):
        super().__init__()

        self.sess = sess
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.word_ctx_size = word_ctx_size
        self.sentence_ctx_size = sentence_ctx_size
        self.num_classes = num_classes

        self.embedded_weights = tf.Variable(tf.random_normal((vocab_size, embedding_size)), name='embedded_weights')

    def predict(self, x):
        embedded_x = tf.nn.embedding_lookup(self.embedded_weights, x)
        word_cell_fw = rnn.GRUCell(num_units=self.hidden_size, input_size=self.embedding_size)
        word_cell_bw = rnn.GRUCell(num_units=self.hidden_size, input_size=self.embedding_size)

    def loss(self, batch_x, batch_y):
        pass

    def optimize(self, batch_x, batch_y):
        pass
