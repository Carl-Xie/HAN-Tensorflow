# encoding=utf-8

from tensorflow.contrib import rnn
from ops import *

import tensorflow as tf


class HAN(object):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_size=200,
                 hidden_size=50,
                 word_ctx_size=100,
                 sentence_ctx_size=100):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.word_ctx_size = word_ctx_size
        self.sentence_ctx_size = sentence_ctx_size
        self.num_classes = num_classes

    def predict(self, x, sequence_length=None):
        embedded_weights = tf.Variable(tf.random_normal((self.vocab_size, self.embedding_size)),
                                       name='embedded_weights')
        embedded_x = tf.nn.embedding_lookup(embedded_weights, x)
        words_with_attention = self._get_word_attention(embedded_x, sequence_length)
        sentences_with_attention = self._get_sentence_attention(words_with_attention)

        v = tf.reduce_sum(sentences_with_attention, axis=1)
        W_c = tf.Variable(tf.truncated_normal([self.num_classes, self.hidden_size * 2]), name='class_weights')
        b_c = tf.Variable(tf.truncated_normal([self.num_classes]))
        score = tf.matmul(W_c, v, transpose_a=False, transpose_b=True) + b_c
        return tf.nn.softmax(score, name='predict')

    def _get_word_attention(self, embedded_x, sequence_length):
        word_cell_fw = rnn.GRUCell(num_units=self.hidden_size, input_size=self.embedding_size)
        word_cell_bw = rnn.GRUCell(num_units=self.hidden_size, input_size=self.embedding_size)
        # outputs shape: [batch_size, max_time, state_size]
        word_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_cell_fw,
                                                          cell_bw=word_cell_bw,
                                                          inputs=embedded_x,
                                                          sequence_length=sequence_length,
                                                          time_major=False)
        # word = [h_fw, h_bw]
        encoded_words = tf.concat(word_outputs, axis=2, name='encoded_words')
        encoded_word_dims = self.hidden_size * 2
        # word attention layer
        word_context = tf.Variable(tf.truncated_normal([self.word_ctx_size]))
        W_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims, encoded_word_dims]),
                             name='word_context_weights')
        b_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims]), name='word_context_bias')

        # U_{it} = tanh(W_w * h_{it} + b_w)
        U_w = tf.tanh(matrix_batch_vectors_mul(W_word, encoded_words) + b_word,
                      name='U_w')
        word_logits = batch_vectors_vector_mul(U_w, word_context)
        word_attention = tf.nn.softmax(logits=word_logits)
        expand_word_attention = tf.expand_dims(word_attention, -1)
        return encoded_words * expand_word_attention

    def _get_sentence_attention(self, words_with_attention):
        sentences = tf.reduce_sum(words_with_attention, axis=1)
        expand_sentences = tf.expand_dims(sentences, 0)
        sentence_cell_fw = rnn.GRUCell(num_units=self.hidden_size, input_size=self.hidden_size * 2)
        sentence_cell_bw = rnn.GRUCell(num_units=self.hidden_size, input_size=self.hidden_size * 2)
        sentence_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=sentence_cell_fw,
                                                              cell_bw=sentence_cell_bw,
                                                              inputs=expand_sentences,
                                                              time_major=False)
        # sentence attention layer
        encoded_sentences = tf.concat(sentence_outputs, axis=2, name='encoded_sentences')
        encoded_sentence_dims = self.hidden_size * 2
        sentence_context = tf.Variable(tf.truncated_normal([encoded_sentence_dims]))
        W_sen = tf.Variable(tf.truncated_normal(shape=[encoded_sentence_dims, encoded_sentence_dims],
                                                name='context_sentence_weights'))
        b_sen = tf.Variable(tf.truncated_normal(shape=[encoded_sentence_dims]), name='context_sentence_bias')
        U_s = tf.tanh(matrix_batch_vectors_mul(W_sen, encoded_sentences) + b_sen, name='U_s')
        sentence_logits = batch_vectors_vector_mul(U_s, sentence_context)
        sentence_attention = tf.nn.softmax(logits=sentence_logits)
        expand_sentence_attention = tf.expand_dims(sentence_attention, -1)
        return encoded_sentences * expand_sentence_attention

    def loss(self, batch_x, batch_seq_len, batch_y):
        total_loss = tf.constant(0, tf.float32, name='loss')
        batch_size = 0
        for x, seq_len, y in zip(batch_x, batch_seq_len, batch_y):
            p = self.predict(x, seq_len)
            total_loss += tf.reduce_sum(tf.log(p) * y)
            batch_size += 1
        return total_loss / batch_size if batch_size > 0 else total_loss


