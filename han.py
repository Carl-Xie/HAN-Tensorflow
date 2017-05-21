# encoding=utf-8

from tensorflow.contrib import rnn
from utils import lazy_property

import tensorflow as tf


def length(sequences):
    """
    :param sequences: shape=[batch_size, max_time_step, embedding_size]
    :return: shape=[batch] whose value is actual length of each sequence in sequences
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


def matrix_batch_vectors_mul(mat, batch_vectors, shape_after_mul):
    """
    :param mat: [N x N] 
    :param batch_vectors: [K x M x N] 
    :param shape_after_mul: [K x M x N]
    :return: new batch vectors: [K x M x N]
    """
    vectors = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
    res = tf.matmul(mat, vectors, transpose_a=False, transpose_b=True)
    return tf.reshape(tf.transpose(res), shape_after_mul)


def batch_vectors_vector_mul(batch_vectors, vector, shape_after_mul):
    """
    :param batch_vectors: [K x M x N]
    :param vector: [N]
    :param shape_after_mul: [K x M]
    :return: [K x M]
    """
    expand_vec = tf.expand_dims(vector, -1)
    mat_vec = tf.reshape(batch_vectors, [-1, batch_vectors.get_shape()[-1].value])
    res = tf.matmul(mat_vec, expand_vec)
    return tf.reshape(res, shape_after_mul)


class HAN(object):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 batch_size=64,
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
        self.batch_size = batch_size

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            # input_x shape = [batch_size, num_sentence, sentence_length]
            self.input_x = tf.placeholder(tf.int32, [batch_size, None, None], name='input_x')
            # input_y shape = [batch_size, one_hot_class_coding]
            self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name='input_y')

        # get the side effect
        self.__prediction = self.prediction
        self.__loss = self.loss

    @lazy_property
    def loss(self):
        # shape = [batch_size, num_classes]
        true_label_prob = self.prediction * self.input_y
        return -tf.reduce_mean(tf.log(tf.reduce_sum(true_label_prob, axis=1)))

    @lazy_property
    def prediction(self):
        with tf.name_scope('embedding'):
            embedded_weights = tf.Variable(tf.random_normal((self.vocab_size, self.embedding_size)),
                                           name='embedded_weights')
            embedded_x = tf.nn.embedding_lookup(embedded_weights, self.input_x)

        with tf.name_scope('sentence_vector_construction'):
            # reshape_embedded_x.shape = [batch_size*num_sentence, sentence_length, embedding_size]
            reshape_embedded_x = tf.reshape(embedded_x, [-1, self.max_sentence_length, self.embedding_size])
            # encoded_words.shape = [batch_size*num_sentence, sentence_length, hidden_size * 2]
            encoded_words = self._word_encoder(reshape_embedded_x)
            # words_attention.shape = [batch_size*num_sentence, sentence_length]
            words_attention = self._word_attention(encoded_words)
            # expand_word_attention.shape = [batch_size*num_sentence, sentence_length, 1]
            expand_word_attention = tf.expand_dims(words_attention, -1)
            # words_with_attention.shape = [batch_size*num_sentence, sentence_length, hidden_size * 2]
            words_with_attention = encoded_words * expand_word_attention
            # sentences.shape = [batch_size*num_sentence, hidden_size * 2]
            sentences = tf.reduce_sum(words_with_attention, axis=1)

        with tf.name_scope('document_vector_construction'):
            # reshape_sentences.shape = [batch_size, num_sentence, hidden_size * 2]
            reshape_sentences = tf.reshape(sentences, shape=[
                self.batch_size, self.max_sentence_num, self.hidden_size * 2])
            # encoded_sentences.shape = [batch_size, num_sentence, hidden_size * 2]
            encoded_sentences = self._sentence_encoder(reshape_sentences)
            # sentence_attention.shape = [batch_size, num_sentence]
            sentence_attention = self._sentence_attention(encoded_sentences)

            expand_sentence_attention = tf.expand_dims(sentence_attention, -1)
            # sentences_with_attention.shape = [batch_size, num_sentence, hidden_size * 2]
            sentences_with_attention = encoded_sentences * expand_sentence_attention

            # document_vectors = [batch_size, hidden_size * 2]
            document_vectors = tf.reduce_sum(sentences_with_attention, axis=1)

        with tf.name_scope('document_prediction'):
            W_c = tf.Variable(tf.truncated_normal([self.hidden_size * 2, self.num_classes]), name='class_weights')
            b_c = tf.Variable(tf.truncated_normal([self.num_classes]), name='class_biases')
            score = tf.matmul(document_vectors, W_c) + b_c
            return tf.nn.softmax(score, name='prediction')

    def _word_encoder(self, embedded_x):
        with tf.variable_scope('word_encoder'):
            word_cell_fw = rnn.GRUCell(num_units=self.hidden_size)
            word_cell_bw = rnn.GRUCell(num_units=self.hidden_size)
            # outputs shape: [batch_size, max_time, state_size]
            word_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_cell_fw,
                                                              cell_bw=word_cell_bw,
                                                              inputs=embedded_x,
                                                              dtype=tf.float32,
                                                              sequence_length=length(embedded_x))
            # word = [h_fw, h_bw]
            return tf.concat(word_outputs, axis=2, name='encoded_words')

    def _word_attention(self, encoded_words):
        with tf.name_scope('word_attention'):
            encoded_word_dims = self.hidden_size * 2
            # word attention layer
            word_context = tf.Variable(tf.truncated_normal([self.word_ctx_size]), name='word_context')
            W_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims, encoded_word_dims]),
                                 name='word_context_weights')
            b_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims]), name='word_context_bias')

            # U_{it} = tanh(W_w * h_{it} + b_w)
            U_w = tf.tanh(matrix_batch_vectors_mul(W_word, encoded_words,
                                                   [-1, self.max_sentence_length, encoded_word_dims]) + b_word,
                          name='U_w')
            word_logits = batch_vectors_vector_mul(U_w, word_context, [-1, self.max_sentence_length])
            return tf.nn.softmax(logits=word_logits)

    def _sentence_encoder(self, sentences):
        with tf.variable_scope('sentence_encoder'):
            sentence_cell_fw = rnn.GRUCell(num_units=self.hidden_size)
            sentence_cell_bw = rnn.GRUCell(num_units=self.hidden_size)
            sentence_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=sentence_cell_fw,
                                                                  cell_bw=sentence_cell_bw,
                                                                  inputs=sentences,
                                                                  dtype=tf.float32,
                                                                  sequence_length=length(sentences))
            return tf.concat(sentence_outputs, axis=2, name='encoded_sentences')

    def _sentence_attention(self, encoded_sentences):
        with tf.name_scope('sentence_attention'):
            encoded_sentence_dims = self.hidden_size * 2
            sentence_context = tf.Variable(tf.truncated_normal([encoded_sentence_dims]), name='sentence_context')
            W_sen = tf.Variable(tf.truncated_normal(shape=[encoded_sentence_dims, encoded_sentence_dims],
                                                    name='context_sentence_weights'))
            b_sen = tf.Variable(tf.truncated_normal(shape=[encoded_sentence_dims]), name='context_sentence_bias')
            U_s = tf.tanh(matrix_batch_vectors_mul(W_sen, encoded_sentences,
                                                   [-1, self.max_sentence_num, encoded_sentence_dims]) + b_sen,
                          name='U_s')
            sentence_logits = batch_vectors_vector_mul(U_s, sentence_context, [-1, self.max_sentence_num])
            return tf.nn.softmax(logits=sentence_logits)

