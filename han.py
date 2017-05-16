# encoding=utf-8

import tensorflow as tf

from tensorflow.contrib import rnn


class HierarchicalAttentionNetwork(object):

    def __init__(self,
                 sequence_len,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 num_units):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [num_classes], name='input_y')
        self.dropout_keep_rate = tf.placeholder(tf.float32, name='dropout_keep_rate')

        # word to vector layer
        with tf.name_scope('embedding_layer'):
            self.w_embed = tf.Variable(tf.random_normal((vocab_size, embedding_size)),
                                       name='embedded_weights')
            self.embedded_x = tf.nn.embedding_lookup(self.w_embed, self.input_x)

        # word encoder
        word_cell_fw = rnn.GRUCell(num_units=num_units, input_size=embedding_size)
        word_cell_bw = rnn.GRUCell(num_units=num_units, input_size=embedding_size)

        # outputs shape: [batch_size, max_time, state_size]
        word_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_cell_fw,
                                                          cell_bw=word_cell_bw,
                                                          inputs=self.embedded_x,
                                                          time_major=False)
        # word = [h_fw, h_bw]
        encoded_words = tf.concat(word_outputs, axis=2, name='encoded_words')
        encoded_word_dims = num_units * 2
        # word attention layer
        self.word_context = tf.Variable(tf.truncated_normal([encoded_word_dims]))
        self.W_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims, encoded_word_dims]),
                                  name='context_word_weights')
        self.b_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims]), name='context_word_bias')

        u_w = tf.tanh(self.matrix_batch_vectors_mul(self.W_word, encoded_words) + self.b_word, name='u_w')
        word_logits = self.batch_vectors_vector_mul(u_w, self.word_context)
        word_attention = tf.nn.softmax(logits=word_logits)
        expand_word_attention = tf.expand_dims(word_attention, -1)
        words_with_attention = encoded_words * expand_word_attention

        # sentence encoder
        sentences = tf.reduce_sum(words_with_attention, axis=1)
        expand_sentences = tf.expand_dims(sentences, 0)
        sentence_cell_fw = rnn.GRUCell(num_units=num_units, input_size=encoded_word_dims)
        sentence_cell_bw = rnn.GRUCell(num_units=num_units, input_size=encoded_word_dims)
        sentence_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=sentence_cell_fw,
                                                              cell_bw=sentence_cell_bw,
                                                              inputs=expand_sentences,
                                                              time_major=False)
        # sentence attention layer
        encoded_sentences = tf.concat(sentence_outputs, axis=2, name='encoded_sentences')
        encoded_sentence_dims = encoded_word_dims * 2
        self.sentence_context = tf.Variable(tf.truncated_normal([encoded_sentence_dims]))
        self.W_sen = tf.Variable(tf.truncated_normal(shape=[encoded_sentence_dims, encoded_sentence_dims],
                                                     name='context_sentence_weights'))
        self.b_sen = tf.Variable(tf.truncated_normal(shape=[encoded_sentence_dims]), name='context_sentence_bias')
        u_s = tf.tanh(self.matrix_batch_vectors_mul(self.W_sen, encoded_sentences) + self.b_sen, name='u_s')
        sentence_logits = self.batch_vectors_vector_mul(u_s, self.sentence_context)
        sentence_attention = tf.nn.softmax(logits=sentence_logits)
        expand_sentence_attention = tf.expand_dims(sentence_attention, -1)
        sentences_with_attention = encoded_sentences * expand_sentence_attention

        document_with_attention = tf.reduce_sum(sentences_with_attention, axis=1)

        self.W_c = tf.Variable(tf.truncated_normal([num_classes, encoded_sentence_dims]), name='class_weights')
        self.b_c = tf.Variable(tf.truncated_normal([num_classes]))
        v = tf.matmul(self.W_c, document_with_attention, transpose_a=False, transpose_b=True)
        p = tf.nn.softmax(v)
        self.loss = -tf.reduce_sum(tf.log(p) * self.input_y)

    @staticmethod
    def matrix_batch_vectors_mul(mat, batch_vectors):
        assert mat.shape[1] == batch_vectors.shape[-1]
        vectors = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
        res = tf.matmul(mat, vectors, transpose_a=False, transpose_b=True)
        shape = batch_vectors.shape.as_list()
        shape[-1] = mat.shape[0].value
        return tf.reshape(tf.transpose(res), shape)

    @staticmethod
    def batch_vectors_vector_mul(batch_vectors, vector):
        assert batch_vectors.shape[-1] == vector.shape[0]
        expand_vec = tf.expand_dims(vector, -1)
        mat_vec = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
        res = tf.matmul(mat_vec, expand_vec)
        return tf.reshape(res, batch_vectors.shape[:-1])
