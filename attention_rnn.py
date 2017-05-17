# encoding=utf-8

from tensorflow.contrib import rnn

import tensorflow as tf
import os


class HAN(object):

    def __init__(self,
                 sess,
                 vocab_size,
                 num_classes,
                 embedding_size=200,
                 hidden_size=50,
                 word_ctx_size=100,
                 sentence_ctx_size=100,
                 checkpoint_dir=None):
        """
            https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
        """
        self.sess = sess
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.word_ctx_size = word_ctx_size
        self.sentence_ctx_size = sentence_ctx_size
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.saver = None

    def predict(self, x):
        embedded_weights = tf.Variable(tf.random_normal((self.vocab_size, self.embedding_size)),
                                       name='embedded_weights')
        embedded_x = tf.nn.embedding_lookup(embedded_weights, x)
        word_cell_fw = rnn.GRUCell(num_units=self.hidden_size, input_size=self.embedding_size)
        word_cell_bw = rnn.GRUCell(num_units=self.hidden_size, input_size=self.embedding_size)

    def loss(self, batch_x, batch_y):
        pass

    def optimize(self, batch_x, batch_y):
        pass

    def save(self, checkpoint_dir, step):
        model_name = "HAN.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_state_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, checkpoint_state_name))
            finditer = re.finditer(pattern="(\d+)(?!.*\d)", string=checkpoint_state_name)
            while finditer:
                finditer = next(finditer)
            counter = int(next(finditer).group(0))
            print(" [*] Success to read {}".format(checkpoint_state_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
