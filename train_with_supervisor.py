# encoding=utf-8

from han import HAN
from utils import decode_batch

import tensorflow as tf
import os
import time


# Data loading params
tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
tf.flags.DEFINE_integer("vocab_size", 4196, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("num_examples", 856, "number of examples")

# Model Hyper parameters
tf.flags.DEFINE_integer("embedding_size", 120, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("word_context_size", 100, "word context vector size at word attention layer")
tf.flags.DEFINE_integer("sentence_context_size", 100, "sentence context vector at sentence attention layer")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("decay_steps", 10, "decay steps")
tf.flags.DEFINE_float("decay_rate", 0.99, "decay rate")
tf.flags.DEFINE_float("max_grad_norm", 10, "max grad norm to prevent gradient explode")
tf.flags.DEFINE_string("log_dir", "./supervisor_logs", "log directory for Supervisor")

FLAGS = tf.flags.FLAGS


def main(_):
    with tf.Graph().as_default() as graph:
        file_queue = tf.train.string_input_producer([FLAGS.data_dir])
        reader = tf.TextLineReader()
        _, line = reader.read(file_queue)
        capacity = 12 * FLAGS.batch_size
        min_after_dequeue = 10 * FLAGS.batch_size
        data_batch = tf.train.shuffle_batch([line], FLAGS.batch_size,
                                            min_after_dequeue=min_after_dequeue, capacity=capacity)

        han = HAN(vocab_size=FLAGS.vocab_size,
                  num_classes=FLAGS.num_classes,
                  embedding_size=FLAGS.embedding_size,
                  hidden_size=FLAGS.hidden_size,
                  word_ctx_size=FLAGS.word_context_size,
                  sentence_ctx_size=FLAGS.sentence_context_size)

        global_step = tf.train.get_or_create_global_step(graph)
        train_op = get_optimization(han, global_step)

        sv = tf.train.Supervisor(logdir=FLAGS.log_dir)
        with sv.managed_session() as sess:
            """ graph is not allow to be modified within this block! """
            train_step = get_train_step(sess=sess,
                                        han=han,
                                        train_op=train_op,
                                        global_step=global_step)

            while not sv.should_stop():
                for epoch in range(FLAGS.num_epochs):
                    print('current epoch %s' % (epoch+1))
                    for _ in range(FLAGS.num_examples // FLAGS.batch_size):
                        x, y = decode_batch(data_batch.eval(session=sess))
                        train_step(x, y)
                sv.request_stop()


def get_optimization(han, global_step):
    with tf.name_scope('optimization'):
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                   FLAGS.decay_steps, FLAGS.decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = tf.gradients(han.loss, tf.trainable_variables())
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
        grads_and_vars = tuple(zip(clipped_gradients, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op


def get_dev_step(sess, han, summaries, out_dir):
    dev_summary_op = tf.summary.merge(summaries)
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    def dev_step(x_batch, y_batch):
        if not x_batch:
            return
        now = time.time()
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.batch_size: FLAGS.batch_size,
            han.max_sentence_length: len(x_batch[0][0]),
            han.max_sentence_num: len(x_batch[0])
        }
        print('current data shape: [%s,%s,%s]' % (FLAGS.batch_size, len(x_batch[0]), len(x_batch[0][0])))
        summary, loss, acc = sess.run([dev_summary_op, han.loss, han.accuracy],
                                      feed_dict=feed_dict)
        time_pass = time.time() - now
        print("takes %s secs, current loss = %s, accuracy=%s"
              % (time_pass, loss, acc))
        dev_summary_writer.add_summary(summary=summary)

    return dev_step


def get_train_step(sess, han, train_op, global_step):

    def train_step(x_batch, y_batch):
        now = time.time()
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.batch_size: FLAGS.batch_size,
            han.max_sentence_length: len(x_batch[0][0]),
            han.max_sentence_num: len(x_batch[0])
        }
        print('current data shape: [%s,%s,%s]' % (FLAGS.batch_size, len(x_batch[0]), len(x_batch[0][0])))
        _, step, loss, acc = sess.run(
            [train_op, global_step, han.loss, han.accuracy],
            feed_dict=feed_dict)
        time_pass = time.time() - now
        print("takes %s secs to run step %s, current loss = %s, accuracy=%s"
              % (time_pass, step, loss, acc))

    return train_step


if __name__ == '__main__':
    tf.app.run()
