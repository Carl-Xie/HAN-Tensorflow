# encoding=utf-8

from han import HAN
from utils import decode_batch
from preprocess import load_dev

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
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 50)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 5, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("decay_steps", 10, "decay steps")
tf.flags.DEFINE_float("decay_rate", 0.99, "decay rate")
tf.flags.DEFINE_float("max_grad_norm", 10, "max grad norm to prevent gradient explode")


FLAGS = tf.flags.FLAGS


file_queue = tf.train.string_input_producer([FLAGS.data_dir])
reader = tf.TextLineReader()
_, line = reader.read(file_queue)
capacity = 12 * FLAGS.batch_size
min_after_dequeue = 10 * FLAGS.batch_size
data_batch = tf.train.shuffle_batch([line], FLAGS.batch_size,
                                    min_after_dequeue=min_after_dequeue, capacity=capacity)

dev_x, dev_y = load_dev()


def main(_):
    with tf.Session() as sess:
        han = HAN(vocab_size=FLAGS.vocab_size,
                  num_classes=FLAGS.num_classes,
                  embedding_size=FLAGS.embedding_size,
                  hidden_size=FLAGS.hidden_size,
                  word_ctx_size=FLAGS.word_context_size,
                  sentence_ctx_size=FLAGS.sentence_context_size)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        train_op, grads_and_vars = get_optimization(han, global_step)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        loss_summary = tf.summary.scalar("loss", han.loss)
        accuracy_summary = tf.summary.scalar("accuracy", han.accuracy)

        sess.run(tf.global_variables_initializer())

        checkpoint = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_dir = os.path.join(checkpoint, "model")
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        dev_step = get_dev_step(sess=sess,
                                han=han,
                                summaries=[loss_summary, accuracy_summary],
                                out_dir=out_dir)

        train_step = get_train_step(sess=sess,
                                    han=han,
                                    saver=saver,
                                    train_op=train_op,
                                    global_step=global_step,
                                    summaries=[loss_summary, accuracy_summary, grad_summaries_merged],
                                    out_dir=out_dir,
                                    checkpoint_dir=checkpoint_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(FLAGS.num_epochs):
            print('current epoch %s' % (epoch+1))
            for _ in range(FLAGS.num_examples // FLAGS.batch_size):
                x, y = decode_batch(data_batch.eval(session=sess))
                step = train_step(x, y)
                if step % FLAGS.evaluate_every == 0:
                    """ we are not really perform evaluation here, just to show the code sketch"""
                    dev_step(dev_x, dev_y)
        coord.request_stop()
        coord.join(threads)


def get_optimization(han, global_step):
    with tf.name_scope('optimization'):
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                   FLAGS.decay_steps, FLAGS.decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = tf.gradients(han.loss, tf.trainable_variables())
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
        grads_and_vars = tuple(zip(clipped_gradients, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op, grads_and_vars


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


def get_train_step(sess, han, saver, train_op, global_step, summaries, out_dir, checkpoint_dir):

    train_summary_op = tf.summary.merge(summaries)
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

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
        _, summary, step, loss, acc = sess.run(
            [train_op, train_summary_op, global_step, han.loss, han.accuracy],
            feed_dict=feed_dict)
        time_pass = time.time() - now
        print("takes %s secs to run step %s, current loss = %s, accuracy=%s"
              % (time_pass, step, loss, acc))

        train_summary_writer.add_summary(summary=summary, global_step=step)
        if step % FLAGS.checkpoint_every == 0:
            save(saver, sess, checkpoint_dir, step)
        return step

    return train_step


def save(saver, sess, checkpoint_dir, step):
    model_name = "HAN.model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name),
               global_step=step)


def load(saver, sess, checkpoint_dir):
    """ if you need to continue to train from previous training """
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False


if __name__ == '__main__':
    tf.app.run()
