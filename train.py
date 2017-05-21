# encoding=utf-8

from han import HAN
from utils import padding_batch_documents

import tensorflow as tf
import os


# Data loading params
tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
tf.flags.DEFINE_string("checkpoint_dir", "./", "check point directory")
tf.flags.DEFINE_string("summary_dir", "./summaries/", "summary directory")
tf.flags.DEFINE_integer("vocab_size", 4196, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("num_examples", 856, "number of examples")
# Model Hyper parameters
tf.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("word_context_size", 100, "word context vector size at word attention layer")
tf.flags.DEFINE_integer("sentence_context_size", 100, "sentence context vector at sentence attention layer")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
tf.flags.DEFINE_float("max_grad_norm", 15, "max grad norm to prevent gradient explode")


FLAGS = tf.flags.FLAGS


file_queue = tf.train.string_input_producer([FLAGS.data_dir])
reader = tf.TextLineReader()
_, line = reader.read(file_queue)
capacity = 10 * FLAGS.batch_size
data_batch = tf.train.batch([line], FLAGS.batch_size, capacity=capacity)


def decode(data):
    x = []
    y = []
    for datum in data:
        label_sentences = datum.decode().split(':')
        label = int(label_sentences[0])
        sentences = label_sentences[1].split('#')
        sentences = [[int(word) for word in sen.split(',')] for sen in sentences]
        x.append(sentences)
        y.append([0, 1] if label == 1 else [1, 0])
    return padding_batch_documents(x), y


def main(_):
    with tf.Session() as sess:
        han = HAN(vocab_size=FLAGS.vocab_size,
                  num_classes=FLAGS.num_classes,
                  batch_size=FLAGS.batch_size,
                  embedding_size=FLAGS.embedding_size,
                  hidden_size=FLAGS.hidden_size,
                  word_ctx_size=FLAGS.word_context_size,
                  sentence_ctx_size=FLAGS.sentence_context_size)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        gradients = tf.gradients(han.loss, tf.trainable_variables())
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
        grads_and_vars = tuple(zip(clipped_gradients, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        loss_summary = tf.summary.scalar("loss", han.loss)
        summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        checkpoint_dir = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, "checkpoints"))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # try to load restore from last train
        load(saver=saver, sess=sess, checkpoint_dir=checkpoint_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(FLAGS.num_epochs):
            for _ in range(FLAGS.num_examples // FLAGS.batch_size):
                x, y = decode(data_batch.eval(session=sess))
                feed_dict = {
                    han.input_x: x,
                    han.input_y: y,
                    han.max_sentence_length: len(x[0][0]),
                    han.max_sentence_num: len(x[0])
                }
                _, summary, step, loss = sess.run(
                    [train_op, summary_op, global_step, han.loss], feed_dict=feed_dict)

                print("step %s, current loss = %s" % (step, loss))

                summary_writer.add_summary(summary=summary, global_step=step)

                if step % FLAGS.checkpoint_every == 0:
                    save(saver, sess, checkpoint_dir, step)

        coord.request_stop()
        coord.join(threads)


def save(saver, sess, checkpoint_dir, step):
    model_name = "HAN.model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name),
               global_step=step)


def load(saver, sess, checkpoint_dir):
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
