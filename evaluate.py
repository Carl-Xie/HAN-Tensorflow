# encoding=utf-8

from utils import decode

import tensorflow as tf
import numpy as np


tf.flags.DEFINE_string("data_path", "data/data.dat", "data file path")
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "checkpoint directory")
tf.flags.DEFINE_integer("num_examples", 856, "number of examples")
FLAGS = tf.flags.FLAGS


file_queue = tf.train.string_input_producer([FLAGS.data_path])
reader = tf.TextLineReader()
_, line = reader.read(file_queue)


def main(_):
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print('restore from %s' % checkpoint_file)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        input_x = sess.graph.get_operation_by_name("placeholder/input_x").outputs[0]
        max_sentence_length = sess.graph.get_operation_by_name("placeholder/max_sentence_length").outputs[0]
        max_sentence_num = sess.graph.get_operation_by_name("placeholder/max_sentence_num").outputs[0]
        batch_size = sess.graph.get_operation_by_name("placeholder/batch_size").outputs[0]
        inference = sess.graph.get_operation_by_name("infer_label").outputs[0]

        actual_labels = np.array([])
        infer_labels = np.array([])
        for _ in range(FLAGS.num_examples):
            x, y = decode(sess.run(line))
            y_hat = sess.run(inference, feed_dict={
                input_x: x,
                batch_size: len(x),
                max_sentence_length: len(x[0][0]),
                max_sentence_num: len(x[0])
            })
            actual_labels = np.append(actual_labels, np.argmax(y, axis=1))
            infer_labels = np.append(infer_labels, y_hat)
            print("infer label = %s, actual label = %s" % (y_hat, np.argmax(y, axis=1)))
        print("training accuracy = %s" % (1-sum(abs(actual_labels-infer_labels))/(FLAGS.num_examples+0.0)))
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

