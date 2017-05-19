# encoding=utf-8

from han import HAN

import tensorflow as tf
import os


# Data loading params
tf.flags.DEFINE_string("positive_data_dir", "data/pos/", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_dir", "data/neg/", "Data source for the negative data.")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")

# Model Hyper parameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("word_context_size", 100, "word context vector size at word attention layer")
tf.flags.DEFINE_integer("sentence_context_size", 100, "sentence context vector at sentence attention layer")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 300, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def save(saver, sess, checkpoint_dir, step):
    model_name = "HAN.model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name),
               global_step=step)


def load(saver, sess, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        finditer = re.finditer(pattern="(\d+)(?!.*\d)", string=ckpt_name)
        counter = int(next(finditer).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def main(_):
    pass

if __name__ == '__main__':
    tf.app.run()
