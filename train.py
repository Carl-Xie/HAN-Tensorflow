# encoding=utf-8

from han import HAN

import tensorflow as tf
import os

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyper parameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 200)")

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
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
        checkpoint_state_name = os.path.basename(checkpoint_state.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_state_name))
        finditer = re.finditer(pattern="(\d+)(?!.*\d)", string=checkpoint_state_name)
        counter = int(next(finditer).group(0))
        print(" [*] Success to read {}".format(checkpoint_state_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0
