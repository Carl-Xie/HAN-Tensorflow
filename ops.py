import tensorflow as tf


def matrix_batch_vectors_mul(mat, batch_vectors):
    assert mat.shape[1] == batch_vectors.shape[-1]
    vectors = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
    res = tf.matmul(mat, vectors, transpose_a=False, transpose_b=True)
    shape = batch_vectors.shape.as_list()
    shape[-1] = mat.shape[0].value
    return tf.reshape(tf.transpose(res), shape)


def batch_vectors_vector_mul(batch_vectors, vector):
    assert batch_vectors.shape[-1] == vector.shape[0]
    expand_vec = tf.expand_dims(vector, -1)
    mat_vec = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
    res = tf.matmul(mat_vec, expand_vec)
    return tf.reshape(res, batch_vectors.shape[:-1])

