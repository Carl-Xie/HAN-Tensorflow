import tensorflow as tf


def matrix_batch_vectors_mul(mat, batch_vectors):
    """
    :param mat: [N x N] 
    :param batch_vectors: [K x M x N] 
    :return: new batch vectors: [K x M x N]
    """
    assert mat.shape[1] == batch_vectors.shape[-1]
    vectors = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
    res = tf.matmul(mat, vectors, transpose_a=False, transpose_b=True)
    shape = batch_vectors.shape.as_list()
    shape[-1] = mat.shape[0].value
    return tf.reshape(tf.transpose(res), shape)


def batch_vectors_vector_mul(batch_vectors, vector):
    """
    :param batch_vectors: [K x M x N]
    :param vector: [N]
    :return: [K x M]
    """
    assert batch_vectors.shape[-1] == vector.shape[0]
    expand_vec = tf.expand_dims(vector, -1)
    mat_vec = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
    res = tf.matmul(mat_vec, expand_vec)
    return tf.reshape(res, batch_vectors.shape[:-1])

