import functools
import numpy as np


def lazy_property(func):
    attribute = '_cache_' + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator


def padding_batch_documents(lst):
    sentence_max_len = max([max([len(sen) for sen in doc]) for doc in lst])
    sentence_max_num = max(map(len, lst))
    result = np.zeros([len(lst), sentence_max_num, sentence_max_len], dtype=np.int32)
    for i, row in enumerate(lst):
        for j, col in enumerate(row):
            for k, val in enumerate(col):
                result[i][j][k] = val
    return result


def decode_batch(data):
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


def decode(datum):
    x = []
    y = []
    label_sentences = datum.decode().split(':')
    label = int(label_sentences[0])
    sentences = label_sentences[1].split('#')
    sentences = [[int(word) for word in sen.split(',')] for sen in sentences]
    x.append(sentences)
    y.append([0, 1] if label == 1 else [1, 0])
    return padding_batch_documents(x), y
