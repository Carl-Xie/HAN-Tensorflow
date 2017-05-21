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
    sentence_max_len = max([max([max(sen) for sen in doc]) for doc in lst])
    sentence_max_num = max(map(len, lst))
    result = np.zeros([len(lst), sentence_max_num, sentence_max_len])
    for i, row in enumerate(lst):
        for j, col in enumerate(row):
            for k, val in enumerate(col):
                result[i][j][k] = val
    return result

