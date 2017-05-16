# encoding=utf-8


class Model(object):

    def __init__(self):
        """init the model with hyper-parameters"""
        pass

    def predict(self, x):
        """forward calculation from x to y"""
        pass

    def loss(self, batch_x, batch_y):
        """calculate model loss"""
        pass

    def optimize(self, batch_x, batch_y):
        """optimize the model loss"""
        pass


class HierarchicalAttentionNetwork(Model):

    def __init__(self):
        super().__init__()
