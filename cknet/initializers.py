import numpy as np

class Initializer():

    def fill(self, param):

        raise NotImplementedError()


class Constant(Initializer):

    def __init__(self, val=0.0, name="constantInit"):
        super(Constant, self).__init__(name=name)
        self.val = val

    def fill(self, param):
        param[:] = self.val

class IdentityInit(Initializer):

    def __init__(self, name="identityInit"):
        super(IdentityInit, self).__init__(name=name)

    def fill(self, param):
        (nin, nout) = param.shape
        w_ary = np.zeros((nin, nout), dtype=np.float32)
        w_ary[:, :nin] = np.eye(nin)
        param[:] = w_ary

class RandomInit(Initializer):

    def __init__(self, name="randomInit"):
        super(RandomInit, self).__init__(name=name)

    def fill(self, param):
        param[:] = np.zeros(param.shape)