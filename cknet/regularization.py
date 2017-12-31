#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : regularization.py
# @Author: kingcong
# @Date  : 31/12/2017
# @Desc  : cost and dW regularization
import numpy as np

class Regularization():

    def __init__(self, lambd = 0.01):
        self.lambd = lambd

    def cost_regularization(self, parameters):
        raise NotImplementedError

    def brop_regularization(self, W, m):
        raise NotImplementedError


class L2Regularization(Regularization):

    def cost_regularization(self, parameters):
        m = len(parameters) // 2
        sum_w = 0
        for i in range(m):
            W = parameters["W" + str(i+1)]
            sum_w += np.sum(np.square(W))
        L2_cost_regularization = self.lambd * sum_w / (2 * m)
        return L2_cost_regularization

    def brop_regularization(self, W, m):
        return self.lambd / m * W
