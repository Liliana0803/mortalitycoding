#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

def categorical_class_balanced_focal_loss(n_instances_per_class, beta, gamma=2.):
    effective_num = 1.0 - np.power(beta, n_instances_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights)
    weights = K.variable(weights)
    def categorical_class_balanced_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = weights * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return categorical_class_balanced_focal_loss_fixed