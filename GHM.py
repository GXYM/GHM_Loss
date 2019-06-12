#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'


"""
The implementation of GHM-C and GHM-R losses.
Details can be found in the paper `Gradient Harmonized Single-stage Detector`:
https://arxiv.org/abs/1811.05181
Copyright (c) 2018 Multimedia Laboratory, CUHK.
Licensed under the MIT License (see LICENSE for details)
Written by Buyu Li
"""

import tensorflow as tf


class GHM_Loss:
    def __init__(self, bins=10, momentum=0.75):
        self.g =None
        self.bins = bins
        self.momentum = momentum
        self.valid_bins = tf.constant(0.0, dtype=tf.float32)
        self.edges_left, self.edges_right = self.get_edges(self.bins)
        if momentum > 0:
            acc_sum = [0.0 for _ in range(bins)]
            self.acc_sum = tf.Variable(acc_sum, trainable=False)

    @staticmethod
    def get_edges(bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left)  # [bins]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-3
        edges_right = tf.constant(edges_right)  # [bins]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1, 1]
        return edges_left, edges_right


    def calc(self, g, valid_mask):
        edges_left, edges_right = self.edges_left, self.edges_right
        alpha = self.momentum
        # valid_mask = tf.cast(valid_mask, dtype=tf.bool)

        tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)
        inds_mask = tf.logical_and(tf.greater_equal(g, edges_left), tf.less(g, edges_right))
        zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        num_in_bin = tf.reduce_sum(inds, axis=[1, 2, 3])  # [bins]
        valid_bins = tf.greater(num_in_bin, 0)  # [bins]

        num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

        if alpha > 0:
            update = tf.assign(self.acc_sum,
                               tf.where(valid_bins, alpha * self.acc_sum + (1 - alpha) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1, 1]
                acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)

        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1, 1]
            num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)

        weights = weights / num_valid_bin

        return weights, tot

    def ghm_class_loss(self, logits, targets, masks=None):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
        self.g = tf.abs(tf.sigmoid(logits) - targets) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0)  # [1, batch_num, class_num]

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0
        weights, tot = self.calc(g, valid_mask)
        print(weights.shape)
        ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets*train_mask,
                                                                 logits=logits)
        ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

        return ghm_class_loss


    def ghm_regression_loss(self, logits, targets, masks):
        """ Args:
        input [batch_num, *(* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num,  *(* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu

        # ASL1 loss
        diff = logits - targets
        # gradient length
        g = tf.abs(diff / tf.sqrt(mu * mu + diff * diff))

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0

        weights, tot = self.calc(g, valid_mask)

        ghm_reg_loss = tf.sqrt(diff * diff + mu * mu) - mu
        ghm_reg_loss = tf.reduce_sum(ghm_reg_loss * weights) / tot

        return ghm_reg_loss

class GHM_Loss2:
    def __init__(self, bins=10, momentum=0.75):
        self.g = None
        self.bins = bins
        self.momentum = momentum
        self.valid_bins = tf.constant(0.0, dtype=tf.float32)
        self.edges_left, self.edges_right = self.get_edges(self.bins)
        if momentum > 0:
            acc_sum = [0.0 for _ in range(bins)]
            self.acc_sum = tf.Variable(acc_sum, trainable=False)

    @staticmethod
    def get_edges(bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left) # [bins]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-6
        edges_right = tf.constant(edges_right) # [bins]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1, 1]
        return edges_left, edges_right


    def calc(self, g, valid_mask):
        edges_left, edges_right = self.edges_left, self.edges_right
        alpha = self.momentum
        # valid_mask = tf.cast(valid_mask, dtype=tf.bool)

        tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)
        inds_mask = tf.logical_and(tf.greater_equal(g, edges_left), tf.less(g, edges_right))
        zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        num_in_bin = tf.reduce_sum(inds, axis=[1, 2])  # [bins]
        valid_bins = tf.greater(num_in_bin, 0)  # [bins]

        num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

        if alpha > 0:
            update = tf.assign(self.acc_sum,
                               tf.where(valid_bins, alpha * self.acc_sum + (1 - alpha) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)

        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)

        weights = weights / num_valid_bin

        return weights, tot


    def ghm_class_loss(self, logits, targets, masks=None):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
        self.g = tf.abs(tf.sigmoid(logits) - targets) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0)  # [1, batch_num, class_num]

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0
        weights, tot = self.calc(g, valid_mask)
        print(weights.shape)
        ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets*train_mask,
                                                                 logits=logits)
        ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

        return ghm_class_loss


    def ghm_regression_loss(self, logits, targets, masks):
        """ Args:
        input [batch_num, *(* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num,  *(* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu

        # ASL1 loss
        diff = logits - targets
        # gradient length
        g = tf.abs(diff / tf.sqrt(mu * mu + diff * diff))

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0

        weights, tot = self.calc(g, valid_mask)

        ghm_reg_loss = tf.sqrt(diff * diff + mu * mu) - mu
        ghm_reg_loss = tf.reduce_sum(ghm_reg_loss * weights) / tot

        return ghm_reg_loss


if __name__ == '__main__':
    ghm = GHM_Loss(momentum=0.75)
    input_1 = tf.constant([[[0.025, 0.35], [0.45, 0.85]]], dtype=tf.float32)
    target_1 = tf.constant([[[1.0, 1.0], [0.0, 1.0]]], dtype=tf.float32)

    input_2 = tf.constant([[[0.55, 0.45], [0.55, 0.65]]], dtype=tf.float32)
    target_2 = tf.constant([[[1.0, 0.0], [0.0, 0.0]]], dtype=tf.float32)
    print(input_1.shape)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        loss = ghm.ghm_class_loss(input_1, target_1)
        print(sess.run([loss, ghm.g, ghm.acc_sum]))
        loss = ghm.ghm_class_loss(input_2, target_2)
        print(sess.run([loss, ghm.g, ghm.acc_sum]))
        loss = ghm.ghm_class_loss(input_2, target_2)
        print(sess.run([loss, ghm.g, ghm.acc_sum]))
        loss = ghm.ghm_class_loss(input_1, target_1)
        print(sess.run([loss, ghm.g, ghm.acc_sum]))
        loss = ghm.ghm_class_loss(input_1, target_1)
        print(sess.run([loss, ghm.g, ghm.acc_sum]))
        loss = ghm.ghm_class_loss(input_1, target_1)
        print(sess.run([loss, ghm.g, ghm.acc_sum]))
