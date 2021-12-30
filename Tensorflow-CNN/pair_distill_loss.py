#!/usr/bin/env python
# encoding: utf-8
'''
@author: lee
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: deamoncao100@gmail.com
@software: garner
@file: pair_distill_loss.py
@time: 2021/12/28 13:56
@desc:
'''
import tensorflow as tf

def L2(f_):
    return tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(f_), -1)), -1) + tf.convert_to_tensor(1e-8)

def similarity(feat):
    tmp = L2(feat)
    feat = feat / tmp
    feat = tf.reshape(feat, [-1, feat.shape[-2]])
    return feat * feat

def sim_dis_compute(f_s, f_t):
    sim_err = (similarity(f_t) - similarity(f_s)) ** 2
    sim_dis = tf.reduce_sum(sim_err)
    return sim_dis


def main(score_flatten, fakes_flatten):
    """
    借鉴的pair论文中loss
    :param score_flatten:
    :param fakes_flatten:
    :return:
    """
    stu_feature = tf.expand_dims(score_flatten, -2)
    tea_feature = tf.expand_dims(fakes_flatten, -2)
    # stu_feature = tf.nn.max_pool(tf.expand_dims(score_flatten, -1), [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    # tea_feature = tf.nn.max_pool(tf.expand_dims(fakes_flatten, -1), [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    return sim_dis_compute(stu_feature, tea_feature)