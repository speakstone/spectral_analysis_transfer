#!/usr/bin/env python
# encoding: utf-8
'''
@author: lee
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: deamoncao100@gmail.com
@software: garner
@file: model.py
@time: 2021/12/24 10:27
@desc:
'''
import tensorflow as tf
import mmd_tensorflow
import pair_distill_loss

FLAGS = tf.app.flags.FLAGS

# 定义model
def model(datas_placeholder, dropout_placeholdr, num_classes):
    # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
    conv0 = tf.layers.conv1d(datas_placeholder, filters=8, kernel_size=30, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.average_pooling1d(conv0, 2, 2, padding="SAME")
    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv1 = tf.layers.conv1d(pool0, filters=16, kernel_size=15, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.average_pooling1d(conv1, 2, 2)
    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv2 = tf.layers.conv1d(pool1, filters=32,kernel_size=10, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool2 = tf.layers.average_pooling1d(conv2, 2, 2)
    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool2)

    # 全连接一次
    fc0 = tf.layers.dense(flatten, 256, activation=tf.nn.relu)

    # 全连接层，转换为长度为100的特征向量
    fc1 = tf.layers.dense(fc0, 64, activation=tf.nn.relu)

    # 全连接层，转换为长度为100的特征向量
    fc = tf.layers.dense(fc1, 16, activation=tf.nn.relu)

    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, num_classes)
    return logits


# 定义model
def model_1(datas_placeholder, dropout_placeholdr, num_classes):
    with tf.name_scope('conv0') as scope:
        # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
        conv0 = tf.layers.conv1d(datas_placeholder, filters=8, kernel_size=30, activation=tf.nn.relu, name='w0')
        # 定义max-pooling层，pooling窗口为2x2，步长为2x2
        pool0 = tf.layers.average_pooling1d(conv0, 2, 2, padding="SAME", name='p0')
        # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    with tf.name_scope('conv1') as scope:
        conv1 = tf.layers.conv1d(pool0, filters=16, kernel_size=15, activation=tf.nn.relu, name='w1')
        # 定义max-pooling层，pooling窗口为2x2，步长为2x2
        pool1 = tf.layers.average_pooling1d(conv1, 2, 2, name='p1')
    with tf.name_scope('conv2') as scope:
        # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
        conv2 = tf.layers.conv1d(pool1, filters=32,kernel_size=10, activation=tf.nn.relu, name='w2')
        # 定义max-pooling层，pooling窗口为2x2，步长为2x2
        pool2 = tf.layers.average_pooling1d(conv2, 2, 2, name='p2')

    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool2, name='reshape')
    # # 输出特征分类
    # with tf.name_scope('class') as scope:
    #     score_cls = model_class(flatten)

    # with tf.name_scope('conv3') as scope:
    #     # 添加一维
    #     conv3f = tf.expand_dims(flatten, -1)
    #     # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    #     conv3 = tf.layers.conv1d(conv3f, filters=3,kernel_size=10, activation=tf.nn.relu, name='w3')
    #     # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    #     pool3 = tf.layers.average_pooling1d(conv3, 2, 2, name='p3')
    #     flatten_c = tf.layers.flatten(pool3, name='reshape')

    with tf.name_scope('dense') as scope:
        # 全连接一次
        fc0 = tf.layers.dense(flatten, 256, activation=tf.nn.relu, name='d0')

        # 全连接层，转换为长度为100的特征向量
        fc1 = tf.layers.dense(fc0, 64, activation=tf.nn.relu, name='d1')

        # 全连接层，转换为长度为100的特征向量
        fc = tf.layers.dense(fc1, 16, activation=tf.nn.relu, name='d2')

    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr, name='drop')

    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, num_classes, name='output')

    # return logits, flatten, score_cls
    # return logits, conv0, tf.expand_dims(fc0, -1)
    return logits, pool1, pool2


def model_class(flatten):
    # 全连接一次
    # fcl0 = tf.layers.dense(flatten, 512, activation=tf.nn.relu, name='dsc_0')
    # fcl1 = tf.layers.dense(fcl0, 400, activation=tf.nn.relu, name='dsc_1')
    # fcl2 = tf.layers.dense(fcl1, 100, activation=tf.nn.relu, name='dsc_2')
    # 未激活的输出层
    # score_cls = tf.layers.dense(fcl2, 1, activation=tf.nn.sigmoid, name='sigmoid_out')
    score_cls = tf.layers.dense(flatten, 1, activation=tf.nn.sigmoid, name='sigmoid_out')
    return score_cls


def class_log_loss(score_true, score_fake):
    """
    分类loss
    :return:
    """
    y_true = tf.ones_like(score_true)
    y_fake = tf.zeros_like(score_fake)
    logs_true = -y_true * tf.log(score_true) - (1 - y_true) * tf.log(1 - score_true + FLAGS.clip_value_min)
    logs_fake = -y_fake * tf.log(score_fake) - (1 - y_fake) * tf.log(1 - score_fake + FLAGS.clip_value_min)

    # logs_loss = tf.reduce_mean(logs)
    return logs_true, logs_fake


def cross_entropy_loss(pred):
    """
    回归loss
    :param pred:
    :return:
    """
    gt0 = tf.zeros([FLAGS.batch_size_per_gpu, 1])
    gt1 = tf.ones([FLAGS.batch_size_per_gpu, 1])
    gt = tf.concat([gt0, gt1], 0)
    cross_entropy = -tf.reduce_sum(pred * tf.log(tf.clip_by_value(gt, 1e-10, 1.0)))
    return cross_entropy


def regress_loss(gt, pred):
    # # mse loss
    # losses = tf.losses.mean_squared_error(gt, pred)
    # hubers loss
    losses = tf.losses.huber_loss(gt, pred)
    # 平均损失
    # mean_loss = tf.reduce_mean(losses)
    return losses


def get_cos_distance(tensor1, tensor2):
    # calculate cos distance between two sets
    # more similar more big
    # 求模长
    tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1), -1))
    tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2), -1))
    # 内积
    tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2), -1)
    cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
    return cosin


def mmd_loss(score_flatten, fakes_flatten):
    """
    mmd_loss tensorflow 实现
    :param score_flatten:
    :param fakes_flatten:
    :return:
    """
    ms, mf = mmd_tensorflow.maximum_mean_discrepancy(score_flatten, fakes_flatten)
    cosin = get_cos_distance(ms, mf)
    return cosin


def loss(datas, fakes, gt, dropout_placeholdr):
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        # pred, score_flatten, score_cls = model_1(datas, dropout_placeholdr, num_classes=FLAGS.class_)
        # scope.reuse_variables() # 共享权重
        # fred, fakes_flatten, fakes_cls = model_1(fakes, dropout_placeholdr, num_classes=FLAGS.class_)
        pred, score_flatten_0, score_flatten_1 = model_1(datas, dropout_placeholdr, num_classes=FLAGS.class_)
        scope.reuse_variables() # 共享权重
        fred, fakes_flatten_0, fakes_flatten_1 = model_1(fakes, dropout_placeholdr, num_classes=FLAGS.class_)
    # 回归loss
    reg_loss = regress_loss(gt, pred)
    # 特征loss
    # dis_loss = get_cos_distance(score_flatten, fakes_flatten)
    # dis_loss = mmd_loss(score_flatten, fakes_flatten)
    dis_loss_0 = pair_distill_loss.main(score_flatten_0, fakes_flatten_0)
    dis_loss_1 = pair_distill_loss.main(score_flatten_1, fakes_flatten_1)
    # 分类loss
    # cla_t, cla_f = class_log_loss(score_cls, fakes_cls)
    # return reg_loss, dis_loss, cla_t, cla_f
    return reg_loss, dis_loss_0, dis_loss_1
