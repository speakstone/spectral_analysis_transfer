#!/usr/bin/env python
# encoding: utf-8
'''
@author: lee
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: deamoncao100@gmail.com
@software: garner
@file: eval.py
@time: 2021/12/24 10:11
@desc:
'''
import numpy as np
import tensorflow as tf
import data_util
import model

# tf.app.flags.DEFINE_string('dataset_path', r"E:\docker\Transfer-Learning\dataset\nir_shootout_MT.mat", '')
# # 修改测试模型路径，gan是使用了迁移学习
# tf.app.flags.DEFINE_string('checkpoint_path', './checkpoint_path/8gan/', '')
# tf.app.flags.DEFINE_integer('class_', 1, '')
# tf.app.flags.DEFINE_string('gpu_list', '0', '')
# FLAGS = tf.app.flags.FLAGS

dataset_path = r"E:\docker\Transfer-Learning\dataset\nir_shootout_MT.mat"
checkpoint_path = './checkpoint_path/8gan/'
class_ = 1
gpu_list = "0"


def mse(y_test, y_true):
    """
    mes指标
    :param y_test:
    :param y_true:
    :return:
    """
    return np.mean((y_test - y_true) ** 2)

def rmse(y_test, y_true):
    """
    rmse指标
    :param y_test:
    :param y_true:
    :return:
    """
    return np.sqrt(np.mean((y_test - y_true) ** 2))

def mae(y_test, y_true):
    """
    mae指标
    :param y_test:
    :param y_true:
    :return:
    """
    return np.sum(np.absolute(y_test - y_true)) / len(y_test)

def r2(y_test, y_true):
    """
    r2指标
    :param y_test:
    :param y_true:
    :return:
    """
    return 1 - ((y_test - y_true) ** 2).sum() / ((y_true - np.mean(y_true)) ** 2).sum()


def eval():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    with tf.get_default_graph().as_default():
        datas_p = tf.placeholder(tf.float32, shape=[None, 597, 1], name='input_images')
        labels_p = tf.placeholder(tf.float32, shape=[None, class_], name='output_maps')
        dropout_placeholdr = tf.placeholder(tf.float32)
        f_score,_,_ = model.model_1(datas_p, dropout_placeholdr, class_)
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            datas_, data_eval, labels_ = data_util.data_load_eval(dataset_path)
            output_p = np.zeros([datas_.shape[0], 1])
            test_feed_dict = {
                datas_p: datas_,
                labels_p: output_p,
                dropout_placeholdr: 0
            }
            pred_val = sess.run(f_score, feed_dict=test_feed_dict)
            print("当前模型测试mse:{}, rmse:{}, mae:{}, r2:{}".format(mse(pred_val, labels_), rmse(pred_val, labels_)
                                                                , mae(pred_val, labels_), r2(pred_val, labels_)))


            output_p = np.zeros([data_eval.shape[0], 1])
            test_feed_dict = {
                datas_p: data_eval,
                labels_p: output_p,
                dropout_placeholdr: 0
            }
            pred_val = sess.run(f_score, feed_dict=test_feed_dict)
            print("当前模型迁移测试mse:{}, rmse:{}, mae:{}, r2:{}".format(mse(pred_val, labels_), rmse(pred_val, labels_)
                                                                , mae(pred_val, labels_), r2(pred_val, labels_)))
            # print("当前模型迁移测试\t{}\t{}\t{}\t{}".format(mse(pred_val, labels_), rmse(pred_val, labels_)
            #                                                     , mae(pred_val, labels_), r2(pred_val, labels_)))
            #
            #

if __name__ == "__main__":
    eval()