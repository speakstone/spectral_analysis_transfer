#!/usr/bin/env python
# encoding: utf-8
'''
@author: lee
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: deamoncao100@gmail.com
@software: garner
@file: data_util.py
@time: 2021/12/24 9:56
@desc:
'''
import numpy as np
import tensorflow as tf
# tf.app.flags.DEFINE_string('dataset_path', r"E:\docker\Transfer-Learning\dataset\nir_shootout_MT.mat", '')
# tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
FLAGS = tf.app.flags.FLAGS

def data_load(data_dir):
    """
    载入数据
    :param data_dir:
    :return:
    """

    from scipy.io import loadmat
    data = loadmat(data_dir)
    Xm, Xs = data['Xm_502'], data['Xs_502']
    Xmtest, Xstest = data['Xtestm'], data['Xtests']
    ytrain, ytest = data['ycal_502'], data['ytest']
    # input image dimensions
    # xlength = len(Xm[0])
    img_rows, img_cols = 1, FLAGS.img_size
    # %%
    in_train = Xs.reshape(Xs.shape[0], img_cols, img_rows)
    on_train = Xm.reshape(Xm.shape[0], img_cols, img_rows)
    # return in_train, on_train, ytrain
    return on_train, in_train, ytrain


def data_load_eval(data_dir):
    """
    载入数据
    :param data_dir:
    :return:
    """

    from scipy.io import loadmat
    data = loadmat(data_dir)
    Xm, Xs = data['Xm_502'], data['Xs_502']
    Xmtest, Xstest = data['Xtestm'], data['Xtests']
    ytrain, ytest = data['ycal_502'], data['ytest']
    # input image dimensions
    xlength = len(Xm[0])
    img_rows, img_cols = 1, xlength
    # %%
    in_test = Xstest.reshape(Xstest.shape[0], img_cols, img_rows)
    on_test = Xmtest.reshape(Xmtest.shape[0], img_cols, img_rows)
    return in_test, on_test, ytest


class dataset():
    def __init__(self):
        self.data_dir = FLAGS.dataset_path
        self.batch_size = FLAGS.batch_size_per_gpu
        self.train_data, self.fake_data, self.train_gt = data_load(self.data_dir)
        self.train_len = len(self.train_data)

    def data_random_arrangement(self):
        """
        数据随机化
        :param data:
        :param label:
        :return:
        """
        data_random_seed = np.array(list(range(self.train_len)))
        np.random.shuffle(data_random_seed)
        # print(data_random_seed)
        data = self.train_data[data_random_seed]
        fake = self.fake_data[data_random_seed]
        label = self.train_gt[data_random_seed]
        # 补齐数据长度
        padding = self.batch_size - (self.train_len % self.batch_size)
        data = np.append(data, data[:padding], 0)
        fake = np.append(fake, fake[:padding], 0)
        label = np.append(label, label[:padding], 0)
        data_step = [data[i: i+self.batch_size] for i in range(0, self.train_len+ padding, self.batch_size)]
        fake_step = [fake[i: i+self.batch_size] for i in range(0, self.train_len+ padding, self.batch_size)]
        label_step = [label[i: i+self.batch_size] for i in range(0, self.train_len+ padding, self.batch_size)]
        return np.array(data_step), np.array(fake_step), np.array(label_step)



if __name__ == "__main__":
    data = dataset()
    data.data_random_arrangement()

