#!/usr/bin/env python
# encoding: utf-8
'''
@author: lee
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: deamoncao100@gmail.com
@software: garner
@file: train.py
@time: 2021/12/24 10:32
@desc:
'''
import time
import tensorflow as tf
import model
import data_util
import numpy as np
import random

tf.app.flags.DEFINE_integer('batch_size_per_gpu', 32, '')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')
tf.app.flags.DEFINE_integer('max_epoch', 10000, '')
tf.app.flags.DEFINE_integer('img_size', 597, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_float('dropout_placeholdr', 0.25, '')
tf.app.flags.DEFINE_float('clip_value_min', 0.000001, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './checkpoint_path/', '')
tf.app.flags.DEFINE_string('logs_path', './logs/', '')
tf.app.flags.DEFINE_string('dataset_path', r"E:\docker\Transfer-Learning\dataset\nir_shootout_MT.mat", '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 500, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_integer('class_', 1, '')
FLAGS = tf.app.flags.FLAGS


def main(random_i, back_gan):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    datas_placeholder = tf.placeholder(tf.float32, shape=[None, 597, 1], name='input_images')
    fakes_placeholder = tf.placeholder(tf.float32, shape=[None, 597, 1], name='fakes_images')
    labels_placeholder = tf.placeholder(tf.float32, shape=[None, FLAGS.class_], name='output_maps')
    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)
    with tf.name_scope('Loss'):
        reg_loss, dis_loss0, dis_loss1 = model.loss(datas_placeholder, fakes_placeholder,
                                                    labels_placeholder, dropout_placeholdr)

        # mean_loss = tf.reduce_mean(reg_loss) - 0.05 * (tf.reduce_mean(cla_t) + tf.reduce_mean(cla_f))
        if back_gan:
            mean_loss = tf.reduce_mean(reg_loss) + 100 * tf.reduce_mean(dis_loss0 + dis_loss1)
        else:
            mean_loss = tf.reduce_mean(reg_loss)
    # 定义优化器，指定要优化的损失函数
    lr = FLAGS.learning_rate
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(mean_loss)
    # 用于保存和载入模型
    saver = tf.train.Saver()

    # 记录张量的数据
    tf.summary.scalar("Loss", mean_loss)
    merged_summary_op = tf.summary.merge_all()  # 定义记录运算


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        print("训练模式")
        # 如果是训练，初始化参数
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.logs_path, graph=tf.get_default_graph())  # 创建写对象
        # 定义输入和Label以填充容器，训练时dropout为0.25
        data = data_util.dataset()
        min_loss = 999999
        for epoch_i in range(FLAGS.max_epoch):
            datas_, fakes_, labels_ = data.data_random_arrangement()
            print(labels_[0].tolist())
            for index, data_label in enumerate(zip(datas_, fakes_, labels_)):
                train_feed_dict = {
                    datas_placeholder: data_label[0],
                    fakes_placeholder: data_label[1],
                    labels_placeholder: data_label[2],
                    dropout_placeholdr: FLAGS.dropout_placeholdr
                }
                _, mloss, rloss, d0loss, d1loss, summary = sess.run([optimizer,
                                                            mean_loss, reg_loss, dis_loss0, dis_loss1
                                                            , merged_summary_op], feed_dict=train_feed_dict)
                # 将日志写入文件
                summary_writer.add_summary(summary, epoch_i * len(data_label[0]) + index)
                if mloss < min_loss:
                    if back_gan:
                        save_dir = FLAGS.checkpoint_path + str(random_i) + "gan"
                    else:
                        save_dir = FLAGS.checkpoint_path + str(random_i)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    # 保留最优模型
                    min_loss = mloss
                    save_path = os.path.join(save_dir, '{}_loss_{}_model.ckpt'.format(epoch_i, min_loss))
                    saver.save(sess, save_path)
                if epoch_i % 1 == 0:
                    # print("rloss", rloss)
                    # print("dloss", dloss)
                    # print("dtloss", d0loss)
                    # print("dfloss", d1loss)
                    print('Epoch {}, step {}, mean_loss {:.4f}, reg_loss {:.4f}, dis_loss {:.4f}, '
                          'clat_loss {:.4f}'.format(
                        epoch_i, index, mloss, np.mean(rloss), np.mean(d0loss), np.mean(d1loss)))

            # if back_gan:
            #     save_dir = FLAGS.checkpoint_path + str(random_i) + "gan"
            # else:
            #     save_dir = FLAGS.checkpoint_path + str(random_i)
            # if not os.path.isdir(save_dir):
            #     os.mkdir(save_dir)
            # # 保留最优模型
            # min_loss = mloss
            # save_path = os.path.join(save_dir, '{}_loss_{}_model.ckpt'.format(epoch_i, min_loss))
            # saver.save(sess, save_path)


if __name__ == "__main__":
    seed = 6
    # 固定化随机种子
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main(seed, 1)
