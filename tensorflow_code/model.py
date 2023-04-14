#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/16 4:36
# @Author : {ZM7}
# @File : model.py
# @Software: PyCharm
import tensorflow as tf
import math

tf.compat.v1.disable_eager_execution()

class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.compat.v1.placeholder(dtype=tf.float32)
        self.alias = tf.compat.v1.placeholder(dtype=tf.int32)  # 给给每个输入重新
        self.item = tf.compat.v1.placeholder(dtype=tf.int32)   # 重新编号的序列构成的矩阵
        self.tar = tf.compat.v1.placeholder(dtype=tf.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.compat.v1.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.compat.v1.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.compat.v1.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.compat.v1.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.compat.v1.zeros_initializer())

    def forward(self, re_embedding, train=True):
        rm = tf.reduce_sum(input_tensor=self.mask, axis=1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.cast(rm, dtype=tf.int32)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(params=re_embedding[i], ids=self.alias[i]) for i in range(self.batch_size)],
                         axis=0)                                                           #batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        b = self.embedding[1:]
        if not self.nonhybrid:
            ma = tf.concat([tf.reduce_sum(input_tensor=tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, axis=1),
                            tf.reshape(last, [-1, self.out_size])], -1)
            self.B = tf.compat.v1.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            logits = tf.matmul(y1, b, transpose_b=True)
        else:
            ma = tf.reduce_sum(input_tensor=tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, axis=1)
            logits = tf.matmul(ma, b, transpose_b=True)
        loss = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.compat.v1.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, tar, item, adj_in, adj_out, alias, mask):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in,
                                                 self.adj_out: adj_out, self.alias: alias, self.mask: mask})


class GGNN(Model):
    def __init__(self,hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False, precision=None, device=None):
        super(GGNN,self).__init__(hidden_size, out_size, batch_size, nonhybrid)
        self.embedding = tf.compat.v1.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.adj_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        self.W_in = tf.compat.v1.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.compat.v1.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.compat.v1.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.compat.v1.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        with tf.compat.v1.variable_scope('ggnn_model', reuse=None):
            self.loss_train, _ = self.forward(self.ggnn())
        with tf.compat.v1.variable_scope('ggnn_model', reuse=True):
            self.loss_test, self.score_test = self.forward(self.ggnn(), train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.compat.v1.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        if precision == "float16" and device == "cuda":
            from tensorflow.core.protobuf import rewriter_config_pb2
            config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
            config.graph_options.rewrite_options.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON
            print("---- enable float16")
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(params=self.embedding, ids=self.item)
        cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.out_size)
        with tf.compat.v1.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.compat.v1.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])


