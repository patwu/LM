import sys
import numpy as np
import tensorflow as tf
import argparse
import os
import threading
import time

class LSTMLM(object):
    def __init__(self, args):
        self.args=args
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        self.graph = tf.Graph()
        self.sess=tf.Session(graph=self.graph, config=config)

    def _loss(self, xs, ys):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def build_model(self):
        with self.graph.as_default():
            batch_x = self.batch_x = tf.placeholder(tf.int64, shape=[self.args.batch_size,self.args.max_length],name='batch_x')
            len_x = self.len_x = tf.placeholder(tf.int64, shape=[self.args.batch_size],name='len_x')
            mask_x = self.mask_x = tf.placeholder(tf.float32, shape=[self.args.batch_size,self.args.max_length-1],name='mask_x')
            global_step = self.global_step= tf.Variable(0, name='global_step', trainable=False)

            embedding = tf.get_variable("embedding", [self.args.vocab_size, self.args.n_emb], dtype=tf.float32)
            batch_emb = tf.nn.embedding_lookup(embedding, batch_x)
            with tf.variable_scope('lstm') as scope:
                cell = tf.contrib.rnn.BasicLSTMCell(self.args.n_hidden)
                outputs,_ = tf.nn.dynamic_rnn(cell=cell,sequence_length=len_x,inputs=batch_emb)
            logits=tf.contrib.layers.fully_connected(outputs,self.args.vocab_size,activation_fn=None)
            self.preds=tf.nn.softmax(logits)    
           
            self.loss=self._loss(logits[:,:self.args.max_length-1],batch_x[:,1:],mask_x)
            opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            grads = opt.compute_gradients(loss)
            self.train_step=opt.apply_gradients(grads, global_step=self.global_step)

    def train(self,sentence):
        len_=[]
        mask=[] 
        batch_x[]
        for s in sentence:
            if len(s)>self.args.max_length:
                s=s[:self.args.max_length]
            len_.append(len(s))
            while len(s)<self.args.max_length:
                s+=[0]
            batch_x.append(s)
            m=[1]*l+[0]*(args.max_length-l)
            mask.append(m)
        _,loss=self.sess.run([self.train_step,self.loss],feed_dict={self.batch_x:batch_x,self.len_x:len_,self.mask_x:mask})
        return loss

    def save_model(self, path=None):
        if path is None:
            path=self.args.model_path
        self.saver.save(self.sess, os.path.join(path,'model.ckpt'), global_step=self.global_step)

    def load_model(self, path=None):
        if path is None:
            path=self.args.model_path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            print "Load model %s" % (ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            print "No model."
            return False
