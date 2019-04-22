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

    def _loss(self, logits, labels,mask):
        logits=tf.reshape(logits,shape=(self.args.batch_size*(self.args.max_length-1),self.args.vocab_size))
        labels=tf.reshape(labels,shape=(self.args.batch_size*(self.args.max_length-1),))
        mask=tf.reshape(mask,shape=(self.args.batch_size*(self.args.max_length-1),))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy = cross_entropy*mask 
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        cross_entropy_mean = cross_entropy_mean/tf.reduce_sum(mask)*(self.args.batch_size*(self.args.max_length-1))
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
                outputs,_ = tf.nn.dynamic_rnn(cell=cell,sequence_length=len_x,inputs=batch_emb,dtype=tf.float32)
            logits=tf.contrib.layers.fully_connected(outputs,self.args.vocab_size,activation_fn=None)
            self.preds=tf.nn.softmax(logits)    
           
            self.loss=self._loss(logits[:,:self.args.max_length-1],batch_x[:,1:],mask_x)
            opt = tf.train.AdamOptimizer()
            grads = opt.compute_gradients(self.loss)
            self.train_step=opt.apply_gradients(grads, global_step=self.global_step)

            self.next=self.preds

            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
             

    def next_char(self,sentence):
        n_valid=len(sentence)
        while len(sentence)<self.args.batch_size:
            sentence.append([2])
        len_=[]
        batch_x=[]
        for s in sentence:
            l=len(s)
            if len(s)>self.args.max_length:
                s=s[:self.args.max_length]
            len_.append(len(s))
            while len(s)<self.args.max_length:
                s.append(0)
            batch_x.append(s)
        feed_dict={self.batch_x:batch_x,self.len_x:len_}
        preds=self.sess.run(self.next,feed_dict=feed_dict)
        next_=[]
        for i in range(n_valid):
            next_.append(preds[i,len_[i]-1])
        return next_
       
    def train(self,sentence):
        len_=[]
        mask=[] 
        batch_x=[]
        for s in sentence:
            l=len(s)
            if len(s)>self.args.max_length:
                s=s[:self.args.max_length]
            len_.append(len(s))
            mask.append([1]*len(s)+[0]*(self.args.max_length-len(s)))
            while len(s)<self.args.max_length:
                s.append(1)
            batch_x.append(s)
        mask=[m[1:] for m in mask]
        feed_dict={self.batch_x:batch_x,self.len_x:len_,self.mask_x:mask}
        _,loss=self.sess.run([self.train_step,self.loss],feed_dict=feed_dict)
        return loss

    def save_model(self, path=None):
        if path is None:
            path=self.args.model_path
        global_step=self.sess.run(self.global_step)
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

