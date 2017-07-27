# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:07:29 2017

@author: chuang
"""


import tensorflow as tf

## clear default graph
tf.reset_default_graph()
## a place holder to pass in variables 
k = tf.placeholder(tf.float32)

## made a normal distrubituion with a shifting mean 
mean_moving_normal = tf.random_normal(shape=[100],mean=(10*k),stddev=1)
## Record that distribution into a histogram summary 
tf.summary.histogram("normal/moving_mean",mean_moving_normal)
summaries = tf.summary.merge_all() 
#%%

## 
sess = tf.Session()
writer = tf.summary.FileWriter("histogram_example")
N = 20
for step in range(N):
    summ = sess.run(summaries,feed_dict={k:step})
    writer.add_summary(summ,global_step=step)
