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
mean_moving_normal = tf.random_normal(shape=[100],mean=(10*k),stddev=10)
## made a normal distrubituion with a expending variance  
expending_var_normal = tf.random_normal(shape=[100],mean=(10),stddev=10*k)
## we can also combine two distributions 
combined_dist = tf.concat([mean_moving_normal,expending_var_normal],axis=0)

## Record that distribution into a histogram summary 
tf.summary.histogram("normal/moving_mean",mean_moving_normal)
tf.summary.histogram("normal/expending_var",expending_var_normal)
tf.summary.histogram("normal/combine",combined_dist)

summaries = tf.summary.merge_all()

## 
sess = tf.Session()
writer = tf.summary.FileWriter("histogram_example")
N = 0 
for step in range(1990,2020):
    N+=1
    summ = sess.run(summaries,feed_dict={k:N})
    writer.add_summary(summ,global_step=step)

#%%
