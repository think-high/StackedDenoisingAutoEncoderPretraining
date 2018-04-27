
# coding: utf-8

# In[1]:

""" Using stacked denoising auto encoders on MNIST dataset of handwritten digit
(http://yann.lecun.com/exdb/mnist/)
"""
from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:

import utils


# In[36]:

import numpy as np


# In[30]:

N_CLASSES = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
#mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

# Step 2: Define paramaters for the model
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
SKIP_STEP = 100
DROPOUT = 0.75
PRETRAINING_N_EPOCHS = 10
N_EPOCHS = 40


# In[23]:

# %reset


# In[123]:

tf.reset_default_graph()


# In[124]:

with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
    Y = tf.placeholder(tf.int32, [None, 10], name="Y_placeholder")

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                           100000, 0.96, staircase=True)


# utils.mkdir('checkpoints')
# utils.make_dir('checkpoints/prob4')
# utils.


# In[125]:

with tf.variable_scope('encoder_1') as scope:
    ew1 = tf.get_variable('Weight',[784,500],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    eb1 = tf.get_variable('biases',[500],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    noise1 = tf.placeholder(tf.float32, [None, 784], name="noise1")
    # Below is the noised data
    C_x = tf.multiply(X,noise1)
    h1_val = tf.add(tf.matmul(C_x,ew1) , eb1)
    h1 = tf.nn.relu(h1_val,name=scope.name)


# In[126]:

with tf.variable_scope('decoder_1') as scope:
    dw1 = tf.get_variable('Weight',[500,784],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    db1 = tf.get_variable('biases',[784],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    #noise1 = tf.placeholder(tf.float32, [None, 784], name="Y_placeholder")
    # Below is the noised data
    #C_x = tf.multiply(X,noise1)
    X_hat_val = tf.add(tf.matmul(h1,dw1) , db1)
    X_hat = tf.nn.relu(X_hat_val,name=scope.name)
    #layer_1_relu = tf.nn.relu(layer_1_val,name=scope.name)


# In[127]:

with tf.name_scope('encoder_cost1'):
#     encoder1_entropy = - tf.reduce_sum( X * tf.log(X_hat) + (1-X) * tf.log(1-X_hat) , axis=1)
#     encoder1_cost = tf.reduce_mean(encoder1_entropy)
    encoder1_cost = - tf.reduce_sum( X * tf.log(X_hat) + (1-X) * tf.log(1-X_hat) )
    
optimizer1 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(encoder1_cost)


# In[128]:

with tf.variable_scope('encoder_2') as scope:
    ew2 = tf.get_variable('Weight',[500,300],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    eb2 = tf.get_variable('biases',[300],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    noise2 = tf.placeholder(tf.float32, [None, 500], name="noise2")
    # Below is the noised data
    C_h1 = tf.multiply(h1,noise2)
    h2_val = tf.add(tf.matmul(C_h1,ew2) , eb2)
    h2 = tf.nn.relu(h2_val,name=scope.name)
    #layer_1_relu = tf.nn.relu(layer_1_val,name=scope.name)


# In[129]:

with tf.variable_scope('decoder_2') as scope:
    dw2 = tf.get_variable('Weight',[300,500],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    db2 = tf.get_variable('biases',[500],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    #noise1 = tf.placeholder(tf.float32, [None, 784], name="Y_placeholder")
    # Below is the noised data
    #C_x = tf.multiply(X,noise1)
    h1_hat_val = tf.add(tf.matmul(h2,dw2) , db2)
    h1_hat = tf.nn.relu(h1_hat_val,name=scope.name)


# In[130]:

with tf.name_scope('encoder_cost2'):
#     encoder2_entropy = - tf.reduce_sum( h1 * tf.log(h1_hat) + (1-h1) * tf.log(1-h1_hat) , axis=1)
#     encoder2_cost = tf.reduce_mean(encoder2_entropy)
    encoder2_cost = - tf.reduce_sum( h1 * tf.log(h1_hat) + (1-h1) * tf.log(1-h1_hat))
    
optimizer2 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(encoder2_cost)


# In[131]:

with tf.variable_scope('encoder_3') as scope:
    ew3 = tf.get_variable('Weight',[300,100],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    eb3 = tf.get_variable('biases',[100],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    noise3 = tf.placeholder(tf.float32, [None,300], name="noise3")
    # Below is the noised data
    C_h2 = tf.multiply(h2,noise3)
    h3_val = tf.add(tf.matmul(C_h2,ew3) , eb3)
    h3 = tf.nn.relu(h3_val,name=scope.name)


# In[132]:

with tf.variable_scope('decoder_3') as scope:
    dw3 = tf.get_variable('Weight',[100,300],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    db3 = tf.get_variable('biases',[300],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    #noise1 = tf.placeholder(tf.float32, [None, 784], name="Y_placeholder")
    # Below is the noised data
    #C_x = tf.multiply(X,noise1)
    h2_hat_val = tf.add(tf.matmul(h3,dw3) , db3)
    h2_hat = tf.nn.relu(h2_hat_val,name=scope.name)


# In[133]:

with tf.name_scope('encoder_cost3'):
    #encoder3_entropy = - tf.reduce_sum( h2 * tf.log(h2_hat) + (1-h2) * tf.log(1-h2_hat) , axis=1)
    #encoder3_cost = tf.reduce_mean(encoder3_entropy)
    encoder3_cost = - tf.reduce_sum( h2 * tf.log(h2_hat) + (1-h2) * tf.log(1-h2_hat))
    
optimizer3 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(encoder3_cost)


# ### Optimization session

# In[137]:

noise_1_mag  = 250
noise_2_mag  = 150
noise_3_mag  = 80


# In[ ]:

with tf.Session() as pre_train_sess:
    pre_train_sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     # to visualize using TensorBoard
#     writer = tf.summary.FileWriter('./my_graph/mnist', sess.graph)
#     ##### You have to create folders to store checkpoints
#     ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/prob4/checkpoint'))
#     # if that checkpoint exists, restore from checkpoint
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_cost1 = 0.0
    total_cost2 = 0.0
    total_cost3 = 0.0
    for index in range(initial_step, n_batches * PRETRAINING_N_EPOCHS):  # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
#         print(type(X_batch))
        noise1_batch = np.append(np.ones([784-noise_1_mag]),np.zeros([noise_1_mag]))
        np.random.shuffle(noise1_batch)
        for i in range(BATCH_SIZE-1):
            p1 = np.append(np.ones([784-noise_1_mag]),np.zeros([noise_1_mag]))
            np.random.shuffle(p1)
            noise1_batch = np.vstack([noise1_batch,p1])
        
        noise2_batch = np.append(np.ones([500-noise_2_mag]),np.zeros([noise_2_mag]))
        np.random.shuffle(noise2_batch)
        for i in range(BATCH_SIZE-1):
            p1 = np.append(np.ones([500-noise_2_mag]),np.zeros([noise_2_mag]))
            np.random.shuffle(p1)
            noise2_batch = np.vstack([noise2_batch,p1])
        
        noise3_batch = np.append(np.ones([300-noise_3_mag]),np.zeros([noise_3_mag]))
        np.random.shuffle(noise3_batch)
        for i in range(BATCH_SIZE-1):
            p1 = np.append(np.ones([300-noise_3_mag]),np.zeros([noise_3_mag]))
            np.random.shuffle(p1)
            noise3_batch = np.vstack([noise3_batch,p1])
        
        _, cost1 = pre_train_sess.run([optimizer1,encoder1_cost],feed_dict={X: X_batch, Y: Y_batch, noise1: noise1_batch})
        _, cost2 = pre_train_sess.run([optimizer2,encoder2_cost],feed_dict={X: X_batch, Y: Y_batch, noise1: noise1_batch, noise2: noise2_batch})
        _, cost3 = pre_train_sess.run([optimizer3,encoder3_cost],feed_dict={X: X_batch, Y: Y_batch, noise1: noise1_batch, noise2: noise2_batch,noise3: noise3_batch})
        
        total_cost1 += cost1
        total_cost2 += cost2
        total_cost3 += cost3
        
        if (index + 1) % SKIP_STEP == 0:
            print('Average cost1 at step {}: {:5.1f}'.format(index + 1, total_cost1 / SKIP_STEP))
            print('Average cost2 at step {}: {:5.1f}'.format(index + 1, total_cost2 / SKIP_STEP))
            print('Average cost3 at step {}: {:5.1f}'.format(index + 1, total_cost3 / SKIP_STEP))
            total_cost1 = 0.0
            total_cost2 = 0.0
            total_cost3 = 0.0
            #saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index)

    print("PreTraining Finished!") 
    print("Total time: {0} seconds".format(time.time() - start_time))


# In[91]:

from matplotlib import pyplot


# In[100]:

get_ipython().magic('matplotlib inline')


# ### Starting the supervised layers

# In[ ]:

with tf.variable_scope('layer1') as scope:
    w1 = tf.get_variable('Weight',[784,300],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    b1 = tf.get_variable('biases',[300],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    
    layer_1_val = tf.add(tf.matmul(X,w1) , b1)
    layer_1_relu = tf.nn.relu(layer_1_val,name=scope.name)


# In[27]:

with tf.variable_scope('layer2') as scope:
    w2 = tf.get_variable('weights',[300,100],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    b2 = tf.get_variable('biases',[100],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    
    layer_2_val = tf.add(tf.matmul(layer_1_relu,w2),b2)
    layer_2_relu = tf.nn.relu(layer_2_val, name=scope.name)

with tf.variable_scope('layer3') as scope:

    w3 = tf.get_variable('weights',[100,16],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    b3 = tf.get_variable('biases',[16],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    
    layer_3_val = tf.add(tf.matmul(layer_2_relu,w3),b3)
    layer_3_relu = tf.nn.relu(layer_3_val, name=scope.name)

with tf.variable_scope('softmax_linear') as scope:
    w_out = tf.get_variable('weights',shape=[16,10],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    b_out = tf.get_variable('biases',shape=[10],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),trainable=True)
    logits = tf.add(tf.matmul(layer_3_relu,w_out),b_out)

with tf.name_scope('loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
    loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)


# In[28]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     # to visualize using TensorBoard
#     writer = tf.summary.FileWriter('./my_graph/mnist', sess.graph)
#     ##### You have to create folders to store checkpoints
#     ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/prob4/checkpoint'))
#     # if that checkpoint exists, restore from checkpoint
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch = sess.run([optimizer, loss],
                                 feed_dict={X: X_batch, Y: Y_batch})
        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            #saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index)

    print("Optimization Finished!") 
    print("Total time: {0} seconds".format(time.time() - start_time))

    # test the model
    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        #optimizer,
        #_, 
        #loss_batch, 
        logits_batch,_ = sess.run([logits,optimizer] ,feed_dict={X: X_batch, Y: Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))


# In[29]:

#%reset


# In[ ]:




# In[ ]:




# In[ ]:



