

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

regularization_strength = 0.002
training_epochs = 50
batch_size = 100

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

dropout_rate = tf.placeholder( "float" )
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, nb_classes])

CW1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))

L1 = tf.nn.relu(tf.nn.conv2d(X_img, CW1, strides=[1,1,1,1], padding='SAME')) # 28 x 28
L1_pool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # 14 x 14 

CW2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
L2 = tf.nn.relu(tf.nn.conv2d(L1_pool, CW2, strides=[1,1,1,1], padding='SAME')) # 14 x 14
L2_pool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # 7 x 7
L2_last = tf.reshape(L2_pool, [-1, 7*7*64])

# FC
W1 = tf.get_variable("W1", shape=[7*7*64, 1024], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([1024]), name='bias')
layer1 = tf.nn.relu(tf.matmul(L2_last,W1) + b)
layer1_ = tf.nn.dropout(layer1, dropout_rate)

W4 = tf.get_variable("W4", shape=[1024, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.zeros([nb_classes]), name='bias')

logits = tf.matmul(layer1_,W4) + b4
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(cost_i) + regularization_strength * tf.reduce_sum(tf.square(W4))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

w4_hist = tf.summary.histogram("weightsW4", W4)
cost_scal = tf.summary.scalar("cost", cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/s_08')
    writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0
        avg_acc = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            acc, c, _ = sess.run([accuracy, cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, dropout_rate:0.7})
            avg_cost += c / total_batch
            avg_acc += acc / total_batch


        print("epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(epoch, avg_cost, avg_acc))


        s = sess.run(summary, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate:0.7})
        writer.add_summary(s, global_step=epoch)

        print("test accuracy : ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, dropout_rate:1}) )

    for step in range(100):
        r = random.randint(0, mnist.test.num_examples -1)
        print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
        print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1], dropout_rate:1}))

        plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
        plt.draw()
        plt.pause(1)
    
'''
    pred, ans = sess.run([prediction, tf.argmax(mnist.test.labels,1)], feed_dict={X: mnist.test.images})
    for p, y in zip(pred,  ans):
        print("[{}] Prediction: {} answer: {}".format( p == int(y), p, int(y)))
'''
