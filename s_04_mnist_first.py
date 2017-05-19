# best cost 0.298 reg 0.0002 epoch 1000 learning_rate 0.03


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

regularization_strength = 0.0002
training_epochs = 50
batch_size = 100

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(cost_i) + regularization_strength * tf.reduce_sum(tf.square(W))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        avg_acc = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            acc, c, _ = sess.run([accuracy, cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
            avg_acc += acc / total_batch


        print("epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(epoch, avg_cost, avg_acc))

    print("test accuracy : ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}) )

    for step in range(1000):
        r = random.randint(0, mnist.test.num_examples -1)
        print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
        print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))

        plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
        plt.draw()
        plt.pause(1)

    
'''
    pred, ans = sess.run([prediction, tf.argmax(mnist.test.labels,1)], feed_dict={X: mnist.test.images})
    for p, y in zip(pred,  ans):
        print("[{}] Prediction: {} answer: {}".format( p == int(y), p, int(y)))
'''
