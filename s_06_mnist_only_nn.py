# why worst result??


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

regularization_strength = 0.0002
training_epochs = 20
batch_size = 100

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, 196]), name='weight')
b = tf.Variable(tf.random_normal([196]), name='bias')
layer3 = tf.matmul(X,W) + b

'''
W2 = tf.Variable(tf.random_normal([196, 49]), name='weight')
b2 = tf.Variable(tf.random_normal([49]), name='bias')
layer2 = tf.matmul(layer1,W2) + b2

W3 = tf.Variable(tf.random_normal([49, 20]), name='weight')
b3 = tf.Variable(tf.random_normal([20]), name='bias')
layer3 = tf.matmul(layer2,W3) + b3

'''
W4 = tf.Variable(tf.random_normal([196, nb_classes]), name='weight')
b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(layer3,W4) + b4
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(cost_i) + regularization_strength * tf.reduce_sum(tf.square(W4))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

w4_hist = tf.summary.histogram("weightsW4", W4)
cost_scal = tf.summary.scalar("cost", cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/s_06')
    writer.add_graph(sess.graph)

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


        s = sess.run(summary, feed_dict={X: batch_xs, Y: batch_ys})
        writer.add_summary(s, global_step=epoch)

    print("test accuracy : ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}) )
'''
    for step in range(1000):
        r = random.randint(0, mnist.test.num_examples -1)
        print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
        print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))

        plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
        plt.draw()
        plt.pause(1)
'''
    
'''
    pred, ans = sess.run([prediction, tf.argmax(mnist.test.labels,1)], feed_dict={X: mnist.test.images})
    for p, y in zip(pred,  ans):
        print("[{}] Prediction: {} answer: {}".format( p == int(y), p, int(y)))
'''
