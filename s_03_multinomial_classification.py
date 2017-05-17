import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tensorflow as tf
import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


regularization_strength = 0.0001

xy = np.loadtxt('shine_data/s_03_training.csv', delimiter=',', dtype=np.float32)
test_xy = np.loadtxt('shine_data/s_03_test.csv', delimiter=',', dtype=np.float32)
x_data = MinMaxScaler(xy[:, 0:-1])
y_data = xy[:, [-1]]
test_x_data = MinMaxScaler(test_xy[:, 0:-1])
test_y_data = test_xy[:, [-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i) + regularization_strength * tf.reduce_sum(tf.square(W))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))


    pred, hy = sess.run([prediction, hypothesis], feed_dict={X: test_x_data})
    for p, h, y in zip(pred, hy, test_y_data.flatten()):
        print("[{}] Prediction: {} answer: {}".format( p == int(y), h, int(y)))
