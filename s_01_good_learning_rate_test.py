import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tensorflow as tf

learning_rates = [0.1, 0.01, 0.001]
x_train = [1,2,3]
y_train = [1,2,3]

for learning_rate in learning_rates:

    W = tf.Variable(1000000.0)
    b = tf.Variable(1000000.0)

    hypothesis = x_train * W + b

    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    for step in range(200001):
        sess.run(train)
        if step % 5000 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))
        if sess.run(cost) < 0.000000000005:
            print(step, sess.run(cost), sess.run(W), sess.run(b))
            break

    print('---------------------')
    time.sleep(1)
