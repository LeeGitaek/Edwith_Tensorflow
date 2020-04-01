import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 동물 데이터 불러오기
xy = np.loadtxt('/Users/gitaeklee/PycharmProjects/untitled2/data-04-zoo.csv', delimiter = ',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Y data가 0 ~ 6으로 7가지
nb_classes = 7

# placeholder
X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])  # shape = (?, 1)

# Y data를, one-hot으로 변경 : shape = (?, 1, 7)
Y_one_hot = tf.one_hot(Y, nb_classes)

# one_hot을 통과하면 차원이 늘어나므로, reshape로 줄여주기 : shape = (?, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

# Hypothesis : softmax function 사용
# softmax = exp(logits) / reduce_sum(exp(logits))
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# cost/loss function : cross entropy
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

# Minimize : Gradient Descent 사용
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# argmax() : [0.1, 0.3, 0.5]의 argmax는 1로 가장 큰 값의 index 출력
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션 시작
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print('Steps: {} Loss: {}, Acc: {}'.format(step + 1, loss, acc))
            # 2000 [0.053728174, 1.0]

    # Predict Test
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data.flatten() : 다차원 배열을 1차원 배열로 쭉 펴준다.
    # zip : pred와 y_data.flatten() 2개의 배열을 하나로 묶어서 p, y로 넘겨줌
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
        # [True] Prediction: 6 True Y: 6
        # [True] Prediction: 1 True Y: 1

