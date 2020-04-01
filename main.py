import tensorflow as tf

tf.executing_eagerly()

x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# x에서 2배를 하면 y 값의 결과를 냄

w = tf.Variable(2.9)
b = tf.Variable(0.5)


learning_rate = 0.01 # learning rate
for i in range(100):
    #100번 Epoch
    with tf.GradientTape() as tape: # 경사하강법을 이용하여 cost 를 최소화한다.
            hypothesis = w * x_data + b
            cost = tf.reduce_mean(tf.square(hypothesis-y_data))
    w_grad,b_grad = tape.gradient(cost,[w,b])
    w.assign_sub(learning_rate*w_grad)
    b.assign_sub(learning_rate*b_grad)
    # w 와 b 값을 할당
    if i%10 == 0:
           print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i,w.numpy(),b.numpy(),cost))

print(w*28+b)
print(w*23+b)
# 예측 테스트
