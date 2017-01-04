# encoding=utf-8

"""
directly run fully net by embedding features but without multi-conv or bi-rnn
"""

from data_process import input_cb513
import tensorflow as tf

cb513 = input_cb513.read_data_sets(truncated=True, embed=True)

batch_size = 32
training_iters = 10 * 10000
display_iters = 1000
learning_rate = 0.001

# > (5534*700=3873800)
print cb513.train.num_examples
assert training_iters * batch_size > cb513.train.num_examples

# 输入变量
x = tf.placeholder(tf.float32, [None, 71])
# 参数变量
W1 = tf.Variable(tf.truncated_normal([71, 128]))
W2 = tf.Variable(tf.truncated_normal([128, 64]))
W3 = tf.Variable(tf.truncated_normal([64, 8]))
# 偏置变量
b1 = tf.Variable(tf.truncated_normal([128]))
b2 = tf.Variable(tf.truncated_normal([64]))
b3 = tf.Variable(tf.truncated_normal([8]))

# 隐藏层输出(784个隐藏单元)
output1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
output2 = tf.nn.tanh(tf.matmul(output1, W2) + b2)
output3 = (tf.matmul(output2, W3) + b3)

y = tf.nn.softmax(output3)

# 定义标签
y_ = tf.placeholder("float", [None, 8])

# 输出一组bool值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 将bool值转化为浮点值求和再取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 定义交叉熵损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 定义训练方法
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# 初始化所有变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
sum = 0
for i in range(1, training_iters + 1):
    batch_xs, batch_ys = cb513.train.next_batch(batch_size)
    _, acc = sess.run([optimizer, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    sum += acc
    if i % display_iters == 0:
        print 'training_iters:', i, 'accuracy:', sum / display_iters
        sum = 0

print 'final accuracy:', sess.run(accuracy, feed_dict={x: cb513.test.raw_datas, y_: cb513.test.labels})
