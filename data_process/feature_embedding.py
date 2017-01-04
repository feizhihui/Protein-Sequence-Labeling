# encoding=utf-8

import input_cb513
import tensorflow as tf
import math
import numpy as np

training_iters = 5 * 10000

learning_rate = 0.005

skip_window = 2
num_skips = 4
batch_size = 128

cb513 = input_cb513.read_data_sets(truncated=True)

TRAIN_LEN = len(cb513.train.datas)
TEST_LEN = len(cb513.test.datas)
print TRAIN_LEN
print 'size of the train copus:', (TRAIN_LEN - 2 * skip_window) * num_skips
print 'size of the test copus:', (TEST_LEN - 2 * skip_window) * num_skips

# 最少要遍历一次语料库
assert training_iters * batch_size > (TRAIN_LEN - 2 * skip_window) * num_skips

# 输入变量
x = tf.placeholder(tf.float32, [None, 21])
# 参数变量
# embeddings = tf.Variable(tf.zeros([21, 50]))
embeddings = tf.Variable(tf.random_uniform([21, 50], -1.0, 1.0))
# W = tf.Variable(tf.zeros([50, 8]))
W = tf.Variable(
    tf.truncated_normal([50, 21],
                        stddev=1.0 / math.sqrt(50)))
# 偏置变量
b1 = tf.Variable(tf.zeros([50]))
b2 = tf.Variable(tf.zeros([21]))

embed = tf.matmul(x, embeddings) + b1
y = tf.nn.softmax(tf.matmul(embed, W) + b2)

# 定义标签
y_ = tf.placeholder("float", [None, 21])

# 输出一组bool值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 将bool值转化为浮点值求和再取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 定义交叉熵损失函数
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
cost = tf.reduce_mean(tf.pow(y - y_, 2))

# 定义训练方法
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
sum = acc = 0
for i in range(1, training_iters + 1):
    # 随机取出一批数据 128*21,128*8
    batch_xs, batch_ys = cb513.train.generate_context_batch(batch_size, num_skips, skip_window)
    _, acc = sess.run([optimizer, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    sum += acc
    if i % 10000 == 0:
        print 'training_iters:', i, 'accuracy:', sum / 10000
        sum = 0

# 输出一组bool值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

test_xs, test_ys = cb513.test.generate_context_batch((TEST_LEN - 2 * skip_window) * num_skips, num_skips, skip_window)
# test_xs, test_ys = cb513.train.generate_context_batch((TRAIN_LEN - 2 * skip_window) * num_skips, num_skips, skip_window)

print sess.run(accuracy,
               feed_dict={x: test_xs, y_: test_ys})

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

# 正则化embeddings
normalized_embeddings = embeddings / norm

feature_embeddings = sess.run(normalized_embeddings)

np.save('../DATA_CB513/embeddings_matrix.npy', feature_embeddings)
