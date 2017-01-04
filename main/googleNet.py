# encoding=utf-8

"""
=============current==============
run multi-conv and bi-rnn by embedding features,
but there is only one softmax layer in fully-net after single-layer bi-rnn.
"""

import tensorflow as tf
from data_process import input_cbfixed
from tensorflow.python.ops import rnn, rnn_cell
import time

# Global Parameter
display_step = 20
training_iters = 2000
batch_size = 8
keep_prob = 0.5

# CNN Parameter
learning_rate = 0.005

# BiRNN Parameters
n_input = 3 * 64
n_steps = 700  # timesteps
n_hidden = 60  # hidden layer num of features

cb513 = input_cbfixed.read_data_sets()
print '---- data input over -----'


# Create some wrappers for simplicity
def conv2d(x, W, b):
    x = tf.reshape(x, shape=[-1, 700, 71])
    # Conv2D wrapper, with bias and relu activation
    # x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.conv1d(x, W, 1, padding='SAME')

    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


def multi_conv(x, weights, biases):
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(x, weights['wc2'], biases['bc2'])
    conv3 = conv2d(x, weights['wc3'], biases['bc3'])

    convs = tf.concat(2, [conv1, conv2, conv3])

    return convs


def BiRNN(x, weights, biases, sequence_len):
    # Current data input shape: (batch_size, n_steps, n_input)
    # Forward direction cell
    lstm_fw_cell = rnn_cell.GRUCell(n_hidden)
    lstm_fw_cell = rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.GRUCell(n_hidden)
    lstm_bw_cell = rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
    # network = rnn_cell.MultiRNNCell([lstm_fw_cell, lstm_bw_cell] * 3)
    # x shape is [batch_size, max_time, input_size]
    out_tuple = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length=sequence_len,
                                              dtype=tf.float32)
    # Linear activation, using rnn inner loop last output 700*(n,2×n_hidden)->(700,n,2*n_hidden)
    # shape is n*700*(n_hidden+n_hidden) because of forward + backward
    outputs = tf.concat(2, out_tuple[0])
    # add the convolutional features,then the shape is (n,700,312)=(n,700,2*n_hidden).concat(n,700,192)
    outputs = tf.concat(2, [outputs, x])
    outputs = tf.reshape(outputs, [-1, 2 * n_hidden + 192])
    return tf.matmul(outputs, weights['out']) + biases['out']


weights = {
    'wc1': tf.Variable(tf.truncated_normal([3, 71, 64])),
    'wc2': tf.Variable(tf.truncated_normal([7, 71, 64])),
    'wc3': tf.Variable(tf.truncated_normal([9, 71, 64])),
    'out': tf.Variable(tf.truncated_normal([2 * n_hidden + 192, 8]))

}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([64])),
    'bc2': tf.Variable(tf.truncated_normal([64])),
    'bc3': tf.Variable(tf.truncated_normal([64])),
    'out': tf.Variable(tf.truncated_normal([8]))
}
# define placehold
x = tf.placeholder(tf.float32, [None, 700, 71])
y = tf.placeholder(tf.float32, [None, 700, 8])
sequence_lengths = tf.placeholder("int32", [None])
mask_weight = tf.placeholder(tf.float32, [None, 700])

# three convolutional, output shape is (n,700,192)
x_convs = multi_conv(x, weights, biases)
print 'after multiply convolutions: ', x_convs

# birnn, output shape is (n*700,192)*(192,8)=(n*700,8)
pred = BiRNN(x_convs, weights, biases, sequence_lengths)

# Define loss and optimizer (n*700)
mask = tf.reshape(mask_weight, [-1])
losses = tf.nn.softmax_cross_entropy_with_logits(pred, y)

# Mask the losses
# (700*n)=(n*700)*(700*n)
masked_losses = mask * losses
# (n,700)
masked_losses = tf.reshape(masked_losses, tf.shape(mask_weight))
# (n)
mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.cast(sequence_lengths, tf.float32)
# (1)
mean_loss = tf.reduce_mean(mean_loss_by_example)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

# Mask the prediction
# (n*700)
mask_pred = mask * tf.to_float(tf.argmax(pred, 1) + 1)
mask_y = tf.to_float(tf.argmax(tf.reshape(y, [-1, 8]), 1) + 1)
# (n*700)
correct_pred = tf.equal(mask_pred, mask_y)
# (n,700)
correct_pred = tf.reshape(correct_pred, tf.shape(mask_weight))
# (n)
accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32), reduction_indices=1) / tf.cast(sequence_lengths, tf.float32)
# (1)
accuracy = tf.reduce_mean(accuracy)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sum = 0
t1 = time.time()
print ' ---begin to training---'
with tf.Session() as sess:
    sess.run(init)
    for i in range(1, training_iters + 1):
        batch_xs, batch_ys, sequence_len, mask_wei = cb513.train.next_batch(batch_size)
        _, acc = sess.run([optimizer, accuracy],
                          feed_dict={x: batch_xs, y: batch_ys, sequence_lengths: sequence_len,
                                     mask_weight: mask_wei})
        sum += acc
        if i % display_step == 0:
            print 'training_iters:', i, 'accuracy:', sum / display_step
            sum = 0

    batch_xs, batch_ys, sequence_len, mask_wei = cb513.test.next_batch(cb513.test.num_examples)
    acc = sess.run(accuracy,
                   feed_dict={x: batch_xs, y: batch_ys, sequence_lengths: sequence_len, mask_weight: mask_wei})
    print 'final accuracy:', acc

t2 = time.time()
print 'Runtime:', '%.2f' % (t2 - t1), 'seconds'
