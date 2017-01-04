# encoding=utf-8
import tensorflow as tf
import numpy as np

# Batch size
B = 4
# (Maximum) number of time steps in this batch
T = 8
RNN_DIM = 128
NUM_CLASSES = 10

# The *acutal* length of the examples
example_len = [1, 2, 3, 8]

# The classes of the examples at each step (between 1 and 9, 0 means padding)
# (4,8)
y = np.random.randint(1, 10, [B, T])
for i, length in enumerate(example_len):
    y[i, length:] = 0

print y

# The RNN outputs(4,8,128)
rnn_outputs = tf.convert_to_tensor(np.random.randn(B, T, RNN_DIM), dtype=tf.float32)

print rnn_outputs

# Output layer weights (128,10)
W = tf.get_variable(
    name="W",
    initializer=tf.random_normal_initializer(),
    shape=[RNN_DIM, NUM_CLASSES])

# Calculate logits and probs
# Reshape so we can calculate them all at once
rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, RNN_DIM])
logits_flat = tf.batch_matmul(rnn_outputs_flat, W)
probs_flat = tf.nn.softmax(logits_flat)

print rnn_outputs_flat, '=>', logits_flat, '=>', probs_flat

# Calculate the losses
y_flat = tf.reshape(y, [-1])
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(probs_flat, y_flat)

# Mask the losses
mask = tf.sign(tf.to_float(y_flat))
masked_losses = mask * losses

# Bring back to [B, T] shape
masked_losses = tf.reshape(masked_losses, tf.shape(y))

# Calculate mean loss
mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / example_len
mean_loss = tf.reduce_mean(mean_loss_by_example)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print '========= mean_loss ========='
    print sess.run(mean_loss)
    print '========= mask ========='
    print sess.run(mask)
    print '========= mean_loss_by_example ========='
    print sess.run(mean_loss_by_example)
    print '========= probs_flat ========='
    print sess.run(probs_flat)
    print '========= y_flat ========='
    print sess.run(y_flat)
