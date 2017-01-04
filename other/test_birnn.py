# encoding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6
# X[1, 6:] = 0
X_lengths = [10, 6]

lstm_fw_cell = rnn_cell.GRUCell(64)
lstm_bw_cell = rnn_cell.GRUCell(64)

outputs, last_states = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X, sequence_length=X_lengths,
                                                     dtype=tf.float64)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

# 取第一次运行结果
res = result[0]
# fw
assert res["outputs"][0].shape == (2, 10, 64)
# bw
assert res["outputs"][1].shape == (2, 10, 64)

# Outputs for the second example past past length 6 should be 0
assert (res["outputs"][0][1, 7, :] == np.zeros(lstm_fw_cell.output_size)).all()

print res["outputs"][0][1, 7, :]

print '============'
print res["outputs"][0]
print '============'
print res["outputs"][1]

print '============='
# bw,batch=1,last_out
print res["outputs"][1][1][-1]
print '============='
