# encoding=utf-8

import numpy as np
import sys
import time

test_data = np.load("../DATA_CB513/cb513+profile_split1.npy")
print  type(test_data), len(test_data), test_data.shape

t1 = time.time()
train_data = np.load("../DATA_CB513/cullpdb+profile_6133_filtered.npy")
t2 = time.time()

print  type(train_data), len(train_data), train_data.shape

print 'Memory:', sys.getsizeof(train_data) / (1024 * 1024), 'M'
print 'Runtime:', '%.2f' % (t2 - t1), 'seconds'

train_data = train_data.reshape((len(train_data), 700, 57))
test_data = test_data.reshape((len(test_data), 700, 57))

# For example in test dataset: 514*700*57 ==> 514*700*21,514*700*21,514*700*8
for i in range(0, len(train_data), 57):
    train_sequence_feature = train_data[:, :, 0:21]
    train_profile_feature = train_data[:, :, 35:56]
    train_label8 = train_data[:, :, 22:30]
    test_sequence_feature = test_data[:, :, 0:21]
    test_profile_feature = test_data[:, :, 35:56]
    test_label8 = test_data[:, :, 22:30]

print '============ labels distribution: ============'
a = {i: 0 for i in range(8)}
for k in test_label8.reshape((-1, 8)):
    m = np.argmax(k)
    a[m] += 1

print a, np.sum(a.values())
# 氨基酸个数
assert np.sum(a.values()) == 514 * 700

np.save('../DATA_CB513/train_sequence_feature.npy', train_sequence_feature.reshape((-1, 21)))
np.save('../DATA_CB513/train_profile_feature.npy', train_profile_feature.reshape((-1, 21)))
np.save('../DATA_CB513/train_label8.npy', train_label8.reshape((-1, 8)))

np.save('../DATA_CB513/test_sequence_feature.npy', test_sequence_feature.reshape((-1, 21)))
np.save('../DATA_CB513/test_profile_feature.npy', test_profile_feature.reshape((-1, 21)))
np.save('../DATA_CB513/test_label8.npy', test_label8.reshape((-1, 8)))
