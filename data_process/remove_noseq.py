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

a1 = train_sequence_feature.reshape((-1, 21))
a2 = train_profile_feature.reshape((-1, 21))
a3 = train_label8.reshape((-1, 8))

b1 = test_sequence_feature.reshape((-1, 21))
b2 = test_profile_feature.reshape((-1, 21))
b3 = test_label8.reshape((-1, 8))

index = {}
rana = set(range(len(a1)))
for i in rana:
    line = a1[i]
    if np.max(line) == 0:
        index[i] = 0
rana = list(rana.difference(set(index.keys())))
a1 = a1[rana]
a2 = a2[rana]
a3 = a3[rana]

index = {}
ranb = set(range(len(b1)))
for i in ranb:
    line = b1[i]
    if np.max(line) == 0:
        index[i] = 0
ranb = list(ranb.difference(set(index.keys())))
b1 = b1[ranb]
b2 = b2[ranb]
b3 = b3[ranb]

np.save('../DATA_NOSEQ/train_sequence_feature.npy', a1)
np.save('../DATA_NOSEQ/train_profile_feature.npy', a2)
np.save('../DATA_NOSEQ/train_label8.npy', a3)

np.save('../DATA_NOSEQ/test_sequence_feature.npy', b1)
np.save('../DATA_NOSEQ/test_profile_feature.npy', b2)
np.save('../DATA_NOSEQ/test_label8.npy', b3)

print 'truncated train dataset:', len(a1), 'truncated train dataset:', len(b1)
