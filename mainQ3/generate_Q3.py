# encoding=utf-8

import numpy as np


def Q823(label_matrix):
    label3d = np.zeros((len(label_matrix), 3))
    for k, line in enumerate(label_matrix):
        if np.max(line) == 0:
            label3d[k] = [0, 0, 0]
            continue
        index = np.argmax(line)
        if index == 3 or index == 5 or index == 4:
            label3d[k] = [1, 0, 0]
        elif index == 2 or index == 1:
            label3d[k] = [0, 1, 0]
        else:
            label3d[k] = [0, 0, 1]
    return label3d


train_label_matrix = np.load('../DATA_CB513/train_label8.npy')
test_label_matrix = np.load('../DATA_CB513/test_label8.npy')

train_label3d = Q823(train_label_matrix)
test_label3d = Q823(test_label_matrix)

print 'before change:'
print test_label_matrix[:10]
print 'after change:'
print test_label3d[:10]

np.save('../DATA_CB513/train_label3.npy', train_label3d)
np.save('../DATA_CB513/test_label3.npy', test_label3d)
