# encoding=utf-8

import numpy as np

array = np.load('../DATA_CB513/test_label8.npy')

count = 0
for i in range(len(array)):
    print array[i], 'max value in array[%d] in protein[%d,%d]:' % (i, i / 700, i % 700), np.max(array[i])
    if np.max(array[i]) == 0:
        count += 1

print 'len(test_label8) ', len(array)
print 'Nosequence:', count, 'Have sequence:', len(array) - count


