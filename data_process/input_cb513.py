# encoding=utf-8

import numpy as np
from matplotlib.pyplot import axis


class DataSet(object):
    def __init__(self, sequence_feature, profile_feature, label8):
        self._sequence_feature = sequence_feature
        self._profile_feature = profile_feature
        self._label8 = label8
        self._num_examples = len(label8)
        self._seq_index = 0
        self._index_in_epoch = 0

    @property
    def datas(self):
        return self._sequence_feature

    @property
    def labels(self):
        return self._label8

    @property
    def raw_datas(self):
        return np.concatenate((self._sequence_feature, self._profile_feature), axis=1)

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        if (self._index_in_epoch == 0):
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._sequence_feature = self._sequence_feature[perm]
            self._profile_feature = self._profile_feature[perm]
            self._label8 = self._label8[perm]

        # 窗口末尾位置超过数据量大小
        if self._index_in_epoch + batch_size > self._num_examples:
            # Start next epoch
            self._index_in_epoch = 0
            return self.next_batch(batch_size)

        start = self._index_in_epoch
        end = self._index_in_epoch + batch_size
        self._index_in_epoch = end
        return np.concatenate((self._sequence_feature[start:end], self._profile_feature[start:end]),
                              axis=1), self._label8[start:end]

    def generate_context_batch(self, batch_size, num_skips, skip_window):
        assert num_skips == 2 * skip_window
        assert batch_size % num_skips == 0

        if self._seq_index - skip_window < 0:
            self._seq_index = skip_window

        target_batch = np.ndarray(shape=(batch_size, 21))
        context_label = np.ndarray(shape=(batch_size, 21))
        i = 0
        # 外循环次数为目标氨基酸的个数
        for h in range(batch_size / (2 * skip_window)):
            # 内循环次数为每个目标氨基酸的上下文氨基酸的个数
            for k in range(self._seq_index - skip_window, self._seq_index) \
                    + range(self._seq_index, self._seq_index + skip_window):
                target_batch[i] = self._sequence_feature[self._seq_index]
                context_label[i] = self._sequence_feature[k]
                i += 1
            self._seq_index += 1
            if self._seq_index + skip_window > self._num_examples:
                self._seq_index = skip_window

        assert i == batch_size
        return target_batch, context_label


def read_data_sets(truncated=False, embed=False):
    class DataSets(object):
        pass

    data_sets = DataSets()
    if truncated == False:
        train_sequence_feature = np.load('../DATA_CB513/train_sequence_feature.npy')
        train_profile_feature = np.load('../DATA_CB513/train_profile_feature.npy')
        train_label8 = np.load('../DATA_CB513/train_label8.npy')
        test_sequence_feature = np.load('../DATA_CB513/test_sequence_feature.npy')
        test_profile_feature = np.load('../DATA_CB513/test_profile_feature.npy')
        test_label8 = np.load('../DATA_CB513/test_label8.npy')
    else:
        train_sequence_feature = np.load('../DATA_NOSEQ/train_sequence_feature.npy')
        train_profile_feature = np.load('../DATA_NOSEQ/train_profile_feature.npy')
        train_label8 = np.load('../DATA_NOSEQ/train_label8.npy')
        test_sequence_feature = np.load('../DATA_NOSEQ/test_sequence_feature.npy')
        test_profile_feature = np.load('../DATA_NOSEQ/test_profile_feature.npy')
        test_label8 = np.load('../DATA_NOSEQ/test_label8.npy')

    if embed:
        feature_embeddings = np.load('../DATA_CB513/embeddings_matrix.npy')
        train_sequence_feature = np.dot(train_sequence_feature, feature_embeddings)
        test_sequence_feature = np.dot(test_sequence_feature, feature_embeddings)
        assert train_sequence_feature.shape[1] == 50

    data_sets.train = DataSet(train_sequence_feature, train_profile_feature, train_label8)
    data_sets.test = DataSet(test_sequence_feature, test_profile_feature, test_label8)

    return data_sets


if __name__ == '__main__':
    datasets = read_data_sets()
    k = 0
    # test 0 ~ 400*700 rows
    for i in range(len(datasets.test.datas[:400 * 700])):
        k += 1
        print "%2d" % np.argmax(datasets.test.datas[i]),
        if k == 700:
            print 'end'
            k = 0

    print datasets.train.datas.shape
    print datasets.test.datas.shape
