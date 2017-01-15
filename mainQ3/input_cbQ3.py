# encoding=utf-8

# encoding=utf-8

import numpy as np


class DataSet(object):
    def __init__(self, feature_embeddings, labelp):
        self._feature_embedding = feature_embeddings
        self._labelp = labelp
        self._num_proteins = len(labelp)
        self._index_in_epoch = 0

    @property
    def datas(self):
        return self._feature_embedding

    @property
    def labels(self):
        return self._labelp

    @property
    def num_examples(self):
        return self._num_proteins

    # 每一个样本是一个蛋白质(700个氨基酸)
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set whose shape is (-1, 700, 71)"""
        if (self._index_in_epoch == 0):
            perm = np.arange(self._num_proteins)
            np.random.shuffle(perm)
            self._feature_embedding = self._feature_embedding[perm]
            self._labelp = self._labelp[perm]

        # 窗口末尾位置超过数据量大小
        if self._index_in_epoch + batch_size > self._num_proteins:
            # Start next epoch
            self._index_in_epoch = 0
            return self.next_batch(batch_size)

        start = self._index_in_epoch
        end = start + batch_size
        self._index_in_epoch = end
        seq_len = [0] * batch_size
        for i, protein in enumerate(self._labelp[start:end]):
            for x in protein:
                if np.max(x) != 0:
                    seq_len[i] += 1

        mask = np.ones((batch_size, 700))
        for i, length in enumerate(seq_len):
            mask[i, length:] = [0]

        return self._feature_embedding[start:end], self._labelp[start:end], seq_len, mask


def read_data_sets():
    class DataSets(object):
        pass

    data_sets = DataSets()

    train_sequence_feature = np.load('../DATA_CB513/train_sequence_feature.npy')
    train_profile_feature = np.load('../DATA_CB513/train_profile_feature.npy')
    train_label3 = np.load('../DATA_CB513/train_label3.npy').reshape((-1, 700, 3))
    test_sequence_feature = np.load('../DATA_CB513/test_sequence_feature.npy')
    test_profile_feature = np.load('../DATA_CB513/test_profile_feature.npy')
    test_label3 = np.load('../DATA_CB513/test_label3.npy').reshape((-1, 700, 3))

    feature_embeddings = np.load('../DATA_CB513/embeddings_matrix.npy')

    train_embeddings = np.dot(train_sequence_feature, feature_embeddings)
    train_sequence_feature = np.concatenate((train_embeddings, \
                                             train_profile_feature), axis=1).reshape((-1, 700, 71))

    test_embeddings = np.dot(test_sequence_feature, feature_embeddings)
    test_sequence_feature = np.concatenate((test_embeddings, \
                                            test_profile_feature), axis=1).reshape((-1, 700, 71))

    data_sets.train = DataSet(train_sequence_feature, train_label3)
    data_sets.test = DataSet(test_sequence_feature, test_label3)

    return data_sets


if __name__ == '__main__':
    datasets = read_data_sets()

    print datasets.train.datas.shape
    print datasets.test.datas.shape
