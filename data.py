from __future__ import print_function
import numpy as np
import cPickle as pickle
import gzip

class Data(object):
    def __init__(self, data):
        self._x, self._y = data
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = self._x.shape[0]
        self._cur_perm = np.arange(self._num_examples)
    
    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def data(self):
        return self._x, self._y
    
    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        """Return the next batch_size examples from the data"""
        # shuffle at the beginning of each epoch
        if self._index_in_epoch == 0:
            np.random.shuffle(self._cur_perm)
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        
        # indices of next batch (might be smaller than batch_size)
        batch_idx = self._cur_perm[start:end]
        
        # after each epoch
        if self._index_in_epoch >= self._num_examples:
            self._epochs_completed += 1
            # reset the index
            self._index_in_epoch = 0
            
        return self._x[batch_idx], self._y[batch_idx]
    
    def reset(self):
        """Reset num of epochs and index in epoch"""
        self._index_in_epoch = 0
        self._epochs_completed = 0

        

def read_dataset(data_path):
    """Construct the dataset given the pickle file"""
    if data_path.endswith('gz'):
        opener = gzip.open
    else:
        opener = open
    with opener(data_path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    return Data(train_set), Data(valid_set), Data(test_set)
        
