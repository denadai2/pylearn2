# -*- coding: utf-8 -*-
"""
A simple general csv dataset wrapper for pylearn2.
Can do automatic one-hot encoding based on labels present in a file.
"""
__authors__ = "Zygmunt Zając, Marco De Nadai"
__copyright__ = "Copyright 2013, Zygmunt Zając"
__credits__ = ["Zygmunt Zając", "Nicholas Dronen", "Marco De Nadai"]
__license__ = "3-clause BSD"
__maintainer__ = "?"
__email__ = "zygmunt@fastml.com"

import numpy as np
import logging

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng

log = logging.getLogger(__name__)

class CSVDataset(DenseDesignMatrix):
    """
    A generic class for accessing CSV files
    labels, if present, should be in the first column
    if there's no labels, set expect_labels to False
    if there's no header line in your file, set expect_headers to False

    Parameters
    ----------
    path : The path to the CSV file.
    task : str, optional
        The type of task in which the dataset will be used -- either "classification" or "regression".
        The task determines the shape of the target variable.  For classification, it is a vector; for regression,
        a matrix.
    one_hot : DEPRECATED. Whether the target variable (i.e. "label") should be encoded as a one-hot vector.
    num_outputs : int, optional
        number of target variables. defaults to 1
    expect_labels : bool, optional
        Whether the CSV file contains a target variable in the first column.
    expect_headers : bool, optional
        Whether the CSV file contains column headers.
    delimiter : str, optional
        The CSV file's delimiter.
    start : int, optional
        The first row of the CSV file to load.
    stop : int, optional
        The last row of the CSV file to load.
    start_fraction: int, optional
        The fraction of rows, starting at the beginning of the file, to load.
    end_fraction: int, optional
        The fraction of rows, starting at the end of the file, to load.
    shuffle: boolean, optional
        Whether the dataset needs to be row shuffled.
    """

    def __init__(self, 
            path = 'train.csv',
            task = 'classification',
            one_hot = False,
            num_outputs = 1,
            expect_labels = True,
            expect_headers = True,
            delimiter = ',',
            start = None,
            stop = None,
            start_fraction = None,
            end_fraction = None,
            shuffle = False):
        """
        .. todo::

            WRITEME
        """
        self.path = path
        self.task = task
        self.one_hot = one_hot
        self.num_outputs = num_outputs
        self.expect_labels = expect_labels
        self.expect_headers = expect_headers
        self.delimiter = delimiter
        self.start = start
        self.stop = stop
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction
        self.shuffle = shuffle
        
        self.view_converter = None

        if task not in ['classification', 'regression']:
            raise ValueError('task must be either "classification" or "regression"; got ' + str(task))

        if start_fraction is not None:
            if end_fraction is not None:
                raise ValueError("Use start_fraction or end_fraction, not both.")
            if start_fraction <= 0:
                raise ValueError("start_fraction should be > 0")

            if start_fraction >= 1:
                raise ValueError("start_fraction should be < 1")

        if end_fraction is not None:
            if end_fraction <= 0:
                raise ValueError("end_fraction should be > 0")

            if end_fraction >= 1:
                raise ValueError("end_fraction should be < 1")

        if start is not None:
            if start_fraction is not None or end_fraction is not None:
                raise ValueError("Use start, start_fraction, or end_fraction, just not together.")

        if stop is not None:
            if start_fraction is not None or end_fraction is not None:
                raise ValueError("Use stop, start_fraction, or end_fraction, just not together.")

        # and go

        self.path = preprocess(self.path)
        X, y = self._load_data()
        
        super(CSVDataset, self).__init__(X=X, y=y)

    def _load_data(self):
        """
        .. todo::

            WRITEME
        """
        assert self.path.endswith('.csv')
    
        if self.expect_headers:
            data = np.loadtxt(self.path, delimiter=self.delimiter, skiprows=1)
        else:
            data = np.loadtxt(self.path, delimiter=self.delimiter)

        def take_subset(X, y):
            if self.start_fraction is not None:
                n = X.shape[0]
                subset_end = int(self.start_fraction * n)
                X = X[0:subset_end, :]
                y = y[0:subset_end]
            elif self.end_fraction is not None:
                n = X.shape[0]
                subset_start = int((1-self.end_fraction) * n)
                X = X[subset_start:, ]
                y = y[subset_start:]
            elif self.start is not None:
                X = X[self.start:self.stop, ]
                if y is not None:
                    y = y[self.start:self.stop]

            return X, y

        if self.shuffle:
            np.random.shuffle(data)
        
        if self.expect_labels:
            y = data[:, 0:self.num_outputs]
            X = data[:, self.num_outputs:]
            
            # get unique labels and map them to one-hot positions
            labels = np.unique(y)
            labels = dict((x, i) for (i, x) in enumerate(labels))

            if self.one_hot:
                log.warning("the `one_hot` parameter is deprecated. To get one-hot encoded targets, request that they \
 live in `VectorSpace` through the `data_specs` parameter of MNIST's iterator method. `one_hot` will be removed on or \
 after December 27, 2014.")
                one_hot = np.zeros((y.shape[0], len(labels)), dtype='float32')
                for i in range(y.shape[0]):
                    label = y[i]
                    label_position = labels[label]
                    one_hot[i, label_position] = 1.
                y = one_hot
            else:
                if self.task == 'regression':
                    y = y.reshape((y.shape[0], 1))

        else:
            X = data
            y = None

        X, y = take_subset(X, y)

        return X, y
