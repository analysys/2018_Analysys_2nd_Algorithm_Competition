import pandas as pd
import numpy as np
import os

import time
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('%s done in {time.time() - t0:.3f} s' % (name))

def read_csv(fname, **kwargs):
        file1=open(fname)
        data_=[]
        for line in file1:
                data_.append(line.replace("\n","").split(","))
        data = pd.DataFrame(data_[1:])
        data.columns=[data_[0]]
        return data

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return [{'num_letter': len(texts)}
                for text in texts]

