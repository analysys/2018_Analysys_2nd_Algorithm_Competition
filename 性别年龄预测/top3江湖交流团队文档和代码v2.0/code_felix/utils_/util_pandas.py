
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from code_felix.utils_.util_log import *


@timed(logger)
def convert_label_encode(sample, exclude=[]):
    try:
        #Label encode
        obj_col = sample.select_dtypes(include=['object']).columns
        obj_col = [ item for item in obj_col if item != 'device' and item not in exclude]
        print(f'{obj_col} will convert to label encode, and fillna with Other')


        sample = sample.apply(lambda x: x.fillna('Other')
                                if x.name in obj_col else x,
                                                      reduce=False)

        label_encode = defaultdict(LabelEncoder)
        sample = sample.apply(lambda x: label_encode[x.name].fit_transform(x.astype(str))
                        if x.name in obj_col else x,
                        reduce=False)


        return sample
    except Exception as e:
        print(f'The columns typs is {sample.dtypes.sort_values()}')
        raise e


import numpy as np
def check_exception(df, index=None):
    df = df.copy(deep=True)
    if index is not None and index in df:
        df.set_index(index,inplace=True)
    df = df.select_dtypes( #include=['float64', 'int'],
                           exclude=['object', 'datetime64[ns]'],)
    try:
        x, y = np.where(np.isinf(df.values) | np.isnan(df.values))
    except Exception as error:
        print(df.dtypes.sort_values())
        raise error
    if len(x)>0:
        print(x.min(), x.max()+1, y.min(), y.max()+1)
        df = df.iloc[x.min():(x.max()+1), y.min():(y.max()+1)]
        return df.iloc[:3, :4]
    else:
        return pd.DataFrame()
