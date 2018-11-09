
import os

import pandas as pd

from code_felix.utils_.util_log import *


class Cache_File:
    def __init__(self):
        self.cache_path='./cache/'
        self.enable=True
        self.date_list = ['start','close','start_base','weekbegin', 'tol_day_cnt_min',	'tol_day_cnt_max']
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

    def get_path(self, key, type):
        return f'{self.cache_path}{key}.{type}'

    def readFile(self, key, file_type):
        if self.enable:
            path = self.get_path(key, file_type)
            if os.path.exists(path):
                logger.debug(f"try to read cache from file:{path}, type:{file_type}")

                #check if the file have the data type column
                if file_type== 'pkl':
                    df = pd.read_pickle(path)
                    logger.debug(f"Load to {type(df)} with density:{df.density} for {file_type}@{path}")
                    df.fillna(0,inplace=True)
                elif file_type == 'h5':
                    df = pd.read_hdf(path, 'key')
                else:
                    df = pd.read_csv(path, nrows=1)
                    tmp_data_list = [item for item in self.date_list if item in df.columns]

                    df =pd.read_csv(path, parse_dates = tmp_data_list)
                logger.debug(f"Return {len(df) } resut from file cache:{path}")
                return df
            else:
                logger.debug(f"Can not find cache from file:{path}")
                return None
        else:
            logger.debug( "disable cache")


    def writeFile(self, key, val, type):
        if isinstance(val, pd.DataFrame ) and len(val)>0:
            path = self.get_path(key, type)
            logger.debug( f"====Write {len(val)} records to File#{path}" )
            if type == 'pkl':
                sparse = val.to_sparse(fill_value=0)
                logger.debug(f'The original sparse.density is {sparse.density}')
                if sparse.density > 0.1:
                    sparse = sparse.to_dense().to_sparse(fill_value=0)
                    logger.debug(f'The new sparse.density is convert to {sparse.density}')

                sparse.to_pickle(path)
                logger.debug(f'The sparse.density is {sparse.density}')
            elif type == 'h5':
                val.to_hdf(path, 'key')
            else:
                if isinstance(val, pd.SparseDataFrame):
                    val = val.to_dense()
                val.to_csv(path, index=False, )
            return val
        else:
            logger.warning('The return is not DataFrame or it is None')
            return None

cache =  Cache_File()

import functools
def file_cache(overwrite=False, type='csv'):
    """
    :param time: How long the case can keep, default is 1 week
    :param overwrite: If force overwrite the cache
    :return:
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            mini_args = get_mini_args(args)
            logger.debug(f'fn:{f.__name__}, para:{str(mini_args)}, kw:{str(kwargs)}')
            key = '_'.join([f.__name__, str(mini_args), str(kwargs)])
            if overwrite==False:
                val = cache.readFile(key, type)
            if overwrite==True or val is None :
                val = f(*args, **kwargs) # call the wrapped function, save in cache
                cache.writeFile(key, val, type)
            return val # read value from cache
        return wrapper
    return decorator

def get_mini_args(args):
    args_mini = [item.split('/')[-1] if isinstance(item, str) else item
                    for item in args
                        if (type(item) in (tuple, list, dict) and len(item) <= 5)
                            or type(item) not in (tuple, list, dict, pd.DataFrame)
                 ]



    df_list  =  [item for item in args if isinstance( item, pd.DataFrame) ]

    i=0
    for df in df_list:
        args_mini.append(f'df{i}_{len(df)}')
        i += 1

    return args_mini

if __name__ == '__main__':

    @timed()
    @file_cache()
    def test_cache(name):
        import time
        import numpy  as np
        time.sleep(3)
        return pd.DataFrame(data= np.arange(0,10).reshape(2,5))

    print(test_cache('Felix'))
    #print(test_cache('Felix'))




