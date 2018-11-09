import pandas as pd

from code_felix.tiny.util import get_category
from  code_felix.utils_.util_log import *

def read_result_for_ensemble(file):
    #file = f'./output/best/{name}.h5'
    store = pd.HDFStore(file)
    logger.debug(store.keys() )
    ensemble = (store["train"] if 'train' in store else None,
                store["label"] if 'label' in store else None,
                store["test"]  if 'test'  in store else None)
    store.close()
    return ensemble

def extract_sub(file, key='test'):
    store = pd.HDFStore(file, mode='r')

    if key in store:
        df = store[key]

        df.index.name='DeviceID'

        #df.columns = get_category()

        df = df[['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
             '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

        new_file = file.replace('h5', 'csv')

        df.to_csv(new_file)
        logger.debug(f'Extract sub to file:{new_file}')
    else:
        logger.debug(f"Can not find test in file:{file}")

    store.close()



if __name__ == '__main__':
    file_name = './output/best/baseline_kfold_xgb.h5'

    extract_sub('./sub/baseline_all_xgb_split_col_.h5')
    extract_sub('./sub/baseline_kfold_xgb.h5')
