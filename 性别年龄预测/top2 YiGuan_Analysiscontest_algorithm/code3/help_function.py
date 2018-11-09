import pandas as pd

# load Train test data
def LoadData(train_path, test_path):
    train_data = pd.read_csv(train_path,sep='\t',names=['device_id','sex','age'])
    test_data = pd.read_csv(test_path,sep='\t',names=['device_id'])
    train_data['label'] = train_data.sex.astype(str)+'-'+train_data.age.astype(str)
    labelencode_dict = {}
    labels = ['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10',
              '2-0', '2-1', '2-2', '2-3','2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']
    for i,label in enumerate(labels):
        labelencode_dict[label] = i
    train_data.label.replace(labelencode_dict,inplace=True)
    return train_data, test_data

