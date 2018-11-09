from code_felix.tiny.util import *

tmp = pd.read_csv('input/deviceid_packages.tsv', sep='\t', header=None, nrows=None)
tmp.columns = ['device', 'package_list']


tmp = tmp[tmp.device.isin(get_test().iloc[:, 0])]

package_list_all = frozenset.union(*tmp.iloc[:, 1].apply(lambda v: frozenset(v.split(','))))

print(len(package_list_all))

i =1
batch = 1000
for package in package_list_all:

    print(f'{i}/{len(package_list_all)}')
    tmp[package] = tmp.apply(lambda _: int(package in _.package_list), axis=1)
    if i%batch == 0 or i == len(package_list_all) :
        #print(type(tmp.columns))
        columns = list(tmp.columns.values)
        columns.remove('package_list')
        tmp[columns].to_csv(f'./output/deviceid_package_{1+(batch*(round(i/batch)-1))}_{i}.tsv',
                            index=False)

        #Make the result set small
        tmp = tmp[['device', 'package_list']]

    i += 1