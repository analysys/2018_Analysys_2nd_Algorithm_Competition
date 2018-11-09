from tiny.usage import *

from code_felix.tiny.tfidf import *

age = pd.read_csv("./sub/baseline_age_2.03974.csv",index_col='DeviceID')

sex = pd.read_csv("./sub/baseline_sex_0.61968.csv",index_col='DeviceID')

age.columns=[f'age_{col}' for col in age.columns]


sex.columns=[f'age_{col}' for col in sex.columns]

sub = pd.DataFrame(index=age.index)
for sex_col in sex.columns:
    for age_col in age.columns:
        sub[f"{sex_col.split('_')[1]}-{age_col.split('_')[1]}"] = sex[sex_col] * age[age_col]

sub.to_csv('./sub/merge_sex_age.csv')
