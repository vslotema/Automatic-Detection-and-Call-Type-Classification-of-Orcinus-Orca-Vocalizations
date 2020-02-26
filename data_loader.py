import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split

#data sets
data_orig = pandas.read_csv("DeepAL_data.csv")
labels = data_orig.label.unique()


#splitting original data into 3 datasets: train, test, val (using numpy.sample)
train_np, validate_np, test_np = np.split(data_orig.sample(frac=1), [int(.6*len(data_orig)), int(.8*len(data_orig))])

# splitting original dataframe into 2, one for orcas and one for noise
df_orcas = data_orig.loc[data_orig['label'] == 'orca']
df_noise = data_orig.loc[data_orig['label'] == 'noise']


# splitting into test and train sklear version
#split arrays or matrices into random train and test subsets
#train, test = train_test_split(data_orig, test_size=0.2)

#printing to view first rows of the datasets
print(labels)
print(train_np[0:10])
print('train_np dataset length ', train_np.shape[0])
print(validate_np[0:10])
print('validation dataset length ', validate_np.shape[0])
print(test_np[0:10])
print('test dataset length ', test_np.shape[0])

#print(train.head(10))
#print(test.head(10))
#print('train dataset length ', train.shape[0])  # gives number of row count
#print('test dataset length', len(test))