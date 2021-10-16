# Import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv('C:/Users/kaila/Desktop/Heart_Diseases_API/data/heart.csv')
#df['id']=range(1, len(df)+1)
print(df.head())

# Do some research on the dataset
print(f'Shape of the dataset:{df.shape}')
print(f'Total data in the dataset:{df.size}')
print(f'Total number of null values:\n{df.isnull().sum()}')
print(f'Information on the dataset:\n{df.dtypes}')




# Splitting into target and feature columns
# Feature column
X = df.drop('target',axis=1)

# Target column
y = df['target']

#Splitting data into train and test. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

# Saving data to a csv
test.to_csv(r'C:/Users/kaila/Desktop/Heart_Diseases_API/data/test.csv', index=False)
train.to_csv(r'C:/Users/kaila/Desktop/Heart_Diseases_API/data/train.csv', index=False)
