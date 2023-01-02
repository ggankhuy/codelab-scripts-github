#one hot encoding implementation.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

df=pd.DataFrame([['green','M',10.1,'class2'],['red','L',13.5, 'class1'],['blue','XL',15.3,'class2']])
df.columns=['color','size', 'price', 'classlabel']
print(df.columns)
print(df)

# convert size mapping to numerical ordinal.

print("Ordinal assignment...")
size_mapping={'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print(df.columns)
print(df)

X=df[['color', 'size','price']].values
color_le=LabelEncoder()

X[:,0] = color_le.fit_transform(X[:,0])
print(X)

# Encode class label.

print("Encodingclass labels...")
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel']=df['classlabel'].map(class_mapping)
print(class_mapping)
print(df)

#encoding class labels.
'''

print("Reversing...")
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel']=df['classlabel'].map(inv_class_mapping)
print(df)

print("Using LabelEncoder from sklearn...")

from sklearn.preprocessing import LabelEncoder
class_le=LabelEncoder()
y=class_le.fit_transform(df['classlabel'].values)
print(y)
'''
#print("Performing one-hot encoding now.")
#color_ohe=OneHotEncoder()
#print(color_ohe.fit_transform(X[:,0].reshape(-1,1)).toarray())

print("Performing selectively transforming multi-feature array...")

X=df[['color','size','price']].values
print(X)
c_transf=ColumnTransformer([\
('onehot', OneHotEncoder(), [0]), \
('nothing', 'passthrough', [1,2]) \
])
print(c_transf.fit_transform(X).astype(float))
'''
---
#mapping ordinal features.

import pandas as pd
df=pd.DataFrame([['green','M',10.1,'class2'],['red','L',13.5, 'class1'],['blue','XL',15.3,'class2']])
df.columns=['color','size', 'price', 'classlabel']
print(df.columns)
print(df)

#ordinal:
print("Ordinal assignment...")
size_mapping={'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print(df.columns)
print(df)

#inverting back  the integer values of size to original string representation

inv_size_mapping={v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))
'''
