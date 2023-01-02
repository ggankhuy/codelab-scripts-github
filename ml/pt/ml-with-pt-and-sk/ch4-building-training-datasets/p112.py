#encoding class labels.

import numpy as np
import pandas as pd
df=pd.DataFrame([['green','M',10.1,'class2'],['red','L',13.5, 'class1'],['blue','XL',15.3,'class2']])
df.columns=['color','size', 'price', 'classlabel']
print(df.columns)
print(df)

print("Encoding class labels...")
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel']=df['classlabel'].map(class_mapping)
print(class_mapping)
print(df)

print("Reversing...")
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel']=df['classlabel'].map(inv_class_mapping)
print(df)

print("Using LabelEncoder from sklearn...")

from sklearn.preprocessing import LabelEncoder
class_le=LabelEncoder()
y=class_le.fit_transform(df['classlabel'].values)
print(y)

print("Inverse transform...")
print(class_le.inverse_transform(y))

