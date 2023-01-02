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


print(pd.get_dummies(df[['price','color','size']]))
print("Drop first...")
print(pd.get_dummies(df[['price','color','size']], drop_first=True))

print("Drop first with OneHotEncoder")
X=df[['color', 'size','price']].values

# convert size column numerical ordinal.

print("Encoding size column...")
size_mapping={'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print(df.columns)
print(df)

X=df[['color', 'size','price']].values
color_le=LabelEncoder()

X[:,0] = color_le.fit_transform(X[:,0])
print(X)

# Encode class label.

print("Encoding classlabels...")
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel']=df['classlabel'].map(class_mapping)
print(class_mapping)
print(df)

color_ohe=OneHotEncoder(categories='auto', drop='first')
c_transf=ColumnTransformer([('onehot', color_ohe, [0]), ('nothing', 'passthrough', [1,2])])
print(c_transf.fit_transform(X).astype(float))
