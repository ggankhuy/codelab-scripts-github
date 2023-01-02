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
