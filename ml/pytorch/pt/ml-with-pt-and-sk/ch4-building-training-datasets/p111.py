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
