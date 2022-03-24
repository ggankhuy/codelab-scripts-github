import os
import tarfile
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ENABLE_PLOT=0
DOWNLOAD_ROOT="http://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH=os.path.join("datasets", "housing")
HOUSING_URL=DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
EXPERIMENTAL_CODE=0

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path=os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#p52
def split_train_test(data, test_ratio):
    DEBUG = 1
    shuffled_indices = np.random.permutation(len(data))

    if DEBUG:
        print("shuffled_indices: ", shuffled_indices)

    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]    


#p52-end
print("housing: ", HOUSING_PATH)
fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing=load_housing_data()

#p48
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

if ENABLE_PLOT:
    housing.hist(bins=50, figsize=(20,15))
    plt.show()

train_set, test_set=split_train_test(housing, 0.2)
print(len(train_set), len(test_set))

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
print(housing["income_cat"])

if ENABLE_PLOT:
    print(housing["income_cat"].hist())
    plt.show()

#p55
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
print(split)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_test_set["income_cat"].value_counts())


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


#p56

housing=strat_train_set.copy()

if ENABLE_PLOT:
    housing.plot(kind="scatter", x="longitude",  y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), \
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.show()

#looking for correlation using median housing value.

corr_matrix=housing.corr()
print(corr_matrix)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]

if ENABLE_PLOT:
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()

# experimental code.

if EXPERIMENTAL_CODE:
    print("households, 1st ten:")
    print(pd.DataFrame(housing).shape)
    print(pd.DataFrame((housing["households"])).shape)
    print(housing)
    print(housing["households"][:10])

#p62
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

if EXPERIMENTAL_CODE or 1:
    for i in ["rooms_per_household", "bedrooms_per_room", "population_per_household"]:
        print(i)
        print(housing[i][:20])

corr_matrix=housing.corr()
print(corr_matrix)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#p63
print("Data cleaning...")
print(housing.info())
housing=strat_train_set.drop("median_house_value", axis=1)
print(housing.info())
housing_labels=strat_train_set["median_house_value"].copy()
print(housing_labels)

#from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
housing_num = housing.drop("ocean_proximity", axis=1)

median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)
print(housing.info())

print(" - convert categorical tex value to encoded numerical values.")

housing_cat=housing[["ocean_proximity"]]
ordinal_encoder= OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(ordinal_encoder.categories_)

cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())

# Custom transformer

# from sklearn.base import BaseEstimator, TransformerMixin
# p68

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

print("housing_extra_attribs:", housing_extra_attribs)

# feature scaling
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
