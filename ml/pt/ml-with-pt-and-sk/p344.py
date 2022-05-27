from sklearn.datasets import fetch_openml
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X=X.values
y=y.astype(int).values
print(X.shape)
print(y.shape)
