from sklearn.datasets import fetch_openml
#import matplotlib.pyplot as plt
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X=X.values
y=y.astype(int).values
print(X.shape)
print(y.shape)

X = ((X/255.)-0.5)*2

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax=ax.flatten()

for i in range(10):
    img = X[y==1][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

