import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta


a = np.arange(5)
hist, bin_edges = np.histogram(a, density=True)
hist
hist.sum()
np.sum(hist * np.diff(bin_edges))


rng = np.random.RandomState(10)  # deterministic random data
a = np.hstack((rng.normal(size=1000),
rng.normal(loc=5, scale=2, size=1000)))
plt.hist(a, bins='auto') 
plt.title("Histogram with 'auto' bins")
plt.show()
			   