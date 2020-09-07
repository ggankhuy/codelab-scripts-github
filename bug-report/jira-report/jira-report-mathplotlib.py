
import numpy as np
import matplotlib.pyplot as plt
import csv

with open('INTERNAL 2020-09-07T14_15_52-0500-filtered.csv', 'r') as f:
    jiradata = list(csv.reader(f, delimiter=','))

'''
t = np.arange(0., 5., 0.2)

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

#plt.plot([1, 2, 3, 4], [10, 20, -4, 100], 'ro')
#plt.ylabel('some numbers')
'''
plt.show()
