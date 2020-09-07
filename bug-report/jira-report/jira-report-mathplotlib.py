
import numpy as np
import matplotlib.pyplot as plt
import csv

with open('INTERNAL 2020-09-07T14_15_52-0500-filtered.csv', 'r') as f:
    jiradata = list(csv.reader(f, delimiter=','))

#for i in jiradata:
#	print(i)

#npjiradata = np.array(jiradata[1:], dtype=np.float)
npjiradata = np.array(jiradata)

npjiradata_date_closed=npjiradata[:,8:9]
npjiradata_date_open=npjiradata[:,9:10]

print(npjiradata_date_closed)
print(type(npjiradata_date_closed))
print(npjiradata_date_closed.shape[0])

for i in range(0, npjiradata_date_closed.shape[0]):
	npjiradata_date_closed[i][0]=npjiradata_date_closed[i][0].split(' ')[0]
	npjiradata_date_open[i][0]=npjiradata_date_open[i][0].split(' ')[0]
	#print(npjiradata_date_closed[i][0])
	#print(type(npjiradata_date_closed[i][0]))
	
print(npjiradata_date_closed)
print(npjiradata_date_open)
'''
t = np.arange(0., 5., 0.2)

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

#plt.plot([1, 2, 3, 4], [10, 20, -4, 100], 'ro')
#plt.ylabel('some numbers')
'''
#plt.show()
