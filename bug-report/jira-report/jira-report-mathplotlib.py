import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta

datetimeToday=datetime.today()

with open('INTERNAL 2020-09-07T14_15_52-0500-filtered.csv', 'r') as f:
    jiradata = list(csv.reader(f, delimiter=','))

#jiradata=jiradata[1][1]

print (jiradata)

for i in range(0, len(jiradata)):
    jiradata[i]= jiradata[i][8:10]

delta=[]

print("converting to datetime format...")
for i in range(1, len(jiradata)):
	closedDate=datetime.strptime(jiradata[i][0], '%m/%d/%Y %H:%M')
	openDate=datetime.strptime(jiradata[i][1], '%m/%d/%Y %H:%M')
	delta.append((closedDate-openDate).days)

for i in jiradata:
	print(i)
print(len(jiradata), len(jiradata[0]))

print("delta:")
for i in delta:
	print(i)

npdelta=np.asarray(delta)
print(npdelta.shape)

'''
#n, bins, patches = plt.hist(delta, num_bins, normed=1, facecolor='blue', alpha=0.5)
n, num_bins, patches = plt.hist(npdelta)

print("n, num_bins, patches: ", n, num_bins, patches)
hist, bin_edges = np.histogram(npdelta, density=True)
print("hist: ", hist)
print("hist.sum(): ", hist.sum())
#print(np.sum(hist * np.diff(bin_edges)))
'''
plt.hist(npdelta, bins=[10, 20, 30, 40, 50, 60, 100], histtype='step')
plt.xlabel('Number of days to resolve')
plt.ylabel('Number of tickets')
plt.title(r'Defect resolution data')
plt.show()
'''
plt.plot(bins, npdelta)
plt.show()
'''

'''
NP RANGE TOO UNFAMILIAR. Instead manipulate list and then convert to np before graphing
#npjiradata = np.array(jiradata[1:], dtype=np.float)
npjiradata = np.array(jiradata)

npjiradata_date_closed=npjiradata[:,8:9]
npjiradata_date_open=npjiradata[:,9:10]

print(npjiradata_date_closed)
print(type(npjiradata_date_closed))
print(npjiradata_date_closed.shape[0])

delta=[]

for i in range(1, npjiradata_date_closed.shape[0]):
	#npjiradata_date_closed[i][0]=npjiradata_date_closed[i][0].split(' ')[0]
	#npjiradata_date_open[i][0]=npjiradata_date_open[i][0].split(' ')[0]
	npjiradata_date_closed[i][0]=datetime.strptime(npjiradata_date_closed[i][0], '%m/%d/%Y %H:%M')
	npjiradata_date_open[i][0]=datetime.strptime(npjiradata_date_open[i][0], '%m/%d/%Y %H:%M')
	delta.append((npjiradata_date_closed[i][0]-npjiradata_date_open[i][0]).days)
print(npjiradata_date_closed)
print(npjiradata_date_open)
print(delta)
''
t = np.arange(0., 5., 0.2)

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

#plt.plot([1, 2, 3, 4], [10, 20, -4, 100], 'ro')
plt.ylabel('some numbers')
'''
