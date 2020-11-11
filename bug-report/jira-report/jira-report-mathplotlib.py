import numpy as np
import matplotlib.pyplot as plt
import csv
import re

from datetime import datetime, timedelta

datetimeToday=datetime.today()
fileName='INTERNAL 2020-10-10T01_44_07-0500-filtered.csv'

# Open exported file: exported from Jira. Column 8-9 must contain closed and open dates, respectively otherwise
# Code will either break or need adjustment. 
# Column 13 must be a priority, otherwise code will either break or need adjustment. 

with open(fileName, 'r') as f:
    jiraData = list(csv.reader(f, delimiter=','))
jiraDataDates=[]

jiraDataP0=[]
jiraDataP0Dates=[]
jiraDataP1=[]
jiraDataP1Dates=[]

# jiraDataP0/P1: Gather p0 and p1 designated tickets.
# jiraDataDates: Gather only column 8, 9 which contains closed and open dates only. 
for i in range(0, len(jiraData)):
    jiraDataDates.append(jiraData[i][8:10])
	
    if re.search("P1", jiraData[i][13]):
	    jiraDataP0.append(jiraData[i])
		
    if re.search("P2", jiraData[i][13]):
        jiraDataP1.append(jiraData[i])

# jiraDataDatesP0/P1: Gather only column 8, 9 for P0/P1 which contains closed and open dates only. 

for i in range(0, len(jiraDataP0)):
    jiraDataP0Dates.append(jiraDataP0[i][8:10])
    
for i in range(0, len(jiraDataP1)):
    jiraDataP1Dates.append(jiraDataP1[i][8:10])
    
print("Total No. of tickets imported: ", len(jiraData))    
print("Total No. of P0 tickets imported: ", len(jiraDataP0))    
print("Total No. of P1 tickets imported: ", len(jiraDataP1))    
    
for i in jiraData:
    print(i)

for i in jiraDataDates:
    print(i)

# holds delta (closed - open) in number of days	
delta=[]
deltaP0=[]
deltaP1=[]

print("converting to datetime format...")

for i in range(1, len(jiraDataDates)):
    closedDate=datetime.strptime(jiraDataDates[i][0], '%m/%d/%Y %H:%M')
    openDate=datetime.strptime(jiraDataDates[i][1], '%m/%d/%Y %H:%M')
    delta.append((closedDate-openDate).days)
	
for i in range(1, len(jiraDataP0Dates)):
    closedDate=datetime.strptime(jiraDataP0Dates[i][0], '%m/%d/%Y %H:%M')
    openDate=datetime.strptime(jiraDataP0Dates[i][1], '%m/%d/%Y %H:%M')
    deltaP0.append((closedDate-openDate).days)

for i in range(1, len(jiraDataP1Dates)):
    closedDate=datetime.strptime(jiraDataP1Dates[i][0], '%m/%d/%Y %H:%M')
    openDate=datetime.strptime(jiraDataP1Dates[i][1], '%m/%d/%Y %H:%M')
    deltaP1.append((closedDate-openDate).days)

for i in jiraDataDates:
    print(i)
print(len(jiraDataDates), len(jiraDataDates[0]))

# Sort deltas

delta.sort()
deltaP0.sort()
deltaP1.sort()

print("delta:")
for i in delta:
    print(i)

# convert delta (closed-open) to numpy format.

npdelta=np.asarray(delta)
npdeltaP0=np.asarray(deltaP0)
npdeltaP1=np.asarray(deltaP1)
print(npdelta.shape)
print(npdeltaP0.shape)
print(npdeltaP1.shape)

# Obsolete code. 
'''
#n, bins, patches = plt.hist(delta, num_bins, normed=1, facecolor='blue', alpha=0.5)

print("n, num_bins, patches: ", n, num_bins, patches)
hist, bin_edges = np.histogram(npdelta, density=True)
print("hist: ", hist)
print("hist.sum(): ", hist.sum())
#print(np.sum(hist * np.diff(bin_edges)))
'''

# Create bins for histogram

bins = [0, 7, 14,21, 28, 400] # your bins

data=npdelta
dataP0=npdeltaP0
dataP1=npdeltaP1

# Create histogram data. 

hist1, bin_edges = np.histogram(data, bins) # make the histogram
hist2, bin_edges = np.histogram(dataP0, bins) # make the histogram
hist3, bin_edges = np.histogram(dataP1, bins) # make the histogram

# Create plot with 3 subplots arranged horizontally, set total size of plot.

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot the histogram heights against integers on the x axis, specify fill and border colors and titles. 

ax1.bar(range(len(hist1)), hist1, width=0.8, color='#aaaaff', edgecolor='#0000ff') 
ax2.bar(range(len(hist2)), hist2, width=0.8, color='#aaffaa', edgecolor='#00ff00') 
ax3.bar(range(len(hist3)), hist3, width=0.8, color='#ffaaaa', edgecolor='#ff0000') 
ax1.set_title("All tickets: P0-Pn")
ax2.set_title("P0 tickets")
ax3.set_title("P1 tickets")

# Set axis labels for x and y axis.	

ax1.set(xlabel='Number of days to resolve', ylabel='Number of tickets')
ax2.set(xlabel='Number of days to resolve', ylabel='Number of tickets')
ax3.set(xlabel='Number of days to resolve', ylabel='Number of tickets')

# Set the ticks to the middle of the bars.

ax1.set_xticks([0.5+i for i,j in enumerate(hist1)])
ax2.set_xticks([0.5+i for i,j in enumerate(hist2)])
ax3.set_xticks([0.5+i for i,j in enumerate(hist3)])

# Set the xticklabels to a string that tells us what the bin edges were.

ax1.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist1)])
ax2.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist2)])
ax3.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist3)])

#	Make Y axis integer only.
yint = []

locs, labels = plt.yticks()
for each in locs:
    yint.append(int(each))
plt.yticks(yint)

plt.show()

# obsolete code
'''
#this was cultprit!!!
plt.hist(npdelta, bins=bins_list)
plt.xlabel('Number of days to resolve')
plt.ylabel('Number of tickets')
plt.title(r'Defect resolution data')
plt.xticks(bins_list, bins_list_x_axis_ticks)
'''
'''
plt.plot(bins, npdelta)
plt.show()
'''

'''
NP RANGE TOO UNFAMILIAR. Instead manipulate list and then convert to np before graphing
#npjiraData = np.array(jiraData[1:], dtype=np.float)
npjiraData = np.array(jiraData)

npjiraData_date_closed=npjiraData[:,8:9]
npjiraData_date_open=npjiraData[:,9:10]

print(npjiraData_date_closed)
print(type(npjiraData_date_closed))
print(npjiraData_date_closed.shape[0])

delta=[]

for i in range(1, npjiraData_date_closed.shape[0]):
    #npjiraData_date_closed[i][0]=npjiraData_date_closed[i][0].split(' ')[0]
    #npjiraData_date_open[i][0]=npjiraData_date_open[i][0].split(' ')[0]
    npjiraData_date_closed[i][0]=datetime.strptime(npjiraData_date_closed[i][0], '%m/%d/%Y %H:%M')
    npjiraData_date_open[i][0]=datetime.strptime(npjiraData_date_open[i][0], '%m/%d/%Y %H:%M')
    delta.append((npjiraData_date_closed[i][0]-npjiraData_date_open[i][0]).days)
print(npjiraData_date_closed)
print(npjiraData_date_open)
print(delta)
''
t = np.arange(0., 5., 0.2)

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

#plt.plot([1, 2, 3, 4], [10, 20, -4, 100], 'ro')
plt.ylabel('some numbers')
'''
