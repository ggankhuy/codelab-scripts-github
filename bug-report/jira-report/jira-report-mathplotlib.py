import numpy as np
import matplotlib.pyplot as plt
import csv
import re

from datetime import datetime, timedelta
DEBUG=1
DEBUGL2=0
IDX_COL_JIRA_ISSUE_KEY=0
IDX_COL_JIRA_SUMMARY=2
IDX_COL_JIRA_CLOSED_DATE=10
IDX_COL_JIRA_OPENED_DATE=11
IDX_COL_JIRA_REJECTED_DATE=12
IDX_COL_JIRA_PRIORITY=15
IDX_COL_JIRA_ASSESSED_DATE=16
IDX_COL_JIRA_ANALYZED_DATE=17

color=['#aaaaff','#aaffaa','#ffaaaa']
edgecolor=['#0000ff','#00ff00','#ff0000']
titles=["All tickets: P0-Pn","P0 tickets","P1 tickets"]

datetimeToday=datetime.today()
fileName='jira-gibraltar.csv'

def printShapeList(pList):
	tmpNp=np.array(pList)
	print("dim: ", np.shape(tmpNp))
	

# Open exported file: exported from Jira. Column 8-9 must contain closed and open dates, respectively otherwise
# Code will either break or need adjustment. 
# Column 13 must be a priority, otherwise code will either break or need adjustment. 

with open(fileName, 'r') as f:
    jiraData = list(csv.reader(f, delimiter=','))

IDX_LIST_JIRA_DATE_OPEN_TO_CLOSE=0
IDX_LIST_JIRA_DATE_OPEN_TO_ASSESS=1
IDX_LIST_JIRA_DATE_OPEN_TO_ANALYZE=2

IDX_LIST_JIRA_DATE_ITER=[IDX_COL_JIRA_REJECTED_DATE, IDX_COL_JIRA_ASSESSED_DATE, IDX_COL_JIRA_ANALYZED_DATE]
jiraDataDates=[]

jiraDataP0=[]
jiraDataP0Dates=[]
jiraDataP1=[]
jiraDataP1Dates=[]

# jiraDataP0/P1: Gather p0 and p1 designated tickets.
# jiraDataDates: Gather only column 8, 9 which contains closed and open dates only. 

tmpList=[]
tmpListP0=[]
tmpListP1=[]

for j in range(0, 3):
	for i in range(0, len(jiraData)):
		'''
# why?	if jiraData[i][IDX_COL_JIRA_CLOSED_DATE].strip() and jiraData[i][IDX_COL_JIRA_OPENED_DATE].strip():
		if jiraData[i][IDX_COL_JIRA_CLOSED_DATE].strip():
			tmpList.append( \
				jiraData[i][IDX_COL_JIRA_OPENED_DATE:IDX_COL_JIRA_OPENED_DATE+1] + \
				jiraData[i][IDX_COL_JIRA_CLOSED_DATE:IDX_COL_JIRA_CLOSED_DATE+1])
		if re.search("P1", jiraData[i][IDX_COL_JIRA_PRIORITY]):
			tmpListP0.append(jiraData[i])
		if re.search("P2", jiraData[i][IDX_COL_JIRA_PRIORITY]):
			tmpListP1.append(jiraData[i])
	jiraDataDates.append(tmpList)
	jiraDataP0.append(tmpListP0)
	jiraDataP1.append(tmpListP1)
print("Shape of jiraData/P0/1: ")	
printShapeList(jiraData)
printShapeList(jiraDataP0)
printShapeList(jiraDataP1)
input("...")
		'''
listName=["all", "p0", "p1"]
# jiraDataDatesP0/P1: Gather only column 8, 9 for P0/P1 which contains closed and open dates only. 

for j in range(0, 3):
	print("Processing ", listName[j])
	tmpList=[]
	for i in range(0, len(jiraData)):
	
		# if cell is not empty.
		
		if jiraData[i][IDX_LIST_JIRA_DATE_ITER[j]].strip():
			tmpList.append( \
				jiraData[i][IDX_COL_JIRA_OPENED_DATE:IDX_COL_JIRA_OPENED_DATE+1] + \
				jiraData[i][IDX_LIST_JIRA_DATE_ITER[j]:IDX_LIST_JIRA_DATE_ITER[j]+1])

	jiraDataDates.append(tmpList)

print("Total No. of tickets imported (all/p0/p1): ")    
for i in range(0, 3):
	printShapeList(jiraDataDates[i])
	
input("...")
    
if DEBUGL2:
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

if DEBUGL2:
	print("delta/all/P0/P1:")
	for k in [delta, deltaP0, deltaP1]:
		print("...")
		for i in k:
			print(i)
		input("...")	
		
# convert delta (closed-open) to numpy format.

npdelta=np.asarray(delta)
npdeltaP0=np.asarray(deltaP0)
npdeltaP1=np.asarray(deltaP1)
print(npdelta.shape)
print(npdeltaP0.shape)
print(npdeltaP1.shape)

# Create bins for histogram

bins = [0, 7, 14, 21, 28, 400] # your bins
data=[npdelta, npdeltaP0, npdeltaP1]

# Create histogram data. 

hist=[]
for i in data:
	hist.append(np.histogram(i, bins)[0])
	
# Create plot with 3 subplots arranged horizontally, set total size of plot.

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot the histogram heights against integers on the x axis, specify fill and border colors and titles. 

ax=[ax1, ax2, ax3]

for i in range(0, len(ax)):
	ax[i].bar(range(len(hist[i])), hist[i], width=0.8, color=color[i], edgecolor=edgecolor[i]) 
	ax[i].set_title(titles[i])
	ax[i].set(xlabel='Number of days to resolve', ylabel='Number of tickets')
	ax[i].set_xticks([0.5+i for i,j in enumerate(hist[i])])
	ax[i].set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist[i])])

#	Make Y axis integer only.
yint = []

locs, labels = plt.yticks()
for each in locs:
    yint.append(int(each))
plt.yticks(yint)
plt.show()