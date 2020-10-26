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

CONFIG_SIZE_WORKFLOWS=3
CONFIG_SIZE_PRIORITIES=3

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

# jiraDataP0/P1: Gather p0 and p1 designated tickets.
# jiraDataDates: Gather only column 8, 9 which contains closed and open dates only. 

tmpList=[]
listNameK=["ALL", "P0", "P1"]
listNameJ=["CLOSED", "ASSESSED", "ANALYZED"]
searchPattern=[".", "P1", "P2"]

# 	Outer loop is for closed/assessed/analyzed loop.

for j in range(0, CONFIG_SIZE_WORKFLOWS):
	print("Processing ", listNameJ[j])

	tmpListJ=[]

	# inner loop for ALL/P0/P1.

	for k in range(0, CONFIG_SIZE_PRIORITIES):
		print("Processing ", listNameK[k])
		tmpListK=[]
		
		for i in range(0, len(jiraData)):
		
			# if cell is not empty and matches the priority.
			
			if jiraData[i][IDX_LIST_JIRA_DATE_ITER[j]].strip() and re.search(searchPattern[k], jiraData[i][IDX_COL_JIRA_PRIORITY].strip()):
				tmpListK.append( \
					jiraData[i][IDX_COL_JIRA_OPENED_DATE:IDX_COL_JIRA_OPENED_DATE+1] + \
					jiraData[i][IDX_LIST_JIRA_DATE_ITER[j]:IDX_LIST_JIRA_DATE_ITER[j]+1])
					
		print("  tmpListK size:", len(tmpListK))
		tmpListJ.append(tmpListK)
		
	print(" tmpListJ size:", len(tmpListJ))
	jiraDataDates.append(tmpListJ)

print("Total No. of tickets imported (all/p0/p1): ")  

for j in range(0, CONFIG_SIZE_WORKFLOWS):
	for k in range(0, CONFIG_SIZE_PRIORITIES):
		printShapeList(jiraDataDates[j][k])    

# holds delta (closed - open) in number of days	

deltas=[]

print("converting to datetime format...")

for j in range(0, CONFIG_SIZE_WORKFLOWS):
	tmpListJ=[]
	
	for k in range(0, CONFIG_SIZE_PRIORITIES):
		tmpListK=[]
		for i in range(1, len(jiraDataDates[j])):
			closedDate=datetime.strptime(jiraDataDates[j][i][0], '%m/%d/%Y %H:%M')
			openDate=datetime.strptime(jiraDataDates[j][i][1], '%m/%d/%Y %H:%M')
			tmpListK.append((closedDate-openDate).days)
		tmpListJ.append(tmpListK)
	deltas.append(tmpListJ)

print("Date deltas constructed (ALL/P0/P1): ")    

for j in range(0, 3):
	for k in range(0, 3):
		printShapeList(deltas[j][k])
	
input("...")

if DEBUGL2:
	for j in range(0, 3):
		for i in jiraDataDates[j]:
			print(i)
		print(len(jiraDataDates[j]), len(jiraDataDates[j][0]))

# Sort deltas
# convert delta (closed-open) to numpy format.

npdelta=[]
for j in range(0, 3):
	deltas[j].sort()
	npdelta.append(np.asarray(deltas[j]))
	
for j in range(0, 3):
	print(npdelta[j].shape)
	
# Create bins for histogram

bins = [0, 7, 14, 21, 28, 400] # your bins
data=npdelta

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