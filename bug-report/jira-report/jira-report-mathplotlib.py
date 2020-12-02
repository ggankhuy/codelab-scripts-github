import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import time

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
titlesI=["All tickets: P0-Pn","P0 tickets","P1 tickets"]
titlesJ=["Resolved time","Assessed time","Analyzed time"]

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

IDX_LIST_JIRA_DATE_ITER=[IDX_COL_JIRA_CLOSED_DATE, IDX_COL_JIRA_ASSESSED_DATE, IDX_COL_JIRA_ANALYZED_DATE]
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

if DEBUGL2:
	for i in jiraDataDates:
		for j in i:
			for k in j:
				print(k)
			
if DEBUG:
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
		for i in range(1, len(jiraDataDates[j][k])):
			print(jiraDataDates[j][k][i])
			closedDate=datetime.strptime(jiraDataDates[j][k][i][1], '%m/%d/%Y %H:%M')
			openDate=datetime.strptime(jiraDataDates[j][k][i][0], '%m/%d/%Y %H:%M')
			tmpListK.append((closedDate-openDate).days)
		tmpListJ.append(tmpListK)
	deltas.append(tmpListJ)

if DEBUG:
	print("Date deltas constructed (ALL/P0/P1): ")    

for j in range(0, CONFIG_SIZE_WORKFLOWS):
	for k in range(0, CONFIG_SIZE_PRIORITIES):
		printShapeList(deltas[j][k])

# Sort deltas
# convert delta (closed-open) to numpy format.

npdelta=[]
for j in range(0, CONFIG_SIZE_WORKFLOWS):
	tmpList=[]
	for k in range(0, CONFIG_SIZE_PRIORITIES):
		deltas[j][k].sort()		
		tmpList.append(np.asarray(deltas[j][k]))
	npdelta.append(tmpList)

for j in range(0, CONFIG_SIZE_WORKFLOWS):
	for k in range(0, CONFIG_SIZE_PRIORITIES):
		print("Shape of np delta: ", np.shape(npdelta[j][k]))

# Create bins for histogram

# bins:
# outer: resolve, assessed, analyzed
# inner: P0-pn], [P1], [P2]]

bins = [\
	[[0, 18, 100, 200], [0, 3, 400], [0, 3, 400]], \
	[[0, 3, 200], [0, 3, 400], [0, 3, 400]], \
	[[0, 3, 200], [0, 3, 400], [0, 3, 400]]\
	] # your bins

enable_flags=[\
	[[1], [0], [0]], \
	[[1], [0], [0]], \
	[[1], [0], [0]]\
	]
data=npdelta

# Create histogram data. 

hist=[]

counterj=0
for j in data:
	
	tmpList=[]
	counterk=0
	for k in j:
		print("bins[counterk]: ", bins[counterj][counterk])
		#time.sleep(1)
		tmpList.append(np.histogram(k, bins[counterj][counterk])[0])
		print("curr hist: ", tmpList)
		counterk+=1
	hist.append(tmpList)
	counterj+=1
	
print("hist: ")
for i in hist:
	print("...")
	for j in i:
		print(j)
	
# Create plot with 3 subplots arranged horizontally, set total size of plot.

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9))  = plt.subplots(3, 3, figsize=(15, 15))
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# Plot the histogram heights against integers on the x axis, specify fill and border colors and titles. 

ax=[[ax1, ax2, ax3],[ax4, ax5, ax6],[ax7, ax8, ax9]]
print(ax)

for j in range(0, len(ax)):
	print("len(ax):", len(ax))
	for i in range(0, len(ax[j])):
		print("len(ax[j]):", len(ax[j]))
		print("ax[j][j]: ", ax[j][i])
		print("len(hist[j][i])/hist[j][i]: ", len(hist[j][i]), ", ", hist[j][i])
		ax[j][i].bar(\
			range(len(hist[j][i])), \
			hist[j][i], width=0.8, \
			color=color[i], edgecolor=edgecolor[i]) 
		ax[j][i].set_title(titlesJ[j] + ", " + titlesI[i])
		ax[j][i].set(xlabel='Number of days', ylabel='Number of tickets')
		ax[j][i].set_xticks([0.5+i for i,k in enumerate(hist[j][i])])
		
		ax[j][i].set_xticklabels(\
			['{} - {}'.format(bins[j][i][m],bins[j][i][m+1]) \
			for m,n in enumerate(hist[j][i])])
		ax[j][i].legend()
		

#	Make Y axis integer only.
yint = []


locs, labels = plt.yticks()
for each in locs:
    yint.append(int(each))
plt.yticks(yint)
#plt.show()

# start second plot containing weekly incoming and weekly fixed rate.

tickets_2d=[[], [], []]
TICKETS_2D_IDX_TICKETS_OPENED=0
TICKETS_2D_IDX_TICKETS_CLOSED=1
TICKETS_2D_IDX_TICKETS_REJECTED=2
TICKETS_2D_LABELS=["opened", "closed", "rejected"]
TICKETS_2D_START_DATE=datetime.strptime("01/01/2019 12:00", '%m/%d/%Y %H:%M')

for i in range(1, len(jiraData)):
	print("converting to date format: ", jiraData[i][IDX_COL_JIRA_OPENED_DATE])
	print("converting to date format: ", jiraData[i][IDX_COL_JIRA_CLOSED_DATE])
	print("converting to date format: ", jiraData[i][IDX_COL_JIRA_REJECTED_DATE])
	
	if (jiraData[i][IDX_COL_JIRA_OPENED_DATE].strip()):
		tickets_2d[TICKETS_2D_IDX_TICKETS_OPENED].append((datetime.strptime(jiraData[i][IDX_COL_JIRA_OPENED_DATE], '%m/%d/%Y %H:%M')-TICKETS_2D_START_DATE).days)
	if (jiraData[i][IDX_COL_JIRA_CLOSED_DATE].strip()):
		tickets_2d[TICKETS_2D_IDX_TICKETS_CLOSED].append((datetime.strptime(jiraData[i][IDX_COL_JIRA_CLOSED_DATE], '%m/%d/%Y %H:%M')-TICKETS_2D_START_DATE).days)
	if (jiraData[i][IDX_COL_JIRA_REJECTED_DATE].strip()):
		tickets_2d[TICKETS_2D_IDX_TICKETS_REJECTED].append((datetime.strptime(jiraData[i][IDX_COL_JIRA_REJECTED_DATE], '%m/%d/%Y %H:%M')-TICKETS_2D_START_DATE).days)	
		
print(tickets_2d)
	
