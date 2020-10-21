import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import time
DOUBLE_BAR="================================"
SINGLE_BAR="--------------------------------"
from datetime import datetime, timedelta
datetimeToday=datetime.today()

DEBUG=1
DEBUGL2=0
fileNameJira='INTERNAL 2020-10-10T01_44_07-0500.csv'
fileNameJira='INTERNAL 2020-09-07T14_15_52-0500.csv'
fileNamesBuganizer=['issuesn12.csv', 'issuesv10.csv']

# Open exported file: exported from Jira. Column 8-9 must contain closed and open dates, respectively otherwise
# Code will either break or need adjustment. 
# Column 13 must be a priority, otherwise code will either break or need adjustment. 
# Column 

IDX_COL_JIRA_CLOSED_DATE=8
IDX_COL_JIRA_OPENED_DATE=9
IDX_COL_JIRA_PRIORITY=8
IDX_COL_JIRA_SUMMARY=2
IDX_COL_JIRA_ISSUE_KEY=0

IDX_COL_BUG_STATUS=5
IDX_COL_BUG_ID=6

with open(fileNameJira, 'r') as f:
    jiraData = list(csv.reader(f, delimiter=','))
jiraDataDates=[]

# jiraDataP0/P1: Gather p0 and p1 designated tickets.
# jiraDataDates: Gather only column 8, 9 which contains closed and open dates only. 

#for i in range(0, len(jiraData)):
#   jiraDataDates.append(jiraData[i][IDX_COL_JIRA_CLOSED_DATE:IDX_COL_JIRA_OPENED_DATE+1])	
	
print("Total No. of tickets imported: ", len(jiraData))    

for i in range(0, len(jiraData)):
	print(jiraData[i][IDX_COL_JIRA_CLOSED_DATE])

#	Open buganizer issues. 

bugData=[]
for fileNameBuganizer in fileNamesBuganizer:
	with open(fileNameBuganizer, 'r') as f:
		bugData+=list(csv.reader(f, delimiter=','))

for i in bugData:
	print(i[IDX_COL_BUG_STATUS:IDX_COL_BUG_ID+1])

print("Total imported buganizer issues: ", len(bugData))

# Iterate through jira bug, get the title column and try extracting the gibraltar issue id. 

jiraDataIssueIdsBuganizer=[]
jiraDataIssueIds=[]
jiraListIllegaSummaryTitle=[]

for i in range(0, len(jiraData)):
	print(SINGLE_BAR)
	issueId=None
	if DEBUG:
		#time.sleep(1)
		print(i, "Summary: ", jiraData[i][IDX_COL_JIRA_SUMMARY])
	summary=jiraData[i][IDX_COL_JIRA_SUMMARY].split()
		
	for j in summary:
		if j.isdigit() and len(j)==9:
			if DEBUG:
				print("Found jira ID: ", str(j))
			issueId=j
			
			try:
				jiraDataIssueIdsBuganizer.append(int(j))
				jiraDataIssueIds.append(jiraData[i][IDX_COL_JIRA_ISSUE_KEY])
			except Exception as msg:
				print("Fatal error: ", j, " is determined to be valid 9-digit Buganizer ID but can not be converted to integer.")
				exit(1)
			print("Added ", jiraData[i][IDX_COL_JIRA_ISSUE_KEY])
			continue
			
	if not issueId:
		print("Unable to find from summary the buganizer ID for ", str(jiraData[i][IDX_COL_JIRA_ISSUE_KEY]))
		jiraDataIssueIdsBuganizer.append('')
		jiraDataIssueIds.append(jiraData[i][IDX_COL_JIRA_ISSUE_KEY])
		jiraListIllegaSummaryTitle.append(jiraData[i][IDX_COL_JIRA_ISSUE_KEY])

input("Press Enter to continue...") 
	
print(DOUBLE_BAR)
print("len of jiradata: ", len(jiraData))
print("Len of jiraDataIssueIdsBuganizer: ", len(jiraDataIssueIdsBuganizer))
print("len of jiraDataIssueIds:", len(jiraDataIssueIds))
print(DOUBLE_BAR)
input("Press Enter to continue...") 
	
print("Following jira summary has illegal title, unable to find buganizer ID-s:")		
for i in jiraListIllegaSummaryTitle:
	print(i)

print(DOUBLE_BAR)

for i in range(0,len(jiraDataIssueIdsBuganizer)):
	if jiraDataIssueIdsBuganizer[i]:
		for j in bugData:
			#print("Checking if ", jiraDataIssueIdsBuganizer[i], " matches ", j)
			if j[IDX_COL_BUG_ID].strip() == str(jiraDataIssueIdsBuganizer[i]):
				print("Found matching issue ID in buganizer: ", str(j[IDX_COL_BUG_ID]))
				continue
		if DEBUGL2:
			print("Unable to find matching issue ID in buganizer for: ", str(jiraDataIssueIdsBuganizer[i]))
	else:
		if DEBUGL2:
			print("Skipping ", jiraDataIssueIds[i], " as it is empty.")
		
	
'''	
    
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

# Create bins for histogram

bins = [0, 7, 14, 21, 28, 400] # your bins

#data=npdelta
#dataP0=npdeltaP0
#dataP1=npdeltaP1
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
'''