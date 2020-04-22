import sys
import csv
import glob
import os 
import time

import numpy as np
from common import *
from numpy import *
from datetime import datetime, timedelta

debug = 1
debugL2 = 0

colIndices=None
validStats=["NEW","ACCEPTED","ASSIGNED"]
PRIORITY_LOWEST = 7

# Open internal jira
# Open issues.csv (buganizer)

fileNameJiraInternal="internal_jira.csv"
fileNameBuganizer="issues.csv"

# 	Open internal jira export csv and external jira (buganizer) issues.

with open(fileNameJiraInternal) as f:
	reader = csv.reader(f, delimiter=',')
	headersJiras = list(next(reader))
	jiras = list(reader)
	np_jiras=np.array(jiras)
		
with open(fileNameBuganizer) as f:
	reader = csv.reader(f, delimiter=',')
	headersBuganizerIssues = list(next(reader))
	buganizerIssues = list(reader)
	np_buganizerIssues=np.array(buganizerIssues)
		
if debug:
	print(type(jiras))
	print(type(buganizerIssues))
	print(type(np_jiras))
	print(type(np_buganizerIssues))

# 	Construct a column indices from headers. 
	
colIndicesMain=setColumnIndices(headersBuganizerIssues)

if not colIndicesMain:
	print("Error: colIndicesMain failed to populate for ", fileName)
	quit(1)

print("colIndicesMain: ", colIndicesMain)

#	Filter out tickets with invalid status

rowsToDel=[]

for i in range(0, len(np_buganizerIssues[:,colIndicesMain[COL_NAME_STATUS]])):
	if debugL2:
		print(i, ":")
		
	if not np_buganizerIssues[i, colIndicesMain[COL_NAME_STATUS]] in validStats:
		if debugL2:
			print("--- INFO: Removing the row with status: ", ", ID: ", np_buganizerIssues[i,colIndicesMain[COL_NAME_ISSUE_ID]], ", STATUS: ", np_buganizerIssues[i,colIndicesMain[COL_NAME_STATUS]])
		rowsToDel.append(i)
	else:
		if debugL2:
			print("--- INFO: Keeping the row with status: ", ", ID: ", np_buganizerIssues[i,colIndicesMain[COL_NAME_ISSUE_ID]], ", STATUS: ", np_buganizerIssues[i,colIndicesMain[COL_NAME_STATUS]])

np_buganizerIssues = np.delete(np_buganizerIssues, rowsToDel, 0)		

	
if debug:
	print("np_buganizerIssues dimension: ", np_buganizerIssues.shape)
	print("np_buganizerIssues type: ", type(np_buganizerIssues))
	printBarSingle()	
	
for i in np_buganizerIssues:
	print(i[COL_ISSUE_ID])
	
# 	Iterate through all bus buganizer issues and search corresponding internal jira.
#		Check to see internal jira opened
#		1. Search ID in title, if found OK.
#		2. Search title string, if found OK.
#		3. If 1. and 2. fails output error.
































