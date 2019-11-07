import sys
fileName=None
COL_PRIORITY=1
COL_TYPE=2
COL_ISSUE_ID=6
COL_STATUS=5

import numpy as np
from numpy import *
import csv
import glob, os

headers=None
debug=0

try:
	fileName=sys.argv[1]
except Exception as msg:
	print("Failed to find p1.")
	print(msg)
	quit(1)
	
with open(fileName) as f:
	reader = csv.reader(f, delimiter=',')
	headers = next(reader)
	data = list(reader)
	data=np.array(data)

	f.close()
	
if debug:
	print("data dimension: ", data.shape)
	print("data type: ", type(data))
	print("------------------")

# 	Extract priorit column and count priorities and display.


priority=list(data[:,COL_PRIORITY])
type=list(data[:,COL_TYPE])
issueId=list(data[:, COL_ISSUE_ID])
statuses=list(data[:, COL_STATUS])

if debug:
	print(type(priority), priority)

print("Total tickets: ", len(priority) )
for i in range(0, 7):
	priority_index='P' + str(i)
	print(priority_index, ": ", priority.count(priority_index))
	
priority_bugs=[]
for i in range(0, len(priority)):
	if type[i]=='BUG':
		priority_bugs.append(priority[i])

print("Total bugs: ", len(priority_bugs))
	
for i in range(0, 7):
	priority_index='P' + str(i)
	print(priority_index, ": ", priority_bugs.count(priority_index))
	
print("Looking for unclassified hotlist tickets:")
print("Gathering tickets in hotlist directory...")
os.chdir(".\hotlist")

fileList=[]
for file in glob.glob("*"):
	fileList.append(file)
		
print(fileList)

hotListPriority=[]
hotListType=[]
hotListIssueId=[]
hotListStatuses=[]

for currFileName in fileList:

	with open(currFileName) as f1:
		reader = csv.reader(f1, delimiter=',')
		headers = next(reader)
		data1 = list(reader)
		data1=np.array(data1)
	
	if debug:
		print("------------------")
		print(currFileName)
		print("data dimension: ", data1.shape)
		#print("data type: ", type(data1))
		print("------------------")

	# 	Extract priorit column and count priorities and display.
		
	currPriority=list(data1[:,COL_PRIORITY])
	currType=list(data1[:,COL_TYPE])
	currIssueId=list(data1[:, COL_ISSUE_ID])
	currStatuses=list(data1[:, COL_STATUS])

	hotListPriority+=currPriority
	hotListType+=currType
	hotListIssueId+=currIssueId
	hotListStatuses+=currStatuses

print("------------------")

print(hotListPriority)
print(hotListType)
print(hotListIssueId)

if debug:
	print(priority)
	print(type)
	print(issueId)
	print(statuses)

print("------------------")

mismatchIssueIds=[]
mismatchStatuses=[]


for i in range(0, len(priority)):
	if not issueId[i] in hotListIssueId:
		mismatchIssueIds.append(issueId[i])
		mismatchStatuses.append(statuses[i])

print("Mismatch issue ID not assigned to hot list: ")

for i in range(0, len(mismatchIssueIds)):
	print(mismatchIssueIds[i], ", ", mismatchStatuses[i])


	
	
	
	
	
	
	
		
		

	
	

	

	


	
	

	