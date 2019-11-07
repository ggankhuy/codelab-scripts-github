import sys
fileName=None

import numpy as np
from numpy import *
import csv
import glob, os

headers=None
debug=0
colIndices=None

def printSingleBar():
	print("-----------------------------------------------")

#	Given headers, populate the dictionary type column based on header read from csv file.
#	Input:
#		pHeaders: header read from csv file (1st row)
#		pListExclude: name of the column to exclude from header.
#	Output:
#		<dict> - dictionary object with keys column name and values column indexes in headers.
#		None - for any errors.
def setColumnIndices(pHeaders, pListExclude=[]):
	debug=0
	COL_PRIORITY=1
	COL_TYPE=2
	COL_ISSUE_ID=6
	COL_STATUS=5
	COL_CREATE_DATETIME=6
	COL_MODIFY_DATETIME=7
	COL_TITLE=3

	COL_INDICES={\
	"PRIORITY": COL_PRIORITY, \
	"TYPE": COL_TYPE, \
	"ISSUE_ID": COL_ISSUE_ID, \
	"STATUS": COL_STATUS, \
	"CREATED_TIME (UTC)": COL_CREATE_DATETIME, \
	"MODIFIED_TIME (UTC)": COL_MODIFY_DATETIME, \
	"TITLE": COL_TITLE \
	}

	
	for i in range(0, len(COL_INDICES)):
		keys=list(COL_INDICES.keys())
		values=list(COL_INDICES.values())
		
		if debug:
			print(keys)
			print(values)

		if keys[i] in pListExclude:
			print("(setColumnIndices) set to exclude: ", keys[i])
			continue
		
		if not keys[i] in pHeaders:
			print("(setColumnIndices) Error: ", keys[i], " is not in the header")
			print("(setColumnIndices)headers: ", pHeaders)
			return None
		else:
			try:
				values[i] = pHeaders.index(keys[i])
				COL_INDICES[keys[i]] = values[i]
			except Exception as msg:
				print("(setColumnIndices)Fatal error: Can not find the index of ", keys[i], " in headers. ")
				print("(setColumnIndices)headers: ", pHeaders)
				return None
			
			if debug:
				print("(setColumnIndices)Column index of ", keys[i], " is set to ", values[i])

	return COL_INDICES
	
try:
	fileName=sys.argv[1]
except Exception as msg:
	print("Failed to find p1.")
	print(msg)
	quit(1)
	
with open(fileName) as f:
	reader = csv.reader(f, delimiter=',')
	headers = list(next(reader))
	data = list(reader)
	data=np.array(data)

	f.close()

colIndices=setColumnIndices(headers)
print("Error: colIndices failed to populate for ", fileName)

print("colIndices: ", colIndices)

if not colIndices:
	print("Error: setting column indices...")
	quit(1)
	
if debug:
	print("data dimension: ", data.shape)
	print("data type: ", type(data))
	printSingleBar()

# 	Extract priority column and count priorities and display.

priority=list(data[:,colIndices["PRIORITY"]])
type=list(data[:,colIndices["TYPE"]])
issueId=list(data[:, colIndices["ISSUE_ID"]])
statuses=list(data[:, colIndices["STATUS"]])
titles=list(data[:, colIndices["TITLE"]])
createDate=list(data[:, colIndices["CREATED_TIME (UTC)"]])
modifyDate=list(data[:, colIndices["MODIFIED_TIME (UTC)"]])

if debug:
	print(type(priority), priority)

print("Total tickets: ", len(priority), priority )
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
		
if debug:
	print(fileList)

hotListPriority=[]
hotListType=[]
hotListIssueId=[]
hotListStatuses=[]
# hotListCreateDate=[]
hotListModifyDate=[]
hotListTitles=[]


for currFileName in fileList:

	with open(currFileName) as f1:
		reader = csv.reader(f1, delimiter=',')
		headers = next(reader)
		data1 = list(reader)
		data1=np.array(data1)
	
	colIndices=None
	colIndices=setColumnIndices(headers, ["CREATED_TIME (UTC)"])
		
	print("colIndices for ", currFileName, ": ", colIndices)

	if not colIndices:
		print("Error: colIndices failed to populate for ", currFileName)
		quit(1)
		
	printSingleBar()
	print(currFileName)
	print("Bugs in ", currFileName, ": ", len(data1[:, 0]))

	# 	Extract priorit column and count priorities and display.
		
	currPriority=list(data1[:,colIndices["PRIORITY"]])
	currType=list(data1[:,colIndices["TYPE"]])
	currIssueId=list(data1[:, colIndices["ISSUE_ID"]])
	currStatuses=list(data1[:, colIndices["STATUS"]])
	currTitles=list(data1[:, colIndices["TITLE"]])
	#currCreateDate=list(data1[:, colIndices["CREATED_TIME (UTC)"]])
	currModifiedDate=list(data1[:, colIndices["MODIFIED_TIME (UTC)"]])

	hotListPriority+=currPriority
	hotListType+=currType
	hotListIssueId+=currIssueId
	hotListStatuses+=currStatuses
	hotListTitles+=currTitles
	#hotListCreateDate+=currCreateDate
	hotListModifyDate+=currModifiedDate


printSingleBar()

if debug:
	print(hotListPriority)
	print(hotListType)
	print(hotListIssueId)

if debug:
	print(priority)
	print(type)
	print(issueId)
	print(statuses)

printSingleBar()

mismatchIssueIds=[]
mismatchStatuses=[]
mismatchCreate=[]
mismatchModified=[]
mismatchTitles=[]


for i in range(0, len(priority)):
	if not issueId[i] in hotListIssueId:

		mismatchIssueIds.append(issueId[i])
		mismatchStatuses.append(statuses[i])
		mismatchTitles.append(titles[i])
		mismatchCreate.append(titles[i])
		mismatchModified.append(titles[i])

print("Mismatch issue ID not assigned to hot list: ")

for i in range(0, len(mismatchIssueIds)):
	print(mismatchIssueIds[i], ", ", mismatchStatuses[i], ", ", mismatchCreate[i], ", ", mismatchModified[i], ", ", \
	mismatchTitles[i][0:10])
	#print(mismatchIssueIds[i], ", ", mismatchStatuses[i], ", ",mismatchModified[i], ", ", \
	#mismatchTitles[i][0:10])


	
	
	
	
	
	
	
		
		

	
	

	

	


	
	

	