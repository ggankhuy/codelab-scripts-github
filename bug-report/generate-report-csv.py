import sys
fileName=None

import numpy as np
from numpy import *
import csv
import glob, os

headers=None
debug=0
colIndices=None

COL_NAME_PRIORITY="PRIORITY"
COL_NAME_TYPE="TYPE"
COL_NAME_ISSUE_ID="ISSUE_ID"
COL_NAME_STATUS="STATUS"
COL_NAME_TITLE="TITLE"
COL_NAME_CREATED_TIME="CREATED_TIME (UTC)"
COL_NAME_MODIFIED_TIME="MODIFIED_TIME (UTC)"

listColumns=[\
	COL_NAME_PRIORITY, COL_NAME_TYPE, COL_NAME_ISSUE_ID, \
	COL_NAME_STATUS, COL_NAME_TITLE, COL_NAME_CREATED_TIME, COL_NAME_MODIFIED_TIME]

def printSingleBar():
	print("-----------------------------------------------")

#	Given headers, populate the dictionary type column based on header read from csv file.
#	Input:
#		pHeaders: header read from csv file (1st row)
#		pListExclude <list>: name of the column(s) to exclude from header.
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
	COL_NAME_ISSUE_ID: COL_ISSUE_ID, \
	COL_NAME_STATUS: COL_STATUS, \
	COL_NAME_CREATED_TIME: COL_CREATE_DATETIME, \
	COL_NAME_MODIFIED_TIME: COL_MODIFY_DATETIME, \
	COL_NAME_TITLE: COL_TITLE \
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

if not colIndices:
	print("Error: colIndices failed to populate for ", fileName)
	quit(1)

print("colIndices: ", colIndices)

if not colIndices:
	print("Error: setting column indices...")
	quit(1)
	
if debug:
	print("data dimension: ", data.shape)
	print("data type: ", type(data))
	printSingleBar()

# 	Extract priority column and count priorities and display.

list2DAllTickets={}

for i in range(0, len(listColumns)):
	list2DAllTickets[listColumns[i]] = list(data[:,colIndices[listColumns[i]]])
	printSingleBar()
	print(listColumns[i], list2DAllTickets[listColumns[i]])

if debug:
	print(type(list2DAllTickets["PRIORITY"]), \
	list2DAllTickets["PRIORITY"])

print(list2DAllTickets["PRIORITY"])
print("Total tickets: ", len(list2DAllTickets["PRIORITY"]), \
	list2DAllTickets["PRIORITY"] )
for i in range(0, 7):
	priority_index='P' + str(i)
	print(priority_index, ": ", list2DAllTickets["PRIORITY"].count(priority_index))
	
priority_bugs=[]
for i in range(0, len(list2DAllTickets["PRIORITY"])):
	if list2DAllTickets["PRIORITY"][i]=='BUG':
		priority_bugs.append(list2DAllTickets["PRIORITY"][i])

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

list2DHotList={}

for i in range(0, len(listColumns)):
	list2DHotList[listColumns[i]] = []

for currFileName in fileList:

	with open(currFileName) as f1:
		reader = csv.reader(f1, delimiter=',')
		headers = next(reader)
		data1 = list(reader)
		data1=np.array(data1)
	
	colIndices=None
	colIndices=setColumnIndices(headers, [COL_NAME_CREATED_TIME])
		
	print("colIndices for ", currFileName, ": ", colIndices)

	if not colIndices:
		print("Error: colIndices failed to populate for ", currFileName)
		quit(1)
		
	printSingleBar()
	print(currFileName)
	print("Bugs in ", currFileName, ": ", len(data1[:, 0]))

	# 	Extract priorit column and count priorities and display.
	
	for i in range(0, len(listColumns)):
		list2DHotList[listColumns[i]] += list(data1[:,colIndices[listColumns[i]]])
	
printSingleBar()

list2DMisMatchList={}

for i in range(0, len(listColumns)):
	list2DMisMatchList[listColumns[i]] = []


for i in range(0, len(list2DAllTickets["PRIORITY"])):
	if not list2DAllTickets[COL_NAME_ISSUE_ID][i] in list2DHotList[COL_NAME_ISSUE_ID]:
		for j in range(0, len(listColumns)):
			try:
				if listColumns[j] in list2DAllTickets.keys():
					list2DMisMatchList[listColumns[j]].\
						append(list2DAllTickets[listColumns[j]][i])	
				else:
					print("Skipping to append: ", listColumns[j])
			except Exception as msg:
				print("Error: Can not append: ", list2DAllTickets[j][i])
				continue
		
		'''
		mismatchIssueIds.append(list2DAllTickets[COL_NAME_ISSUE_ID][i])
		mismatchStatuses.append(list2DAllTickets[COL_NAME_STATUS][i])
		mismatchTitles.append(list2DAllTickets[COL_NAME_TITLE][i])
		mismatchCreate.append(list2DAllTickets[COL_NAME_CREATED_TIME][i])
		mismatchModified.append(list2DAllTickets[COL_NAME_MODIFIED_TIME][i])
		'''

print("Mismatch issue ID not assigned to hot list: ")

for i in range(0, len(list2DMisMatchList[COL_NAME_ISSUE_ID])):
	print(list2DMisMatchList[COL_NAME_ISSUE_ID][i], ", ", list2DMisMatchList[COL_NAME_STATUS][i], ", ", list2DMisMatchList[COL_NAME_CREATED_TIME][i], ", ", list2DMisMatchList[COL_NAME_MODIFIED_TIME][i], ", ", \
	list2DMisMatchList[COL_NAME_TITLE][i][0:50])
	#print(mismatchIssueIds[i], ", ", mismatchStatuses[i], ", ",mismatchModified[i], ", ", \
	#mismatchTitles[i][0:10])


	
	
	
	
	
	
	
		
		

	
	

	

	


	
	

	