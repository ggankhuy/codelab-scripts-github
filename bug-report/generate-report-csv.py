import sys
import csv
import glob
import os 
import time

fileName=None

import numpy as np
from numpy import *

headers=None
debug=0
colIndices=None
validStats=["NEW","ACCEPTED","ASSIGNED"]

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

# 	Start of script execution entry. 
	
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

# 	Construct a column indices from headers. 
	
colIndices=setColumnIndices(headers)

if not colIndices:
	print("Error: colIndices failed to populate for ", fileName)
	quit(1)

print("colIndices: ", colIndices)
	
if debug:
	print("data dimension: ", data.shape)
	print("data type: ", type(data))
	printSingleBar()

# 	Extract priority column and count priorities and display them for 1. all tickets and 2. bugs only.

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

#	Look for unclassified tickets (not in host list)
	
print("Looking for unclassified hotlist tickets:")
print("Gathering tickets in hotlist directory...")
os.chdir(".\hotlist")

#	Scan files in hotlist directory. 

fileList=[]
for file in glob.glob("*"):
	fileList.append(file)
		
if debug:
	print(fileList)

#	Construct 2-d array of hotlist in dict format: 
#	{<string>: <list>}
#	{"<COL_NAME>": [<COL_DATA>]} 
#	Initialize all list empty at first. 

list2DHotList={}

for i in range(0, len(listColumns)):
	list2DHotList[listColumns[i]] = []

#	Iterate through all files in hotlist directory.	
	
for currFileName in fileList:

	with open(currFileName) as f1:
		reader = csv.reader(f1, delimiter=',')
		headers = next(reader)
		data1 = list(reader)
		data1=np.array(data1)
	
	#	For scanned file, construct col names list.
	
	colIndices=None
	colIndices=setColumnIndices(headers, [COL_NAME_CREATED_TIME])
		
	print("colIndices for ", currFileName, ": ", colIndices)

	if not colIndices:
		print("Error: colIndices failed to populate for ", currFileName)
		quit(1)

	#	Filter the assigned list.

	print(data1[:,colIndices["STATUS"]])
	print(data1)
	
	rowsToDel=[]
	
	for i in range(0, len(data1[:,colIndices["STATUS"]])):
		print(i, ":")
		
		if not data1[i, colIndices["STATUS"]] in validStats:
			print("WARNING: Removing the row with status: ", data1[i,:])
			rowsToDel.append(i)
	
	data1 = np.delete(data1, rowsToDel, 0)		
	
	print(data1[:,colIndices["STATUS"]])
	print(data1)
	
	printSingleBar()
	print(currFileName)
	print("Bugs in ", currFileName, ": ", len(data1[:, 0]))

	# 	Iterate through each column and append to hostList.
	
	for i in range(0, len(listColumns)):
		list2DHotList[listColumns[i]] += list(data1[:,colIndices[listColumns[i]]])
	
printSingleBar()

#	Construct mismatch list. The list contains any ticket that is not assigned to any of the hotlist.
#	For that loop will iterate through all tickets and then iterate through list2DHotList which contains
#	cumulative list of all tickets assigned to hotlist. 
#	Therefore: list2DMisMatchList = list2DAllTickets - list2DHotList.

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

#	Print the list of mismatched tickets.
				
print("Mismatch issue ID not assigned to hot list: ")

for i in range(0, len(list2DMisMatchList[COL_NAME_ISSUE_ID])):
	print(list2DMisMatchList[COL_NAME_ISSUE_ID][i], ", ", list2DMisMatchList[COL_NAME_STATUS][i], ", ", list2DMisMatchList[COL_NAME_CREATED_TIME][i], ", ", list2DMisMatchList[COL_NAME_MODIFIED_TIME][i], ", ", \
	list2DMisMatchList[COL_NAME_TITLE][i][0:50])


	
	
	
	
	
	
	
		
		

	
	

	

	


	
	

	