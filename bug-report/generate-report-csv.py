'''
DESCRIPTION:
generate-report-csv.py uses the google tracker exported csv  files to assist in generating the bug report. 

PREREQUISITES:
The script is tested on following system:
Windows 10
python 3.8.0
numpy 1.17.3
pip 19.2.3

PREPARATION of CSV files:
The scripts operates on several specific csv files generated by the export from google tracker hotlists:
<root> directory refers to cwd current directory in which the script is running.
Arrange the csv files in following manner:
<root>\<csv exported from hotList: CST_Triage_N12> - Navi 12 ticket list exported as csv.
<root>\<csv exported from hotList: CST_Triage_VG10> - Vega 10 ticket list exported as csv.
<root>\hotlist\<csv exported from hotList: CST_Triage_Gibraltar> - Gibraltor hotlist ticket list exported as csv.
<root>\hotlist\<csv exported from hotList: CST_Triage_GuestDriver> - Guest driver hotlist ticket list exported as csv.
<root>\hotlist\<csv exported from hotList: CST_Triage_HostDriver> - Host driver hotlist ticket list exported as csv.
<root>\hotlist\<csv exported from hotList: CST_Triage_HW> - HW hotlist ticket list exported as csv.
<root>\hotlist\<csv exported from hotList: CST_Triage_Kernel_ETC> - Kernel ETC hotlist ticket list exported as csv.
<root>\hotlist\<csv exported from hotList: CST_Triage_Tool> - Tools hotlist ticket list exported as csv.
<root>\hotlist\<csv exported from hotList: CST_Triage_Vulkan> - Vulkan hotlist ticket list exported as csv.

OPERATIONAL ASPECT:
Once the hotlist csv files are arranged in an aforementioned manner, you can launch the script with cmd line argument 
of either NAVI12 OR V10 hotlist csv located in <root> directory.

python3 generate-report-csv <CST_Triage_N12 hotlist or CST_Triage_V10 hotlist>

1. The script will import all tickets from N12 or V10 csv file SPECIFIED into numpy format 
	- Displays all P0/P1 ... P6 tickets as well as all tickets for either VG10 or N12 specified in command line.
	- Displays all P0/P1 ... P6 bugs as well as all bugs for either VG10 or N12 specified in command line.
	Note that: bug is one of the ticket type therefore a subset of tickets.
	
2.  The script will scan through hotlist directory and gather all tickets from all files there.
	- Filters out all tickets that are not in ASSIGNED, NEW, ACCEPTED status. That includes all FIXED status including other status.
	- Compares all tickets from VG10/N12 to the ones gathered from hotlist and founds the difference. The differences are the tickets  
	that are not assigned to one of the hotlists:
	CST_Triage_Gibraltar> 
	CST_Triage_GuestDriver
	CST_Triage_HostDriver>
	CST_Triage_HW>
	CST_Triage_Kernel_ETC>
    CST_Triage_Tool>
	CST_Triage_Vulkan>
	This way, it identifies any ticket that are not assigned to one of the above hotlists. 
'''

import sys
import csv
import glob
import os 
import time
from common import *
fileName=None

import numpy as np
from numpy import *
from datetime import datetime, timedelta

headers=None
debug=0
debug_info=0
colIndices=None
validStats=["NEW","ACCEPTED","ASSIGNED"]
PRIORITY_LOWEST = 7


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
	
colIndicesMain=setColumnIndices(headers)

if not colIndicesMain:
	print("Error: colIndicesMain failed to populate for ", fileName)
	quit(1)

print("colIndicesMain: ", colIndicesMain)
	
if debug:
	print("data dimension: ", data.shape)
	print("data type: ", type(data))
	printBarSingle()

#	Filter out tickets with invalid status

rowsToDel=[]

for i in range(0, len(data[:,colIndicesMain[COL_NAME_STATUS]])):
	if debug:
		print(i, ":")
		
	if not data[i, colIndicesMain[COL_NAME_STATUS]] in validStats:
		print("--- INFO: Removing the row with status: ", ", ID: ", data[i,colIndicesMain[COL_NAME_ISSUE_ID]], ", STATUS: ", data[i,colIndicesMain[COL_NAME_STATUS]])
		rowsToDel.append(i)
	else:
		print("--- INFO: Keeping the row with status: ", ", ID: ", data[i,colIndicesMain[COL_NAME_ISSUE_ID]], ", STATUS: ", data[i,colIndicesMain[COL_NAME_STATUS]])

data = np.delete(data, rowsToDel, 0)		

# 	Extract priority column and count priorities and display them for 1. all tickets and 2. bugs only.

list2DAllTickets={}

for i in range(0, len(listColumns)):
	list2DAllTickets[listColumns[i]] = list(data[:,colIndicesMain[listColumns[i]]])

	if debug:
		printBarSingle()
		print(listColumns[i], list2DAllTickets[listColumns[i]])

if debug:
	print(type(list2DAllTickets[COL_NAME_PRIORITY]), \
	list2DAllTickets[COL_NAME_PRIORITY])

if debug:
	print(list2DAllTickets[COL_NAME_PRIORITY])

print("Total tickets: ", len(list2DAllTickets[COL_NAME_PRIORITY]))

for i in range(0, PRIORITY_LOWEST):
	priority_index='P' + str(i)
	print(priority_index, ": ", list2DAllTickets[COL_NAME_PRIORITY].count(priority_index))
	
priority_bugs=[]
for i in range(0, len(list2DAllTickets[COL_NAME_PRIORITY])):
	if list2DAllTickets[COL_NAME_TYPE][i]=='BUG':
		priority_bugs.append(list2DAllTickets[COL_NAME_PRIORITY][i])

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

#	list2DHostList content will be:
#	"<columnName": [columnValues]

list2DHotList={} 

#	list2DHostListIssuesIds content will be:
#	"fileName": [issueIds]
dict2DHotListIssueIds={}
list2DHotListDups=[]
list2DHotListAll=[]
for i in range(0, len(listColumns)):
	list2DHotList[listColumns[i]] = []

#	Iterate through all files in hotlist directory.	
	
priority_bugs_from_hotlist=0
for currFileName in fileList:
	dict2DHotListIssueIds[currFileName] = []
	printBarSingle()
	print(currFileName)

	#	Read the content into np array.
	
	with open(currFileName) as f1:
		reader = csv.reader(f1, delimiter=',')
		headers = next(reader)
		data1 = list(reader)
		data1=np.array(data1)
	
	#	For scanned file, construct col names list.
	
	colIndices=None
	colIndices=setColumnIndices(headers, [COL_NAME_CREATED_TIME])
	
	if debug:
		print("colIndices for ", currFileName, ": ", colIndices)

	if not colIndices:
		print("Error: colIndices failed to populate for ", currFileName)
		quit(1)

	#	Filter the assigned list.

	if debug:
		print(data1[:,colIndices[COL_NAME_STATUS]])
		print(data1)
	
	rowsToDel=[]
	rowsToDelIssueId=[]
	
	#	Iterate over status column. Filter out rows with 1. invalud status 2. issues that is not in all tickets file 3. not a bug.
	#	Those filtered out issues row index are accumulated to rowsToDel list.
		
	for i in range(0, len(data1[:,colIndices[COL_NAME_STATUS]])):
		if debug:
			print(i, ":")
			
		if not data1[i, colIndices[COL_NAME_STATUS]] in validStats:
			if debug_info:
				print("--- INFO: Removing the row with status: ", ", ID: ", data1[i,colIndices[COL_NAME_ISSUE_ID]], ", STATUS: ", data1[i,colIndices[COL_NAME_STATUS]])
			rowsToDel.append(i)
			rowsToDelIssueId.append
		
		if not data1[i, colIndices[COL_NAME_ISSUE_ID]] in list2DAllTickets[COL_NAME_ISSUE_ID]:
			if debug_info:
				print("--- INFO: Removing the row with as it is not in ", fileName, ": ", data1[i, colIndices[COL_NAME_ISSUE_ID]])
			
			if not i in rowsToDel:
				rowsToDel.append(i)
			else:
				if debug_info:
					print("--- WARNING: already marked for delete: ", i)

		if data1[i, colIndices[COL_NAME_TYPE]] != "BUG":
			if debug_info:
				print("--- INFO: Removing the row with as it is not a BUG ", fileName, ": ", data1[i, colIndices[COL_NAME_ISSUE_ID]], ", ", data1[i, colIndices[COL_NAME_TYPE]])
			
			if not i in rowsToDel:
				rowsToDel.append(i)
			else:
				if debug_info:
					print("--- WARNING: already marked for delete: ", i)
	
	#	Remove filtered out rows.
	
	data1 = np.delete(data1, rowsToDel, 0)		
	
	if debug:
		print(data1[:,colIndices[COL_NAME_STATUS]])
		print(data1)
	
	#	Construct bugs in priority order for output.
	
	print("Bugs in ", currFileName, ": ", len(data1[:, 0]))
	priority_bugs_from_hotlist += len(data1[:, 0])
	
	for i in range(0, PRIORITY_LOWEST):
		priority_index='P' + str(i)
		print(priority_index, ": ", list(data1[:, colIndices[COL_NAME_PRIORITY]]).count(priority_index))

	# 	Once all invalid issues aforementioned above are filtered out, export to 2-d array from np array.
		
	for i in range(0, len(listColumns)):
		list2DHotList[listColumns[i]] += list(data1[:,colIndices[listColumns[i]]])
		
	dict2DHotListIssueIds[currFileName]= list(data1[:,colIndices[COL_NAME_ISSUE_ID]])
	
if debug:
	print("dict2DHotListIssueIds: ", dict2DHotListIssueIds)

for i in (list(dict2DHotListIssueIds.keys())):
	list2DHotListAll+=(list(dict2DHotListIssueIds[i]))

if debug:
	print("list2DHotListAll: ", list2DHotListAll)	

for i in list2DHotListAll:
	if list2DHotListAll.count(i) > 1:
		if not i in list2DHotListDups:
			list2DHotListDups.append(i)
	
if list2DHotListDups:
	print("Found following duplicate IDs: ", list2DHotListDups)
else:
	print("No duplicates found in host list: ")

if priority_bugs_from_hotlist != len(priority_bugs):
	print("WARNING!!!: Total bugs gathered from hotlist does not match the bugs in ", fileName)

print("Total bugs gathered from hotlist file: ", priority_bugs_from_hotlist)
print("Total bugs gathered from ", fileName, ": ",len(priority_bugs))

if priority_bugs_from_hotlist != len(priority_bugs):
	time.sleep(10)
else:
	time.sleep(1)
		
printBarSingle()

#	Construct mismatch list. The list contains any ticket that is not assigned to any of the hotlist.
#	For that loop will iterate through all tickets and then iterate through list2DHotList which contains
#	cumulative list of all tickets assigned to hotlist. 
#	Therefore: list2DMisMatchList = list2DAllTickets - list2DHotList.

list2DMisMatchList={}

for i in range(0, len(listColumns)):
	list2DMisMatchList[listColumns[i]] = []

for i in range(0, len(list2DAllTickets[COL_NAME_PRIORITY])):
	if not list2DAllTickets[COL_NAME_ISSUE_ID][i] in list2DHotList[COL_NAME_ISSUE_ID] and list2DAllTickets[COL_NAME_TYPE][i] == "BUG":
		for j in range(0, len(listColumns)):
			try:
				if listColumns[j] in list2DAllTickets.keys():
					list2DMisMatchList[listColumns[j]].append(list2DAllTickets[listColumns[j]][i])	
				else:
					print("Skipping to append: ", listColumns[j])
			except Exception as msg:
				print("Error: Can not append: ", list2DAllTickets[j][i])
				continue

#	Print the list of mismatched tickets.
				
print("Mismatch issue ID not assigned to hot list: ")

if len(list2DMisMatchList[COL_NAME_ISSUE_ID]):
	for i in range(0, len(list2DMisMatchList[COL_NAME_ISSUE_ID])):
		print(list2DMisMatchList[COL_NAME_ISSUE_ID][i], ", ", list2DMisMatchList[COL_NAME_TYPE][i], ", ", list2DMisMatchList[COL_NAME_STATUS][i], ", ", list2DMisMatchList[COL_NAME_CREATED_TIME][i], ", ", list2DMisMatchList[COL_NAME_MODIFIED_TIME][i], ", ", \
		list2DMisMatchList[COL_NAME_TITLE][i][0:50])
else:
	print("None.")

	printBarSingle()

# 	List of tickets opened last 7 days.

datetimeToday=datetime.today()	

if debug:
	print("Today's date: ", datetimeToday)

print("Displaying tickets opened last 7 days")

if debug:
	print(len(data[:,colIndicesMain[COL_NAME_ISSUE_ID]]))

list2DTicketsRecent7days={}

for i in range(0, len(listColumns)):
	list2DTicketsRecent7days[listColumns[i]] = []

issueIdsRecent=[]

for i in range(0, len(data[:,colIndicesMain[COL_NAME_ISSUE_ID]])):
	currTicketDate=data[i,colIndicesMain[COL_NAME_CREATED_TIME]]
	print("CurrTicketDate: ", currTicketDate)
	
	try:
		datetimeCurrTicket=datetime.strptime(currTicketDate, '%m/%d/%Y %H:%M')
	except Exception as msg:
		datetimeCurrTicket=datetime.strptime(currTicketDate, '%Y-%m-%d %H:%M:%S')

	delta=(datetimeToday-datetimeCurrTicket).days
	
	if debug:
		print(currTicketDate)
		print(i, ". date: ", currTicketDate, " datetime rep: ", datetimeCurrTicket, data[i,colIndicesMain[COL_NAME_ISSUE_ID]])
		print("delta from today: ", delta)
		printBarSingle()
	
	if  delta < 8:
		print("Adding to recently opened tickets: ", data[i,colIndicesMain[COL_NAME_ISSUE_ID]], delta)
		issueIdsRecent.append(data[i,colIndicesMain[COL_NAME_ISSUE_ID]])

for i in range(0, len(list2DAllTickets[COL_NAME_ISSUE_ID])):
		
		if debug:
			print(list2DAllTickets[COL_NAME_ISSUE_ID][i])
			
		if 	list2DAllTickets[COL_NAME_ISSUE_ID][i] in issueIdsRecent:
			print("Found in issueIdsRecent: ", list2DAllTickets[COL_NAME_ISSUE_ID][i])
			
			for j in range(0, len(listColumns)):
				try:
					if list2DAllTickets[COL_NAME_ISSUE_ID][i] in issueIdsRecent:
						list2DTicketsRecent7days[listColumns[j]].append(list2DAllTickets[listColumns[j]][i])	
				except Exception as msg:
					print("Error: Can not append: ", list2DAllTickets[j][i])
					continue
print(list2DTicketsRecent7days)
					
#if sum(list2DTicketsRecent7days[COL_NAME_ISSUE_ID]):
#	print(list2DTicketsRecent7days)
#else:
#	print("None.")








