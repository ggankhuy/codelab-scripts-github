import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import time
DOUBLE_BAR="====================================================================================="
SINGLE_BAR="-------------------------------------------------------------------------------------"
from datetime import datetime, timedelta
datetimeToday=datetime.today()

DEBUG=1
DEBUGL2=0
DEBUG_COL_MISMATCH_TEST=0
BUGANIZER_OPEN_STATUSES=['ASSIGNED', 'ASSIGNED']
fileNameJira='INTERNAL 2020-10-10T01_44_07-0500.csv'

# There are still problem with when last of fileNamesBuganizer has mismatched column. Ideally
# all files in the list to be tested individually. Current validation is conjoined list is being tested.

if DEBUG_COL_MISMATCH_TEST:
	fileNameJira='INTERNAL 2020-10-10T01_44_07-0500-bad-col.csv'
	fileNamesBuganizer=['issuesv10-bad-col.csv', 'issuesn12.csv']
	fileNamesBuganizer=['issuesv10.csv', 'issuesn12-bad-col.csv']
else:
	fileNameJira='INTERNAL 2020-09-07T14_15_52-0500.csv'
	fileNamesBuganizer=['issuesv10.csv', 'issuesn12.csv']

# Column name definitions for exports from Jira(internal_jira)/Buganizer(external_jira)
# Code will either break or need adjustment by changing IDX_COL_JIRA_<COLUMN_NAME> for jira export or
# IDX_COL_BUG_<COLUMN_NAME> for buganizer export. 

IDX_COL_JIRA_ISSUE_KEY=0
IDX_COL_JIRA_SUMMARY=2
IDX_COL_JIRA_CLOSED_DATE=8
IDX_COL_JIRA_OPENED_DATE=9
IDX_COL_JIRA_REJECTED_DATE=10
JIRA_DATA_COLUMNS=['Issue key', 'Issue id', 'Summary', 'Labels', 'Labels', 'Labels', 'Labels', 'Labels', 'Custom field (Closed Date)', 'Created', 'Custom field (Rejected Date)']
BUG_DATA_COLUMNS=['POSITION', 'PRIORITY', 'TYPE', 'TITLE', 'ASSIGNEE', 'STATUS', 'ISSUE_ID', 'CREATED_TIME (UTC)', 'MODIFIED_TIME (UTC)']
IDX_COL_BUG_STATUS=5
IDX_COL_BUG_ID=6

with open(fileNameJira, 'r') as f:
    jiraData = list(csv.reader(f, delimiter=','))
jiraDataDates=[]
	
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

jiraDataColumns=jiraData[0]
print("jiraDataColumns:")
print(jiraDataColumns)
bugDataColumns=bugData[0]
print("bugDataColumns:")
print(bugDataColumns)

if DEBUGL2:
	input("..."	)
	
print("Validaing column labels...")		

for i in range(0, len(JIRA_DATA_COLUMNS)):
	if JIRA_DATA_COLUMNS[i] != jiraDataColumns[i]:
		print(DOUBLE_BAR)
		print("Error: imported file(", fileNameJira, ")'s column labels are not matching: ")
		print("{0:<30}".format("Expected"), "{0:<30}".format("Read("+fileNameJira+"):"))
		print(SINGLE_BAR)		
		for j in range(0, max(len(JIRA_DATA_COLUMNS), len(jiraDataColumns))):
			try:
				if JIRA_DATA_COLUMNS[j].strip() != jiraDataColumns[j].strip():
					print("-->", "{0:<30}".format(JIRA_DATA_COLUMNS[j]), "{0:<30}".format(jiraDataColumns[j])) 
				else:
					print("{0:<30}".format(JIRA_DATA_COLUMNS[j]), "{0:<30}".format(jiraDataColumns[j])) 				
			except Exception as msg:
				try:
					print("-->", "{0:<30}".format(JIRA_DATA_COLUMNS[j])) 
				except Exception as msg:
					print("-->", "{0:<30}".format(jiraDataColumns[j])) 
				
		print(DOUBLE_BAR)
		exit(1)


for fileNameBuganizer in fileNamesBuganizer:
	with open(fileNameBuganizer, 'r') as f:
		bugDataTmp=list(csv.reader(f, delimiter=','))

	bugDataColumns=bugDataTmp[0]
	for i in range(0, len(BUG_DATA_COLUMNS)):
		print(i, ": ", BUG_DATA_COLUMNS[i], bugDataColumns[i])		
		if BUG_DATA_COLUMNS[i] != bugDataColumns[i]:
			print(DOUBLE_BAR)
			print("Error: imported file(", str(fileNameBuganizer), ")'s column labels are not matching: ")
			print("{0:<30}".format("Expected"), "{0:<30}".format("Read("+str(fileNamesBuganizer)+"):"))
			print(SINGLE_BAR)		
			for j in range(0, max(len(BUG_DATA_COLUMNS),len(bugDataColumns))):
				try:
					if BUG_DATA_COLUMNS[j] != bugDataColumns[j]:
						print("-->", "{0:<27}".format(BUG_DATA_COLUMNS[j]), "{0:<30}".format(bugDataColumns[j])) 
					else:
						print("{0:<30}".format(BUG_DATA_COLUMNS[j]), "{0:<30}".format(bugDataColumns[j])) 					
				except Exception as msg:
					try:
						print("-->", "{0:<30}".format(BUG_DATA_COLUMNS[j])) 
					except Exception as msg:
						print("-->", "{0:<30}".format(bugDataColumns[j])) 
									
			print(DOUBLE_BAR)
			exit(1)
input("..")

		
# Iterate through jira bug, get the title column and try extracting the gibraltar issue id. 

jiraDataIssueIdsBuganizer=[]
jiraDataIssueIds=[]
jiraListIllegaSummaryTitle=[]

jiraListUnclosedTickets=[]

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

if DEBUGL2:
	input("Press Enter to continue...") 
	
print(DOUBLE_BAR)
print("len of jiradata: ", len(jiraData))
print("Len of jiraDataIssueIdsBuganizer: ", len(jiraDataIssueIdsBuganizer))
print("len of jiraDataIssueIds:", len(jiraDataIssueIds))
print(DOUBLE_BAR)

if DEBUGL2:
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
				if DEBUG:
					print("Found matching issue ID in buganizer: ", str(j[IDX_COL_BUG_ID]))
				
				# Now real check if the jira id is not closed (closed data is empty) and buganizer status is NOT CLOSED.
				
				if not j[IDX_COL_BUG_STATUS] in BUGANIZER_OPEN_STATUSES and jiraData[i][IDX_COL_JIRA_CLOSED_DATE].strip()=='' and  jiraData[i][IDX_COL_JIRA_REJECTED_DATE].strip()=='':
					if DEBUG:
						print("Following jira is not closed: ", jiraData[i][IDX_COL_JIRA_ISSUE_KEY], ", Buganizer ID / status: ", j[IDX_COL_BUG_ID], "/", j[IDX_COL_BUG_STATUS])
						
					jiraListUnclosedTickets.append([jiraData[i][IDX_COL_JIRA_ISSUE_KEY], j[IDX_COL_BUG_ID], j[IDX_COL_BUG_STATUS]])
					
				continue
		if DEBUGL2:
			print("Unable to find matching issue ID in buganizer for: ", str(jiraDataIssueIdsBuganizer[i]))
	else:
		if DEBUGL2:
			print("Skipping ", jiraDataIssueIds[i], " as it is empty.")

print(DOUBLE_BAR)
print("Jira tickets that are not closed (but closed on buganizer): ")
for i in jiraListUnclosedTickets:
		print(i)
print(DOUBLE_BAR)

