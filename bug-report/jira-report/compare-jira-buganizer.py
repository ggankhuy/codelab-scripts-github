'''
#
# Copyright (c) 2014-2019 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

'''
'''
compare-jira-buganizer.py:
The script compares two exported csv files, one from buganizer and one from jira to locate the jira tickets that were not closed 
but closed in buganizer. 
Brief summary of how functionally this script does it:
- open both files and read contents into 2-D list. The column orders of each csv are compared against pre-defined expected column list 
to avoid reading wrong columns.
- read through summary column from jira export and extract 9 digit buganizer ID. For each buganizer ID extracted from each row, it scans the buganizer list for matching ticket in nested loop. Once match is found and if buganizer ticket is not in open state and jira is still open, it 
will add to list of tickets that are not opened. 
'''
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import time
import os
DOUBLE_BAR="====================================================================================="
SINGLE_BAR="-------------------------------------------------------------------------------------"
from datetime import datetime, timedelta
datetimeToday=datetime.today()
datetimeString=datetimeToday.strftime("%m-%d-%Y-%H-%M-%S")
DEBUG=1
DEBUGL2=0
DEBUG_COL_MISMATCH_TEST=0
BUGANIZER_OPEN_STATUSES=['ASSIGNED', 'ACCEPTED']
OUTPUT_PATH="output"
TEXT_EDITOR="notepad++"

# 	Define filenames to be opened. 

if DEBUG_COL_MISMATCH_TEST:
	fileNameJira='INTERNAL 2020-10-10T01_44_07-0500-bad-col.csv'
	fileNamesBuganizer=['issuesv10-bad-col.csv', 'issuesn12.csv']
	fileNamesBuganizer=['issuesv10.csv', 'issuesn12-bad-col.csv']
else:
	fileNameJira='jira-gibraltar.csv'
	fileNamesBuganizer=['issuesv10.csv', 'issuesn12.csv']

# Column label designation for exports from Jira(internal_jira)/Buganizer(external_jira)
# Code will either break or need adjustment by changing IDX_COL_JIRA_<COLUMN_NAME> for jira export or
# IDX_COL_BUG_<COLUMN_NAME> for buganizer export. 

IDX_COL_JIRA_ISSUE_KEY=0
IDX_COL_JIRA_SUMMARY=2
IDX_COL_JIRA_CLOSED_DATE=10
IDX_COL_JIRA_OPENED_DATE=11
IDX_COL_JIRA_REJECTED_DATE=12
IDX_COL_JIRA_ASSESSED_DATE=16
IDX_COL_JIRA_ANALYZED_DATE=17

JIRA_DATA_COLUMNS=['Issue key', 'Issue id', 'Summary', 'Labels', 'Labels', 'Labels', 'Labels', 'Labels','Labels','Labels', 'Custom field (Closed Date)', 'Created', 'Custom field (Rejected Date)', 'Custom field (Triage Category)', 'Custom field (Triage Assignment)', 'Priority', 'Custom field (Assessed Date)',	'Custom field (Analyzed Date)']

BUG_DATA_COLUMNS=['POSITION', 'PRIORITY', 'TYPE', 'TITLE', 'ASSIGNEE', 'STATUS', 'ISSUE_ID', 'CREATED_TIME (UTC)', 'MODIFIED_TIME (UTC)']
IDX_COL_BUG_STATUS=5
IDX_COL_BUG_ID=6

# 	Prepare output directory.

try:
	os.mkdir(OUTPUT_PATH)
except Exception as msg:
	print("directory or file already exist.")
	
print(datetimeString)
OUTPUT_PATH=OUTPUT_PATH+"\output-"+datetimeString+".log"
print(OUTPUT_PATH)

fout=open(OUTPUT_PATH, 'a+')
if not fout:
	print("Error: can not open file ", OUTPUT_PATH, " for writing.")
	time.sleep(3)
	
# 	Open jira issues.

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
bugDataColumns=bugData[0]

if DEBUG:
	print("jiraDataColumns:")
	print(jiraDataColumns)
	print("bugDataColumns:")
	print(bugDataColumns)

if DEBUGL2:
	input("..."	)

# 	Validate column labels. 

print("Validaing column labels...")	

# 	Iterate through each jira column.

for i in range(0, len(JIRA_DATA_COLUMNS)):

	# Throw error if not matching the pre-defined labels. 

	if JIRA_DATA_COLUMNS[i] != jiraDataColumns[i]:
		print(DOUBLE_BAR)
		print("Error: imported file(", fileNameJira, ")'s column labels are not matching: ")
		print("{0:<30}".format("Expected"), "{0:<30}".format("Read("+fileNameJira+"):"))
		print(SINGLE_BAR)		
		
		#	Print each of pre-defined and imported data columns for eye inspection. The nested exception is for taking care of 
		# 	situation where there are unequal number of columns. 
		
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

# 	Iterate through each buganizer import column. Logics are exactly same as for jira column validation
#	except outer loop iterates over multiple import files. 

#	For each buganizer import files.

for fileNameBuganizer in fileNamesBuganizer:
	with open(fileNameBuganizer, 'r') as f:
		bugDataTmp=list(csv.reader(f, delimiter=','))

	# 	Assign first row containing column labels.
	
	bugDataColumns=bugDataTmp[0]
	
	# Throw error if not matching the pre-defined labels. 

	for i in range(0, len(BUG_DATA_COLUMNS)):
		print(i, ": ", BUG_DATA_COLUMNS[i], bugDataColumns[i])		
		if BUG_DATA_COLUMNS[i] != bugDataColumns[i]:
			print(DOUBLE_BAR)
			print("Error: imported file(", str(fileNameBuganizer), ")'s column labels are not matching: ")
			print("{0:<30}".format("Expected"), "{0:<30}".format("Read("+str(fileNamesBuganizer)+"):"))
			print(SINGLE_BAR)		
			
			#	Print each of pre-defined and imported data columns for eye inspection. The nested exception is for taking care of 
			# 	situation where there are unequal number of columns. 

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
		
jiraDataIssueIdsBuganizer=[]
jiraDataIssueIds=[]
jiraListIllegaSummaryTitle=[]
jiraOutput=[]

# Iterate through jira bug, get the title column and try extracting the gibraltar issue id. 

for i in range(0, len(jiraData)):
	print(SINGLE_BAR)
	issueId=None

	if DEBUG:
		print(i, "Summary: ", jiraData[i][IDX_COL_JIRA_SUMMARY])
		
	#	get summary column and assign and tokenize.
	
	summary=jiraData[i][IDX_COL_JIRA_SUMMARY].split()
	
	# 	for each token (split by space), verify it is 9-digit buganizer ID, if not found, throw error.
	#	Either way, found or not, update the list. 
	
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

#	Print out sanity comparison of each list. 
	
print(DOUBLE_BAR)
print("len of jiradata: ", len(jiraData))
print("Len of jiraDataIssueIdsBuganizer: ", len(jiraDataIssueIdsBuganizer))
print("len of jiraDataIssueIds:", len(jiraDataIssueIds))
print(DOUBLE_BAR)

if DEBUGL2:
	input("Press Enter to continue...") 

#	Print out the buganizer tickets whose summary did not contain buganizer IDs. 

print("Following jira summary has illegal title, unable to find buganizer ID-s:")		
for i in jiraListIllegaSummaryTitle:
	print(i)

print(DOUBLE_BAR)

#	Iterate over each buganizer ID extracted from jira summary. 

for i in range(0,len(jiraDataIssueIdsBuganizer)):

	# If valid (able to locate buganizer ID).
	
	if jiraDataIssueIdsBuganizer[i]:
	
		# Now search through each buganizer tickets imported for match.
				
		for j in bugData:
			
			if DEBUGL2:	
				print("Checking if ", jiraDataIssueIdsBuganizer[i], " matches ", j)
				
			if j[IDX_COL_BUG_ID].strip() == str(jiraDataIssueIdsBuganizer[i]):
				if DEBUG:
					print("Found matching issue ID in buganizer: ", str(j[IDX_COL_BUG_ID]))
				
				# Now real check if the jira id is not closed (closed data is empty) and buganizer status is NOT CLOSED.
				
				if not j[IDX_COL_BUG_STATUS] in BUGANIZER_OPEN_STATUSES and jiraData[i][IDX_COL_JIRA_CLOSED_DATE].strip()=='' and  jiraData[i][IDX_COL_JIRA_REJECTED_DATE].strip()=='':
					if DEBUG:
						print("Following jira is not closed: ", jiraData[i][IDX_COL_JIRA_ISSUE_KEY], ", Buganizer ID / status: ", j[IDX_COL_BUG_ID], "/", j[IDX_COL_BUG_STATUS])
						
					#	If not closed, update the list, will print out this list in the end.
						
					jiraOutput.append([jiraData[i][IDX_COL_JIRA_ISSUE_KEY], j[IDX_COL_BUG_ID], j[IDX_COL_BUG_STATUS]])
					
				continue
		if DEBUGL2:
			print("Unable to find matching issue ID in buganizer for: ", str(jiraDataIssueIdsBuganizer[i]))
	else:
		if DEBUGL2:
			print("Skipping ", jiraDataIssueIds[i], " as it is empty.")

#	Print the final list.

print(DOUBLE_BAR)
print("Jira tickets that are not closed (but closed on buganizer): ")
fout.write(DOUBLE_BAR+'\n')
fout.write("Jira tickets that are not closed (but closed on buganizer): "+'\n')

for i in jiraOutput:
		print(i)
		fout.write(str(i)+'\n')
print(DOUBLE_BAR)
fout.write(DOUBLE_BAR+'\n')
fout.close()
print("output file: ")
print(OUTPUT_PATH)
os.system(TEXT_EDITOR + " " + OUTPUT_PATH)