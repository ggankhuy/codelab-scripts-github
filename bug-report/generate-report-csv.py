import sys
fileName=None

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
	#headers = next(reader)
	data = list(reader)
	data=np.array(data)

	f.close()
	
if debug:
	print("data dimension: ", data.shape)
	print("data type: ", type(data))
	print("------------------")

# 	Extract priorit column and count priorities and display.
	
priority=list(data[:,1])
type=list(data[:,2])

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

for currFileName in fileList:
	print("------------------")
	print(currFileName)

	with open(currFileName) as f1:
		reader = csv.reader(f1, delimiter=',')
		#headers = next(reader)
		data1 = list(reader)
		data1=np.array(data)
	
	if debug or 1:
		print("data dimension: ", data1.shape)
		#print("data type: ", type(data1))
		print("------------------")

	# 	Extract priorit column and count priorities and display.
		
	priority=list(data1[:,1])
	type=list(data1[:,2])
		
		

	
	

	

	


	
	

	