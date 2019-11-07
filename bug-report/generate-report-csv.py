import sys
fileName=None

import numpy as np
from numpy import *
import csv
headers=None
debug=0

try:
	fileName=sys.argv[1]
except Exception as msg:
	print("Failed to find p1.")
	print(msg)
	quit(1)
	
fp=open(fileName)

if not fp:
	print("Failed to open file.")
	quit(1)
	
with open(fileName) as f:
	reader = csv.reader(f, delimiter=',')
	#headers = next(reader)
	data = list(reader)
	data=np.array(data)

if debug:
	print("data dimension: ", data.shape)
	print("data type: ", type(data))
	print("------------------")

priority=list(data[:,1])

if debug:
	print(type(priority), priority)

print("Total tickets:")
for i in range(0, 7):
	priority_index='P' + str(i)
	print(priority_index, ": ", priority.count(priority_index))
	


	
	

	