import sys
fileName=None

import numpy as np
from numpy import *
import csv

#import panda as pd

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
	
#content=fp.readlines()
#print(content)
#print(type(content))

with open(fileName) as f:
	reader = csv.reader(f, delimiter=',')
	headers = next(reader)
	data = list(reader)
	data=np.array(data)

print(headers)
print(data.shape)
print(data)
#data=pf.read_csv(fileName, sep=',', header=None)
#print(data.values)


	
	

	