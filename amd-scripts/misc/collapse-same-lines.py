'''
 	the auto test result generates long lines of log with summary is hard to see.
	The grep filter application is hard as even within some of the scripts each loop prints out the 
	test name. 
	This script collapse all those repeated lines and generates summary with  test names, result and
	separator bars.
	 
'''
import sys
import re 

fp = open(sys.argv[1])
 
if not fp:
	print ("can not open ", sys.argv[1])
	quit(1)

lines = fp.read().splitlines()
unique_lines=[]
lines_filtered=[]

for i in lines:
	if re.search("test case:", i):
		print("test case found: ", i)

		if not i in unique_lines:
			lines_filtered.append("-------------------------------------------------------------------------")
			lines_filtered.append(i)
			unique_lines.append(i)

	if re.search("result:", i):
		print("test result found: ", i)
		lines_filtered.append(i)

print ("lines:")

for i in lines_filtered:
	print i
