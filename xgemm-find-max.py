#!/usr/bin/python3
import sys
import os

try:
	filename=sys.argv[1]
except Exception as msg:
	print(msg)
	print("Usage: ", sys.argv[0], " <csv filename containing xgemm scores>")
	quit(1)

print("OK") 
	

fp=open(filename)

if not fp:
	print("Failed to open a file: ", fileName)
	quit(1)

lines=fp.readlines()

score_max=0
score_max_matrix_size=0

for i in lines:
	print(i)

	try:
		score_curr=float(i.split(',')[-1])
	except Exception as msg:
		print(msg)
		continue
	score_curr_matrix_size=i.split(',')[0]
	if score_curr > score_max:
		score_max = score_curr
		score_max_matrix_size=score_curr_matrix_size
		print("Max score updated to: ", score_max, " at matrix size: ", score_curr_matrix_size)

print("Final max score / matrix size: ", score_max, " / ", score_max_matrix_size)


