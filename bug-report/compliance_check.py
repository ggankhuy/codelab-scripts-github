import sys
import csv
import glob
import os 
import time

import numpy as np
from common import *
from numpy import *
from datetime import datetime, timedelta

debug = 1

# Open internal jira
# Open issues.csv (buganizer)

fileNameJiraInternal="internal_jira.csv"
fileNameBuganizer="issues.csv"

# 	Open internal jira export csv and external jira (buganizer) issues.

with open(fileNameJiraInternal) as f:
	reader = csv.reader(f, delimiter=',')
	headersJiras = list(next(reader))
	jiras = list(reader)
	np_jiras=np.array(jiras)
		
with open(fileNameBuganizer) as f:
	reader = csv.reader(f, delimiter=',')
	headersBuganizerIssues = list(next(reader))
	buganizerIssues = list(reader)
	np_buganizerIssues=np.array(buganizerIssues)
		
if debug:
	print(type(jiras))
	print(type(buganizerIssues))
	print(type(np_jiras))
	print(type(np_buganizerIssues))
	
# 	Iterate through all bus buganizer issues and search corresponding internal jira.
#		Check to see internal jira opened
#		1. Search ID in title, if found OK.
#		2. Search title string, if found OK.
#		3. If 1. and 2. fails output error.































