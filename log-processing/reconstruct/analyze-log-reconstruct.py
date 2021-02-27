# Script for scanning the gim/libgv log file and attempt to find matching pattern from log.
# The gim log resulting from issuing certain activity on host or gim is kept as list of pattern.
# The pattern is a dictionary element with 
# key = gpuvsmi command or something similar
# value = matching pattern. The matching pattern can be several lines.

# Matchin dictionary:
# Key 1 = gpuvsmi_reset_gpu
# Value 1 = <libgv log entries of one or more lines resulting from issuing gpuvsmi_reset_gpu>
# Key 2 = gpuvsmi_allocate_vf
# Value 2 = <libgv log entries of one or more lines resulting from issuing gpuvsmi_allocate_vf>
# ...  and so on.

# Because of the variance of factors, matching pattern can not be relied to exact but only approximate.
# To "match" we establish threshold i.e. 80% of the word in dictionary element expected to match.
# The current dictionary element of matching pattern is scanned against the log the same number of lines.
# If threshold is passed, we deduce it is a match and set gpuvsmi or similar activity is candidate.
# If threshold does not pass, we deduce it is not match and loop over next  dictionary element.
# if all the dictionary element passes and nothing match, we move onto next line in the log and start matching.

# Command list: the specific gpuvsmi command is added to this list after pattern match.
# Mismatch list: lines that did not match any pattern in the dictionary values. 

# Actionable code (pseudocode):
# - open log files (libgv or gim)
# - open log files and build dictionary
# - Point to one line past after the line which indicates GIM is initialized.
# - Construct dictionary.
# - start loop:
# - start scanning each line.
# - for each dictionary element
# - - scan and match N-number of lines from libgv log where N is the same as number of line in current dict. value.
# - - if match exceeds threshold, 
# - - - Pick up key and add to command list.
# - - - Move cursor to the line just after the matched lines in the log.
# - - if all the dictionary pattern did not match the with the cursor at current log lines, 
# - - - add this line to mismatch list.
# - - - move cursor one line forward to next line.

import re
import time
from fuzzywuzzy import fuzz

FILE_NAME_MATCH_STRING="match-string.txt"
FILE_NAME_TEST_STRING="test-string.txt"
FILE_NAME_TEST_STRING="test-string-2.txt"
try:
    matchStringBlock=open(FILE_NAME_MATCH_STRING)
    testString=open(FILE_NAME_TEST_STRING)
except Exception as msg:
    print("Failure opening file...")
    print(msg)
    quit(1)

lastLineGimInit=0
cursorTestFile=0
counter=0
testStringBlockContent=testString.readlines()
cursorTestFile=len(testStringBlockContent)
print("No. of lines read: ", cursorTestFile)
testStringBlockContentProcessed=None

for i in reversed(testStringBlockContent):
    if re.search("AMD GIM is Running", i):
        print("Last line gim is finished initialized last time in this log: line: ", str(cursorTestFile), str(i))
        lastLineGimInit = cursorTestFile
        break
    cursorTestFile -= 1
    
testStringBlockContentProcessed=testStringBlockContent[cursorTestFile:]

if counter == 0:
    print("The log does not appear to have gim initialization log.")


counter = 0
print("Printing first few lines of truncated string block:")
for i in testStringBlockContentProcessed:
    print(i)
    counter += 1
    if counter > 5:
        break

# Construct dictionary.

'''
DEBUG: gpuvsmi_set_num_vf_enabled
[amdgv_gpumon_handle_sched_event:1488] process GPUMON event GPUMON_SET_VF_NUM (type:26)
[amdgv_vfmgr_set_vf_num:707] Set enabled VF number to 1
'''

print("Constructing dictionary for match string blocks.")
dictmatchStringBlock={}

currKey=None
currValue=None

testStringBlockContent=matchStringBlock.readlines()
for i in testStringBlockContent:
    print("type/content: ", type(i), i)
    # If line is debug, then, signal new match string block.

    if re.search("DEBUG: ", i):

        # encountered next match string block. Add to the dictionary if it is not first occurrence.
        # Because if it is first occurrence, search just started and nothing to add.

        if currKey and currValue:
            print("currKey",currKey)
            dictmatchStringBlock[currKey] = currValue

        # Reset the currValue to empty and currKey to new key found.

        currKey=re.sub("DEBUG: ", "", i).strip()
        print("currKey generated: ", currKey)
        currValue=[]
    else:
    
        # If not DEBUG line means, it is currValue.

        if i.strip():
            currValue.append(i.strip())
            print("currValue so far: ", currValue)
        
    
keys=list(dictmatchStringBlock.keys())
values=list(dictmatchStringBlock.values())

print("key size/type: ", len(keys), ", ", type(keys))
print("values size/type: ", len(values), ", ", type(values))

#for i in range(0, len(keys)):
for i in range(0, 3):
    print("i: ", i)
    print("key: ", keys[i])
    print("value: ", values[i])
    print("")

#   Outer loop: Start from test string line by line.
#   Inner loop: iterate through dictionary.
#   for each dict values, determine No. of lines 
#   Grab next No. of lines from test string.

for o in testStringBlockContentProcessed:
    for i in list(dictmatchStringBlock.keys()):
        currValue=dictmatchStringBlock[i]
        print("currValue/len:")
        print(currValue)
        print(len(currValue))

        time.sleep(10)
