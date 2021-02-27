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

FILE_NAME_MATCH_STRING="match-string.txt"
FILE_NAME_TEST_STRING="test-string.txt"
FILE_NAME_TEST_STRING="test-string-2.txt"
try:
    matchString=open(FILE_NAME_MATCH_STRING)
    testString=open(FILE_NAME_TEST_STRING)
except Exception as msg:
    print("Failure opening file...")
    print(msg)
    quit(1)

lastLineGimInit=0

testStringContent=testString.readlines()
counter=len(testStringContent)
print("No. of lines read: ", counter)
testStringContentProcessed=None

for i in reversed(testStringContent):
    if re.search("AMD GIM is Running", i):
        print("Last line gim is finished initialized last time in this log: line: ", str(counter), str(i))
        lastLineGimInit = counter
        break
    counter -= 1
    
testStringContentProcessed=testStringContent[counter:]

if counter == 0:
    print("The log does not appear to have gim initialization log.")


counter = 0
for i in testStringContentProcessed:
    print(i)
    counter += 1
    if counter > 10:
        quit(1)

