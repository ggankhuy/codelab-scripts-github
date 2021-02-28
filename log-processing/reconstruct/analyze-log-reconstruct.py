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
FILE_NAME_TEST_STRING="test-string-3.txt"
MAX_CHAR_PER_LINE=120
DEBUG = 0
THRESHOLD_MIN_TOKEN_SET_RATIO=90
cmds=[]

try:
    matchStringBlock=open(FILE_NAME_MATCH_STRING)
    testString=open(FILE_NAME_TEST_STRING)
except Exception as msg:
    print("Failure opening file...")
    print(msg)
    quit(1)

TEST_MODE=1
lastLineGimInit=0
cursorTestFile=0
counter=0
testStringBlockContent=testString.readlines()
cursorTestFile=len(testStringBlockContent)
print("No. of lines read from testFile: ", cursorTestFile)
testStringBlockContentProcessed=None

for i in reversed(testStringBlockContent):
    if re.search("AMD GIM is Running", i):
        print("Last line gim is finished initialized last time in this log: Line No.: ", str(cursorTestFile), str(i))
        lastLineGimInit = cursorTestFile
        break
    cursorTestFile -= 1
    
testStringBlockContentProcessed=testStringBlockContent[cursorTestFile:]

counter = 0
print("Printing first few lines of truncated string block:")
for i in testStringBlockContentProcessed:
    print(i[0:MAX_CHAR_PER_LINE], "...")
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

matchStringBlockContent=matchStringBlock.readlines()
print
for i in matchStringBlockContent:
    # If line is debug, then, signal new match string block.

    if DEBUG: 
        print("currLine:", i)
        
    if re.search("DEBUG: ", i):

        # encountered next match string block. Add to the dictionary if it is not first occurrence.
        # Because if it is first occurrence, search just started and nothing to add.

        if currKey and currValue:
            print("currKey: ",currKey)
            dictmatchStringBlock[currKey] = currValue

        # Reset the currValue to empty and currKey to new key found.

        currKey=re.sub("DEBUG: ", "", i).strip()
        if DEBUG:
            print("currKey generated: ", currKey)
        currValue=[]
    else:
    
        # If not DEBUG line means, it is currValue.

        if i.strip():
            currValue.append(i.strip())
            if DEBUG:
                print("currValue so far: ", currValue)

#   Do this for last entry because, the dictionary assignment is at the start of loop code block.

if currKey and currValue:
    if DEBUG:
        print("currKey: ",currKey)
    dictmatchStringBlock[currKey] = currValue        
    
keys=list(dictmatchStringBlock.keys())
values=list(dictmatchStringBlock.values())

if  DEBUG:
    print("key size/type: ", len(keys), ", ", type(keys))
    print("values size/type: ", len(values), ", ", type(values))

for i in range(0, len(keys)):
    print("i: ", i)
    print("key: ", keys[i])
    if DEBUG:
        print("value: ")
        for j in values[i]:
            print(j)
        print("")

#   Outer loop: Start from test string line by line.
#   Inner loop: iterate through dictionary.
#   for each dict values, determine No. of lines 
#   Grab next No. of lines from test string.

print("Starting match loop...")

LinesToSkip=0
for cursorTestString in range(0, len(testStringBlockContentProcessed)):

    if TEST_MODE:
        if cursorTestString > 100:
            break

    if LinesToSkip-1 > 0:
        print("Skipping line: ", str(cursorTestString))
        LinesToSkip-=1
        continue
    match_found=None
    print("****** iter: cursorTestString: ", str(cursorTestString), "******")
    currTestBlockFirstLine=testStringBlockContentProcessed[cursorTestString:cursorTestString+2]
    for i in currTestBlockFirstLine:
        print(i)

    for i in list(dictmatchStringBlock.keys()):
        currValue=dictmatchStringBlock[i]

        # Print information.

        print("curr dict key:currValue/len:")
        print(i)

        if DEBUG:
            for j in currValue[0:3]:
                print(j[0:MAX_CHAR_PER_LINE], "...")
            print(len(currValue))
        lines=len(currValue)

        # Construct match block.

        currTestBlock=testStringBlockContentProcessed[cursorTestString:cursorTestString+lines]

        if DEBUG:
            print("currTestBlock/len:")
            for k in currTestBlock[0:2]:
                print("...", k[80:MAX_CHAR_PER_LINE+40], "...")
            print(len(currTestBlock))

        # Concatenate string, both match and test string blocks.

        match_string_concat=' '.join(currValue)
        test_string_concat=' '.join(currTestBlock)

        if DEBUG:
            print("match_string_concat: \n", match_string_concat[0:80])
            print("test_string_concat: \n", test_string_concat[0:80])

        token_set_ratio=fuzz.token_set_ratio(match_string_concat, test_string_concat)
        print("match percent (token_set_ratio): ", str(token_set_ratio))

        # If match is above threshold, then move cursor forward in test string same number of lines as current match string block.
        # and break out of inner loop. 
        # (Break out of inner loop could be problematic in situation if two entries in match string blocks are very similar)
        # Consider adding feature to iterate through every entry in the dictionary, gather match ratio for each on current window
        # and pick maximum.

        if token_set_ratio > THRESHOLD_MIN_TOKEN_SET_RATIO:
            print("Match!")
            cmds.append(i)
            match_found=1
            LinesToSkip=len(currTestBlock)
            break

    # If match found is none. Add ? along with cursor Number to its cmds list.  

    if not match_found:
        print("Match is not found for line[LineNo:]: ", "[", str(cursorTestString), "]", str(test_string_concat[0:80]))
        cmds.append("? : LineNo: " + str(lastLineGimInit + cursorTestString))
        LinesToSkip=0
#    time.sleep(1)


print("*** PRINTING CMDS: ***")
for i in cmds:
    print(i)
